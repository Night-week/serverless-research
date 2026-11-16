"""Small, auditable policy-semantic adapters for the R8 preflight.

These functions intentionally do not claim to reproduce the papers' systems.
They make algorithm-level decision contracts testable before the shared event
simulator is connected. Runtime deployment lifecycles, scalable Quilt search,
source-specific workload construction, and named-baseline simulator integration
remain separate implementation gates. The optional GRL-Orch GAT/PPO path is in
``r8_grl_orch.py``; this file retains its score-to-decision adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import combinations, product
from math import ceil
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str  # "sync", "async", or non-call ordering edge
    weight: float = 1.0


@dataclass(frozen=True)
class GroupDecode:
    groups: tuple[tuple[str, ...], ...]
    placement: Mapping[tuple[str, ...], str]
    repair_edges: tuple[tuple[str, str], ...]
    placement_fallbacks: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class RootedBundle:
    """One package candidate in an overlapping rooted DAG cover."""

    root: str
    members: frozenset[str]
    site: str
    resident_mib: float
    cpu_demand: float


@dataclass(frozen=True)
class CostCandidate:
    ident: str
    latency_s: float
    monetary_cost: float
    feasible: bool = True
    edge_to_cloud_only: bool = True
    placement_transitions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostGraphArc:
    src: str
    dst: str
    monetary_cost: float
    latency_s: float
    edge_to_cloud_legal: bool = True


@dataclass(frozen=True)
class QuiltFunctionProfile:
    memory_mib: float
    cpu_units: float


@dataclass(frozen=True)
class QuiltSolution:
    roots: tuple[str, ...]
    groups: tuple[RootedBundle, ...]
    cross_edges: tuple[tuple[str, str], ...]
    cross_weight: float
    root_sets_examined: int
    combinations_examined: int


@dataclass(frozen=True)
class QuiltRoute:
    local_call: bool
    source_root: str
    target_root: str


@dataclass(frozen=True)
class FusionizeSnapshot:
    latency_s: float
    monetary_cost: float


@dataclass(frozen=True)
class ProvuseRollout:
    old_layout: tuple[tuple[str, ...], ...]
    new_layout: tuple[tuple[str, ...], ...]
    evidence_edge: tuple[str, str]
    evidence_at_s: float
    started_at_s: float
    ready_at_s: float
    cutover_at_s: float | None = None
    old_inflight_at_cutover: int | None = None
    old_retired_at_s: float | None = None


@dataclass(frozen=True)
class FunctionConfigScore:
    config: str
    latency_s: float
    monetary_cost: float


@dataclass(frozen=True)
class FusionizePathSearch:
    groups: tuple[tuple[str, ...], ...]
    snapshot: FusionizeSnapshot
    accepted_edges: tuple[tuple[str, str], ...]
    evaluated_candidates: int


def _canonical_components(nodes: Sequence[str], edges: Iterable[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        if left not in parent or right not in parent:
            raise ValueError(f"unknown function in edge {left!r}->{right!r}")
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    grouped: dict[str, list[str]] = {}
    for node in nodes:
        grouped.setdefault(find(node), []).append(node)
    return tuple(sorted(tuple(sorted(group)) for group in grouped.values()))


def provuse_observed_sync_components(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    observed_blocking_edges: set[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    """Contract only synchronous edges for which runtime evidence exists.

    This captures the policy trigger boundary, not Provuse's socket
    interception, image/filesystem merge, gateway update, or health protocol.
    """
    merge_edges = [
        (edge.src, edge.dst)
        for edge in edges
        if edge.kind == "sync" and (edge.src, edge.dst) in observed_blocking_edges
    ]
    return _canonical_components(nodes, merge_edges)


def provuse_begin_rollout(
    active_layout: Sequence[Sequence[str]],
    target_layout: Sequence[Sequence[str]],
    evidence_edge: tuple[str, str],
    evidence_at_s: float,
    decision_at_s: float,
    build_duration_s: float,
) -> ProvuseRollout:
    """Start a deployment only after observed blocking-call evidence exists."""
    if decision_at_s < evidence_at_s:
        raise ValueError("rollout decision cannot precede its observed call evidence")
    if build_duration_s < 0:
        raise ValueError("build duration must be non-negative")
    canonical = lambda layout: tuple(sorted(tuple(sorted(group)) for group in layout))
    return ProvuseRollout(
        old_layout=canonical(active_layout),
        new_layout=canonical(target_layout),
        evidence_edge=evidence_edge,
        evidence_at_s=evidence_at_s,
        started_at_s=decision_at_s,
        ready_at_s=decision_at_s + build_duration_s,
    )


def provuse_cutover(
    rollout: ProvuseRollout, now_s: float, old_inflight_requests: int
) -> ProvuseRollout:
    """Switch only after the new deployment is ready; retain old in-flight work."""
    if rollout.cutover_at_s is not None:
        raise ValueError("rollout has already cut over")
    if now_s < rollout.ready_at_s:
        raise ValueError("new deployment is not ready for cutover")
    if old_inflight_requests < 0:
        raise ValueError("in-flight request count must be non-negative")
    return replace(
        rollout,
        cutover_at_s=now_s,
        old_inflight_at_cutover=old_inflight_requests,
    )


def provuse_layout_for_arrival(
    rollout: ProvuseRollout, request_arrival_s: float
) -> tuple[tuple[str, ...], ...]:
    """Pin pre-cutover arrivals to the old generation and later ones to new."""
    if rollout.cutover_at_s is None or request_arrival_s < rollout.cutover_at_s:
        return rollout.old_layout
    return rollout.new_layout


def provuse_retire_old(
    rollout: ProvuseRollout, now_s: float, old_inflight_requests: int
) -> ProvuseRollout:
    """Retire the previous generation only when its in-flight count drains."""
    if rollout.cutover_at_s is None:
        raise ValueError("cannot retire old generation before cutover")
    if now_s < rollout.cutover_at_s:
        raise ValueError("retirement cannot precede cutover")
    if old_inflight_requests != 0:
        raise ValueError("old generation still has in-flight requests")
    return replace(rollout, old_retired_at_s=now_s)


def grl_orch_decode_from_scores(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    memory_mib: Mapping[str, float],
    cpu_demand: Mapping[str, float],
    site_memory_mib: Mapping[str, float],
    site_cpu_capacity: Mapping[str, float],
    fusion_edge_scores: Mapping[tuple[str, str], float],
    placement_scores: Mapping[tuple[tuple[str, ...], str], float],
) -> GroupDecode:
    """Reconstruct the paper-described score ordering/repair/placement decode.

    This is deliberately the *decoder only*: GAT feature encoding, PPO
    training, reward, and paper-specific sampled workload generation must be
    implemented and audited before this can be called a GRL-Orch baseline.
    """
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def members(root: str) -> tuple[str, ...]:
        return tuple(sorted(node for node in nodes if find(node) == root))

    def feasible_group(group: tuple[str, ...]) -> bool:
        group_mem = sum(memory_mib[node] for node in group)
        group_cpu = sum(cpu_demand[node] for node in group)
        # Fusion repair's model constraint is whether the group can fit at
        # least one eligible node. Placement later applies remaining capacity.
        return any(
            group_mem <= site_memory_mib[site] + 1e-9
            and group_cpu <= site_cpu_capacity[site] + 1e-9
            for site in site_memory_mib
            if site in site_cpu_capacity
        )

    candidate_edges = [
        edge for edge in edges
        if edge.kind in {"sync", "async"} and (edge.src, edge.dst) in fusion_edge_scores
    ]
    candidate_edges.sort(key=lambda e: (-fusion_edge_scores[(e.src, e.dst)], e.src, e.dst))
    rejected_for_capacity: list[tuple[str, str]] = []
    for edge in candidate_edges:
        left, right = find(edge.src), find(edge.dst)
        if left == right:
            continue
        proposed = tuple(sorted(set(members(left)) | set(members(right))))
        if not feasible_group(proposed):
            rejected_for_capacity.append((edge.src, edge.dst))
            continue
        parent[max(left, right)] = min(left, right)

    groups = tuple(sorted({members(find(node)) for node in nodes}))
    remaining = {site: float(capacity) for site, capacity in site_memory_mib.items()}
    placement: dict[tuple[str, ...], str] = {}
    fallbacks: list[tuple[str, ...]] = []
    for group in groups:
        size = sum(memory_mib[node] for node in group)
        feasible_sites = [site for site, free in remaining.items() if free + 1e-9 >= size]
        if feasible_sites:
            selected = max(
                feasible_sites,
                key=lambda site: (placement_scores.get((group, site), float("-inf")), site),
            )
        else:
            # The paper-described decoder has a max-remaining-memory fallback
            # with an overload penalty; preserve and expose that fallback.
            selected = max(remaining, key=lambda site: (remaining[site], site))
            fallbacks.append(group)
        placement[group] = selected
        remaining[selected] -= size
    return GroupDecode(groups, placement, tuple(rejected_for_capacity), tuple(fallbacks))


def validate_rooted_overlapping_cover(
    nodes: Sequence[str], bundles: Sequence[RootedBundle]
) -> None:
    """Validate the cover representation needed by a Quilt adapter.

    It deliberately permits overlap. The separate tiny-graph exact solver
    implements the appendix-style resource/objective semantics; this helper
    only validates a caller-supplied cover representation.
    """
    if not bundles:
        raise ValueError("a rooted cover must contain at least one bundle")
    for bundle in bundles:
        if bundle.root not in bundle.members:
            raise ValueError(f"bundle root {bundle.root!r} is not in its member set")
        if not bundle.members <= set(nodes):
            raise ValueError("bundle contains an unknown function")
        if bundle.resident_mib < 0 or bundle.cpu_demand < 0:
            raise ValueError("resource demands must be non-negative")
    covered = set().union(*(bundle.members for bundle in bundles))
    if covered != set(nodes):
        raise ValueError(f"cover mismatch: missing={sorted(set(nodes)-covered)}")


def fusionize_path_candidates(
    nodes: Sequence[str], edges: Sequence[Edge]
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Return the path-optimization sequence used by Fusionize-style feedback.

    Each prefix incorporates one observed synchronous edge. Async edges are
    deliberately not contracted: their targets become independent entry
    groups. Groups may overlap when a task is synchronously reachable from
    multiple entry groups. This is a policy adaptation of Fusionize++'s path
    phase, not its runtime or CSP-1 implementation.
    """
    node_set = set(nodes)
    sync_edges = [edge for edge in edges if edge.kind == "sync"]
    for edge in edges:
        if edge.src not in node_set or edge.dst not in node_set:
            raise ValueError(f"unknown function in edge {edge.src!r}->{edge.dst!r}")
    prefixes: list[tuple[tuple[str, ...], ...]] = []
    for incorporated in range(len(sync_edges) + 1):
        prefixes.append(_fusionize_groups(nodes, edges, sync_edges[:incorporated]))
    return tuple(prefixes)


def _fusionize_groups(
    nodes: Sequence[str], all_edges: Sequence[Edge], active_sync_edges: Sequence[Edge]
) -> tuple[tuple[str, ...], ...]:
    incoming_sync = {edge.dst for edge in active_sync_edges}
    roots = (set(nodes) - incoming_sync) | {
        edge.dst for edge in all_edges if edge.kind == "async"
    }
    groups = []
    for root in sorted(roots):
        reached = {root}
        frontier = [root]
        while frontier:
            source = frontier.pop()
            for edge in active_sync_edges:
                if edge.src == source and edge.dst not in reached:
                    reached.add(edge.dst)
                    frontier.append(edge.dst)
        groups.append(tuple(sorted(reached)))
    return tuple(sorted(set(groups)))


def fusionize_feedback_path_search(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    measure: Callable[[tuple[tuple[str, ...], ...]], FusionizeSnapshot],
    latency_weight: float = 1.0,
    cost_weight: float = 0.0,
    improvement_epsilon: float = 1e-12,
) -> FusionizePathSearch:
    """Measured one-edge-at-a-time path optimization, then stop at no gain.

    Only observed synchronous edges are candidates. At each iteration the
    routine evaluates every one-edge extension and commits the best measured
    improvement, which reflects Fusionize++'s feedback path-search shape more
    closely than a static sync-component contraction. The external ``measure``
    callback is the simulator/profiling boundary; configuration search remains
    a separate post-path stage.
    """
    if latency_weight < 0 or cost_weight < 0 or latency_weight + cost_weight <= 0:
        raise ValueError("objective weights must be non-negative and not both zero")
    observed_sync = tuple(
        sorted(
            {(edge.src, edge.dst): edge for edge in edges if edge.kind == "sync"}.values(),
            key=lambda edge: (edge.src, edge.dst),
        )
    )
    active: list[Edge] = []
    current_groups = _fusionize_groups(nodes, edges, active)
    current_snapshot = measure(current_groups)
    evaluations = 1
    accepted = []

    def objective(snapshot: FusionizeSnapshot) -> float:
        return latency_weight * snapshot.latency_s + cost_weight * snapshot.monetary_cost

    while len(active) < len(observed_sync):
        proposals = []
        for edge in observed_sync:
            if edge in active:
                continue
            proposal_edges = (*active, edge)
            proposal_groups = _fusionize_groups(nodes, edges, proposal_edges)
            proposal_snapshot = measure(proposal_groups)
            evaluations += 1
            proposals.append((
                objective(proposal_snapshot),
                proposal_snapshot.latency_s,
                proposal_snapshot.monetary_cost,
                edge.src,
                edge.dst,
                edge,
                proposal_groups,
                proposal_snapshot,
            ))
        if not proposals:
            break
        best = min(proposals)
        if best[0] >= objective(current_snapshot) - improvement_epsilon:
            break
        active.append(best[5])
        accepted.append((best[5].src, best[5].dst))
        current_groups = best[6]
        current_snapshot = best[7]
    return FusionizePathSearch(
        groups=current_groups,
        snapshot=current_snapshot,
        accepted_edges=tuple(accepted),
        evaluated_candidates=evaluations,
    )


def fusionize_optimizer_due(
    previous: FusionizeSnapshot | None,
    current: FusionizeSnapshot,
    relative_change_threshold: float = 0.10,
) -> bool:
    """Change-trigger proxy for Fusionize++'s adapted CSP-1 scheduling.

    The paper compares monitored cost/performance snapshots to decide when to
    optimize; this explicit relative-change threshold is a simulator proxy,
    not a faithful CSP-1 recurrence.
    """
    if relative_change_threshold < 0:
        raise ValueError("relative_change_threshold must be non-negative")
    if previous is None:
        return True
    changes = []
    for old, new in (
        (previous.latency_s, current.latency_s),
        (previous.monetary_cost, current.monetary_cost),
    ):
        if old < 0 or new < 0:
            raise ValueError("snapshot metrics must be non-negative")
        scale = max(abs(old), abs(new), 1e-12)
        changes.append(abs(new - old) / scale)
    return max(changes) >= relative_change_threshold


def choose_fusionize_infrastructure(
    groups: Sequence[Sequence[str]],
    measured_scores: Mapping[tuple[str, ...], Sequence[FunctionConfigScore]],
    latency_weight: float = 1.0,
    cost_weight: float = 0.0,
) -> Mapping[tuple[str, ...], FunctionConfigScore]:
    """Choose the measured configuration for each group after path search.

    This mirrors Fusionize++'s separate infrastructure-optimization phase:
    evaluate configurations per final group, rather than jointly searching
    every topology/configuration Cartesian product.
    """
    if latency_weight < 0 or cost_weight < 0 or latency_weight + cost_weight <= 0:
        raise ValueError("objective weights must be non-negative and not both zero")
    selected = {}
    for group in groups:
        key = tuple(sorted(group))
        options = measured_scores.get(key, ())
        if not options:
            raise ValueError(f"no measured infrastructure options for group {key}")
        selected[key] = min(
            options,
            key=lambda score: (
                latency_weight * score.latency_s + cost_weight * score.monetary_cost,
                score.latency_s,
                score.monetary_cost,
                score.config,
            ),
        )
    return selected


def fusionize_select_path_candidate(
    candidates: Sequence[tuple[tuple[str, ...], ...]],
    measured_snapshots: Mapping[tuple[tuple[str, ...], ...], FusionizeSnapshot],
    latency_weight: float = 1.0,
    cost_weight: float = 0.0,
) -> tuple[tuple[str, ...], ...]:
    """Select the best measured path candidate before config optimization."""
    if latency_weight < 0 or cost_weight < 0 or latency_weight + cost_weight <= 0:
        raise ValueError("objective weights must be non-negative and not both zero")
    evaluated = []
    for candidate in candidates:
        key = tuple(sorted(tuple(sorted(group)) for group in candidate))
        snapshot = measured_snapshots.get(key)
        if snapshot is None:
            continue
        evaluated.append((
            latency_weight * snapshot.latency_s + cost_weight * snapshot.monetary_cost,
            snapshot.latency_s,
            snapshot.monetary_cost,
            key,
        ))
    if not evaluated:
        raise ValueError("no path candidate has a measured snapshot")
    return min(evaluated)[-1]


def _quilt_group_resources(
    root: str,
    members: frozenset[str],
    edges: Sequence[Edge],
    profiles: Mapping[str, QuiltFunctionProfile],
    workflow_invocations: int,
) -> tuple[float, float]:
    memory = profiles[root].memory_mib
    cpu = profiles[root].cpu_units
    for edge in edges:
        if edge.src not in members or edge.dst not in members:
            continue
        alpha = ceil(edge.weight / workflow_invocations)
        memory += profiles[edge.dst].memory_mib
        if edge.kind == "async":
            memory += profiles[edge.dst].memory_mib * (alpha - 1)
        cpu += profiles[edge.dst].cpu_units * alpha
    return memory, cpu


def quilt_exact_rooted_cover(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    profiles: Mapping[str, QuiltFunctionProfile],
    memory_capacity_mib: float,
    cpu_capacity: float,
    workflow_invocations: int = 1,
    workflow_root: str | None = None,
    max_nodes: int = 8,
    max_combinations: int = 500_000,
) -> QuiltSolution:
    """Tiny-graph exact rooted-cover solver following Quilt's appendix ILP.

    It enumerates candidate root sets and feasible rooted subgraphs, then
    exactly selects one subgraph per root. The objective is weighted cut-edge
    calls. Memory and CPU use the paper's separate sync/async accounting.
    This is an exact small-instance semantic oracle, not Quilt's scalable
    GRASP/root-selection heuristic or deployment/runtime implementation.
    """
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be non-empty and unique")
    if len(nodes) > max_nodes:
        raise ValueError(f"exact Quilt witness solver is limited to {max_nodes} nodes")
    node_set = set(nodes)
    if set(profiles) != node_set:
        raise ValueError("every function must have exactly one resource profile")
    if workflow_invocations <= 0:
        raise ValueError("workflow_invocations must be positive")
    if memory_capacity_mib <= 0 or cpu_capacity <= 0:
        raise ValueError("container capacities must be positive")
    for edge in edges:
        if edge.src not in node_set or edge.dst not in node_set:
            raise ValueError(f"unknown function in edge {edge.src!r}->{edge.dst!r}")
        if edge.kind not in {"sync", "async"}:
            raise ValueError("Quilt profiles require sync or async call edges")
        if edge.weight <= 0:
            raise ValueError("Quilt call frequencies must be positive")
    if any(p.memory_mib < 0 or p.cpu_units < 0 for p in profiles.values()):
        raise ValueError("resource profiles must be non-negative")

    predecessors = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    adjacency = {node: [] for node in nodes}
    for edge in edges:
        predecessors[edge.dst].add(edge.src)
        adjacency[edge.src].append(edge.dst)
        indegree[edge.dst] += 1
    ready = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise ValueError("Quilt input must be a DAG")
    if workflow_root is None:
        roots_in_graph = [node for node in nodes if not predecessors[node]]
        if len(roots_in_graph) != 1:
            raise ValueError("specify workflow_root when the DAG has multiple roots")
        workflow_root = roots_in_graph[0]
    if workflow_root not in node_set:
        raise ValueError("workflow_root is not in nodes")

    node_order = tuple(nodes)
    others = tuple(node for node in node_order if node != workflow_root)
    root_sets_examined = 0
    combinations_examined = 0
    best: QuiltSolution | None = None
    for root_count in range(len(others) + 1):
        for extra_roots in combinations(others, root_count):
            roots = tuple(sorted((workflow_root, *extra_roots)))
            root_set = set(roots)
            root_sets_examined += 1
            options_by_root: dict[str, list[RootedBundle]] = {}
            possible = True
            for root in roots:
                candidates = []
                remaining = tuple(node for node in node_order if node != root)
                for member_count in range(len(remaining) + 1):
                    for extra_members in combinations(remaining, member_count):
                        member_set = frozenset((root, *extra_members))
                        # Connectivity from the chosen root, matching the
                        # appendix predecessor constraint in a DAG.
                        connected = {root}
                        changed = True
                        while changed:
                            changed = False
                            for node in member_set - connected:
                                if predecessors[node] & connected:
                                    connected.add(node)
                                    changed = True
                        if connected != set(member_set):
                            continue
                        # A non-root target cannot be cut from an included
                        # caller (the Quilt cross-edge-root rule).
                        if any(
                            edge.src in member_set
                            and edge.dst not in root_set
                            and edge.dst not in member_set
                            for edge in edges
                        ):
                            continue
                        mem, cpu = _quilt_group_resources(
                            root, member_set, edges, profiles, workflow_invocations
                        )
                        if mem <= memory_capacity_mib + 1e-9 and cpu <= cpu_capacity + 1e-9:
                            candidates.append(
                                RootedBundle(root, member_set, "", mem, cpu)
                            )
                if not candidates:
                    possible = False
                    break
                options_by_root[root] = candidates
            if not possible:
                continue

            for selected_groups in product(*(options_by_root[root] for root in roots)):
                combinations_examined += 1
                if combinations_examined > max_combinations:
                    raise RuntimeError(
                        "Quilt exact witness search exceeded max_combinations; "
                        "use the paper's root-selection/ILP heuristic for larger cases"
                    )
                union = set().union(*(group.members for group in selected_groups))
                if union != node_set:
                    continue
                cross_edges = tuple(
                    (edge.src, edge.dst)
                    for edge in edges
                    if any(
                        edge.src in group.members and edge.dst not in group.members
                        for group in selected_groups
                    )
                )
                cross_weight = sum(
                    edge.weight
                    for edge in edges
                    if (edge.src, edge.dst) in set(cross_edges)
                )
                solution = QuiltSolution(
                    roots=roots,
                    groups=tuple(selected_groups),
                    cross_edges=cross_edges,
                    cross_weight=cross_weight,
                    root_sets_examined=root_sets_examined,
                    combinations_examined=combinations_examined,
                )
                if best is None or (solution.cross_weight, len(solution.groups), solution.roots) < (
                    best.cross_weight, len(best.groups), best.roots
                ):
                    best = solution
    if best is None:
        raise ValueError("no feasible rooted cover under the supplied resource caps")
    return QuiltSolution(
        best.roots,
        best.groups,
        best.cross_edges,
        best.cross_weight,
        root_sets_examined,
        combinations_examined,
    )


def quilt_route_call(solution: QuiltSolution, current_root: str, src: str, dst: str) -> QuiltRoute:
    """Route one invocation in an overlapping Quilt cover without re-execution.

    An internal edge remains in the current package. A cut edge must target a
    package root by Quilt's cross-edge-root rule; otherwise the cover cannot
    implement the call correctly.
    """
    source_group = next((group for group in solution.groups if group.root == current_root), None)
    if source_group is None:
        raise ValueError(f"unknown current package root {current_root!r}")
    if src not in source_group.members:
        raise ValueError(f"caller {src!r} is not present in package {current_root!r}")
    if dst in source_group.members:
        return QuiltRoute(True, current_root, current_root)
    if dst not in solution.roots:
        raise ValueError(f"cut edge {src!r}->{dst!r} has no package rooted at its callee")
    if not any(group.root == dst and dst in group.members for group in solution.groups):
        raise ValueError(f"no target package exists for cut-edge callee {dst!r}")
    return QuiltRoute(False, current_root, dst)


def costless_native_choice(
    candidates: Sequence[CostCandidate], latency_bound_s: float
) -> CostCandidate:
    """Exact finite-candidate selection for a Costless-style native objective.

    Select minimum monetary cost subject to the external latency bound. If no
    candidate meets the bound, return the minimum-latency feasible candidate
    and mark it as a deadline-infeasible outcome at the caller.
    """
    usable = [candidate for candidate in candidates if candidate.feasible and candidate.edge_to_cloud_only]
    if not usable:
        raise ValueError("no feasible candidate obeys the edge-to-cloud restriction")
    on_time = [candidate for candidate in usable if candidate.latency_s <= latency_bound_s]
    if on_time:
        return min(on_time, key=lambda candidate: (candidate.monetary_cost, candidate.latency_s, candidate.ident))
    return min(usable, key=lambda candidate: (candidate.latency_s, candidate.monetary_cost, candidate.ident))


def costless_larac(
    start: str,
    end: str,
    arcs: Sequence[CostGraphArc],
    latency_bound_s: float,
    max_iterations: int = 1_000,
    epsilon: float = 1e-10,
) -> tuple[tuple[str, ...], float, float]:
    """Solve the native latency-constrained minimum-cost Cost-graph path.

    The caller must build the paper's legal fusion/placement Cost graph. This
    routine implements the LARAC search on its non-negative price/delay arcs;
    it does not fabricate an FnSeq from an arbitrary branching DAG.
    """
    if latency_bound_s < 0 or max_iterations <= 0:
        raise ValueError("latency bound must be non-negative and iterations positive")
    if any(arc.monetary_cost < 0 or arc.latency_s < 0 for arc in arcs):
        raise ValueError("LARAC requires non-negative arc costs and delays")
    legal = [arc for arc in arcs if arc.edge_to_cloud_legal]
    if not legal:
        raise ValueError("Cost graph contains no legal edge-to-cloud path arcs")

    max_cost = max((arc.monetary_cost for arc in legal), default=0.0) or 1.0
    max_delay = max((arc.latency_s for arc in legal), default=0.0) or 1.0

    def shortest(metric: str, lam: float = 0.0):
        adjacency: dict[str, list[CostGraphArc]] = {}
        for arc in legal:
            adjacency.setdefault(arc.src, []).append(arc)
        heap: list[tuple[float, tuple[str, ...], str, float, float]] = [
            (0.0, (start,), start, 0.0, 0.0)
        ]
        best: dict[str, float] = {start: 0.0}
        while heap:
            score, path, node, price, delay = heappop(heap)
            if score > best.get(node, float("inf")) + epsilon:
                continue
            if node == end:
                return path, price, delay
            for arc in sorted(adjacency.get(node, ()), key=lambda item: (item.dst, item.monetary_cost, item.latency_s)):
                if metric == "price":
                    increment = arc.monetary_cost
                elif metric == "delay":
                    increment = arc.latency_s
                else:
                    increment = arc.monetary_cost / max_cost + lam * arc.latency_s / max_delay
                candidate = score + increment
                if candidate + epsilon < best.get(arc.dst, float("inf")):
                    best[arc.dst] = candidate
                    heappush(
                        heap,
                        (
                            candidate,
                            path + (arc.dst,),
                            arc.dst,
                            price + arc.monetary_cost,
                            delay + arc.latency_s,
                        ),
                    )
        return None

    cheapest = shortest("price")
    fastest = shortest("delay")
    if cheapest is None or fastest is None:
        raise ValueError("no legal start-to-end path in Cost graph")
    if cheapest[2] <= latency_bound_s + epsilon:
        return cheapest
    if fastest[2] > latency_bound_s + epsilon:
        raise ValueError("no Cost-graph path satisfies the latency bound")

    infeasible = cheapest
    feasible = fastest
    for _ in range(max_iterations):
        price_gap = feasible[1] / max_cost - infeasible[1] / max_cost
        delay_gap = infeasible[2] / max_delay - feasible[2] / max_delay
        if delay_gap <= epsilon:
            return feasible
        lam = max(0.0, price_gap / delay_gap)
        candidate = shortest("combined", lam)
        if candidate is None:
            return feasible
        if candidate[0] == feasible[0] or candidate[0] == infeasible[0]:
            return feasible
        if candidate[2] <= latency_bound_s + epsilon:
            if candidate[1] >= feasible[1] - epsilon:
                return feasible
            feasible = candidate
        else:
            if candidate[1] >= infeasible[1] - epsilon:
                return feasible
            infeasible = candidate
    return feasible


IMPLEMENTATION_STATUS = {
    "provuse": "observed_trigger_and_rollout_state_machine_done_shared_simulator_integration_pending",
    "fusionize_plus_plus": "path_and_infrastructure_policy_adaptation_done_exact_csp1_and_runtime_pending",
    "quilt": "tiny_exact_rooted_cover_and_call_routing_done_scalable_search_and_continuum_placement_pending",
    "costless": "archived_model_audit_only_excluded_from_r8_direct_comparison",
    "grl_orch": "gat_ppo_train_path_synthetic_smoke_passed_shared_workload_and_simulator_pending",
}
