#!/usr/bin/env python3
"""Generate the frozen GRL and scenario-one rule targets for R336."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
INPUT = HERE / "inputs" / "r328_manifest.json"
WEIGHTS = HERE / "weights"
OUT = HERE / "outputs" / "targets.json"
SEEDS = (601, 631, 659)
METHODS = ("grl_native", "grl_response", "greedy_fusion", "critical_path")
sys.path.insert(0, str(HERE / "vendor"))
import r8_grl_orch as G
from r8_policy_adapters import Edge


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(mapping: tuple[int, ...]) -> list[list[int]]:
    return [[i for i, group in enumerate(mapping) if group == key]
            for key in sorted(set(mapping))]


def canonical(parts: list[tuple[list[int], int]], n: int):
    mapping = [None] * n
    sites = []
    for group, (nodes, site) in enumerate(sorted(parts, key=lambda row: min(row[0]))):
        sites.append(site)
        for node in nodes:
            if mapping[node] is not None:
                raise ValueError("overlapping groups")
            mapping[node] = group
    if any(group is None for group in mapping):
        raise ValueError("groups do not cover all functions")
    return tuple(mapping), tuple(sites)


def valid(p: dict, mapping: tuple[int, ...], sites: tuple[int, ...]) -> bool:
    if len(mapping) != len(p["work"]) or sorted(set(mapping)) != list(range(len(sites))) or any(site not in (0, 1) for site in sites):
        return False
    used = [0., 0.]
    for group, nodes in enumerate(groups(mapping)):
        used[sites[group]] += .3 * sum(p["mem"][i] for i in nodes) + .7 * max(p["mem"][i] for i in nodes)
    if any(amount > capacity + 1e-9 for amount, capacity in zip(used, p["ram"])):
        return False
    edges = {(mapping[u], mapping[v]) for u, v, _ in p["edges"] if mapping[u] != mapping[v]}
    done = set()
    while len(done) < len(sites):
        ready = [g for g in range(len(sites)) if g not in done
                 and all(u in done for u, v in edges if v == g)]
        if not ready:
            return False
        done.update(ready)
    return True


def workflow(p: dict):
    n = len(p["work"])
    return G.GRLWorkflow(
        workflow_id=str(p["id"]),
        nodes=tuple(str(i) for i in range(n)),
        edges=tuple(Edge(str(u), str(v), "dependency", b) for u, v, b in p["edges"]),
        workload_cycles={str(i): value * 1e9 for i, value in enumerate(p["work"])},
        memory_mib={str(i): value for i, value in enumerate(p["mem"])},
        deadline_s={str(i): 60. for i in range(n)},
        site_cpu_cycles_s={str(site): p["speed"][site] * 1e9 for site in (0, 1)},
        site_memory_mib={str(site): p["ram"][site] for site in (0, 1)},
        site_price_per_s={"0": .35, "1": .9},
        link_bandwidth_mbps={("0", "1"): p["mbps"], ("1", "0"): p["mbps"]},
        link_latency_s={("0", "1"): p["rtt"], ("1", "0"): p["rtt"]},
        baseline_latency_s=max(1e-9, sum(p["work"]) / p["speed"][1]),
        baseline_cost=max(1e-9, sum(p["work"]) / p["speed"][1] * .9),
    )


def grl_layout(plan, n: int):
    return canonical([(list(map(int, group)), int(plan.placement[group]))
                     for group in plan.groups], n)


def rules_plan(p: dict, method: str, B):
    """Exact R194 reference-suite greedy-fusion / critical-path planner."""
    n = len(p["work"])
    parent = list(range(n))

    def find(v):
        while parent[v] != v:
            v = parent[v]
        return v

    edges = list(p["edges"])
    if method == "critical_path":
        best, tails = {}, {}
        for u in reversed(range(n)):
            val, v = max(
                [(b * 8 / p["mbps"] + p["rtt"] + best[v], v)
                 for a, v, b in edges if a == u], default=(0.0, None))
            best[u] = p["work"][u] / min(p["speed"]) + val
            tails[u] = [u] + ([] if v is None else tails[v])
        roots = [v for v in range(n)
                 if not any(b == v for a, b, _ in edges)]
        path = tails[max(roots, key=lambda v: (best[v], v))]
        keep = set(zip(path, path[1:]))
        edges = sorted([e for e in edges if tuple(e[:2]) in keep],
                       key=lambda e: (-e[2], e[0], e[1]))
    else:
        edges.sort(key=lambda e: (
            -(e[2] * 8 / p["mbps"] + p["rtt"])
            / (.3 * (p["mem"][e[0]] + p["mem"][e[1]])
               + .7 * max(p["mem"][e[0]], p["mem"][e[1]])),
            e[0], e[1]))

    for u, v, _ in edges:
        a, b = find(u), find(v)
        if a == b:
            continue
        saved = parent[:]
        parent[b] = a
        groups = [sorted(i for i in range(n) if find(i) == root)
                  for root in sorted({find(i) for i in range(n)})]
        candidate = B.canonical([(group, 1) for group in groups], n)
        if not B.valid(p, *candidate):
            parent[:] = saved

    groups = [sorted(i for i in range(n) if find(i) == root)
              for root in sorted({find(i) for i in range(n)})]
    mapping = {node: group for group, nodes in enumerate(groups) for node in nodes}
    deps = {(mapping[u], mapping[v]) for u, v, _ in p["edges"]
            if mapping[u] != mapping[v]}
    todo, order = set(range(len(groups))), []
    while todo:
        ready = sorted(g for g in todo
                       if all(u in order for u, v in deps if v == g))
        if not ready:
            raise RuntimeError("rule planner produced a cyclic group graph")
        order += ready
        todo -= set(ready)

    remaining, assigned, finish = p["ram"][:], {}, {}
    for group in order:
        nodes = groups[group]
        mem = .3 * sum(p["mem"][v] for v in nodes) + .7 * max(p["mem"][v] for v in nodes)
        feasible = [site for site in (0, 1) if remaining[site] + 1e-9 >= mem]
        if not feasible:
            raise RuntimeError("rule planner has no resource-feasible pool")

        def estimate(site):
            arrivals = [0.]
            for prev, prev_site in assigned.items():
                sizes = [b for u, v, b in p["edges"]
                         if u in groups[prev] and v in nodes]
                if sizes:
                    arrivals.append(finish[prev] + (.003 if prev_site == site
                                    else sum(sizes) * 8 / p["mbps"] + p["rtt"]))
            return max(arrivals) + sum(p["work"][v] for v in nodes) / p["speed"][site]

        site = min(feasible, key=lambda s: (estimate(s), s))
        finish[group] = estimate(site)
        assigned[group] = site
        remaining[site] -= mem

    return B.canonical([(nodes, assigned[group])
                        for group, nodes in enumerate(groups)], n)


def main() -> None:
    manifest = json.loads(INPUT.read_text())
    block = next(row for row in manifest["profiles"] if row["profile"].get("id") == 326001)
    profile = block["profile"]
    if len(profile["work"]) != 8 or profile["family"] != "diamond":
        raise RuntimeError("expected the frozen eight-function diamond, profile 326001")

    import torch
    wflow = workflow(profile)
    config = G.GRLRewardConfig()

    targets: dict[str, dict] = {}
    weight_hashes: dict[str, str] = {}
    for method in METHODS:
        targets[method] = {}
        for seed in SEEDS:
            if method in ("grl_native", "grl_response"):
                path = WEIGHTS / method / f"seed_{seed}" / "selected.pt"
                ck = torch.load(path, map_location="cpu", weights_only=False)
                net = G.GRLActorCritic(dataclasses.replace(G.GRLTrainConfig(), seed=seed))
                net.load_state_dict(ck["state"])
                net.eval()
                with torch.no_grad():
                    plan = G.sample_grl_plan(net, wflow, config,
                                             stochastic=False, threshold=ck["threshold"])
                layout = grl_layout(plan, len(profile["work"]))
                detail = {"threshold": ck["threshold"], "selected_groups": [list(g) for g in plan.groups],
                          "selected_placements": [int(plan.placement[g]) for g in plan.groups],
                          "weight_sha256": digest(path)}
                weight_hashes[f"{method}/seed_{seed}"] = detail["weight_sha256"]
            else:
                layout = rules_plan(profile, method, sys.modules[__name__])
                detail = {"rule": "scenario1_frozen_greedy_or_critical_path"}
            if not valid(profile, *layout):
                raise RuntimeError(f"{method} emitted an infeasible layout for seed {seed}")
            if len(groups(layout[0])) > 8:
                raise RuntimeError(f"{method} target exceeds the OpenFaaS service-cap budget")
            targets[method][str(seed)] = {
                "layout": [list(layout[0]), list(layout[1])],
                "groups": [list(g) for g in groups(layout[0])],
                "pool_by_group": list(layout[1]),
                "service_count": len(groups(layout[0])),
                **detail,
            }
            print(f"R336_TARGET method={method} seed={seed} groups={len(groups(layout[0]))} layout={targets[method][str(seed)]}", flush=True)

    if any(targets["grl_native"][str(seed)]["layout"]
           != targets["grl_response"][str(seed)]["layout"] for seed in SEEDS):
        raise RuntimeError("GRL native and response variants differ; revisit the frozen R336 method matrix")
    result = {
        "experiment": "R336_OPENFAAS_METHOD_BASELINE_EXTENSION",
        "profile_id": 326001,
        "profile_family": profile["family"],
        "n_functions": len(profile["work"]),
        "capacity": {k: profile[k] for k in ("cores", "ram", "speed", "mbps", "rtt")},
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "live_methods": ["grl_native", "greedy_fusion", "critical_path"],
        "grl_response_same_layout_as_native_for_all_seeds": True,
        "training_scope": "frozen R194/R209 20-function GRL weights applied as variable-node adapters to the held-out 8-function profile; no retraining on the live confirmation traces",
        "source_bindings": {
            "r328_manifest_sha256": digest(INPUT),
            "greedy_critical_path_source_sha256": digest(Path(__file__)),
            "grl_vendor_sha256": digest(HERE / "vendor" / "r8_grl_orch.py"),
            "grl_adapter_sha256": digest(HERE / "vendor" / "r8_policy_adapters.py"),
            "weight_sha256": weight_hashes,
        },
        "targets": targets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"R336_TARGETS_READY {OUT}", flush=True)


if __name__ == "__main__":
    main()
