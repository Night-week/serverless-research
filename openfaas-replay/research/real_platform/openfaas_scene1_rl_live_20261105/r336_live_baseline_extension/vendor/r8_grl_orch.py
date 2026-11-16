"""GAT/PPO policy path for the R8 GRL-Orch algorithm-level adapter.

The model and decision path follow the supplied GRL-Orch paper. The training
API deliberately accepts a simulator callback: this module does not pretend
that its analytical fallback evaluator is the paper's trace-driven simulator
or the shared R8 event simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import random
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F

from r8_policy_adapters import Edge


@dataclass(frozen=True)
class GRLWorkflow:
    """One DAG and one infrastructure snapshot, with train-only IDs."""

    workflow_id: str
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    workload_cycles: Mapping[str, float]
    memory_mib: Mapping[str, float]
    deadline_s: Mapping[str, float]
    site_cpu_cycles_s: Mapping[str, float]
    site_memory_mib: Mapping[str, float]
    site_price_per_s: Mapping[str, float]
    link_bandwidth_mbps: Mapping[tuple[str, str], float]
    baseline_latency_s: float
    baseline_cost: float
    # Current residual memory snapshot for sequential placement. If omitted,
    # treat the full site capacity as initially free (the unloaded case).
    site_free_memory_mib: Mapping[str, float] = field(default_factory=dict)
    cold_probability: Mapping[str, float] = field(default_factory=dict)
    # Optional fixed propagation/RPC component for each directed site link.
    link_latency_s: Mapping[tuple[str, str], float] = field(default_factory=dict)
    # Optional client/edge boundary profile. The R8 source-shaped workload
    # snapshots do not contain measured endpoint payloads, so these stay off
    # unless a separate, explicit sensitivity profile is supplied.
    client_boundary_enabled: bool = False
    client_request_bytes: float = 0.0
    client_response_bytes: float = 0.0
    client_one_way_latency_s: float = 0.0

    def validate(self) -> None:
        if not self.workflow_id:
            raise ValueError("workflow_id must be non-empty")
        if not self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("nodes must be non-empty and unique")
        known = set(self.nodes)
        for name, values in (
            ("workload_cycles", self.workload_cycles),
            ("memory_mib", self.memory_mib),
            ("deadline_s", self.deadline_s),
        ):
            if set(values) != known:
                raise ValueError(f"{name} must contain exactly the workflow nodes")
            if any(not math.isfinite(float(x)) or float(x) < 0 for x in values.values()):
                raise ValueError(f"{name} must contain finite non-negative values")
        if not self.site_memory_mib or set(self.site_memory_mib) != set(self.site_cpu_cycles_s):
            raise ValueError("site CPU and memory maps must have identical non-empty keys")
        if set(self.site_memory_mib) != set(self.site_price_per_s):
            raise ValueError("site_price_per_s must contain exactly the site keys")
        if self.site_free_memory_mib and set(self.site_free_memory_mib) != set(self.site_memory_mib):
            raise ValueError("site_free_memory_mib must be empty or contain exactly the site keys")
        for site, free in self.site_free_memory_mib.items():
            if (
                not math.isfinite(float(free))
                or not 0 <= float(free) <= float(self.site_memory_mib[site])
            ):
                raise ValueError("site free memory must be within [0, total capacity]")
        if any(not math.isfinite(float(x)) or float(x) <= 0 for x in self.site_cpu_cycles_s.values()):
            raise ValueError("site CPU rates must be positive")
        if any(not math.isfinite(float(x)) or float(x) <= 0 for x in self.site_memory_mib.values()):
            raise ValueError("site memory capacities must be positive")
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in self.site_price_per_s.values()):
            raise ValueError("site prices must be non-negative")
        if (
            not math.isfinite(float(self.baseline_latency_s))
            or not math.isfinite(float(self.baseline_cost))
            or self.baseline_latency_s <= 0
            or self.baseline_cost <= 0
        ):
            raise ValueError("reward normalization baselines must be positive")
        for edge in self.edges:
            if edge.src not in known or edge.dst not in known:
                raise ValueError(f"unknown workflow edge {edge.src!r}->{edge.dst!r}")
            if edge.src == edge.dst or not edge.kind:
                raise ValueError("GRL-Orch edges must be non-self dependencies with a kind")
            if edge.weight < 0 or not math.isfinite(float(edge.weight)):
                raise ValueError("edge payload sizes must be finite and non-negative")
        outgoing = {node: set() for node in self.nodes}
        indegree = {node: 0 for node in self.nodes}
        for edge in self.edges:
            if edge.dst not in outgoing[edge.src]:
                outgoing[edge.src].add(edge.dst)
                indegree[edge.dst] += 1
        ready = [node for node in self.nodes if indegree[node] == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for dst in outgoing[node]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
        if visited != len(self.nodes):
            raise ValueError("workflow dependency graph must be acyclic")
        if set(self.cold_probability) - known:
            raise ValueError("cold_probability contains an unknown function")
        if any(not 0 <= float(p) <= 1 for p in self.cold_probability.values()):
            raise ValueError("cold probabilities must be in [0, 1]")
        for (src, dst), bw in self.link_bandwidth_mbps.items():
            if not src or not dst or not math.isfinite(float(bw)) or bw <= 0:
                raise ValueError("link bandwidth entries must be positive")
        if set(self.link_latency_s) - set(self.link_bandwidth_mbps):
            raise ValueError("link_latency_s entries require corresponding bandwidth entries")
        if any(not math.isfinite(float(latency)) or float(latency) < 0
               for latency in self.link_latency_s.values()):
            raise ValueError("link latencies must be finite and non-negative")
        for name, value in (
            ("client_request_bytes", self.client_request_bytes),
            ("client_response_bytes", self.client_response_bytes),
            ("client_one_way_latency_s", self.client_one_way_latency_s),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class GRLPlan:
    groups: tuple[tuple[str, ...], ...]
    placement: Mapping[tuple[str, ...], str]
    repair_edges: tuple[tuple[str, str], ...]
    placement_fallbacks: tuple[tuple[str, ...], ...]
    fusion_actions: tuple[bool, ...]
    placement_actions: tuple[str | None, ...]
    log_probability: Tensor
    entropy: Tensor
    value: Tensor
    policy_rng_state: Tensor | None = None
    fusion_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class GRLMetrics:
    latency_s: float
    cost: float
    oom_penalty: float
    queue_wait_s: float = 0.0
    wan_bytes_mib: float = 0.0
    cold_starts_expected: float = 0.0


@dataclass(frozen=True)
class GRLRewardConfig:
    latency_weight: float = 1.0
    cost_weight: float = 0.1
    oom_weight: float = 1.0
    memory_sharing_gamma: float = 0.3
    memory_price: float = 0.0
    container_init_s: float = 0.2
    cold_lambda_s_per_mib: float = 0.0
    fusion_threshold: float = 0.5

    def validate(self) -> None:
        if not 0.0 <= self.memory_sharing_gamma <= 1.0:
            raise ValueError("memory_sharing_gamma must be in [0,1]")
        if not 0.0 <= self.fusion_threshold <= 1.0:
            raise ValueError("fusion_threshold must be in [0,1]")
        numeric = (
            self.latency_weight, self.cost_weight, self.oom_weight,
            self.memory_price, self.container_init_s, self.cold_lambda_s_per_mib,
        )
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in numeric):
            raise ValueError("reward weights and cold-start parameters must be finite and non-negative")


@dataclass(frozen=True)
class GRLTrainConfig:
    # Values explicitly reported for GRL-Orch in the supplied paper.
    episodes: int = 4000
    rollout_size: int = 32
    learning_rate: float = 3e-4
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    gat_layers: int = 3
    attention_heads: int = 8
    hidden_dim: int = 64
    dropout: float = 0.1
    validation_interval_episodes: int = 32
    seed: int = 7
    # The paper does not report PPO epochs or minibatch size; make this
    # implementation choice explicit rather than implying exact reproduction.
    ppo_epochs: int = 4
    gradient_clip_norm: float = 1.0
    threshold_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.7, 0.85, 1.0)


@dataclass(frozen=True)
class GRLTrainResult:
    best_validation_reward: float
    selected_fusion_threshold: float
    best_episode: int
    updates: int
    episodes_seen: int
    final_actor_parameter_delta: float
    final_critic_parameter_delta: float
    final_encoder_parameter_delta: float
    history: tuple[Mapping[str, float], ...]
    heldout_reward: float | None
    heldout_latency_s: float | None
    training_seconds: float


class GATLayer(nn.Module):
    """Dense, small-DAG GAT layer with edge-payload-aware attention."""

    def __init__(self, hidden_dim: int, heads: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, 2 * self.head_dim + edge_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: Tensor, adjacency: Tensor, edge_features: Tensor) -> Tensor:
        n = x.shape[0]
        z = self.proj(x).view(n, self.heads, self.head_dim)
        zi = z[:, None, :, :].expand(n, n, self.heads, self.head_dim)
        zj = z[None, :, :, :].expand(n, n, self.heads, self.head_dim)
        edge = edge_features[:, :, None, :].expand(n, n, self.heads, -1)
        pair = torch.cat((zi, zj, edge), dim=-1)
        logits = F.leaky_relu(
            torch.einsum("ijhd,hd->ijh", pair, self.attn), negative_slope=0.2
        )
        logits = logits.masked_fill(~adjacency[:, :, None], torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=1)
        alpha = self.dropout(alpha)
        out = torch.einsum("ijh,jhd->ihd", alpha, z).reshape(n, self.hidden_dim)
        return F.elu(self.norm(out))


class GRLActorCritic(nn.Module):
    """Paper-shaped graph encoder with factorized fusion/placement actor heads."""

    def __init__(self, config: GRLTrainConfig) -> None:
        super().__init__()
        self.config = config
        self.node_projection = nn.Linear(5, config.hidden_dim)
        self.gat = nn.ModuleList(
            GATLayer(config.hidden_dim, config.attention_heads, 1, config.dropout)
            for _ in range(config.gat_layers)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.placement_head = nn.Sequential(
            nn.Linear(config.hidden_dim + 3, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim + 3, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode(self, workflow: GRLWorkflow) -> tuple[Tensor, Tensor]:
        workflow.validate()
        nodes = workflow.nodes
        idx = {name: i for i, name in enumerate(nodes)}
        n = len(nodes)
        max_degree = max(1, n - 1)
        x_values = []
        for name in nodes:
            indegree = sum(edge.dst == name for edge in workflow.edges)
            outdegree = sum(edge.src == name for edge in workflow.edges)
            # Fixed unit scaling preserves the paper's five feature semantics
            # while avoiding magnitude differences between cycles, MiB and s.
            x_values.append([
                workflow.workload_cycles[name] / 1e9,
                workflow.memory_mib[name] / 1024.0,
                workflow.deadline_s[name],
                indegree / max_degree,
                outdegree / max_degree,
            ])
        x = torch.tensor(x_values, dtype=torch.float32)
        adjacency = torch.eye(n, dtype=torch.bool)
        edge_features = torch.zeros((n, n, 1), dtype=torch.float32)
        # The article denotes N(i) without specifying direction. Use both
        # directions so information can propagate across the full DAG.
        for edge in workflow.edges:
            i, j = idx[edge.src], idx[edge.dst]
            adjacency[i, j] = adjacency[j, i] = True
            payload = float(edge.weight) / 100.0
            edge_features[i, j, 0] = edge_features[j, i, 0] = payload
        h = F.elu(self.node_projection(x))
        for layer in self.gat:
            h = layer(h, adjacency, edge_features)
        return h, self._mean_site_features(workflow, None)

    @staticmethod
    def _site_raw_features(workflow: GRLWorkflow, remaining: Mapping[str, float] | None) -> Tensor:
        sites = sorted(workflow.site_memory_mib)
        rows = []
        for site in sites:
            free = (
                workflow.site_free_memory_mib.get(site, workflow.site_memory_mib[site])
                if remaining is None else remaining[site]
            )
            links = [
                bw for (src, dst), bw in workflow.link_bandwidth_mbps.items()
                if src == site and dst != site
            ]
            if not links:
                links = [
                    bw for (src, dst), bw in workflow.link_bandwidth_mbps.items()
                    if dst == site and src != site
                ]
            avg_bw = sum(links) / len(links) if links else 0.0
            rows.append([
                workflow.site_cpu_cycles_s[site] / 1e9,
                free / 1024.0,
                avg_bw / 100.0,
            ])
        return torch.tensor(rows, dtype=torch.float32)

    @classmethod
    def _mean_site_features(cls, workflow: GRLWorkflow, remaining: Mapping[str, float] | None) -> Tensor:
        return cls._site_raw_features(workflow, remaining).mean(dim=0)

    def critic_value(self, h: Tensor, workflow: GRLWorkflow) -> Tensor:
        context = torch.cat((h.mean(dim=0), self._mean_site_features(workflow, None)))
        return self.critic(context).squeeze(-1)

    def fusion_logits(self, h: Tensor, workflow: GRLWorkflow) -> Tensor:
        idx = {name: i for i, name in enumerate(workflow.nodes)}
        if not workflow.edges:
            return h.new_empty((0,))
        pairs = torch.stack([
            torch.cat((h[idx[edge.src]], h[idx[edge.dst]])) for edge in workflow.edges
        ])
        return self.fusion_head(pairs).squeeze(-1)

    def placement_logits(
        self, group_embedding: Tensor, workflow: GRLWorkflow, remaining: Mapping[str, float]
    ) -> tuple[Tensor, Tensor]:
        sites = sorted(workflow.site_memory_mib)
        resource = self._site_raw_features(workflow, remaining).to(group_embedding.device)
        group_rows = group_embedding.expand(len(sites), -1)
        logits = self.placement_head(torch.cat((group_rows, resource), dim=-1)).squeeze(-1)
        return logits, resource


def _memory_of(group: Sequence[str], workflow: GRLWorkflow, gamma: float) -> float:
    members = [float(workflow.memory_mib[node]) for node in group]
    return gamma * sum(members) + (1.0 - gamma) * max(members)


def _topological_groups(
    nodes: Sequence[str], edges: Sequence[Edge], groups: Sequence[Sequence[str]]
) -> tuple[tuple[str, ...], ...]:
    canonical = tuple(tuple(sorted(group)) for group in groups)
    owner = {node: group for group in canonical for node in group}
    outgoing = {group: set() for group in canonical}
    indegree = {group: 0 for group in canonical}
    for edge in edges:
        src, dst = owner[edge.src], owner[edge.dst]
        if src != dst and dst not in outgoing[src]:
            outgoing[src].add(dst)
            indegree[dst] += 1
    node_order = {node: i for i, node in enumerate(nodes)}
    ready = sorted(
        (group for group, deg in indegree.items() if deg == 0),
        key=lambda g: min(node_order[node] for node in g),
    )
    result: list[tuple[str, ...]] = []
    while ready:
        group = ready.pop(0)
        result.append(group)
        for dst in sorted(outgoing[group], key=lambda g: min(node_order[node] for node in g)):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
                ready.sort(key=lambda g: min(node_order[node] for node in g))
    if len(result) != len(canonical):
        raise ValueError("contracted graph is cyclic; input must be a DAG")
    return tuple(result)


def _partition_from_actions(
    workflow: GRLWorkflow,
    actions: Sequence[bool],
    scores: Sequence[float],
    gamma: float,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, str], ...]]:
    parent = {node: node for node in workflow.nodes}

    def find(node: str) -> str:
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def members(root: str) -> tuple[str, ...]:
        return tuple(sorted(node for node in workflow.nodes if find(node) == root))

    if len(actions) != len(workflow.edges) or len(scores) != len(workflow.edges):
        raise ValueError("fusion actions and scores must match workflow edges")
    # Paper Algorithm 1: decreasing probability contraction order.
    # Save behavior scores in the sampled plan so PPO replays its exact decoder.
    selected = sorted(
        [(i, workflow.edges[i].src, workflow.edges[i].dst)
         for i, take in enumerate(actions) if take],
        key=lambda item: (-float(scores[item[0]]), item[0]),
    )
    repairs: list[tuple[str, str]] = []
    max_node_memory = max(workflow.site_memory_mib.values())
    for _idx, src, dst in selected:
        left, right = find(src), find(dst)
        if left == right:
            continue
        proposed = tuple(sorted(set(members(left)) | set(members(right))))
        if _memory_of(proposed, workflow, gamma) > max_node_memory + 1e-8:
            repairs.append((src, dst))
            continue
        # Stable component root is only an implementation detail.
        a, b = sorted((left, right))
        previous_parent = parent
        parent = parent.copy()
        parent[b] = a
        candidate_groups = tuple(sorted({members(find(node)) for node in workflow.nodes}))
        try:
            # Contracting an arbitrary dependency edge can create a cycle in
            # the quotient DAG (e.g. A->B->C plus a direct A->C edge, when
            # only A and C are fused). A plan with such a cycle cannot be
            # executed as a workflow, so reject this action just like a
            # resource-infeasible merge and retain the original partition.
            _topological_groups(workflow.nodes, workflow.edges, candidate_groups)
        except ValueError as exc:
            if "contracted graph is cyclic" not in str(exc):
                raise
            parent = previous_parent
            repairs.append((src, dst))
    raw_groups = tuple(sorted({members(find(node)) for node in workflow.nodes}))
    return _topological_groups(workflow.nodes, workflow.edges, raw_groups), tuple(repairs)


def _plan_from_action_tensors(
    model: GRLActorCritic,
    workflow: GRLWorkflow,
    fusion_actions: Sequence[bool],
    placement_actions: Sequence[str | None],
    fusion_scores: Tensor,
    h: Tensor,
    config: GRLRewardConfig,
) -> tuple[GRLPlan, Tensor, Tensor]:
    if len(fusion_actions) != len(workflow.edges):
        raise ValueError("fusion action count must match dependency edge count")
    placement_actions_supplied = len(placement_actions) > 0
    groups, repairs = _partition_from_actions(
        workflow, fusion_actions, fusion_scores.detach().cpu().tolist(), config.memory_sharing_gamma
    )
    if placement_actions_supplied and len(placement_actions) != len(groups):
        raise ValueError("placement action count must match topologically ordered fused groups")
    if not placement_actions_supplied:
        placement_actions = tuple(None for _ in groups)
    remaining = {
        site: float(workflow.site_free_memory_mib.get(site, cap))
        for site, cap in workflow.site_memory_mib.items()
    }
    site_names = tuple(sorted(workflow.site_memory_mib))
    placement: dict[tuple[str, ...], str] = {}
    fallback_groups: list[tuple[str, ...]] = []
    placement_logps: list[Tensor] = []
    placement_entropies: list[Tensor] = []
    realized_actions: list[str | None] = []
    node_index = {node: i for i, node in enumerate(workflow.nodes)}

    for group_index, group in enumerate(groups):
        group_memory = _memory_of(group, workflow, config.memory_sharing_gamma)
        feasible = [site for site in site_names if remaining[site] + 1e-8 >= group_memory]
        if not feasible:
            selected = max(site_names, key=lambda site: (remaining[site], site))
            fallback_groups.append(group)
            chosen_action = None
        else:
            group_embedding = torch.stack([h[node_index[node]] for node in group]).mean(dim=0)
            raw_logits, _ = model.placement_logits(group_embedding, workflow, remaining)
            mask = torch.tensor([site in feasible for site in site_names], dtype=torch.bool)
            logits = raw_logits.masked_fill(~mask, torch.finfo(raw_logits.dtype).min)
            distribution = Categorical(logits=logits)
            given = placement_actions[group_index]
            if given is None:
                # A missing action means deterministic greedy decoding.
                action_index = int(torch.argmax(logits).item())
                chosen_action = site_names[action_index]
            else:
                if given not in feasible:
                    raise ValueError(f"saved placement action {given!r} is no longer feasible")
                action_index = site_names.index(given)
                chosen_action = given
            selected = chosen_action
            placement_logps.append(distribution.log_prob(torch.tensor(action_index)))
            placement_entropies.append(distribution.entropy())
        placement[group] = selected
        realized_actions.append(chosen_action)
        remaining[selected] -= group_memory

    zero = h.new_zeros(())
    logp = torch.stack(placement_logps).sum() if placement_logps else zero
    entropy = torch.stack(placement_entropies).sum() if placement_entropies else zero
    plan = GRLPlan(
        groups=groups,
        placement=placement,
        repair_edges=repairs,
        placement_fallbacks=tuple(fallback_groups),
        fusion_actions=tuple(bool(x) for x in fusion_actions),
        placement_actions=tuple(realized_actions),
        log_probability=logp,
        entropy=entropy,
        value=model.critic_value(h, workflow),
    )
    return plan, logp, entropy


def sample_grl_plan(
    model: GRLActorCritic,
    workflow: GRLWorkflow,
    reward_config: GRLRewardConfig,
    *,
    stochastic: bool,
    threshold: float | None = None,
    forced_actions: tuple[Sequence[bool], Sequence[str | None]] | None = None,
) -> GRLPlan:
    """Run the paper's factorized fusion/placement decision with repair/masks."""
    reward_config.validate()
    model.eval() if not stochastic else model.train()
    policy_rng_state = torch.random.get_rng_state() if stochastic else None
    h, _ = model.encode(workflow)
    fusion_logits = model.fusion_logits(h, workflow)
    fusion_dist = Bernoulli(logits=fusion_logits)
    fusion_probs = torch.sigmoid(fusion_logits)
    selected_threshold = reward_config.fusion_threshold if threshold is None else threshold

    if forced_actions is not None:
        fusion_actions = tuple(bool(x) for x in forced_actions[0])
        placement_actions = tuple(forced_actions[1])
        if len(fusion_actions) != len(workflow.edges):
            raise ValueError("forced fusion action count mismatch")
        fusion_tensor = torch.tensor(fusion_actions, dtype=torch.float32)
    elif stochastic:
        fusion_tensor = fusion_dist.sample()
        fusion_actions = tuple(bool(x) for x in fusion_tensor.tolist())
        placement_actions = ()
    else:
        fusion_actions = tuple(bool(p > selected_threshold) for p in fusion_probs.tolist())
        fusion_tensor = torch.tensor(fusion_actions, dtype=torch.float32)
        placement_actions = ()

    fusion_logp = fusion_dist.log_prob(fusion_tensor).sum() if len(workflow.edges) else h.new_zeros(())
    fusion_entropy = fusion_dist.entropy().sum() if len(workflow.edges) else h.new_zeros(())
    # First decode topology and resource masks; stochastic placement actions
    # must be generated in the exact same topological order as the decoder.
    groups, _ = _partition_from_actions(
        workflow, fusion_actions, fusion_probs.detach().cpu().tolist(), reward_config.memory_sharing_gamma
    )
    if forced_actions is None and stochastic:
        remaining = {
            site: float(workflow.site_free_memory_mib.get(site, cap))
            for site, cap in workflow.site_memory_mib.items()
        }
        sites = tuple(sorted(workflow.site_memory_mib))
        node_index = {node: i for i, node in enumerate(workflow.nodes)}
        sampled_places: list[str | None] = []
        for group in groups:
            mem = _memory_of(group, workflow, reward_config.memory_sharing_gamma)
            feasible = [site for site in sites if remaining[site] + 1e-8 >= mem]
            if not feasible:
                selected = max(sites, key=lambda site: (remaining[site], site))
                sampled_places.append(None)
            else:
                emb = torch.stack([h[node_index[node]] for node in group]).mean(dim=0)
                raw, _ = model.placement_logits(emb, workflow, remaining)
                mask = torch.tensor([site in feasible for site in sites], dtype=torch.bool)
                dist = Categorical(logits=raw.masked_fill(~mask, torch.finfo(raw.dtype).min))
                index = int(dist.sample().item())
                selected = sites[index]
                sampled_places.append(selected)
            remaining[selected] -= mem
        placement_actions = tuple(sampled_places)

    plan, place_logp, place_entropy = _plan_from_action_tensors(
        model, workflow, fusion_actions, placement_actions, fusion_probs, h, reward_config
    )
    return GRLPlan(
        groups=plan.groups,
        placement=plan.placement,
        repair_edges=plan.repair_edges,
        placement_fallbacks=plan.placement_fallbacks,
        fusion_actions=plan.fusion_actions,
        placement_actions=plan.placement_actions,
        log_probability=fusion_logp + place_logp,
        entropy=fusion_entropy + place_entropy,
        value=plan.value,
        policy_rng_state=policy_rng_state,
        fusion_scores=tuple(float(v) for v in fusion_probs.detach().tolist()),
    )


def _recompute_joint_logp(
    model: GRLActorCritic,
    workflow: GRLWorkflow,
    plan: GRLPlan,
    config: GRLRewardConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    # Reuse the exact dropout mask realization from the sampled behavior
    # policy. Otherwise PPO's likelihood ratio would compare two independent
    # dropout realizations rather than only the updated policy parameters.
    saved_rng_state = torch.random.get_rng_state()
    if plan.policy_rng_state is not None:
        torch.random.set_rng_state(plan.policy_rng_state)
    try:
        return _recompute_joint_logp_inner(model, workflow, plan, config)
    finally:
        torch.random.set_rng_state(saved_rng_state)


def _recompute_joint_logp_inner(
    model: GRLActorCritic,
    workflow: GRLWorkflow,
    plan: GRLPlan,
    config: GRLRewardConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    h, _ = model.encode(workflow)
    logits = model.fusion_logits(h, workflow)
    dist = Bernoulli(logits=logits)
    actions = torch.tensor(plan.fusion_actions, dtype=torch.float32)
    logp = dist.log_prob(actions).sum() if actions.numel() else h.new_zeros(())
    entropy = dist.entropy().sum() if actions.numel() else h.new_zeros(())

    groups, _ = _partition_from_actions(
        workflow, plan.fusion_actions, plan.fusion_scores,
        config.memory_sharing_gamma,
    )
    remaining = {
        site: float(workflow.site_free_memory_mib.get(site, cap))
        for site, cap in workflow.site_memory_mib.items()
    }
    sites = tuple(sorted(workflow.site_memory_mib))
    node_index = {node: i for i, node in enumerate(workflow.nodes)}
    if len(groups) != len(plan.placement_actions):
        raise ValueError("plan placement action count no longer matches decoded groups")
    for i, group in enumerate(groups):
        mem = _memory_of(group, workflow, config.memory_sharing_gamma)
        feasible = [site for site in sites if remaining[site] + 1e-8 >= mem]
        chosen = plan.placement_actions[i]
        if not feasible:
            if chosen is not None:
                raise ValueError("fallback placement must not be represented as a policy action")
            selected = max(sites, key=lambda site: (remaining[site], site))
        else:
            if chosen is None or chosen not in feasible:
                raise ValueError("stored placement action violates its original feasibility mask")
            emb = torch.stack([h[node_index[node]] for node in group]).mean(dim=0)
            raw, _ = model.placement_logits(emb, workflow, remaining)
            mask = torch.tensor([site in feasible for site in sites], dtype=torch.bool)
            d = Categorical(logits=raw.masked_fill(~mask, torch.finfo(raw.dtype).min))
            index = sites.index(chosen)
            logp = logp + d.log_prob(torch.tensor(index))
            entropy = entropy + d.entropy()
            selected = chosen
        remaining[selected] -= mem
    value = model.critic_value(h, workflow)
    return logp, entropy, value


def evaluate_grl_plan(
    workflow: GRLWorkflow,
    plan: GRLPlan,
    reward_config: GRLRewardConfig,
) -> GRLMetrics:
    """Deterministic implementation of the paper's analytical objective model.

    Use a shared event-simulator callback for R8 comparisons. This evaluator
    is only a local mechanics fallback and has no queue contention.
    """
    workflow.validate()
    reward_config.validate()
    owner = {node: group for group in plan.groups for node in group}
    if set(owner) != set(workflow.nodes):
        raise ValueError("plan must partition every workflow node exactly once")
    group_index = {group: i for i, group in enumerate(plan.groups)}
    pred: dict[tuple[str, ...], set[tuple[str, ...]]] = {g: set() for g in plan.groups}
    transfer: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}
    for edge in workflow.edges:
        src_group, dst_group = owner[edge.src], owner[edge.dst]
        if src_group != dst_group:
            pred[dst_group].add(src_group)
            key = (src_group, dst_group)
            transfer[key] = transfer.get(key, 0.0) + float(edge.weight)

    finish: dict[tuple[str, ...], float] = {}
    execution_by_group: dict[tuple[str, ...], float] = {}
    cold_expected = 0.0
    for group in plan.groups:
        site = plan.placement[group]
        workload = sum(float(workflow.workload_cycles[node]) for node in group)
        exec_s = workload / float(workflow.site_cpu_cycles_s[site])
        execution_by_group[group] = exec_s
        incoming_finishes = []
        for parent in pred[group]:
            src_site = plan.placement[parent]
            payload_mib = transfer[(parent, group)]
            if src_site == site or payload_mib <= 0:
                network_s = 0.0
            else:
                bw = workflow.link_bandwidth_mbps.get((src_site, site))
                if bw is None:
                    bw = workflow.link_bandwidth_mbps.get((site, src_site))
                if bw is None or bw <= 0:
                    raise ValueError(f"missing positive bandwidth for {src_site!r}->{site!r}")
                fixed_link_s = workflow.link_latency_s.get(
                    (src_site, site), workflow.link_latency_s.get((site, src_site), 0.0)
                )
                network_s = float(fixed_link_s) + payload_mib * 8.0 / float(bw)
            incoming_finishes.append(finish[parent] + network_s)
        cold_p = sum(float(workflow.cold_probability.get(node, 0.0)) for node in group) / len(group)
        cold_s = cold_p * (
            reward_config.container_init_s
            + reward_config.cold_lambda_s_per_mib * _memory_of(
                group, workflow, reward_config.memory_sharing_gamma
            )
        )
        cold_expected += cold_p
        start_s = max(incoming_finishes, default=0.0) + cold_s
        finish[group] = start_s + exec_s

    exits = [g for g in plan.groups if not any(g in pred[other] for other in plan.groups)]
    latency = max((finish[g] for g in exits), default=0.0)
    cost = 0.0
    for group in plan.groups:
        site = plan.placement[group]
        cost += float(workflow.site_price_per_s[site]) * execution_by_group[group]
        cost += reward_config.memory_price * _memory_of(
            group, workflow, reward_config.memory_sharing_gamma
        )

    used_by_site = {
        site: workflow.site_memory_mib[site] - workflow.site_free_memory_mib.get(
            site, workflow.site_memory_mib[site]
        )
        for site in workflow.site_memory_mib
    }
    for group, site in plan.placement.items():
        used_by_site[site] += _memory_of(group, workflow, reward_config.memory_sharing_gamma)
    oom = sum(
        max(0.0, used - workflow.site_memory_mib[site]) / workflow.site_memory_mib[site]
        for site, used in used_by_site.items()
    )
    wan_bytes = sum(
        transfer[(src_group, dst_group)]
        for src_group, dst_group in transfer
        if plan.placement[src_group] != plan.placement[dst_group]
    )
    return GRLMetrics(latency, cost, oom, wan_bytes_mib=wan_bytes, cold_starts_expected=cold_expected)


def grl_reward(metrics: GRLMetrics, workflow: GRLWorkflow, cfg: GRLRewardConfig) -> float:
    """Equation (28): normalized latency and cost plus residual OOM penalty."""
    workflow.validate()
    cfg.validate()
    values = (metrics.latency_s, metrics.cost, metrics.oom_penalty)
    if any(not math.isfinite(float(x)) for x in values):
        raise ValueError("simulator metrics must be finite")
    return -(
        cfg.latency_weight * metrics.latency_s / workflow.baseline_latency_s
        + cfg.cost_weight * metrics.cost / workflow.baseline_cost
        + cfg.oom_weight * metrics.oom_penalty
    )


def _validate_splits(
    train: Sequence[GRLWorkflow], validation: Sequence[GRLWorkflow], heldout: Sequence[GRLWorkflow]
) -> None:
    for name, rows in (("train", train), ("validation", validation), ("heldout", heldout)):
        if not rows and name != "heldout":
            raise ValueError(f"{name} workflow split must not be empty")
        for workflow in rows:
            workflow.validate()
        ids = [workflow.workflow_id for workflow in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate workflow IDs within the {name} split")
    id_sets = {
        "train": {w.workflow_id for w in train},
        "validation": {w.workflow_id for w in validation},
        "heldout": {w.workflow_id for w in heldout},
    }
    if not train:
        raise ValueError("training workflow split must not be empty")
    for left, right in (("train", "validation"), ("train", "heldout"), ("validation", "heldout")):
        overlap = id_sets[left] & id_sets[right]
        if overlap:
            raise ValueError(f"workflow split leakage between {left} and {right}: {sorted(overlap)}")


def _deterministic_validation(
    model: GRLActorCritic,
    workflows: Sequence[GRLWorkflow],
    reward_cfg: GRLRewardConfig,
    evaluator: Callable[[GRLWorkflow, GRLPlan], GRLMetrics],
    threshold: float,
) -> tuple[float, float]:
    model.eval()
    rewards, latencies = [], []
    with torch.no_grad():
        for workflow in workflows:
            plan = sample_grl_plan(
                model, workflow, reward_cfg, stochastic=False, threshold=threshold
            )
            metrics = evaluator(workflow, plan)
            rewards.append(grl_reward(metrics, workflow, reward_cfg))
            latencies.append(metrics.latency_s)
    return sum(rewards) / len(rewards), sum(latencies) / len(latencies)


def train_grl_orch(
    train_workflows: Sequence[GRLWorkflow],
    validation_workflows: Sequence[GRLWorkflow],
    *,
    heldout_workflows: Sequence[GRLWorkflow] = (),
    evaluator: Callable[[GRLWorkflow, GRLPlan], GRLMetrics] | None = None,
    reward_config: GRLRewardConfig = GRLRewardConfig(),
    train_config: GRLTrainConfig = GRLTrainConfig(),
    checkpoint_path: str | None = None,
) -> tuple[GRLActorCritic, GRLTrainResult]:
    """Train PPO only on train IDs; select checkpoint/threshold on validation.

    The paper reports one workflow-level reward after plan execution. This
    implementation therefore treats the factorized plan as one compound
    terminal action; GAE reduces to the one-step terminal advantage here.
    """
    _validate_splits(train_workflows, validation_workflows, heldout_workflows)
    reward_config.validate()
    evaluate = evaluator or (
        lambda workflow, plan: evaluate_grl_plan(workflow, plan, reward_config)
    )
    if train_config.episodes <= 0 or train_config.rollout_size <= 0:
        raise ValueError("episodes and rollout_size must be positive")
    if train_config.validation_interval_episodes <= 0 or train_config.ppo_epochs <= 0:
        raise ValueError("validation interval and PPO epochs must be positive")
    if train_config.learning_rate <= 0 or train_config.gradient_clip_norm <= 0:
        raise ValueError("learning rate and gradient clip norm must be positive")
    if not (0.0 <= train_config.discount_gamma <= 1.0 and 0.0 <= train_config.gae_lambda <= 1.0):
        raise ValueError("discount gamma and GAE lambda must be in [0,1]")
    if not (0.0 < train_config.clip_epsilon < 1.0 and 0.0 <= train_config.dropout < 1.0):
        raise ValueError("clip epsilon must be in (0,1) and dropout in [0,1)")
    if train_config.hidden_dim <= 0 or train_config.attention_heads <= 0:
        raise ValueError("hidden dimension and attention heads must be positive")
    if train_config.hidden_dim % train_config.attention_heads:
        raise ValueError("hidden_dim must be divisible by attention_heads")
    if any(not 0.0 <= x <= 1.0 for x in train_config.threshold_grid):
        raise ValueError("fusion thresholds must be in [0,1]")

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    device = torch.device("cpu")
    model = GRLActorCritic(train_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    initial_actor = torch.cat([p.detach().flatten() for p in model.fusion_head.parameters()] +
                              [p.detach().flatten() for p in model.placement_head.parameters()]).clone()
    initial_critic = torch.cat([p.detach().flatten() for p in model.critic.parameters()]).clone()
    initial_encoder = torch.cat(
        [p.detach().flatten() for p in model.node_projection.parameters()]
        + [p.detach().flatten() for p in model.gat.parameters()]
    ).clone()
    rng = random.Random(train_config.seed)
    history: list[Mapping[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_reward = -float("inf")
    best_episode = 0
    updates = 0
    episodes_done = 0
    started = time.perf_counter()

    while episodes_done < train_config.episodes:
        count = min(train_config.rollout_size, train_config.episodes - episodes_done)
        samples: list[tuple[GRLWorkflow, GRLPlan, float]] = []
        for _ in range(count):
            workflow = train_workflows[rng.randrange(len(train_workflows))]
            plan = sample_grl_plan(
                model, workflow, reward_config, stochastic=True,
                threshold=reward_config.fusion_threshold,
            )
            metrics = evaluate(workflow, plan)
            reward = grl_reward(metrics, workflow, reward_config)
            samples.append((workflow, plan, reward))
        episodes_done += count

        old_logp = torch.stack([row[1].log_probability.detach() for row in samples])
        returns = torch.tensor([row[2] for row in samples], dtype=torch.float32)
        old_values = torch.stack([row[1].value.detach() for row in samples])
        advantages = returns - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        loss_values: list[float] = []
        for _epoch in range(train_config.ppo_epochs):
            new_logps, entropies, values = [], [], []
            for workflow, plan, _reward in samples:
                logp, entropy, value = _recompute_joint_logp(model, workflow, plan, reward_config)
                new_logps.append(logp)
                entropies.append(entropy)
                values.append(value)
            new_logp = torch.stack(new_logps)
            entropy = torch.stack(entropies).mean()
            value = torch.stack(values)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio, 1.0 - train_config.clip_epsilon, 1.0 + train_config.clip_epsilon
            ) * advantages
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            critic_loss = F.mse_loss(value, returns)
            loss = actor_loss + 0.5 * critic_loss - train_config.entropy_coefficient * entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite PPO loss; aborting training")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("non-finite PPO gradient norm; aborting training")
            optimizer.step()
            loss_values.append(float(loss.detach()))
        updates += 1
        history.append({
            "episode": float(episodes_done),
            "mean_train_reward": float(returns.mean()),
            "mean_actor_advantage": float(advantages.mean()),
            "mean_loss": sum(loss_values) / len(loss_values),
        })

        should_validate = (
            episodes_done % train_config.validation_interval_episodes == 0
            or episodes_done == train_config.episodes
        )
        if should_validate:
            val_reward, _ = _deterministic_validation(
                model, validation_workflows, reward_config, evaluate,
                reward_config.fusion_threshold,
            )
            history[-1] = {**history[-1], "validation_reward_at_0.5": val_reward}
            if val_reward > best_reward:
                best_reward = val_reward
                best_episode = episodes_done
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    threshold_scores = []
    for threshold in train_config.threshold_grid:
        score, _ = _deterministic_validation(
            model, validation_workflows, reward_config, evaluate, threshold
        )
        threshold_scores.append((score, threshold))
    selected_threshold = max(threshold_scores, key=lambda row: (row[0], -row[1]))[1]
    final_actor = torch.cat([p.detach().flatten() for p in model.fusion_head.parameters()] +
                            [p.detach().flatten() for p in model.placement_head.parameters()])
    final_critic = torch.cat([p.detach().flatten() for p in model.critic.parameters()])
    final_encoder = torch.cat(
        [p.detach().flatten() for p in model.node_projection.parameters()]
        + [p.detach().flatten() for p in model.gat.parameters()]
    )
    actor_delta = float(torch.linalg.vector_norm(final_actor - initial_actor))
    critic_delta = float(torch.linalg.vector_norm(final_critic - initial_critic))
    encoder_delta = float(torch.linalg.vector_norm(final_encoder - initial_encoder))

    heldout_reward = heldout_latency = None
    if heldout_workflows:
        heldout_reward, heldout_latency = _deterministic_validation(
            model, heldout_workflows, reward_config, evaluate, selected_threshold
        )
    elapsed = time.perf_counter() - started
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "train_config": vars(train_config),
            "reward_config": vars(reward_config),
            "selected_fusion_threshold": selected_threshold,
            "best_validation_reward": best_reward,
            "best_episode": best_episode,
            "training_workflow_ids": sorted({w.workflow_id for w in train_workflows}),
            "validation_workflow_ids": sorted({w.workflow_id for w in validation_workflows}),
            "heldout_workflow_ids": sorted({w.workflow_id for w in heldout_workflows}),
        }, checkpoint_path)
    result = GRLTrainResult(
        best_validation_reward=best_reward,
        selected_fusion_threshold=selected_threshold,
        best_episode=best_episode,
        updates=updates,
        episodes_seen=episodes_done,
        final_actor_parameter_delta=actor_delta,
        final_critic_parameter_delta=critic_delta,
        final_encoder_parameter_delta=encoder_delta,
        history=tuple(history),
        heldout_reward=heldout_reward,
        heldout_latency_s=heldout_latency,
        training_seconds=elapsed,
    )
    return model, result


__all__ = [
    "GRLActorCritic", "GRLPlan", "GRLMetrics", "GRLRewardConfig", "GRLTrainConfig",
    "GRLTrainResult", "GRLWorkflow", "evaluate_grl_plan", "grl_reward",
    "sample_grl_plan", "train_grl_orch",
]
