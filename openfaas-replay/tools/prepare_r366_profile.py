#!/usr/bin/env python3
"""Regenerate R366's synthetic 8-function diamond profile and initial layouts."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORE = REPO / "easr-simulation/research/rl/r187_twenty_function_scale"
sys.path.insert(0, str(CORE))
import core as B  # noqa: E402

PROFILE_ID = 326001
SOURCE_INDEX = PROFILE_ID - 326000


def raw_r326_profile() -> dict:
    """Reproduce R326's deterministic 20-node profile at index 1."""
    rng = random.Random(3261000 + SOURCE_INDEX)
    n = 20
    edges = []
    for a in range(0, 18, 3):
        edges.extend(((a, a + 1), (a, a + 2), (a + 1, a + 3), (a + 2, a + 3)))
    edges.append((18, 19))
    return {
        "id": PROFILE_ID,
        "family": "diamond",
        "work": [rng.uniform(0.015, 0.16) for _ in range(n)],
        "mem": [rng.choice((32, 64, 96)) for _ in range(n)],
        "edges": [(u, v, rng.choice((0.01, 0.1, 1.0))) for u, v in edges],
        "ram": [1280, 10240],
        "cores": [5, 15],
        "speed": [1.0, rng.uniform(1.5, 3.0)],
        "mbps": rng.choice((10.0, 50.0, 100.0)),
        "rtt": rng.choice((0.01, 0.05, 0.08)),
    }


def topological_order(n: int, edges: list[tuple[int, int, float]]) -> list[int]:
    children = [[] for _ in range(n)]
    indegree = [0] * n
    for u, v, _ in edges:
        children[u].append(v)
        indegree[v] += 1
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    order = []
    while ready:
        u = min(ready)
        ready.remove(u)
        order.append(u)
        for v in children[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                ready.append(v)
    if len(order) != n:
        raise ValueError("profile is not a DAG")
    return order


def build() -> dict:
    source = raw_r326_profile()
    order = topological_order(len(source["work"]), source["edges"])[:8]
    remap = {old: new for new, old in enumerate(order)}
    edges = [[remap[u], remap[v], amount]
             for u, v, amount in source["edges"] if u in remap and v in remap]
    if len(edges) != 9:
        raise RuntimeError(f"expected 9 edges in the 8-function diamond, got {len(edges)}")
    profile = {
        "id": PROFILE_ID,
        "parent_profile_id": PROFILE_ID,
        "family": "diamond",
        "work": [source["work"][i] for i in order],
        "mem": [source["mem"][i] for i in order],
        "edges": edges,
        "selected_source_nodes": order,
        "projection": "first 8 nodes in deterministic topological order; induced sub-DAG",
        "ram": [512, 4096],
        "cores": [2, 6],
        "speed": source["speed"],
        "mbps": source["mbps"],
        "rtt": source["rtt"],
    }
    pool = B.plan_pool(profile, 3282000 + PROFILE_ID, 32)
    rng = random.Random(3284000 + PROFILE_ID)
    initials = rng.sample(pool, min(4, len(pool)))
    for site in (0, 1):
        full = (tuple([0] * len(profile["work"])), (site,))
        if B.valid(profile, *full) and full not in initials:
            initials.append(full)
    return {
        "experiment": "R366_REGENERATED_SYNTHETIC_WORKFLOW_FIXTURE",
        "n_functions": 8,
        "profiles": [{
            "profile": profile,
            "initials": initials,
            "plan_candidates": len(pool),
            "split": "external_real_profiles",
        }],
        "construction": "R326 profile index 1; deterministic topological projection to 8 nodes; R328 resource scaling and initial-layout sampling.",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
    print(f"wrote synthetic profile {PROFILE_ID} to {args.output}")


if __name__ == "__main__":
    main()
