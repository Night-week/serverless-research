#!/usr/bin/env python3
"""R328: train and evaluate the full Scenario 1 mechanism at eight functions."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
R187 = REPO_ROOT / "easr-simulation" / "research" / "rl" / "r187_twenty_function_scale"
sys.path.insert(0, str(R187))
import run as R  # noqa: E402

torch = R.torch
B = R.B
Engine = R.Engine
BASE_POLICY = R.Policy
ROOT = HERE / "outputs"
N = 8
SEEDS = (601, 631, 659)
T0 = 15.0
POOL = 32
TRAIN_PROFILES = 48
VALIDATION_PROFILES = 12
PRIMARY_CONFIRMATION_PROFILES = 48
RF_THRESHOLD = 0.97
PRESSURE_THRESHOLD = 1.0
EXTERNAL_PROFILE_FILE = HERE / "scale8_profiles.json"
EXTERNAL_R326_MANIFEST = HERE / "r326_manifest.json"


class Scale8Policy(BASE_POLICY):
    """R187 policy with its state-step coordinate moved to the eight-node slot."""

    def __init__(self, dim: int):
        super().__init__(dim)
        self.n = N

    def forward(self, x):
        lp, value, entropy = self.policy(x)
        step_index = 2 * self.n + self.n * (self.n - 1) // 2 + 7
        if len(x) > 1 and float(x[0, step_index].detach()) > 0:
            hidden, context = self.context(x)
            delta = self.relative_value(
                torch.cat((context.expand(len(hidden) - 1, -1), hidden[1:] - hidden[0]), 1)
            ).flatten()
            lp = torch.log_softmax(
                lp + torch.cat((delta.new_zeros(1), torch.tanh(delta / 0.1))).detach(), 0
            )
            entropy = torch.distributions.Categorical(logits=lp).entropy()
        return lp, value, entropy


R.ROOT = ROOT
R.Policy = Scale8Policy
R.SEEDS = SEEDS
R.BUDGET = 32768
R.read_manifest = lambda: json.loads((ROOT / "manifest.json").read_text())
# R187's evaluation fits its original initial-state RF. R328 fits the R326/R199
# live-snapshot RF below, so skip only that old evaluation hook during training.
R.evaluate = lambda *args, **kwargs: None


def write(path: Path, value) -> None:
    R.write(path, value)


def source_files() -> dict[str, Path]:
    return {
        "r187_run": R187 / "run.py",
        "r187_core": R187 / "core.py",
        "r187_engine": R187 / "engine.py",
        "scale8_runner": Path(__file__),
        "protocol": HERE / "PROTOCOL.md",
        "external_real_profiles": EXTERNAL_PROFILE_FILE,
        "external_r326_source": EXTERNAL_R326_MANIFEST,
        "frozen_stopping_policy": R.FROZEN / "r151_stopping_ppo_smoke.py",
        "shared_workflow_core": R.FROZEN / "workflow_core.py",
    }


def verify_bindings(manifest: dict) -> None:
    for key, path in source_files().items():
        expected = manifest["source_hashes"][key]
        actual = R.sha(path)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch: {key}: {actual} != {expected}")


def scale_profile(index: int) -> dict:
    rng = random.Random(1871000 + index)
    family = ("chain", "diamond", "tree")[index % 3]
    if family == "chain":
        edges = [(v, v + 1) for v in range(N - 1)]
    elif family == "tree":
        edges = [((v - 1) // 2, v) for v in range(1, N)]
    else:
        edges = []
        last = 0
        for start in range(0, N - 3, 3):
            edges.extend([(start, start + 1), (start, start + 2),
                          (start + 1, start + 3), (start + 2, start + 3)])
            last = start + 3
        edges.extend((v, v + 1) for v in range(last, N - 1))
    cores = math.ceil(N / 4)
    return {
        "id": 328000 + index,
        "family": family,
        "work": [rng.uniform(0.015, 0.16) for _ in range(N)],
        "mem": [rng.choice([32, 64, 96]) for _ in range(N)],
        "edges": [(u, v, rng.choice([0.01, 0.1, 1.0])) for u, v in edges],
        "ram": [64 * N, 512 * N],
        "cores": [cores, 3 * cores],
        "speed": [1.0, rng.uniform(1.5, 3.0)],
        "mbps": rng.choice([10.0, 50.0, 100.0]),
        "rtt": rng.choice([0.01, 0.05, 0.08]),
    }


def external_profiles() -> list[dict]:
    source = json.loads(EXTERNAL_PROFILE_FILE.read_text())
    r326 = json.loads(EXTERNAL_R326_MANIFEST.read_text())
    originals = {block["profile"]["id"]: block["profile"] for block in r326["profiles"]}
    blocks = []
    for source_block in source["profiles"]:
        profile = copy.deepcopy(source_block["profile"])
        original = originals[profile["parent_profile_id"]]
        profile["ram"] = [512, 4096]
        profile["cores"] = [2, 6]
        for key in ("speed", "mbps", "rtt"):
            profile[key] = copy.deepcopy(original[key])
        blocks.append(profile)
    return blocks


def candidate_block(profile: dict, split: str, index: int) -> dict:
    started = time.perf_counter()
    pool = B.plan_pool(profile, 3282000 + int(profile["id"]), POOL)
    candidate_generation_seconds = time.perf_counter() - started
    history_seed = 3283000 + int(profile["id"])
    history = B.flow(history_seed)
    scored = []
    scoring_started = time.perf_counter()
    for layout in pool:
        engine = Engine(profile, layout, history)
        scored.append((engine.finish()["mean"], layout))
    plan = min(scored, key=lambda row: (row[0], row[1]))[1]
    candidate_scoring_seconds = time.perf_counter() - scoring_started
    rng = random.Random(3284000 + int(profile["id"]))
    initials = rng.sample(pool, min(4, len(pool)))
    for site in (0, 1):
        full = (tuple([0] * len(profile["work"])), (site,))
        if B.valid(profile, *full) and full not in initials:
            initials.append(full)
    return {
        "profile": profile,
        "initials": initials,
        "plan": plan,
        "plan_candidates": len(pool),
        "plan_history_seed": history_seed,
        "split": split,
        "index": index,
        "candidate_generation_seconds": candidate_generation_seconds,
        "candidate_scoring_seconds": candidate_scoring_seconds,
    }


def prepare() -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    dest = ROOT / "manifest.json"
    if dest.exists():
        manifest = json.loads(dest.read_text())
        verify_bindings(manifest)
        return manifest

    current_hashes = {key: R.sha(path) for key, path in source_files().items()}
    cache_path = ROOT / "profile_cache.json"
    cached = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("source_hashes") == current_hashes:
            cached = {block["profile"]["id"]: block for block in cache["profiles"]}
    profiles = []
    for index in range(TRAIN_PROFILES + VALIDATION_PROFILES + PRIMARY_CONFIRMATION_PROFILES):
        split = ("train" if index < TRAIN_PROFILES else
                 "validation" if index < TRAIN_PROFILES + VALIDATION_PROFILES else
                 "confirmation")
        profile = scale_profile(index)
        block = cached.get(profile["id"])
        if block is None:
            block = candidate_block(profile, split, index)
            cached[profile["id"]] = block
            write(cache_path, {"source_hashes": current_hashes,
                               "profiles": list(cached.values())})
        profiles.append(block)
        if (index + 1) % 12 == 0:
            write(ROOT / "status.json", {"phase": "planning", "profiles": index + 1,
                                          "total": 108})
            print(f"R328_PLAN {index + 1}/108", flush=True)

    for index, profile in enumerate(external_profiles()):
        block = cached.get(profile["id"])
        if block is None:
            block = candidate_block(profile, "external_real_profiles", index)
            cached[profile["id"]] = block
            write(cache_path, {"source_hashes": current_hashes,
                               "profiles": list(cached.values())})
        profiles.append(block)
        print(f"R328_EXTERNAL_PLAN {index + 1}/9", flush=True)

    manifest = {
        "experiment": "R328_SCENE1_SCALE8_FULL_MECHANISM",
        "n_functions": N,
        "seeds": list(SEEDS),
        "ppo_budget_per_seed": R.BUDGET,
        "plan_candidate_count": POOL,
        "local_action_cap": 2,
        "decision_snapshot_seconds": T0,
        "rf_label": f"historical whole32 mean < {RF_THRESHOLD} * local mean",
        "pressure_override_threshold": PRESSURE_THRESHOLD,
        "resource_budgets": {"cores": [2, 6], "ram_mib": [512, 4096]},
        "profiles": profiles,
        "source_hashes": current_hashes,
        "external_profile_sha256": R.sha(EXTERNAL_PROFILE_FILE),
        "external_r326_manifest_sha256": R.sha(EXTERNAL_R326_MANIFEST),
        "real_platform_reference": {
            "source_experiment": "OpenFaaS 8-function reduced Scenario 1 validation",
            "profile_count": 9,
            "physical_nodes": 1,
            "pool_budgets_match": True,
            "external_profiles_are_held_out": True,
        },
        "scope": "Matched 8-function simulator confirmation plus a held-out replay of the 9 profiles used on OpenFaaS. The OpenFaaS replay is still simulation; live control is not claimed.",
    }
    write(dest, manifest)
    verify_bindings(manifest)
    return manifest


def load_policy(seed: int):
    path = ROOT / f"seed_{seed}" / "selected.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = Scale8Policy(checkpoint["dim"])
    net.load_state_dict(checkpoint["state"])
    net.eval()
    return net, R.sha(path)


def feature_vector(engine, net) -> list[float]:
    profile = engine.p
    pool = B.neighbors(profile, engine.layout)
    observed = R.observe(engine, pool, 0)
    with torch.no_grad():
        logits = net(torch.tensor(observed, dtype=torch.float32))[0]
        action = int(logits.argmax())
    past = [t - 60 for t in B.flow(152300000 + profile["id"])]
    known = past + [t for t in engine.arrivals if t <= engine.now]
    rate = sum(engine.now - 60 < t <= engine.now for t in known) / 60
    pressures = [R.pressure(profile, layout, rate) for layout in pool]
    best_pressure = min(pressures, key=lambda item: item[2])
    chosen_pressure = pressures[action]
    return observed[0] + [
        math.log1p(best_pressure[2]),
        sum(item[2] < 1 for item in pressures) / len(pressures),
        *[math.asinh(value) for value in best_pressure[0] + best_pressure[1]],
        float(action == 0),
        *[math.asinh(value) for value in chosen_pressure[0] + chosen_pressure[1]],
        math.asinh(chosen_pressure[2]),
        math.log1p(B.cost(profile, engine.layout, pool[action])),
        float(logits[action].exp()),
    ]


def finish(engine, t0: float, pending: list[int]) -> dict:
    result = engine.finish()
    by_request = dict(engine.responses)
    for name, ids in (
        ("pending", pending),
        ("future", [i for i, arrival in enumerate(engine.arrivals) if arrival > t0]),
        ("affected", sorted(set(pending) | {i for i, arrival in enumerate(engine.arrivals)
                                           if arrival > t0})),
    ):
        values = sorted(by_request[i] for i in ids)
        result[name] = {
            "n": len(values),
            "mean": statistics.mean(values) if values else None,
            "p95": values[math.ceil(0.95 * len(values)) - 1] if values else None,
        }
    return result


def run_local(engine, net, t0: float, pending: list[int]) -> dict:
    actions = []
    for step in range(2):
        pool = [engine.layout] + [layout for layout in B.neighbors(engine.p, engine.layout)[1:]
                                  if layout not in engine.visited]
        with torch.no_grad():
            logits = net(torch.tensor(R.observe(engine, pool, step), dtype=torch.float32))[0]
            action = int(logits.argmax())
        actions.append(action)
        if action == 0:
            break
        engine.change(pool[action])
        if step == 0:
            engine.advance(max(t0 + 15.0, engine.now))
            if engine.now >= 60:
                break
    result = finish(engine, t0, pending)
    result["actions"] = actions
    return result


def snapshot(profile: dict, old, stream_seed: int):
    arrivals = B.flow(stream_seed)
    engine = Engine(copy.deepcopy(profile), R.layout(old), arrivals)
    engine.advance(T0)
    completed_before = {index for index, _ in engine.responses}
    pending = [index for index, arrival in enumerate(arrivals)
               if arrival <= T0 and index not in completed_before]
    return engine, pending, arrivals


def simulate(seed: int, net, gate, block: dict, initial_index: int, stream: int) -> dict:
    profile = block["profile"]
    old = block["initials"][initial_index]
    stream_seed = 3285000 + int(profile["id"]) * 10 + stream
    engine, pending, arrivals = snapshot(profile, old, stream_seed)

    started = time.perf_counter()
    features = feature_vector(engine, net)
    feature_seconds = time.perf_counter() - started
    started = time.perf_counter()
    rf_route = bool(gate.predict([features])[0])
    rf_seconds = time.perf_counter() - started
    pressure = math.sinh(features[-3])

    local_engine = copy.deepcopy(engine)
    started = time.perf_counter()
    local = run_local(local_engine, net, T0, pending)
    local_seconds = time.perf_counter() - started

    whole_engine = copy.deepcopy(engine)
    started = time.perf_counter()
    whole_engine.change(R.layout(block["plan"]))
    whole = finish(whole_engine, T0, pending)
    whole_seconds = time.perf_counter() - started

    keep = finish(copy.deepcopy(engine), T0, pending)
    base = whole if rf_route else local
    pressure_override = pressure > PRESSURE_THRESHOLD
    system = whole if (rf_route or pressure_override) else local
    for name, result in (("local", local), ("whole32", whole), ("keep", keep),
                         ("system_base", base), ("system", system)):
        if result["completed"] != len(arrivals):
            raise RuntimeError(f"incomplete request stream: {seed}/{profile['id']}/{name}")
    if any(result["peak_ram_fraction"] > 1 + 1e-9
           for result in (local, whole, keep, system)):
        raise RuntimeError(f"resource overflow: {seed}/{profile['id']}/{initial_index}/{stream}")
    return {
        "seed": seed,
        "profile": profile["id"],
        "family": profile["family"],
        "initial": initial_index,
        "stream": stream,
        "stream_seed": stream_seed,
        "rf_route_whole": int(rf_route),
        "pressure": pressure,
        "pressure_override": int(pressure_override),
        "full_system_whole": int(rf_route or pressure_override),
        "decision_timing_seconds": {
            "feature_construction_and_ppo_first_action": feature_seconds,
            "rf_inference": rf_seconds,
            "two_step_local_rollout": local_seconds,
            "whole32_transition_and_finish": whole_seconds,
        },
        "methods": {"system": system, "system_base": base, "local": local,
                    "whole32": whole, "keep": keep},
    }


def fit_gate(seed: int, net, manifest: dict):
    dest = ROOT / f"seed_{seed}"
    gate_path = dest / "rf_live_snapshot.joblib"
    if gate_path.exists():
        return R.joblib.load(gate_path)
    features, labels, audit = [], [], []
    train_states = R.states(manifest, "train")
    for number, (block, old, initial_index) in enumerate(train_states):
        profile = block["profile"]
        for stream in range(2):
            stream_seed = 3286000 + int(profile["id"]) * 10 + stream
            engine, pending, arrivals = snapshot(profile, old, stream_seed)
            x = feature_vector(engine, net)
            local = run_local(copy.deepcopy(engine), net, T0, pending)
            whole_engine = copy.deepcopy(engine)
            whole_engine.change(R.layout(block["plan"]))
            whole = finish(whole_engine, T0, pending)
            label = int(whole["mean"] < RF_THRESHOLD * local["mean"])
            features.append(x)
            labels.append(label)
            audit.append({"profile": profile["id"], "initial": initial_index,
                          "stream": stream, "local_mean": local["mean"],
                          "whole32_mean": whole["mean"], "label_whole": label,
                          "pressure": math.sinh(x[-3])})
        if (number + 1) % 40 == 0:
            write(dest / "rf_status.json", {"phase": "labeling_train_states",
                                              "states": number + 1,
                                              "total": len(train_states)})
    gate = R.RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=6,
        class_weight="balanced", random_state=121, n_jobs=1,
    ).fit(features, labels)
    R.joblib.dump(gate, gate_path)
    write(dest / "rf_training_audit.json", audit)
    write(dest / "rf_fit.json", {
        "training_states": len(train_states),
        "training_examples": len(labels),
        "positive_whole_rate": statistics.mean(labels),
        "feature_dim": len(features[0]),
        "threshold": RF_THRESHOLD,
        "model_seed": seed,
    })
    return gate


def evaluate_seed(seed: int, manifest: dict) -> None:
    torch.set_num_threads(1)
    net, model_hash = load_policy(seed)
    gate = fit_gate(seed, net, manifest)
    dest = ROOT / f"seed_{seed}"
    for split in ("confirmation", "external_real_profiles"):
        blocks = [block for block in manifest["profiles"] if block["split"] == split]
        for profile_number, block in enumerate(blocks, 1):
            profile = block["profile"]
            output = dest / f"{split}_{profile['id']}.json"
            if output.exists():
                continue
            rows = []
            for initial_index, _ in enumerate(block["initials"]):
                for stream in range(3):
                    rows.append(simulate(seed, net, gate, block, initial_index, stream))
            write(output, {"seed": seed, "profile": profile["id"],
                           "split": split, "model_sha256": model_hash, "rows": rows})
            status = {"phase": "evaluation", "seed": seed, "split": split,
                      "profiles": profile_number, "total": len(blocks)}
            write(ROOT / "status.json", status)
            print(f"R328_EVAL {seed} {split} {profile_number}/{len(blocks)} "
                  f"profile={profile['id']}", flush=True)


def percentile(values, q=0.95):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def clustered_bootstrap(rows: list[dict], method: str, iterations: int = 10000) -> dict:
    by_profile = {}
    for row in rows:
        by_profile.setdefault(row["profile"], []).append(row)
    profile_ids = sorted(by_profile)
    profile_diff = {
        pid: statistics.mean(r["methods"][method]["mean"] -
                             r["methods"]["whole32"]["mean"] for r in by_profile[pid])
        for pid in profile_ids
    }
    profile_values = {
        pid: (statistics.mean(r["methods"]["whole32"]["mean"] for r in by_profile[pid]),
              statistics.mean(r["methods"][method]["mean"] for r in by_profile[pid]))
        for pid in profile_ids
    }
    rng = random.Random(3289900 + len(profile_ids))
    differences, gains = [], []
    for _ in range(iterations):
        chosen = [rng.choice(profile_ids) for _ in profile_ids]
        differences.append(statistics.mean(profile_diff[pid] for pid in chosen))
        whole_mean = statistics.mean(profile_values[pid][0] for pid in chosen)
        method_mean = statistics.mean(profile_values[pid][1] for pid in chosen)
        gains.append(100 * (1 - method_mean / whole_mean))
    differences.sort()
    gains.sort()
    return {
        "profiles": len(profile_ids),
        "mean_difference_seconds": statistics.mean(profile_diff.values()),
        "difference_ci95": [differences[249], differences[9749]],
        "relative_improvement_pct": statistics.mean(
            100 * (1 - value[1] / value[0]) for value in profile_values.values()),
        "relative_improvement_ci95": [gains[249], gains[9749]],
    }


def summarize_split(manifest: dict, split: str) -> dict:
    rows = []
    for seed in SEEDS:
        seed_dir = ROOT / f"seed_{seed}"
        for block in manifest["profiles"]:
            if block["split"] != split:
                continue
            path = seed_dir / f"{split}_{block['profile']['id']}.json"
            rows.extend(json.loads(path.read_text())["rows"])
    methods = {}
    for name in ("system", "system_base", "local", "whole32", "keep"):
        metrics = {key: statistics.mean(row["methods"][name][key] for row in rows)
                   for key in ("mean", "p95", "deployment", "drain", "changes")}
        metrics["over_5s_vs_whole32"] = sum(
            row["methods"][name]["mean"] - row["methods"]["whole32"]["mean"] > 5
            for row in rows)
        metrics["bootstrap_vs_whole32"] = (None if name == "whole32" else
                                            clustered_bootstrap(rows, name))
        methods[name] = metrics
    route_rows = [row for row in rows]
    decision_timing = {
        key: statistics.mean(row["decision_timing_seconds"][key] for row in rows)
        for key in next(iter(rows))["decision_timing_seconds"]
    }
    per_seed = []
    for seed in SEEDS:
        subset = [row for row in rows if row["seed"] == seed]
        for name in ("system", "system_base", "local", "whole32", "keep"):
            per_seed.append({"seed": seed, "method": name,
                             "mean_response_seconds": statistics.mean(
                                 row["methods"][name]["mean"] for row in subset)})
    all_layouts = [block for block in manifest["profiles"] if block["split"] == split]
    search_times = [block["candidate_generation_seconds"] +
                    block["candidate_scoring_seconds"] for block in all_layouts]
    return {
        "split": split,
        "profiles": len(all_layouts),
        "records": len(rows),
        "methods": methods,
        "routing": {
            "rf_whole_rate": statistics.mean(row["rf_route_whole"] for row in route_rows),
            "pressure_override_rate": statistics.mean(row["pressure_override"]
                                                       for row in route_rows),
            "whole_route_rate": statistics.mean(row["full_system_whole"]
                                                 for row in route_rows),
        },
        "decision_compute_wall_seconds_per_record": decision_timing,
        "whole32_search_wall_seconds_per_profile": {
            "mean": statistics.mean(search_times),
            "p95": percentile(search_times),
            "max": max(search_times),
            "candidates_mean": statistics.mean(block["plan_candidates"]
                                                for block in all_layouts),
        },
        "per_seed_means": per_seed,
        "all_requests_completed": all(
            row["methods"][name]["completed"] == row["methods"]["whole32"]["completed"]
            for row in rows for name in ("system", "system_base", "local", "keep")),
        "resource_legal": all(
            row["methods"][name]["peak_ram_fraction"] <= 1 + 1e-9
            for row in rows for name in ("system", "system_base", "local", "whole32", "keep")),
    }


def train_all() -> None:
    for seed in SEEDS:
        seed_dir = ROOT / f"seed_{seed}"
        if (seed_dir / "result.json").exists():
            print(f"R328_TRAIN_SKIP {seed} already completed", flush=True)
            continue
        write(ROOT / "status.json", {"phase": "training", "seed": seed,
                                      "budget": R.BUDGET, "seeds": list(SEEDS)})
        print(f"R328_TRAIN_START seed={seed} budget={R.BUDGET}", flush=True)
        R.train(seed)
        if not (seed_dir / "selected.pt").exists():
            raise RuntimeError(f"training did not produce selected model for seed {seed}")
        print(f"R328_TRAIN_DONE seed={seed}", flush=True)


def run_all() -> None:
    try:
        manifest = prepare()
        verify_bindings(manifest)
        write(ROOT / "status.json", {"phase": "prepared", "profiles": len(manifest["profiles"]),
                                      "seeds": list(SEEDS)})
        train_all()
        for seed in SEEDS:
            write(ROOT / "status.json", {"phase": "fit_rf_and_evaluate", "seed": seed})
            evaluate_seed(seed, manifest)
        summary = {
            "experiment": manifest["experiment"],
            "seeds": list(SEEDS),
            "ppo_budget_per_seed": R.BUDGET,
            "primary_confirmation": summarize_split(manifest, "confirmation"),
            "external_real_profile_replay": summarize_split(manifest, "external_real_profiles"),
            "protocol_sha256": R.sha(HERE / "PROTOCOL.md"),
            "manifest_sha256": R.sha(ROOT / "manifest.json"),
            "source_hashes": manifest["source_hashes"],
            "interpretation": "The external replay uses the 9 OpenFaaS workload profiles, but runs through the simulator. It validates scale transfer, not live closed-loop control on OpenFaaS.",
        }
        write(ROOT / "summary.json", summary)
        write(ROOT / "status.json", {"phase": "completed", "summary": str(ROOT / "summary.json")})
        print("R328_ALL_FINISHED", flush=True)
    except Exception:
        write(ROOT / "status.json", {"phase": "failed", "traceback": traceback.format_exc()})
        raise


def preflight() -> dict:
    manifest = prepare()
    verify_bindings(manifest)
    train_block = next(block for block in manifest["profiles"] if block["split"] == "train")
    profile = train_block["profile"]
    layout = R.layout(train_block["initials"][0])
    neighbors = B.neighbors(profile, layout)
    assert all(B.valid(profile, *candidate) for candidate in neighbors)
    engine = Engine(profile, layout, B.flow(3287000 + profile["id"]))
    first = R.observe(engine, neighbors, 0)
    second = R.observe(engine, neighbors, 1)
    step_index = 2 * N + N * (N - 1) // 2 + 7
    assert first[0][step_index] == 0 and abs(second[0][step_index] - math.asinh(1.0)) < 1e-9
    net = Scale8Policy(len(first[0]))
    with torch.no_grad():
        logits, _, _ = net(torch.tensor(first, dtype=torch.float32))
    assert torch.isfinite(logits).all()
    external = next(block for block in manifest["profiles"]
                    if block["split"] == "external_real_profiles")
    current, pending, arrivals = snapshot(external["profile"], external["initials"][0],
                                          3287001 + external["profile"]["id"])
    feat = feature_vector(current, net)
    local = run_local(copy.deepcopy(current), net, T0, pending)
    whole_engine = copy.deepcopy(current)
    whole_engine.change(R.layout(external["plan"]))
    whole = finish(whole_engine, T0, pending)
    assert local["completed"] == whole["completed"] == len(arrivals)
    result = {"status": "passed", "n_functions": N, "candidate_count": len(neighbors),
              "ppo_dim": len(first[0]), "rf_feature_dim": len(feat),
              "step_index_is_variable_n": True, "legal_neighbors": True,
              "full_system_primitives_complete": True, "profiles": len(manifest["profiles"])}
    write(ROOT / "PREFLIGHT.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.all:
        run_all()
    else:
        parser.error("choose --preflight or --all")
