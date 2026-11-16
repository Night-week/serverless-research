#!/usr/bin/env python3
"""Evaluate completed R328 policy seeds on Azure minute-trace ingress windows."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
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
R328_DIR = HERE.parent / "r328_scene1_scale8_mechanism"
R328_OUTPUTS = R328_DIR / "outputs"
TRACE_FILE = HERE / "inputs" / "azure_trace_streams.json"
OUT = HERE / "outputs"
spec = importlib.util.spec_from_file_location("r328_trace_replay_source", R328_DIR / "run.py")
E = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = E
spec.loader.exec_module(E)
SEEDS = tuple(range(601, 701))
METHODS = ("system", "system_base", "local", "whole32", "keep")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def install_trace_observer() -> None:
    B = E.R.B
    it = E.R.it

    def observe(engine, pool, step):
        p, now = engine.p, engine.now
        n = len(p["work"])
        seen = [t for t in engine.arrivals if t <= now]
        rate = len(seen) / max(1.0, now)
        recent = sum(t > now - 10 for t in seen) / min(10.0, max(now, 1.0))
        queue = [sum(max(0.0, end - max(now, begin))
                     for site, begin, end in engine.task_intervals if site == s)
                 for s in (0, 1)]
        links = [sum(max(0.0, end - max(now, begin))
                     for site, begin, end in engine.link_intervals if site == s)
                 for s in (0, 1)]
        edge = {(u, v): b for u, v, b in p["edges"]}
        common = p["work"] + [v / 256 for v in p["mem"]]
        common += [edge.get(pair, 0.0) for pair in it.combinations(range(n), 2)]
        common += p["speed"] + list(map(float, p["cores"])) + [p["mbps"] / 100, p["rtt"]]
        common += [now / 60, float(step), engine.changes / 2,
                   engine.deployment_total / 10, engine.drain_total / 10,
                   engine.active / 100, len(engine.backlog) / 100,
                   rate / 10, recent / 10] + queue + links
        known = list(getattr(engine, "_trace_past", ())) + seen
        hr = sum(now - 60 < t <= now for t in known) / 60
        rr = sum(now - 10 < t <= now for t in known) / 10
        features = []
        for layout in pool:
            used = [0.0, 0.0]
            work = [sum(p["work"][v] for v in range(n)
                        if layout[1][layout[0][v]] == s) for s in (0, 1)]
            for g, nodes in enumerate(B.groups(layout[0])):
                used[layout[1][g]] += (
                    0.3 * sum(p["mem"][v] for v in nodes)
                    + 0.7 * max(p["mem"][v] for v in nodes)
                )
            raw = common + B.vector(p, engine.layout) + B.vector(p, layout)
            raw += [B.cost(p, engine.layout, layout), float(layout == engine.layout)]
            raw += [1 - used[s] / p["ram"][s] for s in (0, 1)]
            raw += [rate * work[s] / p["speed"][s] / p["cores"][s] for s in (0, 1)]
            raw += [hr, rr]
            wan = [0.0, 0.0]
            for u, v, amount in p["edges"]:
                left = layout[1][layout[0][u]]
                right = layout[1][layout[0][v]]
                if left != right:
                    wan[left] += amount
            features.append([math.asinh(v) for v in raw]
                            + [math.asinh(hr * value * 8 / p["mbps"]) for value in wan])
        return features

    E.R.observe = observe


def trace_feature_vector(engine, net) -> list[float]:
    profile = engine.p
    pool = E.R.B.neighbors(profile, engine.layout)
    observed = E.R.observe(engine, pool, 0)
    with E.torch.no_grad():
        logits = net(E.torch.tensor(observed, dtype=E.torch.float32))[0]
        action = int(logits.argmax())
    known = list(getattr(engine, "_trace_past", ()))
    known.extend(t for t in engine.arrivals if t <= engine.now)
    rate = sum(engine.now - 60 < t <= engine.now for t in known) / 60
    pressures = [E.R.pressure(profile, layout, rate) for layout in pool]
    best_pressure = min(pressures, key=lambda item: item[2])
    chosen_pressure = pressures[action]
    return observed[0] + [
        math.log1p(best_pressure[2]),
        sum(item[2] < 1 for item in pressures) / len(pressures),
        *[math.asinh(value) for value in best_pressure[0] + best_pressure[1]],
        float(action == 0),
        *[math.asinh(value) for value in chosen_pressure[0] + chosen_pressure[1]],
        math.asinh(chosen_pressure[2]),
        math.log1p(E.R.B.cost(profile, engine.layout, pool[action])),
        float(logits[action].exp()),
    ]


def compact(result: dict) -> dict:
    keys = ("mean", "p95", "completed", "deployment", "drain", "changes",
            "peak_ram_fraction", "pending", "future", "affected")
    return {key: result[key] for key in keys if key in result}


def simulate_one(seed: int, net, gate, block: dict, initial_index: int,
                 stream: dict) -> dict:
    profile = block["profile"]
    initial = E.R.layout(block["initials"][initial_index])
    engine = E.Engine(copy.deepcopy(profile), initial, stream["arrivals_seconds"])
    engine._trace_past = stream["history_seconds"]
    engine.advance(E.T0)
    completed_before = {index for index, _ in engine.responses}
    pending = [index for index, arrival in enumerate(engine.arrivals)
               if arrival <= E.T0 and index not in completed_before]

    features = trace_feature_vector(engine, net)
    rf_route = bool(gate.predict([features])[0])
    pressure = math.sinh(features[-3])

    local = E.run_local(copy.deepcopy(engine), net, E.T0, pending)
    whole_engine = copy.deepcopy(engine)
    whole_engine.change(E.R.layout(block["plan"]))
    whole = E.finish(whole_engine, E.T0, pending)
    keep = E.finish(copy.deepcopy(engine), E.T0, pending)
    system_base = whole if rf_route else local
    pressure_override = pressure > E.PRESSURE_THRESHOLD
    system = whole if (rf_route or pressure_override) else local
    methods = {"system": system, "system_base": system_base, "local": local,
               "whole32": whole, "keep": keep}
    expected = len(stream["arrivals_seconds"])
    if any(result["completed"] != expected for result in methods.values()):
        raise RuntimeError(f"incomplete stream: seed={seed} profile={profile['id']} "
                           f"stream={stream['stream_id']}")
    if any(result["peak_ram_fraction"] > 1 + 1e-9 for result in methods.values()):
        raise RuntimeError(f"resource limit violation: seed={seed} profile={profile['id']} "
                           f"stream={stream['stream_id']}")
    return {
        "seed": seed,
        "profile": profile["id"],
        "family": profile["family"],
        "initial_index": initial_index,
        "stream_id": stream["stream_id"],
        "trace_class": stream["trace_class"],
        "source_minute_index_1based": stream["source_minute_index_1based"],
        "source_minute_count": stream["source_minute_count"],
        "scale_divisor": stream["scale_divisor"],
        "arrival_count": expected,
        "rf_route_whole": int(rf_route),
        "pressure_override": int(pressure_override),
        "methods": {name: compact(value) for name, value in methods.items()},
        "local_actions": local.get("actions", []),
    }


def run_seed(seed: int, streams: list[dict], profiles: list[dict]) -> dict:
    # ProcessPool workers must install the trace-aware observation function in
    # their own process before any PPO/RF inference. Otherwise R187's default
    # synthetic-history features leak into this trace-driven evaluation.
    install_trace_observer()
    E.torch.set_num_threads(1)
    seed_dir = R328_OUTPUTS / f"seed_{seed}"
    net, model_hash = E.load_policy(seed)
    gate_path = seed_dir / "rf_live_snapshot.joblib"
    gate = E.R.joblib.load(gate_path)
    rows = []
    for block in profiles:
        for initial_index in range(len(block["initials"])):
            for stream in streams:
                rows.append(simulate_one(seed, net, gate, block, initial_index, stream))
    path = OUT / f"seed_{seed}_trace_rows.json.gz"
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        json.dump({"seed": seed, "model_sha256": model_hash, "rows": rows}, handle,
                  ensure_ascii=False, separators=(",", ":"))
    os.replace(temp, path)
    receipt = {
        "seed": seed,
        "status": "completed_and_verified",
        "row_count": len(rows),
        "profiles": len(profiles),
        "streams": len(streams),
        "all_requests_completed": True,
        "all_layouts_resource_legal": True,
        "model_sha256": model_hash,
        "rf_sha256": sha(gate_path),
        "raw_output_sha256": sha(path),
    }
    atomic_json(OUT / f"seed_{seed}_trace_receipt.json", receipt)
    return receipt


def bootstrap_by_stream(rows: list[dict], method: str, iterations: int = 10000) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row["stream_id"], []).append(row)
    ids = sorted(groups)
    means = {}
    for stream_id in ids:
        subset = groups[stream_id]
        means[stream_id] = {
            name: statistics.mean(row["methods"][name]["mean"] for row in subset)
            for name in (method, "whole32")
        }
    differences, gains = [], []
    rng = random.Random(3299917 + len(ids))
    for _ in range(iterations):
        chosen = [rng.choice(ids) for _ in ids]
        whole_mean = statistics.mean(means[item]["whole32"] for item in chosen)
        method_mean = statistics.mean(means[item][method] for item in chosen)
        differences.append(method_mean - whole_mean)
        gains.append(100 * (1 - method_mean / whole_mean))
    differences.sort()
    gains.sort()
    return {
        "clusters": len(ids),
        "mean_difference_seconds": statistics.mean(
            means[item][method] - means[item]["whole32"] for item in ids),
        "difference_ci95": [differences[int(.025 * iterations)],
                            differences[int(.975 * iterations) - 1]],
        "relative_improvement_pct": statistics.mean(
            100 * (1 - means[item][method] / means[item]["whole32"]) for item in ids),
        "relative_improvement_ci95": [gains[int(.025 * iterations)],
                                      gains[int(.975 * iterations) - 1]],
        "interpretation_limit": "interval resamples 12 minute windows from only 3 underlying Azure function series; it is exploratory, not population-level Azure uncertainty",
    }


def summarize(manifest: dict, streams: list[dict], profiles: list[dict]) -> dict:
    rows = []
    for seed in SEEDS:
        path = OUT / f"seed_{seed}_trace_rows.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        rows.extend(value["rows"])
    method_summary = {}
    for name in METHODS:
        metrics = {key: statistics.mean(row["methods"][name][key]
                                         for row in rows if row["methods"][name].get(key) is not None)
                   for key in ("mean", "p95", "deployment", "drain", "changes")}
        if name != "whole32":
            metrics["bootstrap_vs_whole32_by_trace_window"] = bootstrap_by_stream(rows, name)
        method_summary[name] = metrics
    return {
        "experiment": "R329_AZURE_2019_MINUTE_TRACE_INGRESS_REPLAY",
        "seeds": list(SEEDS),
        "seed_count": len(SEEDS),
        "profiles": len(profiles),
        "stream_count": len(streams),
        "rows": len(rows),
        "methods": method_summary,
        "routing": {
            "rf_whole_rate": statistics.mean(row["rf_route_whole"] for row in rows),
            "pressure_override_rate": statistics.mean(row["pressure_override"] for row in rows),
            "full_system_whole_rate": statistics.mean(
                row["rf_route_whole"] or row["pressure_override"] for row in rows),
        },
        "trace_classes": manifest["selected_functions"],
        "all_requests_completed": all(
            row["methods"][name]["completed"] == row["arrival_count"]
            for row in rows for name in METHODS),
        "resource_legal": all(
            row["methods"][name]["peak_ram_fraction"] <= 1 + 1e-9
            for row in rows for name in METHODS),
        "source_trace_sha256": manifest["source"]["sha256"],
        "trace_bundle_sha256": sha(TRACE_FILE),
        "r328_manifest_sha256": sha(R328_OUTPUTS / "manifest.json"),
        "r328_checkpoint_seed_range": [min(SEEDS), max(SEEDS)],
        "scope": "Azure minute-count-driven workflow-ingress proxy replay in the R328 simulator. The public Azure trace contains neither this workflow DAG nor exact request timestamps; minute intervals are uniformly jittered and counts are downscaled as specified. This is not live traffic or a production workflow trace.",
    }


def preflight() -> dict:
    if not TRACE_FILE.exists():
        raise FileNotFoundError(TRACE_FILE)
    manifest = json.loads((R328_OUTPUTS / "manifest.json").read_text())
    profile_blocks = [block for block in manifest["profiles"]
                      if block["split"] == "external_real_profiles"]
    if len(profile_blocks) != 9:
        raise RuntimeError(f"expected nine external R326 profiles, found {len(profile_blocks)}")
    status = json.loads((R328_OUTPUTS / "queue_status_100.json").read_text())
    if status.get("phase") != "completed" or status.get("completed_count") != 100:
        raise RuntimeError("R328 100-seed experiment is not fully complete")
    for seed in SEEDS:
        if not (R328_OUTPUTS / f"seed_{seed}" / "selected.pt").exists():
            raise RuntimeError(f"missing selected checkpoint for seed {seed}")
        if not (R328_OUTPUTS / f"seed_{seed}" / "rf_live_snapshot.joblib").exists():
            raise RuntimeError(f"missing fitted RF for seed {seed}")
    bundle = json.loads(TRACE_FILE.read_text())
    streams = bundle["streams"]
    if len(streams) != 12:
        raise RuntimeError(f"expected 12 trace windows, found {len(streams)}")
    for block in profile_blocks:
        profile = block["profile"]
        for initial in block["initials"]:
            layout = E.R.layout(initial)
            if not E.R.B.valid(profile, *layout):
                raise RuntimeError(f"illegal initial layout in profile {profile['id']}")
    return {"seed_count": len(SEEDS), "external_profiles": len(profile_blocks),
            "trace_windows": len(streams),
            "simulations_per_seed": sum(len(block["initials"]) for block in profile_blocks) * len(streams),
            "methods": list(METHODS), "source_trace_sha256": bundle["source"]["sha256"],
            "r328_status": status["phase"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    check = preflight()
    atomic_json(OUT / "preflight.json", check)
    print("R329_PREFLIGHT", json.dumps(check, ensure_ascii=False), flush=True)
    if args.preflight:
        return
    bundle = json.loads(TRACE_FILE.read_text())
    r328_manifest = json.loads((R328_OUTPUTS / "manifest.json").read_text())
    profiles = [block for block in r328_manifest["profiles"]
                if block["split"] == "external_real_profiles"]
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "run_status.json", {"phase": "running", "seeds": list(SEEDS),
                                          "workers": args.workers,
                                          "preflight": check})
    failures = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_seed, seed, bundle["streams"], profiles): seed
                   for seed in SEEDS}
        done = 0
        for future in cf.as_completed(futures):
            seed = futures[future]
            try:
                receipt = future.result()
                done += 1
                print(f"R329_SEED_VERIFIED {seed} {done}/{len(SEEDS)} "
                      f"rows={receipt['row_count']}", flush=True)
            except Exception:
                failures[seed] = traceback.format_exc()
                print(f"R329_SEED_FAILED {seed}", flush=True)
            atomic_json(OUT / "run_status.json", {
                "phase": "running", "completed": done, "total": len(SEEDS),
                "failed_seeds": sorted(failures), "last_seed": seed,
                "workers": args.workers,
            })
    if failures:
        atomic_json(OUT / "failures.json", failures)
        atomic_json(OUT / "run_status.json", {"phase": "failed", "completed": done,
                                                "failed_seeds": sorted(failures)})
        raise SystemExit(f"{len(failures)} seed(s) failed")
    summary = summarize(bundle, bundle["streams"], profiles)
    summary.update({
        "simulation_source_hashes": {
            "driver": sha(Path(__file__)),
            "protocol": sha(HERE / "R329_TRACE_REPLAY_PROTOCOL.md"),
            "r328_driver": sha(R328_DIR / "run.py"),
            "r187_engine": sha(R328_DIR.parent / "r187_twenty_function_scale" / "engine.py"),
            "trace_preparer": sha(HERE / "prepare_azure_traces.py"),
        },
    })
    atomic_json(OUT / "summary.json", summary)
    atomic_json(OUT / "run_status.json", {"phase": "completed", "completed": len(SEEDS),
                                            "failed_seeds": [],
                                            "summary": str(OUT / "summary.json")})
    print("R329_ALL_FINISHED", json.dumps({"rows": summary["rows"],
                                           "methods": summary["methods"]},
                                          ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

