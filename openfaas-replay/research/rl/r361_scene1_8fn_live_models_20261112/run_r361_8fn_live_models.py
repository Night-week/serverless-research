#!/usr/bin/env python3
"""Train the 8-function live-replay models on R332 Azure flows.

Runs the simulator on an isolated compute process. The mid-CV p50 sequence is
held out; the script never sends requests to or changes OpenFaaS/Kubernetes.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R328_DIR = REPO / "research/rl/r328_scene1_scale8_mechanism"
R329_DIR = REPO / "research/rl/r329_trace_driven_replay"
R336_DIR = REPO / "research/real_platform/openfaas_scene1_rl_live_20261105/r336_live_baseline_extension"
STREAM_FILE = HERE / "inputs/azure_sequence_streams.json"
MANIFEST_FILE = HERE / "inputs/r328_manifest.json"
OUT = HERE / "outputs"
SEEDS = (601, 631, 659)
PPO_BUDGET = 32768
RF_THRESHOLD = 0.97
PRESSURE_THRESHOLD = 1.0
TEST_SEQUENCE = "mid_cv_p50"
VALIDATION_SEQUENCE = "mid_cv_p10"
TARGET_PROFILE_ID = 326001
READY_PROXY_SECONDS = 4.02


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


class AzureArrivals(list):
    def __init__(self, stream: dict):
        super().__init__(float(t) for t in stream["arrivals_seconds"])
        self.history = [float(t) for t in stream["history_seconds"]]
        self.stream_id = stream["stream_id"]


def prepare_data():
    stream_bundle = json.loads(STREAM_FILE.read_text())
    streams = stream_bundle["streams"]
    test = [row for row in streams if row["sequence_id"] == TEST_SEQUENCE]
    validation = [row for row in streams if row["sequence_id"] == VALIDATION_SEQUENCE]
    train = [row for row in streams if row["sequence_id"] not in
             {TEST_SEQUENCE, VALIDATION_SEQUENCE}]
    if (len(train), len(validation), len(test)) != (28, 4, 4):
        raise RuntimeError(f"unexpected Azure split: train={len(train)} val={len(validation)} test={len(test)}")

    source_manifest = json.loads(MANIFEST_FILE.read_text())
    profiles = [row for row in source_manifest["profiles"]
                if row["split"] in {"train", "validation"}
                or (row["profile"]["id"] == TARGET_PROFILE_ID
                    and row["split"] == "external_real_profiles")]
    if not any(row["profile"]["id"] == TARGET_PROFILE_ID for row in profiles):
        raise RuntimeError("held-out R338 profile 326001 missing")
    OUT.mkdir(parents=True, exist_ok=True)
    return stream_bundle, train, validation, test, source_manifest, profiles


def score_whole32(E, profiles, train_streams):
    """Regenerate 32 feasible plans and score a switch at the 15s snapshot."""
    R = E.R
    B = E.R.B
    scored_manifest_profiles = []
    target_audits = {}
    for number, original in enumerate(profiles, 1):
        block = copy.deepcopy(original)
        profile = block["profile"]
        candidates = B.plan_pool(profile, 3282000 + int(profile["id"]), 32)
        if len(candidates) != 32:
            raise RuntimeError(f"profile {profile['id']} has {len(candidates)} candidates, expected 32")
        ranking = []
        initial = R.layout(block["initials"][0])
        for index, layout in enumerate(candidates):
            per_window = []
            for stream in train_streams:
                engine = E.Engine(copy.deepcopy(profile), initial, stream["arrivals_seconds"])
                engine.advance(15.0)
                completed_before = {request for request, _ in engine.responses}
                pending = [request for request, arrival in enumerate(engine.arrivals)
                           if arrival <= 15.0 and request not in completed_before]
                engine.change(layout)
                result = E.finish(engine, 15.0, pending)
                per_window.append(float(result["mean"]))
            ranking.append({
                "candidate_index": index,
                "layout": [list(layout[0]), list(layout[1])],
                "mean_window_latency_s": statistics.mean(per_window),
                "window_means_s": per_window,
            })
        best = min(ranking, key=lambda row: (row["mean_window_latency_s"], row["layout"]))
        block["plan"] = best["layout"]
        block["plan_candidates"] = len(candidates)
        block["azure_training_window_count"] = len(train_streams)
        block["azure_training_selected_candidate"] = best["candidate_index"]
        scored_manifest_profiles.append(block)
        if profile["id"] == TARGET_PROFILE_ID:
            target_audits = {
                "selected_candidate_index": best["candidate_index"],
                "selected_layout": best["layout"],
                "selected_mean_window_latency_s": best["mean_window_latency_s"],
                "candidate_scores": ranking,
            }
        if number % 10 == 0 or number == len(profiles):
            print(f"R361_WHOLE_PROGRESS {number}/{len(profiles)}", flush=True)
    return scored_manifest_profiles, target_audits


def install_azure_training(E, D, train_streams, validation_streams):
    R = E.R
    OriginalEngine = E.Engine

    class TraceEngine(OriginalEngine):
        def __init__(self, profile, layout, arrivals):
            super().__init__(profile, layout, arrivals)
            self._trace_past = list(getattr(arrivals, "history", ()))

    E.Engine = TraceEngine
    R.Engine = TraceEngine
    R.rollout = None  # replaced below; R187's rollout otherwise drops history

    def trace_rollout(profile, old, arrivals, net, stochastic=False):
        engine = TraceEngine(profile, R.layout(old), arrivals)
        # Training and validation start at the same online decision snapshot used
        # by R329 evaluation: 15 seconds of arrivals have already been observed.
        engine.advance(15.0)
        dataset = []
        pair_state = None
        for step in range(2):
            pool = [engine.layout] + [layout for layout in R.B.neighbors(profile, engine.layout)[1:]
                                      if layout not in engine.visited]
            features = R.observe(engine, pool, step)
            if step == 1 and len(pool) > 1:
                pair_state = (copy.deepcopy(engine), pool, features)
            with R.torch.no_grad():
                logp, value, _entropy = net(R.torch.tensor(features, dtype=R.torch.float32))
                action = (int(R.torch.distributions.Categorical(logits=logp).sample())
                          if stochastic else int(logp.argmax()))
            dataset.append({"x": features, "index": action, "logp": float(logp[action]),
                            "value": float(value), "reward": 0.0})
            if action == 0:
                break
            engine.change(pool[action])
            if step == 0:
                engine.advance(max(30.0, engine.now))
                if engine.now >= 60.0:
                    break
        result = engine.finish()
        dataset[-1]["reward"] = -float(result["mean"]) / 100.0
        result["indices"] = [row["index"] for row in dataset]
        return result, dataset, pair_state

    R.rollout = trace_rollout
    original_validate = R.validate

    def trace_validate(net, manifest):
        values = []
        for block, old, _initial_index in R.states(manifest, "validation"):
            for stream in validation_streams:
                result, _dataset, _pair = trace_rollout(
                    block["profile"], old, AzureArrivals(stream), net, stochastic=False
                )
                values.append(float(result["mean"]))
        if not values:
            raise RuntimeError("no validation rollouts")
        return statistics.mean(values)

    R.validate = trace_validate
    seed_namespaces = {"validation": (187700000, 200000000)}

    def azure_flow(seed: int):
        choices = validation_streams if seed_namespaces["validation"][0] <= seed < seed_namespaces["validation"][1] else train_streams
        digest = hashlib.sha256(f"r339:{seed}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(choices)
        return AzureArrivals(choices[index])

    R.B.flow = azure_flow
    return TraceEngine, original_validate


def fit_azure_rf(E, D, manifest, train_streams, seed: int):
    R = E.R
    net, _model_hash = E.load_policy(seed)
    xs, ys, audit = [], [], []
    states = R.states(manifest, "train")
    for state_number, (block, old, initial_index) in enumerate(states, 1):
        profile = block["profile"]
        for stream in train_streams:
            arrivals = AzureArrivals(stream)
            engine = E.Engine(copy.deepcopy(profile), R.layout(old), arrivals)
            engine._trace_past = arrivals.history
            engine.advance(15.0)
            completed_before = {index for index, _ in engine.responses}
            pending = [index for index, arrival in enumerate(engine.arrivals)
                       if arrival <= 15.0 and index not in completed_before]
            features = D.trace_feature_vector(engine, net)
            local = E.run_local(copy.deepcopy(engine), net, 15.0, pending)
            whole_engine = copy.deepcopy(engine)
            whole_engine.change(R.layout(block["plan"]))
            whole = E.finish(whole_engine, 15.0, pending)
            label = bool(whole["mean"] < RF_THRESHOLD * local["mean"])
            xs.append(features)
            ys.append(int(label))
            audit.append({"profile": profile["id"], "initial_index": initial_index,
                          "stream_id": stream["stream_id"], "local_mean_s": local["mean"],
                          "whole32_mean_s": whole["mean"], "label_whole32": label,
                          "pressure": math.sinh(features[-3])})
        if state_number % 20 == 0 or state_number == len(states):
            print(f"R361_RF_PROGRESS seed={seed} states={state_number}/{len(states)} examples={len(xs)}", flush=True)
    gate = R.RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=6,
                                    class_weight="balanced", random_state=121, n_jobs=1)
    gate.fit(xs, ys)
    seed_dir = OUT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    R.joblib.dump(gate, seed_dir / "rf_live_snapshot.joblib")
    write(seed_dir / "rf_fit_azure.json", {
        "training_examples": len(ys), "positive_whole_rate": statistics.mean(ys),
        "rf_label": f"whole32 mean < {RF_THRESHOLD} * local mean",
        "pressure_threshold": PRESSURE_THRESHOLD,
        "training_sequence_ids": sorted({s["sequence_id"] for s in train_streams}),
    })
    write(seed_dir / "rf_training_audit_azure.json", audit)
    return gate


def build_grl_workflows(E, G, workflow_builder, profiles, train_streams, validation_streams, test_streams):
    workflow_map = {}

    def make(rows, stream_rows, split):
        result = []
        for block in rows:
            profile = block["profile"]
            base = workflow_builder(profile)
            for stream in stream_rows:
                key = f"{profile['id']}|{stream['stream_id']}"
                wf = dataclasses.replace(base, workflow_id=key)
                result.append(wf)
                workflow_map[key] = (profile, stream, split, block["initials"][0])
        return result

    train_rows = [b for b in profiles if b["split"] == "train"]
    val_rows = [b for b in profiles if b["split"] == "validation"]
    test_rows = [b for b in profiles if b["profile"]["id"] == TARGET_PROFILE_ID]
    return (make(train_rows, train_streams, "train"),
            make(val_rows, validation_streams, "validation"),
            make(test_rows, test_streams, "test"), workflow_map)


def grl_layout(E, plan, n: int):
    parts = [(list(map(int, group)), int(plan.placement[group])) for group in plan.groups]
    return E.R.B.canonical(parts, n)


def train_grl(E, train_streams, validation_streams, test_streams, profiles,
              seed: int, objective: str = "native"):
    sys.path.insert(0, str(R336_DIR / "vendor"))
    target_module = load_module("r339_r336_target_helpers", R336_DIR / "prepare_targets.py")
    G = target_module.G
    train_wfs, val_wfs, test_wfs, workflow_map = build_grl_workflows(
        E, G, target_module.workflow, profiles, train_streams, validation_streams, test_streams
    )
    if objective == "native":
        reward = G.GRLRewardConfig()
    elif objective == "response":
        # Keep the R336 GRL architecture and constraints, while aligning the
        # learned reward with response time as in the established R209 variant.
        reward = dataclasses.replace(G.GRLRewardConfig(), cost_weight=0.0)
    else:
        raise ValueError(f"unknown GRL objective: {objective}")

    def evaluator(workflow, plan):
        profile, stream, _split, initial = workflow_map[workflow.workflow_id]
        layout = grl_layout(E, plan, len(profile["work"]))
        engine = E.Engine(copy.deepcopy(profile), E.R.layout(initial),
                          AzureArrivals(stream))
        engine._trace_past = list(stream["history_seconds"])
        engine.advance(15.0)
        completed_before = {request for request, _ in engine.responses}
        pending = [request for request, arrival in enumerate(engine.arrivals)
                   if arrival <= 15.0 and request not in completed_before]
        engine.change(layout)
        result = E.finish(engine, 15.0, pending)
        analytical = G.evaluate_grl_plan(workflow, plan, reward)
        return G.GRLMetrics(latency_s=float(result["mean"]), cost=analytical.cost,
                            oom_penalty=analytical.oom_penalty,
                            queue_wait_s=analytical.queue_wait_s,
                            wan_bytes_mib=analytical.wan_bytes_mib,
                            cold_starts_expected=analytical.cold_starts_expected)

    train_cfg = dataclasses.replace(G.GRLTrainConfig(), seed=seed)
    seed_dir = OUT / f"grl_{objective}" / f"seed_{seed}"
    model, result = G.train_grl_orch(
        train_wfs, val_wfs, heldout_workflows=test_wfs, evaluator=evaluator,
        reward_config=reward, train_config=train_cfg,
        checkpoint_path=str(seed_dir / "selected.pt"),
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    write(seed_dir / "training_summary.json", {
        "seed": seed, "episodes": result.episodes_seen,
        "objective": objective,
        "reward_config": dataclasses.asdict(reward),
        "best_episode": result.best_episode,
        "selected_fusion_threshold": result.selected_fusion_threshold,
        "best_validation_reward": result.best_validation_reward,
        "heldout_reward": result.heldout_reward,
        "heldout_mean_latency_s": result.heldout_latency_s,
        "training_seconds": result.training_seconds,
        "train_workflow_instances": len(train_wfs),
        "validation_workflow_instances": len(val_wfs),
        "heldout_workflow_instances": len(test_wfs),
        "training_sequence_ids": sorted({s["sequence_id"] for s in train_streams}),
    })
    plans = {}
    for stream in test_streams:
        wf = next(w for w in test_wfs if w.workflow_id.endswith("|" + stream["stream_id"]))
        with G.torch.no_grad():
            plan = G.sample_grl_plan(model, wf, reward, stochastic=False,
                                     threshold=result.selected_fusion_threshold)
        chosen = grl_layout(E, plan, 8)
        plans[stream["stream_id"]] = [list(chosen[0]), list(chosen[1])]
    write(seed_dir / "heldout_layouts.json", plans)
    return plans


def evaluate_heldout(E, D, manifest, test_streams):
    rows = []
    target = next(b for b in manifest["profiles"] if b["profile"]["id"] == TARGET_PROFILE_ID)
    initial_index = 0
    for seed in SEEDS:
        net, model_hash = E.load_policy(seed)
        gate = E.R.joblib.load(OUT / f"seed_{seed}" / "rf_live_snapshot.joblib")
        grl_plans = {
            objective: json.loads((OUT / f"grl_{objective}" / f"seed_{seed}" / "heldout_layouts.json").read_text())
            for objective in ("native", "response")
        }
        for stream in test_streams:
            full = D.simulate_one(seed, net, gate, target, initial_index, stream)
            initial = E.R.layout(target["initials"][initial_index])
            arrivals = AzureArrivals(stream)
            engine = E.Engine(copy.deepcopy(target["profile"]), initial, arrivals)
            engine._trace_past = arrivals.history
            engine.advance(15.0)
            completed_before = {index for index, _ in engine.responses}
            pending = [index for index, arrival in enumerate(engine.arrivals)
                       if arrival <= 15.0 and index not in completed_before]
            grl_results = {}
            for objective in ("native", "response"):
                plan_layout = tuple(tuple(int(v) for v in part)
                                    for part in grl_plans[objective][stream["stream_id"]])
                grl_engine = copy.deepcopy(engine)
                grl_engine.change(plan_layout)
                grl_results[objective] = E.finish(grl_engine, 15.0, pending)
            rows.append({
                "seed": seed, "profile": TARGET_PROFILE_ID,
                "stream_id": stream["stream_id"], "source_sequence": stream["sequence_id"],
                "initial_index": initial_index, "training_excludes_test_sequence": TEST_SEQUENCE,
                "full_system_route": "whole32" if full["rf_route_whole"] or full["pressure_override"] else "candidate_adjustment",
                "rf_route_whole": bool(full["rf_route_whole"]),
                "pressure_override": bool(full["pressure_override"]),
                "full_system": full["methods"]["system"],
                "whole32": full["methods"]["whole32"],
                "grl_native": grl_results["native"],
                "grl_response": grl_results["response"],
                "grl_native_layout": grl_plans["native"][stream["stream_id"]],
                "grl_response_layout": grl_plans["response"][stream["stream_id"]],
                "whole32_layout": target["plan"],
                "policy_sha256": model_hash,
            })
    write(OUT / "heldout_evaluation.json", rows)
    return rows


def install_parallel_ready_cost_model(E):
    """R343 calibrated cost: one 4.02s parallel-Ready proxy per changed target with fresh services."""
    B = E.R.B
    def calibrated_cost(profile, old, new):
        if old == new:
            return 0.0
        old_services = {(tuple(nodes), int(old[1][group]))
                        for group, nodes in enumerate(B.groups(old[0]))}
        fresh_services = [(nodes, int(new[1][group]))
                          for group, nodes in enumerate(B.groups(new[0]))
                          if (tuple(nodes), int(new[1][group])) not in old_services]
        return READY_PROXY_SECONDS if fresh_services else 0.0
    B.cost = calibrated_cost
    return calibrated_cost


def build_live_targets(E, D, manifest, test_streams):
    """Freeze the first t=15 decision and four fixed baseline targets for real replay."""
    target_module = load_module("r352_rule_target_helpers", R336_DIR / "prepare_targets.py")
    block = next(row for row in manifest["profiles"]
                 if row["profile"].get("id") == TARGET_PROFILE_ID)
    profile = block["profile"]
    initial = E.R.layout(block["initials"][0])
    whole_layout = E.R.layout(block["plan"])
    greedy_layout = target_module.rules_plan(profile, "greedy_fusion", E.R.B)
    critical_layout = target_module.rules_plan(profile, "critical_path", E.R.B)
    rows = []
    for seed in SEEDS:
        net, model_hash = E.load_policy(seed)
        gate = E.R.joblib.load(OUT / f"seed_{seed}" / "rf_live_snapshot.joblib")
        grl_plans = {
            objective: json.loads((OUT / f"grl_{objective}" / f"seed_{seed}" / "heldout_layouts.json").read_text())
            for objective in ("native", "response")
        }
        for stream in test_streams:
            arrivals = AzureArrivals(stream)
            engine = E.Engine(copy.deepcopy(profile), initial, arrivals)
            engine._trace_past = arrivals.history
            engine.advance(15.0)
            features = D.trace_feature_vector(engine, net)
            rf_route = bool(gate.predict([features])[0])
            pressure = math.sinh(features[-3])
            pool = E.R.B.neighbors(profile, engine.layout)
            observed = E.R.observe(engine, pool, 0)
            with E.torch.no_grad():
                logits = net(E.torch.tensor(observed, dtype=E.torch.float32))[0]
                ppo_action = int(logits.argmax())
            system_route = "whole32" if (rf_route or pressure > PRESSURE_THRESHOLD) else "ppo"
            system_layout = whole_layout if system_route == "whole32" else pool[ppo_action]
            grl_layout_values = {
                objective: tuple(tuple(int(v) for v in part)
                                 for part in grl_plans[objective][stream["stream_id"]])
                for objective in ("native", "response")
            }
            methods = {
                "full_system": system_layout,
                "whole32": whole_layout,
                "grl_native": grl_layout_values["native"],
                "grl_response": grl_layout_values["response"],
                "greedy_fusion": greedy_layout,
                "critical_path": critical_layout,
            }
            serialized = {}
            for name, layout in methods.items():
                if not E.R.B.valid(profile, *layout):
                    raise RuntimeError(f"infeasible target {name}/seed={seed}/{stream['stream_id']}")
                services = len(E.R.B.groups(layout[0]))
                if services > 8:
                    raise RuntimeError(f"OpenFaaS function cap exceeded: {name}/{services}")
                serialized[name] = {
                    "layout": [list(layout[0]), list(layout[1])],
                    "service_count": services,
                }
            rows.append({
                "seed": seed, "profile": TARGET_PROFILE_ID,
                "stream_id": stream["stream_id"],
                "source_sequence": stream["sequence_id"],
                "arrival_count": len(stream["arrivals_seconds"]),
                "decision_time_s": 15.0,
                "initial_layout": [list(initial[0]), list(initial[1])],
                "rf_route_whole": rf_route,
                "pressure": pressure,
                "pressure_override": pressure > PRESSURE_THRESHOLD,
                "ppo_action_index": ppo_action,
                "system_route": system_route,
                "policy_sha256": model_hash,
                "rf_sha256": D.sha(OUT / f"seed_{seed}" / "rf_live_snapshot.joblib"),
                "grl_native_sha256": D.sha(OUT / "grl_native" / f"seed_{seed}" / "selected.pt"),
                "grl_response_sha256": D.sha(OUT / "grl_response" / f"seed_{seed}" / "selected.pt"),
                "methods": serialized,
            })
    write(OUT / "live_targets.json", rows)
    return rows


def main():
    started = time.time()
    D = load_module("r339_r329_driver", R329_DIR / "run_r329_simulation.py")
    D.install_trace_observer()
    E = D.E
    R = E.R
    cost_model = install_parallel_ready_cost_model(E)
    _stream_bundle, train_streams, validation_streams, test_streams, source_manifest, source_profiles = prepare_data()
    write(OUT / "status.json", {
        "phase": "azure_whole32_candidate_rescoring",
        "profiles": len(source_profiles), "candidates_per_profile": 32,
        "training_windows": len(train_streams),
    })
    selected_profiles, whole_audit = score_whole32(E, source_profiles, train_streams)

    # R328 training expects the current experiment manifest under its output root.
    manifest = copy.deepcopy(source_manifest)
    manifest["profiles"] = selected_profiles
    manifest["traffic_training"] = {
        "source": "R332 Azure Functions per-minute sequence bundle",
        "train_sequences": sorted({s["sequence_id"] for s in train_streams}),
        "validation_sequences": sorted({s["sequence_id"] for s in validation_streams}),
        "heldout_test_sequence": TEST_SEQUENCE,
        "train_window_count": len(train_streams),
        "validation_window_count": len(validation_streams),
        "heldout_window_count": len(test_streams),
    }
    write(OUT / "manifest.json", manifest)
    write(OUT / "whole32_target_profile_326001.json", whole_audit)
    E.ROOT = OUT
    R.ROOT = OUT
    R.SEEDS = SEEDS
    R.BUDGET = PPO_BUDGET
    R.read_manifest = lambda: json.loads((OUT / "manifest.json").read_text())
    # R187's trainer automatically fits a second RF and evaluates a confirmation
    # split when it returns. That path is tied to the old synthetic-flow protocol;
    # R361 fits its RF explicitly below from Azure train windows instead.
    R.evaluate = lambda *_args, **_kwargs: None
    TraceEngine, _old_validate = install_azure_training(E, D, train_streams, validation_streams)
    write(OUT / "status.json", {"phase": "ppo_training", "seeds": list(SEEDS), "budget": PPO_BUDGET})

    for seed in SEEDS:
        print(f"R361_PPO_START seed={seed} budget={PPO_BUDGET}", flush=True)
        R.train(seed)
        print(f"R361_PPO_DONE seed={seed}", flush=True)
        net, _model_hash = E.load_policy(seed)
        write(OUT / "status.json", {"phase": "rf_training", "seed": seed})
        fit_azure_rf(E, D, manifest, train_streams, seed)
        print(f"R361_RF_DONE seed={seed}", flush=True)

    sys.path.insert(0, str(R336_DIR / "vendor"))
    target_module = load_module("r339_r336_target_helpers_for_grl", R336_DIR / "prepare_targets.py")
    for seed in SEEDS:
        write(OUT / "status.json", {"phase": "grl_native_training", "seed": seed})
        train_grl(E, train_streams, validation_streams, test_streams, selected_profiles, seed,
                  objective="native")
        print(f"R361_GRL_NATIVE_DONE seed={seed}", flush=True)

    for seed in SEEDS:
        write(OUT / "status.json", {"phase": "grl_response_training", "seed": seed})
        train_grl(E, train_streams, validation_streams, test_streams, selected_profiles, seed,
                  objective="response")
        print(f"R361_GRL_RESPONSE_DONE seed={seed}", flush=True)

    write(OUT / "status.json", {"phase": "heldout_evaluation"})
    rows = evaluate_heldout(E, D, manifest, test_streams)
    live_targets = build_live_targets(E, D, manifest, test_streams)
    summary = {
        "experiment": "R361_SCENE1_8FN_LIVE_REPLAY_MODEL_TRAINING",
        "cost_model": {"service_ready_proxy_seconds": READY_PROXY_SECONDS,
                       "scope": "one parallel Ready proxy per changed target that has at least one fresh target service; event-driven drain and queue; no teardown/warm-up cost"},
        "live_target_count": len(live_targets),
        "live_targets_sha256": sha(OUT / "live_targets.json"),
        "seeds": list(SEEDS), "ppo_budget_per_seed": PPO_BUDGET,
        "training_sequences": sorted({s["sequence_id"] for s in train_streams}),
        "validation_sequences": sorted({s["sequence_id"] for s in validation_streams}),
        "heldout_sequence": TEST_SEQUENCE,
        "heldout_window_count": len(test_streams),
        "heldout_record_count": len(rows),
        "target_profile_whole32_candidate_count": len(whole_audit["candidate_scores"]),
        "target_profile_whole32_selected_candidate_index": whole_audit["selected_candidate_index"],
        "elapsed_seconds": time.time() - started,
        "data_sha256": sha(STREAM_FILE),
        "source_manifest_sha256": sha(MANIFEST_FILE),
        "training_script_sha256": sha(Path(__file__)),
        "source_code_sha256": {
            "r187_run": sha(REPO / "research/rl/r187_twenty_function_scale/run.py"),
            "r187_core": sha(REPO / "research/rl/r187_twenty_function_scale/core.py"),
            "r187_engine": sha(REPO / "research/rl/r187_twenty_function_scale/engine.py"),
            "r328_run": sha(R328_DIR / "run.py"),
            "r329_driver": sha(R329_DIR / "run_r329_simulation.py"),
            "grl_orch": sha(R336_DIR / "vendor/r8_grl_orch.py"),
            "grl_adapters": sha(R336_DIR / "vendor/r8_policy_adapters.py"),
        },
        "method_scope": "Eight-function full system PPO+RF with pressure gate fixed on, GRL native-reward and response-objective adapters, Whole32, and frozen Greedy Fusion/Critical Path target rules; first t=15 decision frozen for paired real-platform replay",
    }
    write(OUT / "summary.json", summary)
    write(OUT / "status.json", {"phase": "completed", "summary": str(OUT / "summary.json")})
    print("R361_ALL_FINISHED", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
