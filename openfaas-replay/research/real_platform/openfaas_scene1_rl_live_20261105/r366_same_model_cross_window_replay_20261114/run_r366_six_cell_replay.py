#!/usr/bin/env python3
"""Preflight and run three interleaved repetitions for two fixed held-out windows."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CONTROLLER = HERE / "run_r366_live_controller.py"
SEED = 631
SUFFIX = "262a9981"
WINDOWS = {
    "q25": ("mid_cv_p50_q25_minute0024", 39, "ppo"),
    "q75": ("mid_cv_p50_q75_minute0294", 57, "whole32"),
}
ORDER = [(window, rep) for rep in (1, 2, 3) for window in ("q25", "q75")]
EXPECTED_POLICY_SHA256 = "7af45ba1cd35bf37053dc683e39ec184a331c279ce522cd74f1c546d906b937f"
EXPECTED_RF_SHA256 = "20c13ff9bd150b81365e635aad8311198f782bb816d22cddf80e964a233028f2"


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def identity(window_key: str, rep: int) -> tuple[str, str, str]:
    stream, _n, _route = WINDOWS[window_key]
    short = "q25" if window_key == "q25" else "q75"
    tag = f"r366-631-{short}-r{rep}-{SUFFIX}"
    return stream, tag, short


def env_for(window_key: str, rep: int) -> dict:
    stream, tag, _short = identity(window_key, rep)
    e = os.environ.copy()
    e.update({
        "KUBECONFIG": os.environ.get("KUBECONFIG", "/etc/rancher/k3s/k3s.yaml"),
        "R366_SEED": str(SEED),
        "R366_PROFILE_ID": "326001",
        "R366_STREAM_ID": stream,
        "R366_METHOD": "easr",
        "R366_RUN_TAG": tag,
        "R366_QUOTA_NAME": f"q-{tag}",
        "R366_SERVICE_PREFIX": f"s1-{tag}",
        "R366_OUTPUT_FILE": str(OUT / "runs" / f"{tag}.json"),
        "R366_PREFLIGHT_FILE": str(OUT / "preflights" / f"{tag}.json"),
    })
    return e


def expected_decision(window_key: str, model_decision: dict) -> None:
    _stream, expected_n, expected_route = WINDOWS[window_key]
    if model_decision["route"] != expected_route:
        raise RuntimeError(f"{window_key}: expected {expected_route}, got {model_decision['route']}")
    if model_decision["arrival_count"] != expected_n:
        raise RuntimeError(f"{window_key}: expected {expected_n} arrivals, got {model_decision['arrival_count']}")
    if model_decision.get("policy_sha256") != EXPECTED_POLICY_SHA256:
        raise RuntimeError(f"{window_key}: seed-631 PPO checkpoint hash changed")
    if model_decision.get("rf_sha256") != EXPECTED_RF_SHA256:
        raise RuntimeError(f"{window_key}: seed-631 calibrated RF hash changed")
    if model_decision.get("rf_probability_threshold") != 0.34:
        raise RuntimeError(f"{window_key}: RF threshold changed")
    if window_key == "q25":
        if model_decision.get("pressure_override") or model_decision.get("ppo_action_index") != 0:
            raise RuntimeError("q25 expected PPO keep without a pressure override")
    else:
        if not model_decision.get("pressure_override") or model_decision.get("rf_route_whole"):
            raise RuntimeError("q75 expected pressure-only Whole32 route")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflights").mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    status_path = OUT / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") != "prepared":
        raise RuntimeError("R366 status is not prepared; refusing duplicate execution")
    if not CONTROLLER.is_file():
        raise FileNotFoundError(CONTROLLER)

    expected_total = len(ORDER)
    preflights = []
    write(status_path, {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY", "status": "preflight",
                        "preflight_passed": 0, "total": expected_total, "completed": 0,
                        "execution_order": [f"{w}-{r}" for w, r in ORDER], "updated_unix": time.time()})
    for i, (window_key, rep) in enumerate(ORDER, 1):
        stream, tag, _short = identity(window_key, rep)
        receipt = OUT / "preflights" / f"{tag}.json"
        output = OUT / "runs" / f"{tag}.json"
        log = OUT / "preflights" / f"{tag}.log"
        if receipt.exists() or output.exists():
            raise RuntimeError(f"R366 output already exists for {tag}; refusing overwrite")
        with log.open("w") as f:
            proc = subprocess.run([sys.executable, str(CONTROLLER), "--preflight"],
                                  env=env_for(window_key, rep), stdout=f,
                                  stderr=subprocess.STDOUT)
        if proc.returncode:
            write(status_path, {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY",
                                "status": "preflight_failed", "failed_cell": f"{window_key}-{rep}",
                                "preflight_passed": len(preflights), "total": expected_total,
                                "updated_unix": time.time()})
            print(f"R366_PREFLIGHT_FAILED {i}/{expected_total} {window_key}-{rep}", flush=True)
            return proc.returncode
        data = json.loads(receipt.read_text())
        expected_decision(window_key, data["model_decision"])
        preflights.append({"order": i, "window": stream, "replicate": rep,
                           "run_tag": tag, "route": data["model_decision"]["route"],
                           "request_count": data["requests"], "receipt": str(receipt)})
        write(status_path, {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY", "status": "preflight",
                            "preflight_passed": len(preflights), "total": expected_total,
                            "completed": 0, "preflights": preflights,
                            "execution_order": [f"{w}-{r}" for w, r in ORDER],
                            "updated_unix": time.time()})
        print(f"R366_PREFLIGHT_OK {i}/{expected_total} {window_key}-{rep} route={data['model_decision']['route']} "
              f"requests={data['requests']}", flush=True)

    completed = []
    for i, (window_key, rep) in enumerate(ORDER, 1):
        stream, tag, _short = identity(window_key, rep)
        log = OUT / "runs" / f"{tag}.log"
        write(status_path, {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY", "status": "running",
                            "phase": "live_runs", "preflight_passed": expected_total,
                            "completed": len(completed), "total": expected_total,
                            "current_cell": f"{window_key}-{rep}", "preflights": preflights,
                            "completed_runs": completed, "updated_unix": time.time()})
        print(f"R366_RUN_START {i}/{expected_total} {window_key}-{rep} seed={SEED}", flush=True)
        with log.open("w") as f:
            proc = subprocess.run([sys.executable, str(CONTROLLER)], env=env_for(window_key, rep),
                                  stdout=f, stderr=subprocess.STDOUT)
        result_path = OUT / "runs" / f"{tag}.json"
        if proc.returncode or not result_path.is_file():
            write(status_path, {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY", "status": "failed",
                                "failed_cell": f"{window_key}-{rep}", "completed": len(completed),
                                "total": expected_total, "failure_log": str(log),
                                "completed_runs": completed, "updated_unix": time.time()})
            print(f"R366_RUN_FAILED {window_key}-{rep}; stopping", flush=True)
            return proc.returncode or 2
        d = json.loads(result_path.read_text())
        expected_decision(window_key, d["controller_decision"])
        valid = (d.get("all_requests_succeeded") and d.get("all_measured_checksums_match")
                 and d.get("checksum_mismatches") == 0
                 and d["platform"].get("new_services_removed_after_run")
                 and d["platform"].get("old_pilot_services_untouched"))
        if not valid or len(d["requests"]) != WINDOWS[window_key][1]:
            write(status_path, {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY", "status": "integrity_failed",
                                "failed_cell": f"{window_key}-{rep}", "completed": len(completed),
                                "total": expected_total, "failure_log": str(log),
                                "completed_runs": completed, "updated_unix": time.time()})
            print(f"R366_INTEGRITY_FAILED {window_key}-{rep}; stopping", flush=True)
            return 2
        item = {"order": i, "window": stream, "replicate": rep, "seed": SEED,
                "run_tag": tag, "result": str(result_path),
                "route": d["controller_decision"]["route"],
                "mean_response_s": d["summary"]["all"]["mean_response_from_scheduled_arrival_s"],
                "p95_response_s": d["summary"]["all"]["p95_response_from_scheduled_arrival_s"],
                "actual_dispatch_pause_s": d["transition"].get("actual_dispatch_pause_from_t15_seconds"),
                "target_ready_s": d["transition"].get("target_layout_ready_seconds"),
                "successes": d["summary"]["all"]["successes"]}
        completed.append(item)
        write(status_path, {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY", "status": "running",
                            "phase": "live_runs", "preflight_passed": expected_total,
                            "completed": len(completed), "total": expected_total,
                            "last_cell": f"{window_key}-{rep}", "preflights": preflights,
                            "completed_runs": completed, "updated_unix": time.time()})
        print(f"R366_RUN_DONE {len(completed)}/{expected_total} {window_key}-{rep} route={item['route']} "
              f"mean={item['mean_response_s']:.4f}s p95={item['p95_response_s']:.4f}s", flush=True)

    by_window = {}
    for window_key in WINDOWS:
        stream = WINDOWS[window_key][0]
        rows = [x for x in completed if x["window"] == stream]
        by_window[stream] = {
            "replicates": len(rows),
            "routes": [x["route"] for x in rows],
            "mean_response_run_mean_s": statistics.mean(x["mean_response_s"] for x in rows),
            "sd_response_run_mean_s": statistics.stdev(x["mean_response_s"] for x in rows),
            "mean_of_run_p95_s": statistics.mean(x["p95_response_s"] for x in rows),
            "sd_of_run_p95_s": statistics.stdev(x["p95_response_s"] for x in rows),
            "mean_dispatch_pause_s": statistics.mean(x["actual_dispatch_pause_s"] for x in rows),
            "mean_target_ready_s": statistics.mean(x["target_ready_s"] for x in rows),
        }
    summary = {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY", "seed": SEED,
               "same_controller_both_windows": True, "preflights_passed": expected_total,
               "completed": expected_total, "execution_order": [f"{w}-{r}" for w, r in ORDER],
               "runs": completed, "by_window": by_window}
    write(OUT / "summary.json", summary)
    lines = ["# R366 六格真机回放汇总", "", "同一个 seed-631 控制器；顺序为交替回放。每行是一次 39 或 57 请求的实际回放。", "",
             "| 顺序 | 窗口 | 重复 | 决策 | 平均响应(s) | P95(s) | t=15 派发暂停(s) | 目标 Ready(s) | 成功请求 |",
             "|---:|---|---:|---|---:|---:|---:|---:|---:|"]
    for x in completed:
        lines.append(f"| {x['order']} | {x['window']} | {x['replicate']} | {x['route']} | {x['mean_response_s']:.4f} | {x['p95_response_s']:.4f} | {x['actual_dispatch_pause_s']:.3f} | {x['target_ready_s']:.3f} | {x['successes']} |")
    lines += ["", "每个窗口的跨重复均值、标准差和决策输入见同目录 `summary.json` 与逐次原始 JSON。P95 是逐次回放的 P95；表中不把不同到达窗口的总体均值当作同分布比较。"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    write(status_path, {"experiment": "R366_CALIBRATED_CROSS_WINDOW_REPLAY", "status": "complete",
                        "phase": "done", "preflight_passed": expected_total,
                        "completed": expected_total, "total": expected_total,
                        "preflights": preflights, "completed_runs": completed,
                        "updated_unix": time.time()})
    print("R366_ALL_FINISHED 6/6", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
