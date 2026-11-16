#!/usr/bin/env python3
"""Run an 8-function Scenario 1 fused/split validation on OpenFaaS CE."""
from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KUBECONFIG = os.environ.get("KUBECONFIG", "/etc/rancher/k3s/k3s.yaml")
GATEWAY = os.environ.get("OPENFAAS_GATEWAY", "http://127.0.0.1:18080")
PROFILE_SOURCE = Path(os.environ.get("R326_MANIFEST", str(HERE / "scale8_profiles.json")))
OUTPUT = Path(os.environ.get("SCENE1_OUTPUT", str(HERE / "outputs" / "scene1_scale8_real.json")))
LOG = Path(os.environ.get("SCENE1_LOG", str(HERE / "run.log")))
REPEATS = int(os.environ.get("SCENE1_REPEATS", "10"))
N_FUNCTIONS = 8
POOL_SPLIT = 4
QUOTA = "scene1-8fn-budget-20261104"
IMAGE = os.environ.get("OPENFAAS_FUNCTION_IMAGE", "")
POOL_NAMES = ("edge", "cloud")
POOL_CPU_M = {"edge": 500, "cloud": 1500}
POOL_MEM_MIB = {"edge": 128, "cloud": 1024}
TOPOLOGY_NAMES = ("chain", "diamond", "tree")
MAX_READY_CONCURRENCY = 10


def say(message: str) -> None:
    print(message, flush=True)


def kubectl(*args: str, check: bool = True, capture: bool = True, input_text: str | None = None):
    cmd = ["kubectl", "--kubeconfig", KUBECONFIG, *args]
    return subprocess.run(cmd, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None,
                          input=input_text)


def get_auth_header() -> str:
    encoded = kubectl("-n", "openfaas", "get", "secret", "basic-auth",
                      "-o", "jsonpath={.data.basic-auth-password}").stdout.strip()
    password = base64.b64decode(encoded).decode()
    return "Basic " + base64.b64encode(f"admin:{password}".encode()).decode()


AUTH = get_auth_header()


def api(method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(
        GATEWAY + path, data=data,
        headers={"Authorization": AUTH, "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as response:
        return response.code, response.read()


def topology_edges(profile: dict, family: str) -> list[tuple[int, int, int]]:
    edges = []
    for source, target, amount in profile["edges"]:
        # The image only accepts payload size as accounting metadata. Keep the
        # values within the range already exercised by the four-node pilot.
        byte_count = 128 if amount <= .01 else 256 if amount <= .1 else 512
        edges.append((int(source), int(target), byte_count))
    expected_edges = {"chain": 7, "diamond": 9, "tree": 7}
    if family not in expected_edges:
        raise ValueError(f"unknown DAG family {family}")
    if len(edges) != expected_edges[family]:
        raise ValueError(f"unexpected 8-function {family} edge count: {len(edges)}")
    return edges


def topological_order(n: int, edges: list[tuple[int, int, int]]) -> list[int]:
    indegree = [0] * n
    children = [[] for _ in range(n)]
    for source, target, _ in edges:
        children[source].append(target)
        indegree[target] += 1
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    order = []
    while ready:
        node = min(ready)
        ready.remove(node)
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != n:
        raise ValueError("DAG has a cycle or missing node")
    return order


def load_profiles() -> list[dict]:
    source = json.loads(PROFILE_SOURCE.read_text())
    selected = {family: [] for family in TOPOLOGY_NAMES}
    for block in source["profiles"]:
        profile = block["profile"]
        family = profile["family"]
        if family in selected and len(selected[family]) < 3:
            selected[family].append(profile)
    profiles = [profile for family in TOPOLOGY_NAMES for profile in selected[family]]
    if any(len(selected[family]) != 3 for family in TOPOLOGY_NAMES):
        raise RuntimeError("R326 manifest does not contain three profiles per DAG family")
    return profiles


def function_name(layout: str, index: int | None = None, pool: str | None = None) -> str:
    if layout == "fused":
        return f"s1-8-fused-{pool}-20261104"
    assert index is not None and pool is not None
    return f"s1-8-n{index:02d}-{pool}-20261104"


def layout_functions(layout: str) -> list[dict]:
    if layout == "fused":
        return [
            {"name": function_name("fused", pool="edge"), "pool": "edge",
             "nodes": list(range(0, POOL_SPLIT)), "cpu_m": 2000, "memory_mib": 512},
            {"name": function_name("fused", pool="cloud"), "pool": "cloud",
             "nodes": list(range(POOL_SPLIT, N_FUNCTIONS)), "cpu_m": 6000, "memory_mib": 4096},
        ]
    return [
        {"name": function_name("split", index=i, pool="edge" if i < POOL_SPLIT else "cloud"),
         "pool": "edge" if i < POOL_SPLIT else "cloud", "nodes": [i],
         "cpu_m": POOL_CPU_M["edge" if i < POOL_SPLIT else "cloud"],
         "memory_mib": POOL_MEM_MIB["edge" if i < POOL_SPLIT else "cloud"]}
        for i in range(N_FUNCTIONS)
    ]


def all_names() -> list[str]:
    return [entry["name"] for layout in ("fused", "split") for entry in layout_functions(layout)]


def ensure_quota() -> None:
    quota = {
        "apiVersion": "v1", "kind": "ResourceQuota",
        "metadata": {"name": QUOTA, "namespace": "openfaas-fn"},
        "spec": {"hard": {"requests.cpu": "9", "limits.cpu": "10",
                            "requests.memory": "6Gi", "limits.memory": "7Gi"}},
    }
    kubectl("apply", "-f", "-", capture=True,
            input_text=json.dumps(quota, separators=(",", ":")))


def deploy_layout(layout: str) -> dict:
    if not IMAGE:
        raise RuntimeError("Set OPENFAAS_FUNCTION_IMAGE to an immutable image digest before deployment")
    entries = layout_functions(layout)
    phase_started = time.perf_counter()
    records = []
    for entry in entries:
        name = entry["name"]
        body = {
            "service": name,
            "image": IMAGE,
            "envVars": {},
            "labels": {"com.openfaas.scale.min": "1", "com.openfaas.scale.max": "1"},
            "annotations": {},
            "requests": {"cpu": f"{entry['cpu_m']}m", "memory": f"{entry['memory_mib']}Mi"},
            "limits": {"cpu": f"{entry['cpu_m']}m", "memory": f"{entry['memory_mib']}Mi"},
            "secrets": [],
        }
        started = time.perf_counter()
        status, response = api("POST", "/system/functions", body)
        if status not in (200, 201, 202):
            raise RuntimeError(f"OpenFaaS POST {name}: HTTP {status}: {response[:240]!r}")
        kubectl("-n", "openfaas-fn", "rollout", "status", f"deployment/{name}",
                "--timeout=300s")
        row = {"name": name, "pool": entry["pool"], "http_status": status,
               "ready_seconds": time.perf_counter() - started,
               "cpu_m": entry["cpu_m"], "memory_mib": entry["memory_mib"]}
        records.append(row)
        say(f"S1_DEPLOY_READY layout={layout} function={name} seconds={row['ready_seconds']:.3f}")
    return {"functions": records, "ready_wall_seconds": time.perf_counter() - phase_started}


def remove_layout(layout: str) -> float:
    started = time.perf_counter()
    for entry in layout_functions(layout):
        name = entry["name"]
        kubectl("-n", "openfaas-fn", "delete", f"deployment/{name}", f"service/{name}",
                "--ignore-not-found=true", "--wait=true", "--timeout=180s", check=False)
    return time.perf_counter() - started


def remove_everything() -> None:
    for layout in ("fused", "split"):
        remove_layout(layout)
    kubectl("-n", "openfaas-fn", "delete", "resourcequota", QUOTA,
            "--ignore-not-found=true", check=False)


def work_node(profile: dict, index: int) -> dict:
    # The benchmark image has deterministic bounded work units. These scales
    # preserve the R326 per-function work/memory ordering while staying below
    # the 128 MiB edge service limit.
    cpu_mib = max(1, min(8, round(profile["work"][index] * 32)))
    memory_mib = max(4, min(16, int(profile["mem"][index]) // 8))
    return {"id": f"n{index:02d}", "kind": "mixed",
            "parameters": {"cpu_mib": cpu_mib, "memory_mib": memory_mib,
                           "memory_passes": 1}}


def checksum_map(result: dict) -> dict[str, dict]:
    rows = {}
    for node in result["node_trace"]:
        value = node["result"]
        if value["kind"] == "mixed":
            checksums = {"cpu": value["cpu"]["checksum"],
                         "memory": value["memory"]["checksum"]}
        else:
            checksums = {value["kind"]: value["checksum"]}
        rows[node["node_id"]] = checksums
    return rows


def invoke(function: str, workflow_id: str, group_index: int,
           profile: dict, indices: list[int], input_bytes: int) -> dict:
    payload = {
        "schema_version": 2, "mode": "execute_group", "workflow_id": workflow_id,
        "group_index": group_index, "input_payload_bytes": input_bytes,
        "nodes": [work_node(profile, i) for i in indices],
    }
    request = urllib.request.Request(
        f"{GATEWAY}/function/{function}/execute-group",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Authorization": AUTH, "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            value = json.loads(response.read())
            return value["result"]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{function}: HTTP {exc.code}: {exc.read()[:240]!r}") from exc


def run_workflow(layout: str, family: str, profile: dict, workflow_id: str) -> dict:
    edges = topology_edges(profile, family)
    order = topological_order(N_FUNCTIONS, edges)
    incoming = [sum(size for source, target, size in edges if target == i) for i in range(N_FUNCTIONS)]
    started = time.perf_counter()
    checksums: dict[str, dict] = {}
    calls = 0
    if layout == "fused":
        for group_index, indices in enumerate((order[:POOL_SPLIT], order[POOL_SPLIT:])):
            node_set = set(indices)
            metadata_bytes = sum(size for source, target, size in edges
                                 if target in node_set and source not in node_set)
            pool = "edge" if group_index == 0 else "cloud"
            result = invoke(function_name("fused", pool=pool), workflow_id,
                            group_index, profile, indices, metadata_bytes)
            checksums.update(checksum_map(result))
            calls += 1
    else:
        parents = [[] for _ in range(N_FUNCTIONS)]
        for source, target, _ in edges:
            parents[target].append(source)
        finished: set[int] = set()
        while len(finished) < N_FUNCTIONS:
            ready = [i for i in order if i not in finished and all(p in finished for p in parents[i])]
            if not ready:
                raise RuntimeError("DAG scheduler made no progress")
            with cf.ThreadPoolExecutor(max_workers=min(MAX_READY_CONCURRENCY, len(ready))) as pool:
                futures = {
                    pool.submit(invoke,
                                function_name("split", index=i,
                                              pool="edge" if i < POOL_SPLIT else "cloud"),
                                workflow_id, i, profile, [i], incoming[i]): i
                    for i in ready
                }
                for future in cf.as_completed(futures):
                    checksums.update(checksum_map(future.result()))
                    finished.add(futures[future])
                    calls += 1
    return {"elapsed_seconds": time.perf_counter() - started,
            "checksums": checksums, "function_calls": calls}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def save(report: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def node_info() -> dict:
    value = json.loads(kubectl("get", "nodes", "-o", "json").stdout)
    items = value["items"]
    if len(items) != 1:
        raise RuntimeError(f"Expected one physical K3s node, found {len(items)}")
    node = items[0]
    return {"node": node["metadata"]["name"],
            "architecture": node["status"]["nodeInfo"]["architecture"],
            "kubernetes": node["status"]["nodeInfo"]["kubeletVersion"],
            "allocatable": node["status"]["allocatable"]}


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    by_family = {family: [p for p in profiles if p["family"] == family]
                 for family in TOPOLOGY_NAMES}
    report = {
        "experiment": "OPENFAAS_SCENE1_8_FUNCTION_TWO_LOGICAL_POOL_VALIDATION_20261104",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"provider": "OpenFaaS CE", "physical_nodes": 1,
                     "placement": "single K3s node; logical pool budgets only",
                     **node_info()},
        "source_profiles": str(PROFILE_SOURCE),
        "source_sha256": {"script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "r326_manifest": hashlib.sha256(PROFILE_SOURCE.read_bytes()).hexdigest()},
        "profile_ids": {family: [p["id"] for p in blocks]
                        for family, blocks in by_family.items()},
        "pool_budgets": {"edge": {"cpu": "2", "memory": "512Mi"},
                         "cloud": {"cpu": "6", "memory": "4096Mi"}},
        "layout": {"fused": "two sequential four-node groups; first group edge-like, second cloud-like",
                   "split": "eight services; same per-function requests/limits and topological pool assignment",
                   "edge_functions": 4, "cloud_functions": 4,
                   "requests_equal_limits": True},
        "workload": {"profiles": 9, "families": list(TOPOLOGY_NAMES),
                     "measured_workflows_per_profile_layout": REPEATS,
                     "mixed_work_per_node": True,
                     "work_scaling": "cpu_mib=clamp(round(work*32),1,8); memory_mib=clamp(mem//8,4,16)",
                     "payload_bytes_are_metadata_only": True,
                     "computed_output_forwarded_between_functions": False},
        "phases": [], "measurements": [], "errors": [],
        "boundary": "Real OpenFaaS deployment and 8-function DAG invocation check. Per-function resource requests match the 20-function pilot, while aggregate pool budgets scale down. It does not run the frozen RF/PPO controller, model two physical pools, pass computed outputs between functions, or measure WAN/image pulls.",
    }
    save(report)
    try:
        # Clear only names from this experiment, in case a prior attempt was interrupted.
        remove_everything()
        ensure_quota()
        # Alternate deployment blocks to reduce time-of-day/order drift. Each
        # layout is torn down before the next one so the two pool budgets do
        # not coexist on the single physical node.
        fused_checksums: dict[str, dict] = {}
        fused_records, split_records = [], []
        active_layout = None
        for phase_index, (layout, repeats) in enumerate((
                ("fused", list(range(0, REPEATS, 2))),
                ("split", list(range(0, REPEATS, 2))),
                ("fused", list(range(1, REPEATS, 2))),
                ("split", list(range(1, REPEATS, 2))))):
            if active_layout is not None:
                delete_seconds = remove_layout(active_layout)
                report["phases"].append({"phase": phase_index, "deleted_layout": active_layout,
                                         "delete_seconds": delete_seconds})
            say(f"S1_PHASE {phase_index + 1}/4 deploy_{layout} repeats={repeats}")
            deployment = deploy_layout(layout)
            phase_row = {"phase": phase_index + 1, "layout": layout,
                         "repeats": repeats, "deployment": deployment}
            report["phases"].append(phase_row)
            active_layout = layout
            for family in TOPOLOGY_NAMES:
                for profile in by_family[family]:
                    key = f"{family}:{profile['id']}"
                    warm = run_workflow(layout, family, profile, f"warm-{key}")
                    if len(warm["checksums"]) != N_FUNCTIONS:
                        raise RuntimeError(f"{layout} warmup returned {len(warm['checksums'])}/{N_FUNCTIONS} nodes for {key}")
                    if key not in fused_checksums:
                        fused_checksums[key] = warm["checksums"]
                    elif warm["checksums"] != fused_checksums[key]:
                        raise RuntimeError(f"warmup checksum mismatch for {layout} {key}")
                    for repeat in repeats:
                        workflow_id = f"s1-{family}-p{profile['id']}-r{repeat:02d}"
                        result = run_workflow(layout, family, profile, workflow_id)
                        if len(result["checksums"]) != N_FUNCTIONS:
                            raise RuntimeError(f"{layout} run missing node traces: {key} {workflow_id}")
                        row = {"layout": layout, "family": family,
                               "profile": profile["id"], "workflow_id": workflow_id,
                               "repeat": repeat, **result}
                        if layout == "fused":
                            fused_records.append(row)
                        else:
                            fused = next((item for item in fused_records
                                          if item["workflow_id"] == workflow_id), None)
                            if fused is not None:
                                row["checksum_match"] = result["checksums"] == fused["checksums"]
                                if not row["checksum_match"]:
                                    raise RuntimeError(f"checksum mismatch for {workflow_id}")
                            split_records.append(row)
                    report["measurements"] = fused_records + split_records
                    save(report)
                    say(f"S1_MEASURED layout={layout} family={family} profile={profile['id']} repeats={repeats}")
        if active_layout is not None:
            report["phases"].append({"deleted_layout": active_layout,
                                     "delete_seconds": remove_layout(active_layout)})
            active_layout = None
        summaries = {}
        for family in TOPOLOGY_NAMES:
            summaries[family] = {}
            for layout, rows in (("fused", fused_records), ("split", split_records)):
                values = [r["elapsed_seconds"] for r in rows if r["family"] == family]
                summaries[family][layout] = {
                    "workflows": len(values), "mean_seconds": statistics.mean(values),
                    "p50_seconds": statistics.median(values), "p95_seconds": percentile(values, .95),
                    "function_calls_per_workflow": 2 if layout == "fused" else N_FUNCTIONS,
                }
        paired = []
        for frow in fused_records:
            srow = next(row for row in split_records if row["workflow_id"] == frow["workflow_id"])
            paired.append({"family": frow["family"], "profile": frow["profile"],
                           "workflow_id": frow["workflow_id"],
                           "fused_seconds": frow["elapsed_seconds"],
                           "split_seconds": srow["elapsed_seconds"],
                           "split_minus_fused_seconds": srow["elapsed_seconds"] - frow["elapsed_seconds"]})
        report["summaries"] = summaries
        report["paired_workflows"] = paired
        report["checksums_match"] = len(split_records) == len(fused_records) and all(
            row.get("checksum_match", False) for row in split_records)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["status"] = "completed"
        save(report)
        say("S1_REAL_ALL_FINISHED " + json.dumps(summaries, ensure_ascii=False))
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(type(exc).__name__ + ": " + str(exc)[:500])
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save(report)
        say(f"S1_REAL_FAILED {type(exc).__name__}: {str(exc)[:500]}")
        raise
    finally:
        cleanup_started = time.perf_counter()
        remove_everything()
        report["phases"].append({"cleanup_seconds": time.perf_counter() - cleanup_started})
        save(report)


if __name__ == "__main__":
    main()
