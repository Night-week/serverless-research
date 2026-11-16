#!/usr/bin/env python3
"""Build reproducible minute-window ingress streams from Azure Functions 2019."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


QUANTILES = (0.25, 0.50, 0.75, 0.95)
MAX_SCALED_COUNT_PER_MINUTE = 480


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"HashOwner", "HashApp", "HashFunction"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"missing trace fields: {sorted(required - set(reader.fieldnames or ())) }")
        minute_columns = [name for name in (reader.fieldnames or ()) if name.isdigit()]
        if len(minute_columns) != 1440:
            raise ValueError(f"expected 1440 minute columns, found {len(minute_columns)}")
        result = []
        for row in reader:
            counts = [int(row.get(column) or 0) for column in minute_columns]
            total = sum(counts)
            if total <= 0:
                continue
            result.append({
                "ids": [row["HashOwner"], row["HashApp"], row["HashFunction"]],
                "counts": counts,
                "total": total,
                "active_fraction": sum(value > 0 for value in counts) / len(counts),
                "cv": statistics.pstdev(counts) / statistics.mean(counts),
                "maximum": max(counts),
            })
    return result


def choose_functions(rows: list[dict]) -> dict[str, dict]:
    ordered = sorted(rows, key=lambda row: (-row["total"], row["ids"]))
    top_count = max(1, math.ceil(0.01 * len(ordered)))
    eligible = [row for row in ordered[:top_count] if row["active_fraction"] >= 0.90]
    if len(eligible) < 3:
        raise RuntimeError(f"only {len(eligible)} eligible traces in top 1%")
    by_cv = sorted(eligible, key=lambda row: (row["cv"], -row["total"], row["ids"]))
    selected = {
        "steady": by_cv[0],
        "variable": by_cv[len(by_cv) // 2],
        "bursty": by_cv[-1],
    }
    if len({tuple(row["ids"]) for row in selected.values()}) != 3:
        raise RuntimeError("trace selection rules did not yield three distinct functions")
    return selected


def nearest_minute(counts: list[int], q: float, used: set[int]) -> int:
    ordered = sorted(counts)
    target = ordered[round(q * (len(ordered) - 1))]
    choices = [i for i, value in enumerate(counts) if i not in used]
    return min(choices, key=lambda i: (abs(counts[i] - target), i))


def scaled_count(count: int, divisor: int) -> int:
    return (count + divisor // 2) // divisor


def jitter(count: int, seed: int, offset: float) -> list[float]:
    rng = random.Random(seed)
    return sorted(offset + 60.0 * rng.random() for _ in range(count))


def build(path: Path, output: Path) -> dict:
    rows = read_rows(path)
    selected = choose_functions(rows)
    streams = []
    selected_metadata = {}
    for kind, row in selected.items():
        divisor = max(1, math.ceil(row["maximum"] / MAX_SCALED_COUNT_PER_MINUTE))
        selected_metadata[kind] = {
            "hashed_function_key": row["ids"],
            "daily_invocations": row["total"],
            "active_minute_fraction": row["active_fraction"],
            "minute_count_cv": row["cv"],
            "maximum_source_minute_count": row["maximum"],
            "scale_divisor": divisor,
            "expected_scaled_max_per_minute": row["maximum"] / divisor,
        }
        used: set[int] = set()
        for q_index, q in enumerate(QUANTILES):
            minute_index = nearest_minute(row["counts"], q, used)
            used.add(minute_index)
            count_raw = row["counts"][minute_index]
            count_scaled = scaled_count(count_raw, divisor)
            previous_raw = row["counts"][minute_index - 1] if minute_index > 0 else 0
            previous_scaled = scaled_count(previous_raw, divisor)
            stream_seed = 32900000 + len(streams)
            streams.append({
                "stream_id": f"{kind}_q{int(q * 100):02d}_minute{minute_index + 1:04d}",
                "trace_class": kind,
                "target_count_quantile": q,
                "source_function_key": row["ids"],
                "source_minute_index_1based": minute_index + 1,
                "source_minute_count": count_raw,
                "scale_divisor": divisor,
                "current_scaled_count": count_scaled,
                "previous_source_minute_count": previous_raw,
                "previous_scaled_count": previous_scaled,
                "arrivals_seconds": jitter(count_scaled, stream_seed, 0.0),
                "history_seconds": jitter(previous_scaled, stream_seed + 1, -60.0),
                "jitter_seed": stream_seed,
            })
    if len(streams) != 12 or any(not row["arrivals_seconds"] for row in streams):
        raise RuntimeError("expected 12 non-empty minute streams")
    bundle = {
        "experiment": "R329_AZURE_2019_MINUTE_TRACE_INGRESS_REPLAY",
        "source": {
            "name": "Azure Functions 2019 public trace, invocations_per_function_md.anon.d01.csv",
            "sha256": sha256(path),
            "functions_with_nonzero_daily_count": len(rows),
            "minutes_per_function": 1440,
            "fields": "per-function invocation counts per minute",
        },
        "selection": {
            "eligible_pool": "top 1% by daily invocation count, then active in at least 90% of minutes",
            "classes": "minimum, median, and maximum minute-count coefficient of variation",
            "quantiles": list(QUANTILES),
            "within_minute_interpolation": "uniform jitter because source has minute counts only",
            "scaling": f"integer divisor ceil(max_count/{MAX_SCALED_COUNT_PER_MINUTE}); nearest-integer count scaling",
            "workflow_mapping": "one selected Azure function series drives ingress to a synthetic R326 eight-function DAG; the Azure trace contains no DAG relationship",
        },
        "selected_functions": selected_metadata,
        "streams": streams,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    temp.replace(output)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = build(args.input, args.output)
    print(json.dumps({
        "stream_count": len(bundle["streams"]),
        "selected_functions": bundle["selected_functions"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

