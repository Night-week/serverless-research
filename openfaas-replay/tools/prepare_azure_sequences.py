#!/usr/bin/env python3
"""Select nine outcome-blind Azure Functions minute-count sequences."""
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
MAX_PER_MINUTE = 480


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        minute_names = [name for name in names if name.isdigit()]
        if not {"HashOwner", "HashApp", "HashFunction"}.issubset(names):
            raise ValueError("Azure CSV is missing hashed function identifiers")
        if len(minute_names) != 1440:
            raise ValueError(f"expected 1440 minute columns, found {len(minute_names)}")
        result = []
        for row in reader:
            counts = [int(row.get(name) or 0) for name in minute_names]
            total = sum(counts)
            if total <= 0:
                continue
            result.append({
                "key": [row["HashOwner"], row["HashApp"], row["HashFunction"]],
                "counts": counts,
                "total": total,
                "active_fraction": sum(value > 0 for value in counts) / len(counts),
                "cv": statistics.pstdev(counts) / statistics.mean(counts),
                "maximum": max(counts),
            })
    return result


def select_nine(rows: list[dict]) -> list[tuple[str, dict, float]]:
    ordered = sorted(rows, key=lambda item: (-item["total"], item["key"]))
    top = ordered[:max(1, math.ceil(0.01 * len(ordered)))]
    eligible = sorted((row for row in top if row["active_fraction"] >= 0.90),
                      key=lambda row: (row["cv"], -row["total"], row["key"]))
    if len(eligible) < 9:
        raise RuntimeError(f"only {len(eligible)} eligible series; nine required")
    # Divide the eligible CV ranks into three equal strata; choose p10/p50/p90
    # within each stratum. This yields nine distinct, outcome-blind sequences.
    selected = []
    for stratum, lo, hi in (("low_cv", 0, 1), ("mid_cv", 1, 2), ("high_cv", 2, 3)):
        left = len(eligible) * lo // 3
        right = len(eligible) * hi // 3
        group = eligible[left:right]
        if len(group) < 3:
            raise RuntimeError("CV stratum is too small")
        for quantile, suffix in ((0.10, "p10"), (0.50, "p50"), (0.90, "p90")):
            index = round(quantile * (len(group) - 1))
            selected.append((f"{stratum}_{suffix}", group[index], quantile))
    if len({tuple(row["key"]) for _, row, _ in selected}) != 9:
        raise RuntimeError("selection did not yield nine distinct function series")
    return selected


def nearest_minute(counts: list[int], q: float, used: set[int]) -> int:
    ordered = sorted(counts)
    target = ordered[round(q * (len(ordered) - 1))]
    choices = [index for index, value in enumerate(counts) if index not in used]
    return min(choices, key=lambda index: (abs(counts[index] - target), index))


def round_scaled(count: int, divisor: int) -> int:
    return (count + divisor // 2) // divisor


def jitter(count: int, seed: int, offset: float) -> list[float]:
    rng = random.Random(seed)
    return sorted(offset + 60.0 * rng.random() for _ in range(count))


def build(source: Path, output: Path) -> dict:
    rows = read_rows(source)
    selected = select_nine(rows)
    streams, metadata = [], {}
    for sequence_index, (sequence_id, row, cv_quantile) in enumerate(selected):
        divisor = max(1, math.ceil(row["maximum"] / MAX_PER_MINUTE))
        metadata[sequence_id] = {
            "hashed_function_key": row["key"],
            "daily_invocations": row["total"],
            "active_minute_fraction": row["active_fraction"],
            "minute_count_cv": row["cv"],
            "cv_stratum_quantile": cv_quantile,
            "maximum_source_minute_count": row["maximum"],
            "scale_divisor": divisor,
        }
        used: set[int] = set()
        for q_index, q in enumerate(QUANTILES):
            minute_index = nearest_minute(row["counts"], q, used)
            used.add(minute_index)
            current_raw = row["counts"][minute_index]
            previous_raw = row["counts"][minute_index - 1] if minute_index else 0
            current = round_scaled(current_raw, divisor)
            previous = round_scaled(previous_raw, divisor)
            seed = 33200000 + sequence_index * 100 + q_index
            streams.append({
                "stream_id": f"{sequence_id}_q{int(q * 100):02d}_minute{minute_index + 1:04d}",
                "sequence_id": sequence_id,
                "trace_class": sequence_id.split("_")[0],
                "cv": row["cv"],
                "cv_quantile": cv_quantile,
                "window_quantile": q,
                "source_minute_index_1based": minute_index + 1,
                "source_minute_count": current_raw,
                "previous_source_minute_count": previous_raw,
                "scale_divisor": divisor,
                "current_scaled_count": current,
                "previous_scaled_count": previous,
                "arrivals_seconds": jitter(current, seed, 0.0),
                "history_seconds": jitter(previous, seed + 1, -60.0),
                "jitter_seed": seed,
            })
    if len(streams) != 36 or any(not stream["arrivals_seconds"] for stream in streams):
        raise RuntimeError("expected 36 nonempty windows")
    bundle = {
        "experiment": "R332_AZURE_FUNCTION_SEQUENCE_EXPANSION",
        "source": {
            "name": source.name,
            "sha256": sha256(source),
            "nonempty_function_series": len(rows),
            "minutes_per_series": 1440,
            "fields": "anonymized per-function invocation count per minute",
        },
        "selection": {
            "eligible_pool": "top 1% by daily count; active >=90% of minutes",
            "selection_rule": "CV-sort eligible rows; split into three equal strata; choose within-stratum p10/p50/p90",
            "minute_windows": list(QUANTILES),
            "maximum_scaled_requests_per_minute": MAX_PER_MINUTE,
            "scaling": "smallest integer divisor with rounded scaled max <=480/minute",
            "within_minute_interpolation": "uniform jitter, deterministic per sequence/window",
            "previous_minute": "uniformly jittered over [-60,0] seconds as history",
            "workflow_mapping": "each Azure function series drives ingress to existing held-out synthetic 20-function DAGs; source has no DAG membership",
        },
        "selected_sequences": metadata,
        "streams": streams,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.input, args.output)
    print(json.dumps({"sequence_count": len(result["selected_sequences"]),
                      "stream_count": len(result["streams"]),
                      "sha256": result["source"]["sha256"],
                      "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
