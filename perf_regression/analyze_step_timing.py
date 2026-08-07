#!/usr/bin/env python3
"""Summarize one OCL GPU-gap timing CSV without third-party dependencies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def json_list(row: dict[str, str], key: str) -> list:
    try:
        value = json.loads(row.get(key, "") or "[]")
        return value if isinstance(value, list) else [value]
    except (json.JSONDecodeError, TypeError):
        return []


def update_succeeded(row: dict[str, str]) -> bool:
    value = (row.get("update_succeeded") or "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measure", type=int, default=500)
    parser.add_argument("--output")
    parser.add_argument("--require-no-cache-miss", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.measure <= 0:
        parser.error("--warmup must be non-negative and --measure must be positive")

    source = Path(args.csv_path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    successful_indices = [
        index for index, row in enumerate(all_rows) if update_succeeded(row)
    ]
    required_successes = args.warmup + args.measure
    if len(successful_indices) < required_successes:
        raise SystemExit(
            f"not enough successful optimizer updates in {source}; "
            f"successful={len(successful_indices)}, required={required_successes}, "
            f"attempts={len(all_rows)}"
        )
    start_index = successful_indices[args.warmup - 1] + 1 if args.warmup else 0
    end_index = successful_indices[required_successes - 1] + 1
    rows = all_rows[start_index:end_index]
    successful_rows = [row for row in rows if update_succeeded(row)]

    total = [as_float(row, "total_wall_time") for row in rows]
    update_cycle_times = []
    pending_cycle_ms = 0.0
    for row in rows:
        pending_cycle_ms += as_float(row, "total_wall_time")
        if update_succeeded(row):
            update_cycle_times.append(pending_cycle_ms)
            pending_cycle_ms = 0.0
    iterator = [as_float(row, "iterator_wait_time") for row in rows]
    inter_step_gap = [as_float(row, "inter_step_cuda_gap_time") for row in rows]
    worker_getitem_times = [
        float(value)
        for row in rows
        for value in json_list(row, "worker_getitem_time_ms")
    ]
    raw_samples = sum(as_float(row, "batch_size") for row in successful_rows)
    elapsed_seconds = sum(total) / 1000.0
    host_gap_fields = (
        "iterator_wait_time",
        "cpu_batch_prepare_time",
        "logging_time",
        "checkpoint_time",
        "validation_time",
        "unattributed_wall_time",
    )
    host_gap_candidate_ms = sum(
        sum(as_float(row, field) for field in host_gap_fields) for row in rows
    )
    classification_counts = Counter(
        row.get("gap_classification", "UNKNOWN") for row in rows
    )
    slow_rows = [row for row in rows if row.get("slow_step", "").lower() == "true"]
    slow_classification_counts = Counter(
        row.get("gap_classification", "UNKNOWN") for row in slow_rows
    )
    epoch_boundary_slow = sum(
        row.get("epoch_boundary", "").lower() == "true" for row in slow_rows
    )
    cache_miss_steps = sum(
        any(int(value) == 1 for value in json_list(row, "cache_miss"))
        for row in rows
    )
    cache_write_steps = sum(
        row.get("triggered_cache_write", "").lower() == "true" for row in rows
    )
    slow_volumes = sorted(
        (
            {
                "global_step": row.get("global_step"),
                "max_worker_getitem_time_ms": as_float(
                    row, "max_worker_getitem_time_ms"
                ),
                "paths": json_list(row, "data_file_path"),
                "cache_status": json_list(row, "cache_status"),
                "workers": json_list(row, "dataloader_worker_id"),
            }
            for row in slow_rows
        ),
        key=lambda item: item["max_worker_getitem_time_ms"],
        reverse=True,
    )[:10]
    summary = {
        "source": str(source),
        "warmup_steps": args.warmup,
        "measured_steps": len(successful_rows),
        "measured_attempts": len(rows),
        "measurement_start_unix": (
            as_float(rows[0], "step_start_unix") if rows else None
        ),
        "measurement_end_unix": (
            as_float(rows[-1], "step_end_unix") if rows else None
        ),
        "amp_overflow_attempts": len(rows) - len(successful_rows),
        "median_step_time_ms": statistics.median(update_cycle_times),
        "p95_step_time_ms": percentile(update_cycle_times, 0.95),
        "iterator_wait_p95_ms": percentile(iterator, 0.95),
        "worker_getitem_p95_ms": percentile(worker_getitem_times, 0.95),
        "samples_per_second": raw_samples / elapsed_seconds if elapsed_seconds else 0.0,
        "host_gap_candidate_total_ms": host_gap_candidate_ms,
        "inter_step_cuda_gap_candidate_total_ms": sum(inter_step_gap),
        "inter_step_cuda_gap_p95_ms": percentile(inter_step_gap, 0.95),
        "gpu_idle_gap_total_ms": None,
        "gpu_idle_gap_note": (
            "Exact device-idle duration requires the saved profiler/Nsight timeline; "
            "the CUDA-event inter-step value spans the previous logging-side CUDA "
            "tail to the next H2D-start and can include checkpoint D2H or tiny "
            "non-training kernels, so it is a gap candidate, not an exact "
            "idle-state integral."
        ),
        "slow_step_count": len(slow_rows),
        "slow_epoch_boundary_count": epoch_boundary_slow,
        "cache_miss_step_count": cache_miss_steps,
        "cache_write_step_count": cache_write_steps,
        "classification_counts": dict(classification_counts),
        "slow_classification_counts": dict(slow_classification_counts),
        "dominant_slow_classification": (
            slow_classification_counts.most_common(1)[0][0]
            if slow_classification_counts
            else "NONE"
        ),
        "top_slow_volume_batches": slow_volumes,
    }
    destination = (
        Path(args.output)
        if args.output
        else source.with_name(source.stem + "_summary.json")
    )
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.require_no_cache_miss and cache_miss_steps:
        raise SystemExit(
            f"{cache_miss_steps} measured steps contained cache misses; "
            "prewarm/freeze the cache before comparing A/B runs"
        )


if __name__ == "__main__":
    main()
