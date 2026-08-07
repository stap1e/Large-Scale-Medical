#!/usr/bin/env python3
"""Aggregate completed A/B timing and throughput summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir")
    args = parser.parse_args()
    root = Path(args.runs_dir)
    if not root.is_dir():
        raise SystemExit(f"runs directory does not exist: {root}")
    rows = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        timing_path = case_dir / "summary.json"
        throughput_path = case_dir / "throughput.json"
        if not timing_path.is_file() or not throughput_path.is_file():
            continue
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case": case_dir.name,
                "production_median_step_ms": throughput["median_step_cycle_ms"],
                "production_p95_step_ms": throughput["p95_step_cycle_ms"],
                "diagnostic_median_step_ms": timing["median_step_time_ms"],
                "diagnostic_p95_step_ms": timing["p95_step_time_ms"],
                "iterator_wait_p95_ms": timing["iterator_wait_p95_ms"],
                "inter_step_gap_candidate_total_ms": timing[
                    "inter_step_cuda_gap_candidate_total_ms"
                ],
                "samples_per_second_global": throughput[
                    "samples_per_second_global"
                ],
                "slow_steps": timing["slow_step_count"],
                "exact_gpu_idle_gap_ms": "",
                "utilization_series": str(case_dir / "nvidia_dmon.txt"),
            }
        )

    if not rows:
        raise SystemExit(
            f"no complete case contains both summary.json and throughput.json under {root}"
        )

    baseline = next((row for row in rows if row["case"] == "A1_current"), None)
    for row in rows:
        if baseline is None:
            row["samples_per_second_vs_A1_percent"] = ""
            row["median_step_time_vs_A1_percent"] = ""
            continue
        baseline_samples = float(baseline["samples_per_second_global"])
        baseline_step = float(baseline["production_median_step_ms"])
        row["samples_per_second_vs_A1_percent"] = (
            (float(row["samples_per_second_global"]) / baseline_samples - 1.0)
            * 100.0
            if baseline_samples
            else ""
        )
        row["median_step_time_vs_A1_percent"] = (
            (float(row["production_median_step_ms"]) / baseline_step - 1.0)
            * 100.0
            if baseline_step
            else ""
        )

    csv_path = root / "ab_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# OCL A/B results",
        "",
        "| Case | production median ms | production p95 ms | iterator p95 ms | inter-step gap candidate ms | global samples/s | slow steps | exact GPU idle ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {case} | {production_median_step_ms:.3f} | {production_p95_step_ms:.3f} | "
            "{iterator_wait_p95_ms:.3f} | {inter_step_gap_candidate_total_ms:.3f} | "
            "{samples_per_second_global:.4f} | {slow_steps} | pending Nsight |".format(
                **row
            )
        )
    fixed = next((row for row in rows if row["case"] == "A6_fixed"), None)
    if (
        baseline is not None
        and fixed is not None
        and isinstance(fixed["samples_per_second_vs_A1_percent"], float)
        and isinstance(fixed["median_step_time_vs_A1_percent"], float)
    ):
        markdown.extend(
            [
                "",
                "A6 相对 A1：samples/s "
                f"{fixed['samples_per_second_vs_A1_percent']:+.2f}%，"
                "median step time "
                f"{fixed['median_step_time_vs_A1_percent']:+.2f}%。",
            ]
        )
    markdown.extend(
        [
            "",
            "`exact GPU idle` 必须从各 case 的 Nsight timeline 填写；CUDA event gap "
            "candidate 不是精确 idle integral。",
        ]
    )
    (root / "ab_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"wrote {csv_path} and {root / 'ab_results.md'}")


if __name__ == "__main__":
    main()
