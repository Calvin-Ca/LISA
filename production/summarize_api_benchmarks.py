from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one performance summary is required")
    versions = {item["model_version"] for item in summaries}
    models = {item["model_path"] for item in summaries}
    if len(versions) != 1 or len(models) != 1:
        raise ValueError("all rounds must use the same model version and path")

    rounds = []
    for index, item in enumerate(summaries, start=1):
        measured = item["phases"]["measured"]
        memory = item["gpu_memory_mib"]
        rounds.append(
            {
                "round": index,
                "created_at": item["created_at"],
                "repo_git_commit": item["repo_git_commit"],
                "acceptance_passed": item["acceptance"]["passed"],
                "ready_time_ms": item["ready_time_ms"],
                "measured_p50_ms": measured["client_latency_p50_ms"],
                "measured_p95_ms": measured["client_latency_p95_ms"],
                "measured_p99_ms": measured["client_latency_p99_ms"],
                "throughput_requests_per_second": measured[
                    "throughput_requests_per_second"
                ],
                "baseline_memory_mib": memory["baseline"],
                "peak_memory_mib": memory["peak"],
                "remaining_at_peak_mib": memory["remaining_at_peak"],
                "peak_increment_over_baseline_mib": memory[
                    "peak_increment_over_baseline"
                ],
                "post_warmup_drift_mib": memory["post_warmup_drift"],
                "total_requests": item["total_requests"],
                "total_failed": item["total_failed"],
                "existing_compute_processes_before": item[
                    "existing_compute_processes_before"
                ],
            }
        )

    ready_times = [float(item["ready_time_ms"]) for item in rounds]
    p95_values = [float(item["measured_p95_ms"]) for item in rounds]
    throughput_values = [
        float(item["throughput_requests_per_second"]) for item in rounds
    ]
    peak_values = [int(item["peak_memory_mib"]) for item in rounds]
    remaining_values = [int(item["remaining_at_peak_mib"]) for item in rounds]
    drift_values = [int(item["post_warmup_drift_mib"]) for item in rounds]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": next(iter(versions)),
        "model_path": next(iter(models)),
        "round_count": len(rounds),
        "shared_gpu": all(item.get("shared_gpu", False) for item in summaries),
        "rounds": rounds,
        "ready_time_ms": {
            "min": min(ready_times),
            "mean": statistics.fmean(ready_times),
            "max": max(ready_times),
        },
        "measured_client_p95_ms": {
            "min": min(p95_values),
            "mean": statistics.fmean(p95_values),
            "max": max(p95_values),
        },
        "throughput_requests_per_second": {
            "min": min(throughput_values),
            "mean": statistics.fmean(throughput_values),
            "max": max(throughput_values),
        },
        "gpu_memory_mib": {
            "max_peak": max(peak_values),
            "min_remaining_at_peak": min(remaining_values),
            "max_post_warmup_drift": max(drift_values),
        },
        "total_requests": sum(int(item["total_requests"]) for item in rounds),
        "total_failed": sum(int(item["total_failed"]) for item in rounds),
        "acceptance": {
            "passed": all(
                item["acceptance_passed"] for item in rounds
            ),
            "rounds_passed": sum(
                bool(item["acceptance_passed"]) for item in rounds
            ),
            "rounds_total": len(rounds),
        },
    }


def write_markdown(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# LISA Shared-GPU Performance Aggregate",
        "",
        f"- Created at: `{aggregate['created_at']}`",
        f"- Model: `{aggregate['model_version']}`",
        f"- Rounds: `{aggregate['round_count']}`",
        f"- Shared GPU: `{aggregate['shared_gpu']}`",
        "",
        "| Round | Pass | Ready ms | P50 ms | P95 ms | P99 ms | Req/s | "
        "Baseline MiB | Peak MiB | Remaining MiB | Drift MiB |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]
    for item in aggregate["rounds"]:
        lines.append(
            "| {round} | {passed} | {ready:.3f} | {p50:.3f} | "
            "{p95:.3f} | {p99:.3f} | {throughput:.6f} | {baseline} | "
            "{peak} | {remaining} | {drift} |".format(
                round=item["round"],
                passed="PASS" if item["acceptance_passed"] else "FAIL",
                ready=item["ready_time_ms"],
                p50=item["measured_p50_ms"],
                p95=item["measured_p95_ms"],
                p99=item["measured_p99_ms"],
                throughput=item["throughput_requests_per_second"],
                baseline=item["baseline_memory_mib"],
                peak=item["peak_memory_mib"],
                remaining=item["remaining_at_peak_mib"],
                drift=item["post_warmup_drift_mib"],
            )
        )
    lines.extend(
        [
            "",
            "## Worst-case summary",
            "",
            f"- Maximum ready time: `{aggregate['ready_time_ms']['max']:.3f} ms`",
            f"- Maximum measured P95: `{aggregate['measured_client_p95_ms']['max']:.3f} ms`",
            f"- Minimum throughput: `{aggregate['throughput_requests_per_second']['min']:.6f} req/s`",
            f"- Maximum GPU peak: `{aggregate['gpu_memory_mib']['max_peak']} MiB`",
            f"- Minimum remaining memory: `{aggregate['gpu_memory_mib']['min_remaining_at_peak']} MiB`",
            f"- Maximum post-warmup drift: `{aggregate['gpu_memory_mib']['max_post_warmup_drift']} MiB`",
            f"- Total requests: `{aggregate['total_requests']}`",
            f"- Total failed: `{aggregate['total_failed']}`",
            "",
            "## Acceptance",
            "",
            f"Overall: `{'PASS' if aggregate['acceptance']['passed'] else 'FAIL'}` "
            f"({aggregate['acceptance']['rounds_passed']}/"
            f"{aggregate['acceptance']['rounds_total']} rounds)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple production API benchmark summaries."
    )
    parser.add_argument(
        "--round-summary",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "aggregate_summary.json"
    markdown_path = output_dir / "aggregate_summary.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"aggregate outputs already exist under: {output_dir}"
        )
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.round_summary
    ]
    aggregate = aggregate_summaries(summaries)
    json_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_path, aggregate)
    print(f"Saved aggregate outputs to: {output_dir}")
    print(
        "Acceptance: "
        + ("PASS" if aggregate["acceptance"]["passed"] else "FAIL")
    )


if __name__ == "__main__":
    main()
