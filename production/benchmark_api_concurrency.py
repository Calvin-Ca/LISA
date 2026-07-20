from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from production.benchmark_api import (
    ensure_port_available,
    existing_compute_processes,
    get_json,
    portable_path,
    query_gpu_memory,
    query_gpu_total_memory,
    read_peak_gpu_memory,
    send_segment_request,
    stop_process,
    summarize_phase,
    wait_until_ready,
    write_requests_csv,
)


DEFAULT_PHASES = (
    ("warmup-c1", 1, 5),
    ("measured-c1", 1, 30),
    ("measured-c2", 2, 30),
    ("measured-c4", 4, 30),
    ("stability-c4", 4, 100),
)


def metric_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    name: str,
) -> float:
    return float(after.get(name, 0)) - float(before.get(name, 0))


def summarize_concurrency_phase(
    *,
    rows: list[dict[str, Any]],
    phase: str,
    concurrency: int,
    elapsed_seconds: float,
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
) -> dict[str, Any]:
    summary = summarize_phase(rows, phase, elapsed_seconds)
    started = int(metric_delta(
        metrics_before, metrics_after, "requests_started_total"
    ))
    queue_wait = metric_delta(
        metrics_before, metrics_after, "queue_wait_seconds_total"
    )
    gpu_seconds = metric_delta(
        metrics_before, metrics_after, "gpu_inference_seconds_total"
    )
    summary.update(
        {
            "client_concurrency": concurrency,
            "runtime_requests_received": int(metric_delta(
                metrics_before,
                metrics_after,
                "requests_received_total",
            )),
            "gpu_requests_started": started,
            "queue_wait_seconds_total": round(queue_wait, 6),
            "average_queue_wait_ms": round(
                queue_wait * 1000 / started if started else 0.0,
                3,
            ),
            "gpu_inference_seconds_total": round(gpu_seconds, 6),
            "average_gpu_inference_ms": round(
                gpu_seconds * 1000 / started if started else 0.0,
                3,
            ),
            "queue_timeout_delta": int(metric_delta(
                metrics_before, metrics_after, "queue_timeout_total"
            )),
            "queue_rejected_delta": int(metric_delta(
                metrics_before, metrics_after, "queue_rejected_total"
            )),
            "queue_cancelled_delta": int(metric_delta(
                metrics_before, metrics_after, "queue_cancelled_total"
            )),
        }
    )
    return summary


def build_acceptance_checks(
    *,
    phase_summaries: dict[str, dict[str, Any]],
    metrics_initial: dict[str, Any],
    metrics_final: dict[str, Any],
    total_requests: int,
    expected_requests: int,
    total_failed: int,
    gpu_memory_mib: dict[str, int],
    final_ready: dict[str, Any],
    missing_initial_process_ids: list[int],
    server_log: str,
    max_p95_ms: dict[str, float],
    min_throughput: float,
    max_peak_memory_mib: int,
    min_remaining_memory_mib: int,
    max_memory_drift_mib: int,
) -> dict[str, dict[str, Any]]:
    runtime_received = int(metric_delta(
        metrics_initial, metrics_final, "requests_received_total"
    ))
    runtime_started = int(metric_delta(
        metrics_initial, metrics_final, "requests_started_total"
    ))
    runtime_succeeded = int(metric_delta(
        metrics_initial, metrics_final, "requests_succeeded_total"
    ))
    gpu_succeeded = int(metric_delta(
        metrics_initial, metrics_final, "gpu_inference_succeeded_total"
    ))
    masks_returned = int(metric_delta(
        metrics_initial, metrics_final, "masks_returned_total"
    ))
    checks: dict[str, dict[str, Any]] = {
        "all_requests_succeeded": {
            "actual": total_requests - total_failed,
            "expected": expected_requests,
            "passed": (
                total_requests == expected_requests and total_failed == 0
            ),
        },
        "runtime_requests_received": {
            "actual": runtime_received,
            "expected": expected_requests,
            "passed": runtime_received == expected_requests,
        },
        "gpu_requests_started": {
            "actual": runtime_started,
            "expected": expected_requests,
            "passed": runtime_started == expected_requests,
        },
        "requests_succeeded": {
            "actual": runtime_succeeded,
            "expected": expected_requests,
            "passed": runtime_succeeded == expected_requests,
        },
        "gpu_inference_succeeded_total": {
            "actual": gpu_succeeded,
            "expected": expected_requests,
            "passed": gpu_succeeded == expected_requests,
        },
        "masks_returned": {
            "actual": masks_returned,
            "expected": f">= {expected_requests}",
            "passed": masks_returned >= expected_requests,
        },
        "gpu_inference_in_flight_max": {
            "actual": int(metrics_final.get(
                "gpu_inference_in_flight_max", 0
            )),
            "expected": 1,
            "passed": int(metrics_final.get(
                "gpu_inference_in_flight_max", 0
            )) == 1,
        },
        "gpu_inference_in_flight_final": {
            "actual": int(metrics_final.get("gpu_inference_in_flight", 0)),
            "expected": 0,
            "passed": int(
                metrics_final.get("gpu_inference_in_flight", 0)
            ) == 0,
        },
        "queue_timeout_total": {
            "actual": int(metric_delta(
                metrics_initial, metrics_final, "queue_timeout_total"
            )),
            "expected": 0,
            "passed": int(metric_delta(
                metrics_initial, metrics_final, "queue_timeout_total"
            )) == 0,
        },
        "queue_rejected_total": {
            "actual": int(metric_delta(
                metrics_initial, metrics_final, "queue_rejected_total"
            )),
            "expected": 0,
            "passed": int(metric_delta(
                metrics_initial, metrics_final, "queue_rejected_total"
            )) == 0,
        },
        "queue_cancelled_total": {
            "actual": int(metric_delta(
                metrics_initial, metrics_final, "queue_cancelled_total"
            )),
            "expected": 0,
            "passed": int(metric_delta(
                metrics_initial, metrics_final, "queue_cancelled_total"
            )) == 0,
        },
        "requests_timeout_total": {
            "actual": int(metric_delta(
                metrics_initial, metrics_final, "requests_timeout_total"
            )),
            "expected": 0,
            "passed": int(metric_delta(
                metrics_initial, metrics_final, "requests_timeout_total"
            )) == 0,
        },
        "gpu_inference_failed_total": {
            "actual": int(metric_delta(
                metrics_initial,
                metrics_final,
                "gpu_inference_failed_total",
            )),
            "expected": 0,
            "passed": int(metric_delta(
                metrics_initial,
                metrics_final,
                "gpu_inference_failed_total",
            )) == 0,
        },
        "unexpected_errors_total": {
            "actual": int(metric_delta(
                metrics_initial, metrics_final, "unexpected_errors_total"
            )),
            "expected": 0,
            "passed": int(metric_delta(
                metrics_initial, metrics_final, "unexpected_errors_total"
            )) == 0,
        },
        "cuda_oom_total": {
            "actual": int(metric_delta(
                metrics_initial, metrics_final, "cuda_oom_total"
            )),
            "expected": 0,
            "passed": int(metric_delta(
                metrics_initial, metrics_final, "cuda_oom_total"
            )) == 0,
        },
        "peak_gpu_memory_mib": {
            "actual": gpu_memory_mib["peak"],
            "expected": f"<= {max_peak_memory_mib}",
            "passed": gpu_memory_mib["peak"] <= max_peak_memory_mib,
        },
        "remaining_gpu_memory_mib": {
            "actual": gpu_memory_mib["remaining_at_peak"],
            "expected": f">= {min_remaining_memory_mib}",
            "passed": (
                gpu_memory_mib["remaining_at_peak"]
                >= min_remaining_memory_mib
            ),
        },
        "post_warmup_memory_drift_mib": {
            "actual": gpu_memory_mib["post_warmup_drift"],
            "expected": f"<= {max_memory_drift_mib}",
            "passed": (
                gpu_memory_mib["post_warmup_drift"]
                <= max_memory_drift_mib
            ),
        },
        "service_ready_after_requests": {
            "actual": final_ready.get("status"),
            "expected": "ready",
            "passed": final_ready.get("status") == "ready",
        },
        "existing_compute_processes_remained": {
            "actual": missing_initial_process_ids,
            "expected": [],
            "passed": not missing_initial_process_ids,
        },
        "no_cuda_oom_in_log": {
            "actual": "cuda out of memory" in server_log.lower(),
            "expected": False,
            "passed": "cuda out of memory" not in server_log.lower(),
        },
    }
    for phase, limit in max_p95_ms.items():
        actual = float(
            phase_summaries[phase].get("client_latency_p95_ms", 0.0)
        )
        checks[f"{phase}_client_p95_ms"] = {
            "actual": actual,
            "expected": f"<= {limit}",
            "passed": actual <= limit,
        }
    for phase in max_p95_ms:
        actual = float(
            phase_summaries[phase]["throughput_requests_per_second"]
        )
        checks[f"{phase}_throughput"] = {
            "actual": actual,
            "expected": f">= {min_throughput}",
            "passed": actual >= min_throughput,
        }
    return checks


def run_concurrent_phase(
    *,
    url: str,
    image_base64: str,
    prompt: str,
    phase: str,
    count: int,
    concurrency: int,
    timeout_seconds: float,
    sequence_start: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for index in range(count):
            sequence = sequence_start + index + 1
            request_id = f"concurrency-{phase}-{index + 1:03d}"
            futures.append(
                executor.submit(
                    send_segment_request,
                    url=url,
                    image_base64=image_base64,
                    prompt=prompt,
                    request_id=request_id,
                    timeout_seconds=timeout_seconds,
                    phase=phase,
                    sequence=sequence,
                )
            )
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            if row["success"] and int(row["mask_count"]) < 1:
                row["success"] = False
                row["error"] = "segment response did not contain a mask"
            rows.append(row)
            completed += 1
            status = "ok" if row["success"] else "failed"
            print(
                f"{phase} {completed}/{count}: {status}, "
                f"{row['client_latency_ms']} ms",
                flush=True,
            )
    return rows, time.perf_counter() - started


def wait_until_idle(
    *,
    base_url: str,
    server: subprocess.Popen,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_seconds
    last: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"Uvicorn exited with code {server.returncode}"
            )
        try:
            last = get_json(f"{base_url}/metrics", timeout=2.0)
        except (OSError, ValueError, urllib.error.URLError):
            time.sleep(0.05)
            continue
        if (
            int(last.get("queue_size", 0)) == 0
            and int(last.get("gpu_inference_in_flight", 0)) == 0
        ):
            return last
        time.sleep(0.05)
    raise TimeoutError(
        f"runtime did not become idle after {timeout_seconds}s: {last}"
    )


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LISA Production API Concurrency Verification",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Model: `{summary['model_version']}`",
        f"- Shared GPU: `{summary['shared_gpu']}`",
        f"- Ready time: `{summary['ready_time_ms']:.3f} ms`",
        f"- GPU worker count: `{summary['max_concurrency']}`",
        f"- Queue size: `{summary['max_queue_size']}`",
        "",
        "## Phases",
        "",
        "| Phase | Client concurrency | Requests | Pass | P50 ms | P95 ms | "
        "P99 ms | Req/s | Avg queue ms | Avg GPU ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: |",
    ]
    for phase, item in summary["phases"].items():
        lines.append(
            f"| {phase} | {item['client_concurrency']} | "
            f"{item['requests']} | {item['succeeded']} | "
            f"{item.get('client_latency_p50_ms', 0):.3f} | "
            f"{item.get('client_latency_p95_ms', 0):.3f} | "
            f"{item.get('client_latency_p99_ms', 0):.3f} | "
            f"{item['throughput_requests_per_second']:.6f} | "
            f"{item['average_queue_wait_ms']:.3f} | "
            f"{item['average_gpu_inference_ms']:.3f} |"
        )
    gpu = summary["gpu_memory_mib"]
    lines.extend(
        [
            "",
            "## GPU memory",
            "",
            f"- Baseline: `{gpu['baseline']} MiB`",
            f"- Model loaded: `{gpu['loaded']} MiB`",
            f"- After warmup: `{gpu['after_warmup']} MiB`",
            f"- Peak: `{gpu['peak']} MiB`",
            f"- Remaining at peak: `{gpu['remaining_at_peak']} MiB`",
            f"- After requests: `{gpu['after_requests']} MiB`",
            f"- Post-warmup drift: `{gpu['post_warmup_drift']} MiB`",
            "",
            "## Final runtime metrics",
            "",
            "- Requests received: "
            f"`{summary['runtime_deltas']['requests_received_total']}`",
            "- Requests succeeded: "
            f"`{summary['runtime_deltas']['requests_succeeded_total']}`",
            "- GPU inference succeeded: "
            f"`{summary['runtime_deltas']['gpu_inference_succeeded_total']}`",
            "- Maximum GPU in flight: "
            f"`{summary['metrics_final'].get('gpu_inference_in_flight_max', 0)}`",
            f"- Queue timeout: `{summary['runtime_deltas']['queue_timeout_total']}`",
            f"- Queue rejected: `{summary['runtime_deltas']['queue_rejected_total']}`",
            f"- CUDA OOM: `{summary['runtime_deltas']['cuda_oom_total']}`",
            "",
            "## Acceptance",
            "",
        ]
    )
    for name, item in summary["acceptance"]["checks"].items():
        marker = "PASS" if item["passed"] else "FAIL"
        lines.append(
            f"- {marker} `{name}`: actual `{item['actual']}`, "
            f"expected `{item['expected']}`"
        )
    lines.extend(
        [
            "",
            f"Overall: `{'PASS' if summary['acceptance']['passed'] else 'FAIL'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark serialized LISA GPU inference under HTTP concurrency."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", default="lisa13b-clean030-v1")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--client-timeout", type=float, default=150.0)
    parser.add_argument("--max-queue-size", type=int, default=8)
    parser.add_argument("--queue-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--min-throughput", type=float, default=2.0)
    parser.add_argument("--max-c1-p95-ms", type=float, default=1000.0)
    parser.add_argument("--max-c2-p95-ms", type=float, default=2000.0)
    parser.add_argument("--max-c4-p95-ms", type=float, default=4000.0)
    parser.add_argument("--max-peak-memory-mib", type=int, default=36864)
    parser.add_argument("--min-remaining-memory-mib", type=int, default=4096)
    parser.add_argument("--max-memory-drift-mib", type=int, default=500)
    parser.add_argument(
        "--require-existing-process-substring",
        default="VLLM::EngineCore",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_path = args.model_path.resolve()
    vision_tower = args.vision_tower.resolve()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()

    for path, description in [
        (model_path / "config.json", "model config"),
        (vision_tower / "config.json", "CLIP config"),
        (vision_tower / "preprocessor_config.json", "CLIP preprocessor"),
        (image_path, "benchmark image"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"missing {description}: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_port_available(args.host, args.port)

    processes_before = existing_compute_processes(args.gpu_index)
    required = args.require_existing_process_substring.lower()
    if not any(
        required in item["process_name"].lower()
        for item in processes_before
    ):
        raise RuntimeError(
            "required shared GPU process was not found: "
            f"{args.require_existing_process_substring}"
        )

    baseline_memory = query_gpu_memory(args.gpu_index)
    total_gpu_memory = query_gpu_total_memory(args.gpu_index)
    expected_requests = sum(count for _, _, count in DEFAULT_PHASES)
    max_p95_ms = {
        "measured-c1": args.max_c1_p95_ms,
        "measured-c2": args.max_c2_p95_ms,
        "measured-c4": args.max_c4_p95_ms,
        "stability-c4": args.max_c4_p95_ms,
    }
    runtime_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "model_version": args.model_version,
        "model_path": portable_path(model_path, repo_root),
        "vision_tower": portable_path(vision_tower, repo_root),
        "image": portable_path(image_path, repo_root),
        "prompt": args.prompt,
        "precision": "bf16",
        "gpu_index": args.gpu_index,
        "shared_gpu": True,
        "existing_compute_processes_before": processes_before,
        "max_concurrency": 1,
        "max_queue_size": args.max_queue_size,
        "queue_timeout_seconds": args.queue_timeout,
        "request_timeout_seconds": args.request_timeout,
        "host": args.host,
        "port": args.port,
        "phases": [
            {
                "name": phase,
                "client_concurrency": concurrency,
                "requests": count,
            }
            for phase, concurrency, count in DEFAULT_PHASES
        ],
        "expected_requests": expected_requests,
        "thresholds": {
            "max_p95_ms": max_p95_ms,
            "min_throughput_requests_per_second": args.min_throughput,
            "max_peak_memory_mib": args.max_peak_memory_mib,
            "min_remaining_memory_mib": args.min_remaining_memory_mib,
            "max_memory_drift_mib": args.max_memory_drift_mib,
        },
    }
    (output_dir / "runtime_config.json").write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "LISA_MODEL_VERSION": args.model_version,
            "LISA_MODEL_PATH": str(model_path),
            "LISA_VISION_TOWER": str(vision_tower),
            "LISA_PRECISION": "bf16",
            "LISA_LOAD_IN_8BIT": "false",
            "LISA_LOAD_IN_4BIT": "false",
            "LISA_GPU_INDEX": "0",
            "LISA_MAX_CONCURRENCY": "1",
            "LISA_MAX_QUEUE_SIZE": str(args.max_queue_size),
            "LISA_QUEUE_TIMEOUT_SECONDS": str(args.queue_timeout),
            "LISA_REQUEST_TIMEOUT_SECONDS": str(args.request_timeout),
            "LISA_EAGER_LOAD": "true",
            "LISA_API_KEY": "",
        }
    )
    server_log_path = output_dir / "server.log"
    gpu_csv_path = output_dir / "gpu_metrics.csv"
    server: subprocess.Popen | None = None
    monitor: subprocess.Popen | None = None
    server_handle = None
    monitor_handle = None
    rows: list[dict[str, Any]] = []
    metrics_snapshots: dict[str, Any] = {}
    try:
        monitor_handle = gpu_csv_path.open("w", encoding="utf-8")
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={args.gpu_index}",
                "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ],
            stdout=monitor_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        server_handle = server_log_path.open("w", encoding="utf-8")
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "production.app:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--workers",
                "1",
                "--no-access-log",
            ],
            cwd=repo_root,
            env=env,
            stdout=server_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base_url = f"http://{args.host}:{args.port}"
        ready_time_ms = wait_until_ready(
            base_url, server, args.startup_timeout
        )
        loaded_memory = query_gpu_memory(args.gpu_index)
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        metrics_initial = get_json(f"{base_url}/metrics", timeout=2.0)
        metrics_snapshots["initial"] = metrics_initial

        sequence = 0
        phase_summaries: dict[str, dict[str, Any]] = {}
        after_warmup_memory = loaded_memory
        for phase, concurrency, count in DEFAULT_PHASES:
            metrics_before = get_json(
                f"{base_url}/metrics", timeout=2.0
            )
            phase_rows, elapsed = run_concurrent_phase(
                url=f"{base_url}/v1/segment",
                image_base64=image_base64,
                prompt=args.prompt,
                phase=phase,
                count=count,
                concurrency=concurrency,
                timeout_seconds=args.client_timeout,
                sequence_start=sequence,
            )
            sequence += count
            rows.extend(phase_rows)
            metrics_after = wait_until_idle(
                base_url=base_url,
                server=server,
                timeout_seconds=args.client_timeout,
            )
            metrics_snapshots[phase] = {
                "before": metrics_before,
                "after": metrics_after,
            }
            phase_summaries[phase] = summarize_concurrency_phase(
                rows=phase_rows,
                phase=phase,
                concurrency=concurrency,
                elapsed_seconds=elapsed,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
            )
            if phase == "warmup-c1":
                after_warmup_memory = query_gpu_memory(args.gpu_index)

        metrics_final = wait_until_idle(
            base_url=base_url,
            server=server,
            timeout_seconds=args.client_timeout,
        )
        final_ready = get_json(f"{base_url}/ready", timeout=2.0)
        after_requests_memory = query_gpu_memory(args.gpu_index)
        processes_after = existing_compute_processes(args.gpu_index)
        time.sleep(0.5)
        stop_process(monitor, timeout=5)
        monitor = None
        monitor_handle.close()
        monitor_handle = None
        peak_memory = read_peak_gpu_memory(gpu_csv_path)
        if peak_memory <= 0:
            raise RuntimeError(
                "GPU monitor produced no valid memory samples"
            )

        rows.sort(key=lambda item: int(item["sequence"]))
        write_requests_csv(output_dir / "requests.csv", rows)
        (output_dir / "requests.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metrics_snapshots["final"] = metrics_final
        (output_dir / "metrics_snapshots.json").write_text(
            json.dumps(metrics_snapshots, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        initial_process_ids = {item["pid"] for item in processes_before}
        final_process_ids = {item["pid"] for item in processes_after}
        missing_initial_process_ids = sorted(
            initial_process_ids - final_process_ids
        )
        drift = after_requests_memory - after_warmup_memory
        gpu_memory_mib = {
            "baseline": baseline_memory,
            "loaded": loaded_memory,
            "after_warmup": after_warmup_memory,
            "peak": peak_memory,
            "after_requests": after_requests_memory,
            "total": total_gpu_memory,
            "remaining_at_peak": total_gpu_memory - peak_memory,
            "peak_increment_over_baseline": peak_memory - baseline_memory,
            "post_warmup_drift": drift,
        }
        total_failed = sum(not bool(row["success"]) for row in rows)
        server_log = server_log_path.read_text(
            encoding="utf-8", errors="replace"
        )
        checks = build_acceptance_checks(
            phase_summaries=phase_summaries,
            metrics_initial=metrics_initial,
            metrics_final=metrics_final,
            total_requests=len(rows),
            expected_requests=expected_requests,
            total_failed=total_failed,
            gpu_memory_mib=gpu_memory_mib,
            final_ready=final_ready,
            missing_initial_process_ids=missing_initial_process_ids,
            server_log=server_log,
            max_p95_ms=max_p95_ms,
            min_throughput=args.min_throughput,
            max_peak_memory_mib=args.max_peak_memory_mib,
            min_remaining_memory_mib=args.min_remaining_memory_mib,
            max_memory_drift_mib=args.max_memory_drift_mib,
        )
        runtime_delta_names = (
            "requests_received_total",
            "requests_started_total",
            "requests_succeeded_total",
            "gpu_inference_succeeded_total",
            "gpu_inference_failed_total",
            "masks_returned_total",
            "queue_timeout_total",
            "queue_rejected_total",
            "queue_cancelled_total",
            "requests_timeout_total",
            "unexpected_errors_total",
            "cuda_oom_total",
            "queue_wait_seconds_total",
            "gpu_inference_seconds_total",
        )
        runtime_deltas = {
            name: round(
                metric_delta(metrics_initial, metrics_final, name),
                6,
            )
            for name in runtime_delta_names
        }
        gpu_name = subprocess.run(
            [
                "nvidia-smi",
                f"--id={args.gpu_index}",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        summary = {
            **runtime_config,
            "gpu_name": gpu_name,
            "ready_time_ms": round(ready_time_ms, 3),
            "gpu_memory_mib": gpu_memory_mib,
            "existing_compute_processes_after": processes_after,
            "missing_initial_process_ids": missing_initial_process_ids,
            "phases": phase_summaries,
            "total_requests": len(rows),
            "total_failed": total_failed,
            "runtime_deltas": runtime_deltas,
            "metrics_final": metrics_final,
            "final_ready": final_ready,
            "acceptance": {
                "passed": all(item["passed"] for item in checks.values()),
                "checks": checks,
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_markdown(output_dir / "summary.md", summary)
        print(f"Saved concurrency outputs to: {output_dir}")
        print(
            "Acceptance: "
            + ("PASS" if summary["acceptance"]["passed"] else "FAIL")
        )
        return 0 if summary["acceptance"]["passed"] else 2
    finally:
        stop_process(server)
        stop_process(monitor, timeout=5)
        if server_handle is not None:
            server_handle.close()
        if monitor_handle is not None:
            monitor_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
