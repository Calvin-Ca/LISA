from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from production.benchmark_api import (
    ensure_port_available,
    existing_compute_processes,
    get_json,
    portable_path,
    query_gpu_memory,
    stop_process,
    wait_until_ready,
)


def send_expected_timeout(
    *,
    url: str,
    image_base64: str,
    prompt: str,
    request_id: str,
    client_timeout_seconds: float,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "request_id": request_id,
            "prompt": prompt,
            "image_base64": image_base64,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
        },
        method="POST",
    )
    barrier.wait(timeout=10)
    started = time.perf_counter()
    status = 0
    payload: dict[str, Any] = {}
    error = ""
    try:
        with urllib.request.urlopen(
            request, timeout=client_timeout_seconds
        ) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
            error = "request unexpectedly succeeded"
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as decode_exc:
            error = f"invalid error response: {decode_exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    passed = (
        status == 504
        and payload.get("code") == "inference_timeout"
        and payload.get("request_id") == request_id
        and not error
    )
    return {
        "request_id": request_id,
        "http_status": status,
        "error_code": payload.get("code"),
        "response_request_id": payload.get("request_id"),
        "client_latency_ms": round(
            (time.perf_counter() - started) * 1000, 3
        ),
        "passed": passed,
        "error": error,
    }


def wait_for_gpu_jobs(
    *,
    base_url: str,
    server: subprocess.Popen,
    expected_jobs: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], bool]:
    deadline = time.perf_counter() + timeout_seconds
    last_metrics: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"Uvicorn exited with code {server.returncode}"
            )
        try:
            last_metrics = get_json(f"{base_url}/metrics", timeout=2.0)
        except (OSError, ValueError, urllib.error.URLError):
            time.sleep(0.1)
            continue
        completed = int(
            last_metrics.get("gpu_inference_succeeded_total", 0)
        ) + int(last_metrics.get("gpu_inference_failed_total", 0))
        in_flight = int(last_metrics.get("gpu_inference_in_flight", 0))
        if completed >= expected_jobs and in_flight == 0:
            return last_metrics, True
        time.sleep(0.1)
    return last_metrics, False


def build_acceptance_checks(
    *,
    requests: list[dict[str, Any]],
    metrics: dict[str, Any],
    jobs_completed: bool,
    missing_initial_process_ids: list[int],
    server_log: str,
) -> dict[str, dict[str, Any]]:
    expected_requests = len(requests)
    correct_timeout_responses = sum(
        bool(item.get("passed")) for item in requests
    )
    checks = {
        "timeout_responses": {
            "actual": correct_timeout_responses,
            "expected": expected_requests,
            "passed": correct_timeout_responses == expected_requests,
        },
        "requests_received_total": {
            "actual": int(metrics.get("requests_received_total", 0)),
            "expected": expected_requests,
            "passed": int(metrics.get("requests_received_total", 0))
            == expected_requests,
        },
        "requests_started_total": {
            "actual": int(metrics.get("requests_started_total", 0)),
            "expected": expected_requests,
            "passed": int(metrics.get("requests_started_total", 0))
            == expected_requests,
        },
        "requests_timeout_total": {
            "actual": int(metrics.get("requests_timeout_total", 0)),
            "expected": expected_requests,
            "passed": int(metrics.get("requests_timeout_total", 0))
            == expected_requests,
        },
        "gpu_inference_succeeded_total": {
            "actual": int(
                metrics.get("gpu_inference_succeeded_total", 0)
            ),
            "expected": expected_requests,
            "passed": int(
                metrics.get("gpu_inference_succeeded_total", 0)
            )
            == expected_requests,
        },
        "gpu_inference_failed_total": {
            "actual": int(metrics.get("gpu_inference_failed_total", 0)),
            "expected": 0,
            "passed": int(metrics.get("gpu_inference_failed_total", 0)) == 0,
        },
        "gpu_inference_in_flight_max": {
            "actual": int(metrics.get("gpu_inference_in_flight_max", 0)),
            "expected": 1,
            "passed": int(
                metrics.get("gpu_inference_in_flight_max", 0)
            )
            == 1,
        },
        "gpu_inference_in_flight_final": {
            "actual": int(metrics.get("gpu_inference_in_flight", 0)),
            "expected": 0,
            "passed": int(metrics.get("gpu_inference_in_flight", 0)) == 0,
        },
        "queue_wait_recorded": {
            "actual": float(metrics.get("queue_wait_seconds_total", 0.0)),
            "expected": "> 0",
            "passed": float(
                metrics.get("queue_wait_seconds_total", 0.0)
            )
            > 0.0,
        },
        "jobs_completed_after_http_timeout": {
            "actual": jobs_completed,
            "expected": True,
            "passed": jobs_completed,
        },
        "existing_compute_processes_remained": {
            "actual": missing_initial_process_ids,
            "expected": [],
            "passed": not missing_initial_process_ids,
        },
        "no_cuda_oom": {
            "actual": (
                "CUDA out of memory" in server_log
                or "OutOfMemoryError" in server_log
            ),
            "expected": False,
            "passed": (
                "CUDA out of memory" not in server_log
                and "OutOfMemoryError" not in server_log
            ),
        },
    }
    return checks


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LISA Timeout Serialization Verification",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Model: `{summary['model_version']}`",
        f"- Shared GPU: `{summary['shared_gpu']}`",
        f"- Ready time: `{summary['ready_time_ms']:.3f} ms`",
        f"- Inference timeout: `{summary['request_timeout_seconds']} s`",
        f"- Queue timeout: `{summary['queue_timeout_seconds']} s`",
        "",
        "## Requests",
        "",
        "| Request | HTTP | Error code | Client latency ms | Pass |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for item in summary["requests"]:
        lines.append(
            f"| {item['request_id']} | {item['http_status']} | "
            f"{item['error_code']} | {item['client_latency_ms']:.3f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Runtime metrics",
            "",
            f"- Requests received: `{summary['metrics'].get('requests_received_total', 0)}`",
            f"- Requests started: `{summary['metrics'].get('requests_started_total', 0)}`",
            f"- Requests timed out: `{summary['metrics'].get('requests_timeout_total', 0)}`",
            f"- GPU inference succeeded: `{summary['metrics'].get('gpu_inference_succeeded_total', 0)}`",
            f"- GPU inference failed: `{summary['metrics'].get('gpu_inference_failed_total', 0)}`",
            f"- Maximum GPU in flight: `{summary['metrics'].get('gpu_inference_in_flight_max', 0)}`",
            f"- Final GPU in flight: `{summary['metrics'].get('gpu_inference_in_flight', 0)}`",
            f"- Total queue wait: `{summary['metrics'].get('queue_wait_seconds_total', 0):.6f} s`",
            "",
            "## Acceptance",
            "",
        ]
    )
    for name, item in summary["acceptance"]["checks"].items():
        lines.append(
            f"- {'PASS' if item['passed'] else 'FAIL'} `{name}`: "
            f"actual `{item['actual']}`, expected `{item['expected']}`"
        )
    lines.extend(
        [
            "",
            "Overall: "
            f"`{'PASS' if summary['acceptance']['passed'] else 'FAIL'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that timed-out HTTP requests remain serialized on GPU."
        )
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", default="lisa13b-clean030-v1")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=0.1)
    parser.add_argument("--queue-timeout", type=float, default=5.0)
    parser.add_argument("--completion-timeout", type=float, default=30.0)
    parser.add_argument("--client-timeout", type=float, default=30.0)
    parser.add_argument("--max-queue-size", type=int, default=8)
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
        (image_path, "smoke image"),
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
    created_at = datetime.now(timezone.utc).isoformat()
    runtime_config = {
        "created_at": created_at,
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
    server: subprocess.Popen | None = None
    server_handle = None
    try:
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
        image_base64 = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

        barrier = threading.Barrier(3)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    send_expected_timeout,
                    url=f"{base_url}/v1/segment",
                    image_base64=image_base64,
                    prompt=args.prompt,
                    request_id=f"timeout-guard-{index}",
                    client_timeout_seconds=args.client_timeout,
                    barrier=barrier,
                )
                for index in (1, 2)
            ]
            barrier.wait(timeout=10)
            requests = [future.result() for future in futures]

        metrics, jobs_completed = wait_for_gpu_jobs(
            base_url=base_url,
            server=server,
            expected_jobs=2,
            timeout_seconds=args.completion_timeout,
        )
        after_requests_memory = query_gpu_memory(args.gpu_index)
        processes_after = existing_compute_processes(args.gpu_index)
        initial_process_ids = {item["pid"] for item in processes_before}
        final_process_ids = {item["pid"] for item in processes_after}
        missing_initial_process_ids = sorted(
            initial_process_ids - final_process_ids
        )

        server_handle.flush()
        server_log = server_log_path.read_text(
            encoding="utf-8", errors="replace"
        )
        checks = build_acceptance_checks(
            requests=requests,
            metrics=metrics,
            jobs_completed=jobs_completed,
            missing_initial_process_ids=missing_initial_process_ids,
            server_log=server_log,
        )
        summary = {
            **runtime_config,
            "ready_time_ms": round(ready_time_ms, 3),
            "gpu_memory_mib": {
                "baseline": baseline_memory,
                "loaded": loaded_memory,
                "after_requests": after_requests_memory,
            },
            "requests": requests,
            "metrics": metrics,
            "jobs_completed": jobs_completed,
            "existing_compute_processes_after": processes_after,
            "missing_initial_process_ids": missing_initial_process_ids,
            "acceptance": {
                "passed": all(item["passed"] for item in checks.values()),
                "checks": checks,
            },
        }
        (output_dir / "requests.json").write_text(
            json.dumps(requests, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_markdown(output_dir / "summary.md", summary)
        print(f"Saved timeout verification outputs to: {output_dir}")
        print(
            "Acceptance: "
            + ("PASS" if summary["acceptance"]["passed"] else "FAIL")
        )
        return 0 if summary["acceptance"]["passed"] else 2
    finally:
        stop_process(server)
        if server_handle is not None:
            server_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
