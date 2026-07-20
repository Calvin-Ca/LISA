from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from production.benchmark_api import (
    ensure_port_available,
    existing_compute_processes,
    get_json,
    portable_path,
    query_gpu_memory,
    stop_process,
    validate_segment_response,
    wait_until_ready,
)
from production.image_io import PNG_SIGNATURE


API_KEY_SENTINEL = "api-key-SENSITIVE-DO-NOT-LOG"
INVALID_API_KEY_SENTINEL = "invalid-api-key-SENSITIVE-DO-NOT-LOG"
PROMPT_SENTINEL = "prompt-SENSITIVE-DO-NOT-LOG"
IMAGE_SENTINEL = "image-SENSITIVE-DO-NOT-LOG"
PRIVATE_PATH_SENTINEL = "/private/sensitive/path/DO-NOT-LOG"


def get_authenticated_json(
    url: str,
    *,
    api_key: str,
    timeout: float = 2.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"X-API-Key": api_key},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def send_case(
    *,
    url: str,
    image_base64: str,
    prompt: str,
    request_id: str,
    api_key: str,
    expected_status: int,
    expected_code: str | None,
    require_mask: bool,
    client_timeout_seconds: float,
    sensitive_values: tuple[str, ...],
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
            "X-API-Key": api_key,
        },
        method="POST",
    )
    started = time.perf_counter()
    status = 0
    raw = b""
    payload: dict[str, Any] = {}
    error = ""
    try:
        with urllib.request.urlopen(
            request, timeout=client_timeout_seconds
        ) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if raw:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = f"invalid JSON response: {exc}"

    response_text = raw.decode("utf-8", errors="replace")
    sensitive_response = any(
        value and value in response_text for value in sensitive_values
    )
    response_valid = True
    if status == 200:
        try:
            validate_segment_response(payload, request_id)
        except (TypeError, ValueError, KeyError) as exc:
            response_valid = False
            error = f"invalid segment response: {exc}"
        if require_mask and int(payload.get("mask_count", 0)) < 1:
            response_valid = False
            error = "segment response did not contain a mask"
    else:
        response_valid = (
            payload.get("request_id") == request_id
            and payload.get("code") == expected_code
        )

    passed = (
        status == expected_status
        and response_valid
        and not sensitive_response
        and not error
    )
    return {
        "case": request_id,
        "http_status": status,
        "error_code": payload.get("code"),
        "mask_count": int(payload.get("mask_count", 0)),
        "client_latency_ms": round(
            (time.perf_counter() - started) * 1000, 3
        ),
        "response_bytes": len(raw),
        "sensitive_response": sensitive_response,
        "passed": passed,
        "error": error,
    }


def wait_for_metrics(
    *,
    base_url: str,
    api_key: str,
    server: subprocess.Popen,
    predicate: Callable[[dict[str, Any]], bool],
    timeout_seconds: float,
    description: str,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_seconds
    last: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"Uvicorn exited with code {server.returncode}"
            )
        try:
            last = get_authenticated_json(
                f"{base_url}/metrics",
                api_key=api_key,
            )
        except (OSError, ValueError, urllib.error.URLError):
            time.sleep(0.01)
            continue
        if predicate(last):
            return last
        time.sleep(0.01)
    raise TimeoutError(
        f"timed out waiting for {description}; last metrics: {last}"
    )


def build_acceptance_checks(
    *,
    cases: list[dict[str, Any]],
    metrics_after_invalid: dict[str, Any],
    metrics_final: dict[str, Any],
    final_ready: dict[str, Any],
    leaked_log_labels: list[str],
    missing_initial_process_ids: list[int],
    server_log: str,
) -> dict[str, dict[str, Any]]:
    passed_cases = sum(bool(item.get("passed")) for item in cases)
    sensitive_responses = [
        item["case"] for item in cases if item.get("sensitive_response")
    ]
    checks = {
        "all_api_cases_passed": {
            "actual": passed_cases,
            "expected": len(cases),
            "passed": passed_cases == len(cases),
        },
        "invalid_requests_did_not_enter_runtime": {
            "actual": int(
                metrics_after_invalid.get("requests_received_total", 0)
            ),
            "expected": 0,
            "passed": int(
                metrics_after_invalid.get("requests_received_total", 0)
            )
            == 0,
        },
        "runtime_requests_received": {
            "actual": int(metrics_final.get("requests_received_total", 0)),
            "expected": 5,
            "passed": int(metrics_final.get("requests_received_total", 0))
            == 5,
        },
        "gpu_requests_started": {
            "actual": int(metrics_final.get("requests_started_total", 0)),
            "expected": 3,
            "passed": int(metrics_final.get("requests_started_total", 0))
            == 3,
        },
        "requests_succeeded": {
            "actual": int(metrics_final.get("requests_succeeded_total", 0)),
            "expected": 3,
            "passed": int(metrics_final.get("requests_succeeded_total", 0))
            == 3,
        },
        "queue_timeout_total": {
            "actual": int(metrics_final.get("queue_timeout_total", 0)),
            "expected": 1,
            "passed": int(metrics_final.get("queue_timeout_total", 0)) == 1,
        },
        "queue_rejected_total": {
            "actual": int(metrics_final.get("queue_rejected_total", 0)),
            "expected": 1,
            "passed": int(metrics_final.get("queue_rejected_total", 0)) == 1,
        },
        "queue_cancelled_total": {
            "actual": int(metrics_final.get("queue_cancelled_total", 0)),
            "expected": 1,
            "passed": int(metrics_final.get("queue_cancelled_total", 0)) == 1,
        },
        "gpu_inference_succeeded_total": {
            "actual": int(
                metrics_final.get("gpu_inference_succeeded_total", 0)
            ),
            "expected": 3,
            "passed": int(
                metrics_final.get("gpu_inference_succeeded_total", 0)
            )
            == 3,
        },
        "gpu_inference_failed_total": {
            "actual": int(
                metrics_final.get("gpu_inference_failed_total", 0)
            ),
            "expected": 0,
            "passed": int(
                metrics_final.get("gpu_inference_failed_total", 0)
            )
            == 0,
        },
        "gpu_inference_in_flight_max": {
            "actual": int(
                metrics_final.get("gpu_inference_in_flight_max", 0)
            ),
            "expected": 1,
            "passed": int(
                metrics_final.get("gpu_inference_in_flight_max", 0)
            )
            == 1,
        },
        "gpu_inference_in_flight_final": {
            "actual": int(
                metrics_final.get("gpu_inference_in_flight", 0)
            ),
            "expected": 0,
            "passed": int(
                metrics_final.get("gpu_inference_in_flight", 0)
            )
            == 0,
        },
        "masks_returned": {
            "actual": int(metrics_final.get("masks_returned_total", 0)),
            "expected": ">= 3",
            "passed": int(metrics_final.get("masks_returned_total", 0)) >= 3,
        },
        "cuda_oom_total": {
            "actual": int(metrics_final.get("cuda_oom_total", 0)),
            "expected": 0,
            "passed": int(metrics_final.get("cuda_oom_total", 0)) == 0,
        },
        "unexpected_errors_total": {
            "actual": int(metrics_final.get("unexpected_errors_total", 0)),
            "expected": 0,
            "passed": int(metrics_final.get("unexpected_errors_total", 0))
            == 0,
        },
        "service_ready_after_errors": {
            "actual": final_ready.get("status"),
            "expected": "ready",
            "passed": final_ready.get("status") == "ready",
        },
        "sensitive_values_absent_from_responses": {
            "actual": sensitive_responses,
            "expected": [],
            "passed": not sensitive_responses,
        },
        "sensitive_values_absent_from_log": {
            "actual": leaked_log_labels,
            "expected": [],
            "passed": not leaked_log_labels,
        },
        "existing_compute_processes_remained": {
            "actual": missing_initial_process_ids,
            "expected": [],
            "passed": not missing_initial_process_ids,
        },
        "no_cuda_oom_in_log": {
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
        "# LISA Production API Robustness Verification",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Model: `{summary['model_version']}`",
        f"- Shared GPU: `{summary['shared_gpu']}`",
        f"- Ready time: `{summary['ready_time_ms']:.3f} ms`",
        f"- Queue size: `{summary['max_queue_size']}`",
        f"- Queue timeout: `{summary['queue_timeout_seconds']} s`",
        "",
        "## Cases",
        "",
        "| Case | HTTP | Code | Masks | Latency ms | Pass |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for item in summary["cases"]:
        lines.append(
            f"| {item['case']} | {item['http_status']} | "
            f"{item['error_code'] or '-'} | {item['mask_count']} | "
            f"{item['client_latency_ms']:.3f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Final runtime metrics",
            "",
            f"- Requests received: `{summary['metrics_final'].get('requests_received_total', 0)}`",
            f"- GPU requests started: `{summary['metrics_final'].get('requests_started_total', 0)}`",
            f"- Requests succeeded: `{summary['metrics_final'].get('requests_succeeded_total', 0)}`",
            f"- Queue timeout: `{summary['metrics_final'].get('queue_timeout_total', 0)}`",
            f"- Queue rejected: `{summary['metrics_final'].get('queue_rejected_total', 0)}`",
            f"- Queue cancelled: `{summary['metrics_final'].get('queue_cancelled_total', 0)}`",
            f"- Maximum GPU in flight: `{summary['metrics_final'].get('gpu_inference_in_flight_max', 0)}`",
            f"- CUDA OOM: `{summary['metrics_final'].get('cuda_oom_total', 0)}`",
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


def png_header(width: int, height: int) -> bytes:
    return (
        PNG_SIGNATURE
        + b"\x00\x00\x00\x0d"
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify production API input and queue robustness."
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
    parser.add_argument("--client-timeout", type=float, default=150.0)
    parser.add_argument("--queue-timeout", type=float, default=0.15)
    parser.add_argument("--max-queue-size", type=int, default=1)
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
        "request_timeout_seconds": 120,
        "max_image_bytes": 20 * 1024 * 1024,
        "max_image_pixels": 25_000_000,
        "max_prompt_chars": 1000,
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
            "LISA_MAX_IMAGE_BYTES": str(20 * 1024 * 1024),
            "LISA_MAX_IMAGE_PIXELS": "25000000",
            "LISA_MAX_PROMPT_CHARS": "1000",
            "LISA_MAX_CONCURRENCY": "1",
            "LISA_MAX_QUEUE_SIZE": str(args.max_queue_size),
            "LISA_QUEUE_TIMEOUT_SECONDS": str(args.queue_timeout),
            "LISA_REQUEST_TIMEOUT_SECONDS": "120",
            "LISA_EAGER_LOAD": "true",
            "LISA_API_KEY": API_KEY_SENTINEL,
        }
    )
    server_log_path = output_dir / "server.log"
    server: subprocess.Popen | None = None
    server_handle = None
    cases: list[dict[str, Any]] = []
    sensitive_values = (
        API_KEY_SENTINEL,
        INVALID_API_KEY_SENTINEL,
        PROMPT_SENTINEL,
        IMAGE_SENTINEL,
        PRIVATE_PATH_SENTINEL,
    )
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
        segment_url = f"{base_url}/v1/segment"
        ready_time_ms = wait_until_ready(
            base_url, server, args.startup_timeout
        )
        loaded_memory = query_gpu_memory(args.gpu_index)
        jpeg_base64 = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

        def run_case(
            case: str,
            *,
            image_base64: str = jpeg_base64,
            prompt: str = args.prompt,
            api_key: str = API_KEY_SENTINEL,
            expected_status: int,
            expected_code: str | None,
            require_mask: bool = False,
        ) -> dict[str, Any]:
            result = send_case(
                url=segment_url,
                image_base64=image_base64,
                prompt=prompt,
                request_id=case,
                api_key=api_key,
                expected_status=expected_status,
                expected_code=expected_code,
                require_mask=require_mask,
                client_timeout_seconds=args.client_timeout,
                sensitive_values=sensitive_values,
            )
            cases.append(result)
            print(
                f"{case}: {'PASS' if result['passed'] else 'FAIL'}, "
                f"HTTP {result['http_status']}, "
                f"{result['client_latency_ms']} ms",
                flush=True,
            )
            return result

        invalid_cases = [
            {
                "case": "auth-invalid",
                "api_key": INVALID_API_KEY_SENTINEL,
                "expected_status": 401,
                "expected_code": "unauthorized",
            },
            {
                "case": "prompt-empty",
                "prompt": "",
                "expected_status": 422,
                "expected_code": "validation_error",
            },
            {
                "case": "prompt-blank",
                "prompt": "   ",
                "expected_status": 400,
                "expected_code": "invalid_request",
            },
            {
                "case": "prompt-too-long",
                "prompt": (
                    PROMPT_SENTINEL
                    + PRIVATE_PATH_SENTINEL
                    + "x" * 1001
                ),
                "expected_status": 400,
                "expected_code": "invalid_request",
            },
            {
                "case": "image-invalid-base64",
                "image_base64": IMAGE_SENTINEL,
                "expected_status": 400,
                "expected_code": "invalid_request",
            },
            {
                "case": "image-gif",
                "image_base64": base64.b64encode(
                    b"GIF89a\x01\x00\x01\x00"
                ).decode("ascii"),
                "expected_status": 400,
                "expected_code": "invalid_request",
            },
            {
                "case": "image-webp",
                "image_base64": base64.b64encode(
                    b"RIFF\x08\x00\x00\x00WEBP"
                ).decode("ascii"),
                "expected_status": 400,
                "expected_code": "invalid_request",
            },
            {
                "case": "image-corrupt-png",
                "image_base64": base64.b64encode(
                    PNG_SIGNATURE
                ).decode("ascii"),
                "expected_status": 400,
                "expected_code": "invalid_request",
            },
            {
                "case": "image-corrupt-jpeg",
                "image_base64": base64.b64encode(
                    b"\xff\xd8\xff\xe0\x00\x02"
                ).decode("ascii"),
                "expected_status": 400,
                "expected_code": "invalid_request",
            },
            {
                "case": "image-oversized-header",
                "image_base64": base64.b64encode(
                    png_header(10_000, 10_000)
                ).decode("ascii"),
                "expected_status": 400,
                "expected_code": "invalid_request",
            },
        ]
        for item in invalid_cases:
            run_case(**item)

        metrics_after_invalid = get_authenticated_json(
            f"{base_url}/metrics",
            api_key=API_KEY_SENTINEL,
        )

        import cv2
        import numpy as np

        decoded = cv2.imdecode(
            np.frombuffer(image_path.read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if decoded is None:
            raise RuntimeError("fixed JPEG could not be decoded for PNG case")
        png_ok, png_bytes = cv2.imencode(".png", decoded)
        if not png_ok:
            raise RuntimeError("failed to encode fixed image as PNG")
        png_base64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")

        run_case(
            "valid-jpeg",
            expected_status=200,
            expected_code=None,
            require_mask=True,
        )
        run_case(
            "valid-png",
            image_base64=png_base64,
            expected_status=200,
            expected_code=None,
            require_mask=True,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            primary_future = executor.submit(
                send_case,
                url=segment_url,
                image_base64=jpeg_base64,
                prompt=args.prompt,
                request_id="queue-primary",
                api_key=API_KEY_SENTINEL,
                expected_status=200,
                expected_code=None,
                require_mask=True,
                client_timeout_seconds=args.client_timeout,
                sensitive_values=sensitive_values,
            )
            wait_for_metrics(
                base_url=base_url,
                api_key=API_KEY_SENTINEL,
                server=server,
                predicate=lambda item: int(
                    item.get("gpu_inference_in_flight", 0)
                )
                == 1,
                timeout_seconds=10,
                description="primary GPU inference",
            )
            queued_future = executor.submit(
                send_case,
                url=segment_url,
                image_base64=jpeg_base64,
                prompt=args.prompt,
                request_id="queue-timeout",
                api_key=API_KEY_SENTINEL,
                expected_status=504,
                expected_code="inference_queue_timeout",
                require_mask=False,
                client_timeout_seconds=args.client_timeout,
                sensitive_values=sensitive_values,
            )
            wait_for_metrics(
                base_url=base_url,
                api_key=API_KEY_SENTINEL,
                server=server,
                predicate=lambda item: int(item.get("queue_size", 0)) == 1,
                timeout_seconds=5,
                description="one queued request",
            )
            overflow = send_case(
                url=segment_url,
                image_base64=jpeg_base64,
                prompt=args.prompt,
                request_id="queue-full",
                api_key=API_KEY_SENTINEL,
                expected_status=503,
                expected_code="inference_queue_full",
                require_mask=False,
                client_timeout_seconds=args.client_timeout,
                sensitive_values=sensitive_values,
            )
            primary = primary_future.result()
            queued = queued_future.result()

        for result in (primary, queued, overflow):
            cases.append(result)
            print(
                f"{result['case']}: "
                f"{'PASS' if result['passed'] else 'FAIL'}, "
                f"HTTP {result['http_status']}, "
                f"{result['client_latency_ms']} ms",
                flush=True,
            )

        metrics_final = wait_for_metrics(
            base_url=base_url,
            api_key=API_KEY_SENTINEL,
            server=server,
            predicate=lambda item: (
                int(item.get("gpu_inference_in_flight", 0)) == 0
                and int(item.get("queue_size", 0)) == 0
                and int(item.get("queue_cancelled_total", 0)) == 1
            ),
            timeout_seconds=30,
            description="queue drain and GPU idle",
        )
        final_ready = get_json(f"{base_url}/ready", timeout=2.0)
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
        sentinel_labels = {
            "api_key": API_KEY_SENTINEL,
            "invalid_api_key": INVALID_API_KEY_SENTINEL,
            "prompt": PROMPT_SENTINEL,
            "image": IMAGE_SENTINEL,
            "private_path": PRIVATE_PATH_SENTINEL,
        }
        leaked_log_labels = [
            label
            for label, value in sentinel_labels.items()
            if value in server_log
        ]
        checks = build_acceptance_checks(
            cases=cases,
            metrics_after_invalid=metrics_after_invalid,
            metrics_final=metrics_final,
            final_ready=final_ready,
            leaked_log_labels=leaked_log_labels,
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
            "cases": cases,
            "metrics_after_invalid": metrics_after_invalid,
            "metrics_final": metrics_final,
            "final_ready": final_ready,
            "existing_compute_processes_after": processes_after,
            "missing_initial_process_ids": missing_initial_process_ids,
            "leaked_log_labels": leaked_log_labels,
            "acceptance": {
                "passed": all(item["passed"] for item in checks.values()),
                "checks": checks,
            },
        }
        (output_dir / "cases.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics_after_invalid.json").write_text(
            json.dumps(
                metrics_after_invalid, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics_final.json").write_text(
            json.dumps(metrics_final, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_markdown(output_dir / "summary.md", summary)
        print(f"Saved API robustness outputs to: {output_dir}")
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
