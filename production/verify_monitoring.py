from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from production.benchmark_api import (
    existing_compute_processes,
    portable_path,
    query_gpu_memory,
    query_gpu_total_memory,
    read_peak_gpu_memory,
    stop_process,
)
from production.verify_container_smoke import (
    container_exists,
    docker_inspect,
    ensure_port_available,
    find_log_issues,
    get_json,
    run_logged,
    sanitize_container_inspect,
    send_smoke_request,
    validate_docker_ignore_files,
    wait_for_container_ready,
)


PROMETHEUS_SAMPLE_RE = re.compile(
    r"^(lisa_[a-zA-Z0-9_:]+)"
    r'\{model_version="((?:\\.|[^"])*)"\}\s+'
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def get_authenticated_text(
    url: str,
    *,
    api_key: str,
    timeout: float = 3.0,
) -> str:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def get_authenticated_json(
    url: str,
    *,
    api_key: str,
    timeout: float = 3.0,
) -> dict[str, Any]:
    return json.loads(
        get_authenticated_text(
            url,
            api_key=api_key,
            timeout=timeout,
        )
    )


def parse_prometheus_samples(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid Prometheus sample line: {line!r}")
        metric_name, _, value = match.groups()
        if metric_name in samples:
            raise ValueError(f"duplicate Prometheus metric: {metric_name}")
        samples[metric_name] = float(value)
    if not samples:
        raise ValueError("Prometheus output contains no numeric samples")
    return samples


def send_unauthorized_request(
    *,
    url: str,
    image_base64: str,
    prompt: str,
    request_id: str,
    invalid_api_key: str,
    sensitive_values: tuple[str, ...],
    timeout_seconds: float,
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
            "X-API-Key": invalid_api_key,
        },
        method="POST",
    )
    started = time.perf_counter()
    status = 0
    raw = b""
    error = ""
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    payload: dict[str, Any] = {}
    if raw:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = error or f"invalid JSON response: {exc}"
    response_text = raw.decode("utf-8", errors="replace")
    sensitive_response = any(
        value and value in response_text for value in sensitive_values
    )
    passed = (
        status == 401
        and payload.get("request_id") == request_id
        and payload.get("code") == "unauthorized"
        and not sensitive_response
        and not error
    )
    return {
        "case": request_id,
        "http_status": status,
        "error_code": payload.get("code"),
        "client_latency_ms": round(
            (time.perf_counter() - started) * 1000,
            3,
        ),
        "response_bytes": len(raw),
        "sensitive_response": sensitive_response,
        "passed": passed,
        "error": error,
    }


def prometheus_json_mismatches(
    metrics: dict[str, Any],
    prometheus_samples: dict[str, float],
    names: tuple[str, ...],
) -> list[str]:
    mismatches: list[str] = []
    for name in names:
        prometheus_name = f"lisa_{name}"
        if name not in metrics or prometheus_name not in prometheus_samples:
            mismatches.append(name)
            continue
        expected = float(metrics[name])
        actual = prometheus_samples[prometheus_name]
        if abs(expected - actual) > 1e-6:
            mismatches.append(name)
    return mismatches


def build_acceptance_checks(
    *,
    build_exit_code: int,
    unit_test_exit_code: int,
    unit_test_count: int,
    minimum_unit_tests: int,
    initial_alerts: dict[str, Any],
    firing_alerts: dict[str, Any],
    recovered_alerts: dict[str, Any],
    valid_requests: list[dict[str, Any]],
    unauthorized_request: dict[str, Any],
    metrics: dict[str, Any],
    prometheus_mismatches: list[str],
    monitoring_leaks: list[str],
    configured_user: str,
    runtime_uid: int,
    runtime_gid: int,
    mounts_read_only: bool,
    gpu_memory_mib: dict[str, int],
    max_peak_memory_mib: int,
    min_remaining_memory_mib: int,
    max_post_stop_drift_mib: int,
    missing_initial_process_ids: list[int],
    log_issues: dict[str, Any],
    ready_after: dict[str, Any],
    stopped_exit_code: int,
) -> dict[str, dict[str, Any]]:
    valid_passed = sum(
        bool(item.get("passed")) for item in valid_requests
    )
    firing_codes = {
        str(item.get("code"))
        for item in firing_alerts.get("alerts", [])
    }
    numeric_window_names = (
        "http_request_latency_ms_p50",
        "http_request_latency_ms_p95",
        "http_request_latency_ms_p99",
        "queue_wait_ms_p50",
        "queue_wait_ms_p95",
        "queue_wait_ms_p99",
        "gpu_inference_ms_p50",
        "gpu_inference_ms_p95",
        "gpu_inference_ms_p99",
    )
    missing_or_invalid_windows = [
        name
        for name in numeric_window_names
        if not isinstance(metrics.get(name), (int, float))
        or float(metrics.get(name, 0)) < 0
    ]
    expected_exact_metrics = {
        "http_requests_total": 6,
        "http_responses_2xx_total": 5,
        "http_responses_4xx_total": 1,
        "http_responses_5xx_total": 0,
        "authentication_failed_total": 1,
        "requests_received_total": 5,
        "requests_started_total": 5,
        "requests_succeeded_total": 5,
        "gpu_inference_succeeded_total": 5,
        "gpu_inference_failed_total": 0,
        "http_request_latency_ms_window_samples": 6,
        "queue_wait_ms_window_samples": 5,
        "gpu_inference_ms_window_samples": 5,
        "gpu_inference_in_flight_max": 1,
        "gpu_inference_in_flight": 0,
        "queue_timeout_total": 0,
        "queue_rejected_total": 0,
        "queue_cancelled_total": 0,
        "requests_timeout_total": 0,
        "cuda_oom_total": 0,
        "unexpected_errors_total": 0,
    }
    metric_mismatches = {
        name: {
            "actual": metrics.get(name),
            "expected": expected,
        }
        for name, expected in expected_exact_metrics.items()
        if int(metrics.get(name, 0)) != expected
    }
    checks = {
        "docker_build_succeeded": {
            "actual": build_exit_code,
            "expected": 0,
            "passed": build_exit_code == 0,
        },
        "container_unit_tests": {
            "actual": {
                "exit_code": unit_test_exit_code,
                "tests": unit_test_count,
            },
            "expected": {
                "exit_code": 0,
                "tests": f">= {minimum_unit_tests}",
            },
            "passed": (
                unit_test_exit_code == 0
                and unit_test_count >= minimum_unit_tests
            ),
        },
        "initial_alert_status": {
            "actual": initial_alerts.get("status"),
            "expected": "ok",
            "passed": initial_alerts.get("status") == "ok",
        },
        "high_4xx_alert_fired": {
            "actual": {
                "status": firing_alerts.get("status"),
                "codes": sorted(firing_codes),
            },
            "expected": {
                "status": "firing",
                "code": "http_4xx_rate_high",
            },
            "passed": (
                firing_alerts.get("status") == "firing"
                and "http_4xx_rate_high" in firing_codes
            ),
        },
        "high_4xx_alert_recovered": {
            "actual": recovered_alerts.get("status"),
            "expected": "ok",
            "passed": recovered_alerts.get("status") == "ok",
        },
        "valid_requests_passed": {
            "actual": valid_passed,
            "expected": 5,
            "passed": valid_passed == 5,
        },
        "unauthorized_request_passed": {
            "actual": unauthorized_request.get("passed"),
            "expected": True,
            "passed": bool(unauthorized_request.get("passed")),
        },
        "exact_runtime_metrics": {
            "actual": metric_mismatches,
            "expected": {},
            "passed": not metric_mismatches,
        },
        "latency_windows_present": {
            "actual": missing_or_invalid_windows,
            "expected": [],
            "passed": not missing_or_invalid_windows,
        },
        "prometheus_matches_json": {
            "actual": prometheus_mismatches,
            "expected": [],
            "passed": not prometheus_mismatches,
        },
        "monitoring_outputs_no_sensitive_values": {
            "actual": monitoring_leaks,
            "expected": [],
            "passed": not monitoring_leaks,
        },
        "configured_non_root_user": {
            "actual": configured_user,
            "expected": "lisa",
            "passed": configured_user == "lisa",
        },
        "runtime_uid": {
            "actual": runtime_uid,
            "expected": 10001,
            "passed": runtime_uid == 10001,
        },
        "runtime_gid_non_root": {
            "actual": runtime_gid,
            "expected": "!= 0",
            "passed": runtime_gid != 0,
        },
        "model_and_clip_mounts_read_only": {
            "actual": mounts_read_only,
            "expected": True,
            "passed": mounts_read_only,
        },
        "peak_gpu_memory_mib": {
            "actual": gpu_memory_mib["peak"],
            "expected": f"<= {max_peak_memory_mib}",
            "passed": (
                gpu_memory_mib["peak"] <= max_peak_memory_mib
            ),
        },
        "remaining_gpu_memory_mib": {
            "actual": gpu_memory_mib["remaining_at_peak"],
            "expected": f">= {min_remaining_memory_mib}",
            "passed": (
                gpu_memory_mib["remaining_at_peak"]
                >= min_remaining_memory_mib
            ),
        },
        "post_stop_memory_drift_mib": {
            "actual": gpu_memory_mib["post_stop_drift"],
            "expected": f"abs <= {max_post_stop_drift_mib}",
            "passed": (
                abs(gpu_memory_mib["post_stop_drift"])
                <= max_post_stop_drift_mib
            ),
        },
        "existing_compute_processes_remained": {
            "actual": missing_initial_process_ids,
            "expected": [],
            "passed": not missing_initial_process_ids,
        },
        "container_logs_clean": {
            "actual": log_issues,
            "expected": {
                "cuda_oom": False,
                "traceback": False,
                "error_log": False,
                "sensitive_value_labels": [],
                "private_path_labels": [],
            },
            "passed": (
                not bool(log_issues.get("cuda_oom"))
                and not bool(log_issues.get("traceback"))
                and not bool(log_issues.get("error_log"))
                and not log_issues.get("sensitive_value_labels")
                and not log_issues.get("private_path_labels")
            ),
        },
        "service_ready_after_requests": {
            "actual": ready_after.get("status"),
            "expected": "ready",
            "passed": ready_after.get("status") == "ready",
        },
        "container_stopped_cleanly": {
            "actual": stopped_exit_code,
            "expected": 0,
            "passed": stopped_exit_code == 0,
        },
    }
    return checks


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics_final"]
    gpu = summary["gpu_memory_mib"]
    lines = [
        "# LISA Production Monitoring Verification",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Git commit: `{summary['repo_git_commit']}`",
        f"- Image: `{summary['image_tag']}`",
        f"- Model: `{summary['model_version']}`",
        f"- GPU: `{summary['gpu_name']}`",
        f"- Shared GPU: `{summary['shared_gpu']}`",
        f"- Build time: `{summary['build_time_seconds']:.3f} s`",
        f"- Container unit tests: `{summary['unit_test_count']}`",
        f"- Ready time: `{summary['startup']['ready_time_ms']:.3f} ms`",
        f"- Runtime UID/GID: `{summary['container']['runtime_uid']}` / "
        f"`{summary['container']['runtime_gid']}`",
        f"- Read-only model mounts: "
        f"`{summary['container']['mounts_read_only']}`",
        "",
        "## Alert lifecycle",
        "",
        "| Phase | Status | Alert codes |",
        "| --- | --- | --- |",
    ]
    for name, payload in (
        ("initial", summary["initial_alerts"]),
        ("after one 4xx", summary["firing_alerts"]),
        ("after recovery traffic", summary["recovered_alerts"]),
    ):
        codes = ", ".join(
            str(item.get("code"))
            for item in payload.get("alerts", [])
        ) or "-"
        lines.append(
            f"| {name} | {payload.get('status')} | {codes} |"
        )
    lines.extend(
        [
            "",
            "## Final request and runtime metrics",
            "",
            f"- HTTP requests: `{metrics.get('http_requests_total')}`",
            f"- HTTP 2xx / 4xx / 5xx: "
            f"`{metrics.get('http_responses_2xx_total')}` / "
            f"`{metrics.get('http_responses_4xx_total')}` / "
            f"`{metrics.get('http_responses_5xx_total')}`",
            f"- Authentication failures: "
            f"`{metrics.get('authentication_failed_total')}`",
            f"- GPU inference succeeded / failed: "
            f"`{metrics.get('gpu_inference_succeeded_total')}` / "
            f"`{metrics.get('gpu_inference_failed_total')}`",
            f"- HTTP P50 / P95 / P99: "
            f"`{metrics.get('http_request_latency_ms_p50')}` / "
            f"`{metrics.get('http_request_latency_ms_p95')}` / "
            f"`{metrics.get('http_request_latency_ms_p99')} ms`",
            f"- Queue wait P50 / P95 / P99: "
            f"`{metrics.get('queue_wait_ms_p50')}` / "
            f"`{metrics.get('queue_wait_ms_p95')}` / "
            f"`{metrics.get('queue_wait_ms_p99')} ms`",
            f"- GPU inference P50 / P95 / P99: "
            f"`{metrics.get('gpu_inference_ms_p50')}` / "
            f"`{metrics.get('gpu_inference_ms_p95')}` / "
            f"`{metrics.get('gpu_inference_ms_p99')} ms`",
            f"- GPU maximum / final in flight: "
            f"`{metrics.get('gpu_inference_in_flight_max')}` / "
            f"`{metrics.get('gpu_inference_in_flight')}`",
            "",
            "## GPU memory",
            "",
            f"- Baseline: `{gpu['baseline']} MiB`",
            f"- Peak: `{gpu['peak']} MiB`",
            f"- Remaining at peak: `{gpu['remaining_at_peak']} MiB`",
            f"- After stop: `{gpu['after_stop']} MiB`",
            f"- Post-stop drift: `{gpu['post_stop_drift']} MiB`",
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
        description="Verify LISA container monitoring and alert lifecycle."
    )
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--vision-model-dir", type=Path, required=True)
    parser.add_argument("--vision-snapshot", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", default="lisa13b-clean030-v1")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--startup-timeout", type=float, default=360.0)
    parser.add_argument("--client-timeout", type=float, default=150.0)
    parser.add_argument("--minimum-unit-tests", type=int, default=65)
    parser.add_argument("--max-peak-memory-mib", type=int, default=36864)
    parser.add_argument("--min-remaining-memory-mib", type=int, default=4096)
    parser.add_argument("--max-post-stop-drift-mib", type=int, default=500)
    parser.add_argument(
        "--require-existing-process-substring",
        default="VLLM::EngineCore",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dockerfile = args.dockerfile.resolve()
    model_artifact = args.model_artifact.resolve()
    vision_model_dir = args.vision_model_dir.resolve()
    vision_tower = (
        vision_model_dir / "snapshots" / args.vision_snapshot
    )
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()

    required_files = (
        dockerfile,
        model_artifact / "merged_hf" / "config.json",
        model_artifact / "SHA256SUMS",
        vision_tower / "config.json",
        vision_tower / "preprocessor_config.json",
        image_path,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")
    validate_docker_ignore_files(repo_root)
    if not (vision_model_dir / "blobs").is_dir():
        raise FileNotFoundError("CLIP cache blobs directory is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_port_available(args.host, args.port)
    if shutil.which("docker") is None:
        raise RuntimeError("docker command is not available")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi command is not available")
    if container_exists(args.container_name):
        raise RuntimeError(
            f"container name already exists: {args.container_name}"
        )

    processes_before = existing_compute_processes(args.gpu_index)
    required_process = args.require_existing_process_substring.lower()
    if not any(
        required_process in item["process_name"].lower()
        for item in processes_before
    ):
        raise RuntimeError(
            "required shared GPU process was not found: "
            f"{args.require_existing_process_substring}"
        )
    baseline_memory = query_gpu_memory(args.gpu_index)
    total_gpu_memory = query_gpu_total_memory(args.gpu_index)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_git_commit": commit,
        "dockerfile": portable_path(dockerfile, repo_root),
        "image_tag": args.image_tag,
        "container_name": args.container_name,
        "model_version": args.model_version,
        "model_artifact": portable_path(model_artifact, repo_root),
        "vision_tower": (
            "huggingface://openai/clip-vit-large-patch14@"
            f"{args.vision_snapshot}"
        ),
        "smoke_image": portable_path(image_path, repo_root),
        "prompt": args.prompt,
        "precision": "bf16",
        "gpu_index": args.gpu_index,
        "host": args.host,
        "port": args.port,
        "valid_requests": 5,
        "unauthorized_requests": 1,
        "alert_minimum_requests": 2,
        "alert_max_4xx_rate": 0.2,
        "existing_compute_processes_before": processes_before,
        "thresholds": {
            "minimum_unit_tests": args.minimum_unit_tests,
            "max_peak_memory_mib": args.max_peak_memory_mib,
            "min_remaining_memory_mib": args.min_remaining_memory_mib,
            "max_post_stop_drift_mib": args.max_post_stop_drift_mib,
        },
    }
    (output_dir / "runtime_config.json").write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    build_started = time.perf_counter()
    build_exit_code, _ = run_logged(
        [
            "docker",
            "build",
            "--file",
            str(dockerfile),
            "--tag",
            args.image_tag,
            "--build-arg",
            f"LISA_MODEL_VERSION={args.model_version}",
            "--build-arg",
            f"LISA_SOURCE_COMMIT={commit}",
            str(repo_root),
        ],
        output_dir / "build.log",
    )
    build_time_seconds = time.perf_counter() - build_started
    if build_exit_code != 0:
        raise RuntimeError(
            f"docker build failed with exit code {build_exit_code}"
        )

    unit_test_exit_code, unit_test_output = run_logged(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            args.image_tag,
            "-m",
            "unittest",
            "discover",
            "-s",
            "production/tests",
            "-v",
        ],
        output_dir / "unit_tests.log",
    )
    match = re.search(r"Ran\s+(\d+)\s+tests?", unit_test_output)
    unit_test_count = int(match.group(1)) if match else 0
    if unit_test_exit_code != 0:
        raise RuntimeError(
            "container unit tests failed with exit code "
            f"{unit_test_exit_code}"
        )

    api_key = secrets.token_urlsafe(32)
    invalid_api_key = secrets.token_urlsafe(32)
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    gpu_csv_path = output_dir / "gpu_metrics.csv"
    server_log_path = output_dir / "server.log"
    monitor: subprocess.Popen | None = None
    monitor_handle = None
    env_file_path: Path | None = None
    removed = False
    stopped_exit_code = -1
    inspected_sanitized: dict[str, Any] = {}
    configured_user = ""
    runtime_uid = -1
    runtime_gid = -1
    mounts_read_only = False
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
        container_model_path = f"/models/{args.model_version}"
        container_clip_root = "/models/clip-cache"
        container_clip_path = (
            f"{container_clip_root}/snapshots/{args.vision_snapshot}"
        )
        environment = {
            "LISA_MODEL_VERSION": args.model_version,
            "LISA_MODEL_PATH": f"{container_model_path}/merged_hf",
            "LISA_VISION_TOWER": container_clip_path,
            "LISA_PRECISION": "bf16",
            "LISA_LOAD_IN_8BIT": "false",
            "LISA_LOAD_IN_4BIT": "false",
            "LISA_GPU_INDEX": "0",
            "LISA_MASK_THRESHOLD": "0.0",
            "LISA_MAX_IMAGE_BYTES": "20971520",
            "LISA_MAX_IMAGE_PIXELS": "25000000",
            "LISA_MAX_PROMPT_CHARS": "1000",
            "LISA_MAX_REQUEST_BYTES": "31457280",
            "LISA_MAX_CONCURRENCY": "1",
            "LISA_MAX_QUEUE_SIZE": "8",
            "LISA_QUEUE_TIMEOUT_SECONDS": "30",
            "LISA_REQUEST_TIMEOUT_SECONDS": "120",
            "LISA_METRICS_WINDOW_SIZE": "100",
            "LISA_ALERT_MINIMUM_REQUESTS": "2",
            "LISA_ALERT_MAX_4XX_RATE": "0.20",
            "LISA_ALERT_MAX_5XX_RATE": "0.01",
            "LISA_ALERT_MAX_P95_LATENCY_MS": "2000",
            "LISA_ALERT_MAX_QUEUE_UTILIZATION": "0.80",
            "LISA_EAGER_LOAD": "true",
            "LISA_API_KEY": api_key,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        env_file_descriptor, env_file_name = tempfile.mkstemp(
            prefix="lisa-monitoring-",
            suffix=".env",
            text=True,
        )
        env_file_path = Path(env_file_name)
        os.fchmod(env_file_descriptor, 0o600)
        with os.fdopen(
            env_file_descriptor,
            "w",
            encoding="utf-8",
        ) as env_handle:
            for name, value in environment.items():
                env_handle.write(f"{name}={value}\n")
        docker_run = [
            "docker",
            "run",
            "--detach",
            "--name",
            args.container_name,
            "--gpus",
            f"device={args.gpu_index}",
            "--shm-size",
            "8g",
            "--restart",
            "no",
            "--env-file",
            str(env_file_path),
            "--publish",
            f"{args.host}:{args.port}:8000",
            "--mount",
            (
                f"type=bind,source={model_artifact},"
                f"target={container_model_path},readonly"
            ),
            "--mount",
            (
                f"type=bind,source={vision_model_dir},"
                f"target={container_clip_root},readonly"
            ),
            args.image_tag,
        ]
        try:
            container_id = subprocess.run(
                docker_run,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        finally:
            env_file_path.unlink(missing_ok=True)
            env_file_path = None
        print(f"Started container: {container_id[:12]}", flush=True)

        base_url = f"http://{args.host}:{args.port}"
        startup = wait_for_container_ready(
            container_name=args.container_name,
            base_url=base_url,
            timeout_seconds=args.startup_timeout,
        )
        inspected = docker_inspect(args.container_name)
        inspected_sanitized = sanitize_container_inspect(inspected)
        configured_user = str(
            inspected.get("Config", {}).get("User", "")
        )
        runtime_uid = int(
            subprocess.run(
                ["docker", "exec", args.container_name, "id", "-u"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        runtime_gid = int(
            subprocess.run(
                ["docker", "exec", args.container_name, "id", "-g"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        required_destinations = {
            container_model_path,
            container_clip_root,
        }
        matching_mounts = [
            item
            for item in inspected_sanitized["mounts"]
            if item["destination"] in required_destinations
        ]
        mounts_read_only = (
            {item["destination"] for item in matching_mounts}
            == required_destinations
            and all(item["read_only"] for item in matching_mounts)
        )
        (output_dir / "container_inspect.json").write_text(
            json.dumps(
                inspected_sanitized,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        initial_metrics = get_json(
            f"{base_url}/metrics",
            api_key=api_key,
        )
        initial_prometheus = get_authenticated_text(
            f"{base_url}/metrics/prometheus",
            api_key=api_key,
        )
        initial_alerts = get_authenticated_json(
            f"{base_url}/alerts",
            api_key=api_key,
        )
        (output_dir / "metrics-initial.json").write_text(
            json.dumps(initial_metrics, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "prometheus-initial.txt").write_text(
            initial_prometheus,
            encoding="utf-8",
        )
        (output_dir / "alerts-initial.json").write_text(
            json.dumps(initial_alerts, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

        valid_requests = [
            send_smoke_request(
                url=f"{base_url}/v1/segment",
                api_key=api_key,
                image_base64=image_base64,
                prompt=args.prompt,
                request_id="monitoring-valid-1",
                expected_model_version=args.model_version,
                timeout_seconds=args.client_timeout,
                output_dir=output_dir,
                cycle=1,
            )
        ]
        unauthorized_request = send_unauthorized_request(
            url=f"{base_url}/v1/segment",
            image_base64=image_base64,
            prompt=args.prompt,
            request_id="monitoring-auth-invalid",
            invalid_api_key=invalid_api_key,
            sensitive_values=(api_key, invalid_api_key),
            timeout_seconds=args.client_timeout,
        )
        firing_alerts = get_authenticated_json(
            f"{base_url}/alerts",
            api_key=api_key,
        )
        (output_dir / "alerts-firing.json").write_text(
            json.dumps(firing_alerts, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

        for request_index in range(2, 6):
            valid_requests.append(
                send_smoke_request(
                    url=f"{base_url}/v1/segment",
                    api_key=api_key,
                    image_base64=image_base64,
                    prompt=args.prompt,
                    request_id=f"monitoring-valid-{request_index}",
                    expected_model_version=args.model_version,
                    timeout_seconds=args.client_timeout,
                    output_dir=output_dir,
                    cycle=request_index,
                )
            )
        metrics_final = get_json(
            f"{base_url}/metrics",
            api_key=api_key,
        )
        prometheus_final = get_authenticated_text(
            f"{base_url}/metrics/prometheus",
            api_key=api_key,
        )
        recovered_alerts = get_authenticated_json(
            f"{base_url}/alerts",
            api_key=api_key,
        )
        ready_after = get_json(f"{base_url}/ready")
        prometheus_samples = parse_prometheus_samples(
            prometheus_final
        )
        compared_names = (
            "ready",
            "http_requests_total",
            "http_responses_2xx_total",
            "http_responses_4xx_total",
            "http_responses_5xx_total",
            "authentication_failed_total",
            "requests_received_total",
            "requests_succeeded_total",
            "gpu_inference_succeeded_total",
            "gpu_inference_in_flight_max",
            "http_request_latency_ms_p95",
            "queue_wait_ms_p95",
            "gpu_inference_ms_p95",
        )
        prometheus_mismatches = prometheus_json_mismatches(
            metrics_final,
            prometheus_samples,
            compared_names,
        )
        monitoring_text = "\n".join(
            (
                initial_prometheus,
                prometheus_final,
                json.dumps(initial_alerts, ensure_ascii=False),
                json.dumps(firing_alerts, ensure_ascii=False),
                json.dumps(recovered_alerts, ensure_ascii=False),
            )
        )
        sensitive_candidates = {
            "api_key": api_key,
            "invalid_api_key": invalid_api_key,
            "image_base64": image_base64,
            "prompt": args.prompt,
            "repo_root": str(repo_root),
            "model_path": str(model_artifact),
            "clip_path": str(vision_model_dir),
        }
        monitoring_leaks = [
            label
            for label, value in sensitive_candidates.items()
            if value and value in monitoring_text
        ]
        for name, value in (
            ("metrics-final.json", metrics_final),
            ("alerts-recovered.json", recovered_alerts),
            (
                "request-results.json",
                {
                    "valid": valid_requests,
                    "unauthorized": unauthorized_request,
                },
            ),
        ):
            (output_dir / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        (output_dir / "prometheus-final.txt").write_text(
            prometheus_final,
            encoding="utf-8",
        )

        subprocess.run(
            [
                "docker",
                "stop",
                "--timeout",
                "30",
                args.container_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stopped_inspect = docker_inspect(args.container_name)
        stopped_exit_code = int(
            stopped_inspect.get("State", {}).get("ExitCode", -1)
        )
        server_log = subprocess.run(
            ["docker", "logs", "--timestamps", args.container_name],
            check=True,
            capture_output=True,
            text=True,
        )
        server_log_text = server_log.stdout + server_log.stderr
        server_log_path.write_text(server_log_text, encoding="utf-8")
        time.sleep(1.0)
        stop_process(monitor, timeout=5)
        monitor = None
        monitor_handle.close()
        monitor_handle = None

        peak_memory = read_peak_gpu_memory(gpu_csv_path)
        if peak_memory <= 0:
            raise RuntimeError(
                "GPU monitor produced no valid memory samples"
            )
        after_stop_memory = query_gpu_memory(args.gpu_index)
        processes_after = existing_compute_processes(args.gpu_index)
        initial_process_ids = {item["pid"] for item in processes_before}
        final_process_ids = {item["pid"] for item in processes_after}
        missing_initial_process_ids = sorted(
            initial_process_ids - final_process_ids
        )
        log_issues = find_log_issues(
            server_log_text,
            sensitive_values=[api_key, invalid_api_key, image_base64],
            private_paths=[
                str(repo_root),
                str(model_artifact),
                str(vision_model_dir),
                str(image_path),
            ],
        )
        gpu_memory_mib = {
            "baseline": baseline_memory,
            "peak": peak_memory,
            "total": total_gpu_memory,
            "remaining_at_peak": total_gpu_memory - peak_memory,
            "after_stop": after_stop_memory,
            "post_stop_drift": after_stop_memory - baseline_memory,
        }
        checks = build_acceptance_checks(
            build_exit_code=build_exit_code,
            unit_test_exit_code=unit_test_exit_code,
            unit_test_count=unit_test_count,
            minimum_unit_tests=args.minimum_unit_tests,
            initial_alerts=initial_alerts,
            firing_alerts=firing_alerts,
            recovered_alerts=recovered_alerts,
            valid_requests=valid_requests,
            unauthorized_request=unauthorized_request,
            metrics=metrics_final,
            prometheus_mismatches=prometheus_mismatches,
            monitoring_leaks=monitoring_leaks,
            configured_user=configured_user,
            runtime_uid=runtime_uid,
            runtime_gid=runtime_gid,
            mounts_read_only=mounts_read_only,
            gpu_memory_mib=gpu_memory_mib,
            max_peak_memory_mib=args.max_peak_memory_mib,
            min_remaining_memory_mib=args.min_remaining_memory_mib,
            max_post_stop_drift_mib=args.max_post_stop_drift_mib,
            missing_initial_process_ids=missing_initial_process_ids,
            log_issues=log_issues,
            ready_after=ready_after,
            stopped_exit_code=stopped_exit_code,
        )
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
            "shared_gpu": bool(processes_before),
            "build_time_seconds": round(build_time_seconds, 3),
            "unit_test_exit_code": unit_test_exit_code,
            "unit_test_count": unit_test_count,
            "startup": startup,
            "initial_alerts": initial_alerts,
            "firing_alerts": firing_alerts,
            "recovered_alerts": recovered_alerts,
            "valid_requests": valid_requests,
            "unauthorized_request": unauthorized_request,
            "metrics_initial": initial_metrics,
            "metrics_final": metrics_final,
            "prometheus_compared_names": list(compared_names),
            "prometheus_mismatches": prometheus_mismatches,
            "monitoring_leaks": monitoring_leaks,
            "ready_after": ready_after,
            "container": {
                "configured_user": configured_user,
                "runtime_uid": runtime_uid,
                "runtime_gid": runtime_gid,
                "mounts_read_only": mounts_read_only,
            },
            "gpu_memory_mib": gpu_memory_mib,
            "existing_compute_processes_after": processes_after,
            "missing_initial_process_ids": missing_initial_process_ids,
            "log_issues": log_issues,
            "stopped_exit_code": stopped_exit_code,
            "acceptance": {
                "passed": all(
                    item["passed"] for item in checks.values()
                ),
                "checks": checks,
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_markdown(output_dir / "summary.md", summary)
        subprocess.run(
            ["docker", "rm", args.container_name],
            check=True,
            capture_output=True,
            text=True,
        )
        removed = True
        print(f"Saved monitoring outputs to: {output_dir}")
        print(
            "Acceptance: "
            + ("PASS" if summary["acceptance"]["passed"] else "FAIL")
        )
        return 0 if summary["acceptance"]["passed"] else 2
    finally:
        stop_process(monitor, timeout=5)
        if monitor_handle is not None:
            monitor_handle.close()
        if env_file_path is not None:
            env_file_path.unlink(missing_ok=True)
        if container_exists(args.container_name) and not removed:
            fallback_logs = subprocess.run(
                ["docker", "logs", "--timestamps", args.container_name],
                capture_output=True,
                text=True,
            )
            server_log_path.write_text(
                fallback_logs.stdout + fallback_logs.stderr,
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "docker",
                    "stop",
                    "--timeout",
                    "30",
                    args.container_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["docker", "rm", args.container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    raise SystemExit(main())
