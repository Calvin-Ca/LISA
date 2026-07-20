from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from production.benchmark_api import (
    existing_compute_processes,
    portable_path,
    query_gpu_memory,
)
from production.verify_container_smoke import (
    container_exists,
    docker_inspect,
    ensure_port_available,
    find_log_issues,
    get_json,
    run_logged,
    sanitize_container_inspect,
    validate_docker_ignore_files,
)


def decode_chunked_body(body: bytes) -> bytes:
    decoded = bytearray()
    position = 0
    while True:
        line_end = body.find(b"\r\n", position)
        if line_end < 0:
            raise ValueError("chunked response is missing a size line")
        size_text = body[position:line_end].split(b";", 1)[0]
        try:
            size = int(size_text, 16)
        except ValueError as exc:
            raise ValueError("chunked response has invalid size") from exc
        position = line_end + 2
        if size == 0:
            return bytes(decoded)
        chunk_end = position + size
        if chunk_end + 2 > len(body):
            raise ValueError("chunked response body is truncated")
        decoded.extend(body[position:chunk_end])
        if body[chunk_end:chunk_end + 2] != b"\r\n":
            raise ValueError("chunked response has invalid terminator")
        position = chunk_end + 2


def parse_http_response(raw: bytes) -> dict[str, Any]:
    header_end = raw.find(b"\r\n\r\n")
    if header_end < 0:
        raise ValueError("HTTP response is missing header terminator")
    header_lines = raw[:header_end].split(b"\r\n")
    status_parts = header_lines[0].decode("latin-1").split(" ", 2)
    if len(status_parts) < 2:
        raise ValueError("HTTP response has invalid status line")
    status = int(status_parts[1])
    headers: dict[str, str] = {}
    for line in header_lines[1:]:
        name, separator, value = line.partition(b":")
        if not separator:
            raise ValueError("HTTP response has invalid header")
        headers[name.decode("latin-1").lower()] = (
            value.decode("latin-1").strip()
        )
    body = raw[header_end + 4:]
    if "chunked" in headers.get("transfer-encoding", "").lower():
        body = decode_chunked_body(body)
    elif "content-length" in headers:
        expected = int(headers["content-length"])
        if len(body) < expected:
            raise ValueError("HTTP response body is truncated")
        body = body[:expected]
    payload: dict[str, Any] = {}
    if body:
        payload = json.loads(body.decode("utf-8"))
    return {
        "status": status,
        "headers": headers,
        "payload": payload,
        "body_bytes": len(body),
    }


def raw_http_exchange(
    *,
    host: str,
    port: int,
    request: bytes,
    timeout_seconds: float,
) -> dict[str, Any]:
    chunks: list[bytes] = []
    started = time.perf_counter()
    with socket.create_connection(
        (host, port),
        timeout=timeout_seconds,
    ) as connection:
        connection.settimeout(timeout_seconds)
        connection.sendall(request)
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    result = parse_http_response(b"".join(chunks))
    result["client_latency_ms"] = round(
        (time.perf_counter() - started) * 1000,
        3,
    )
    return result


def build_request(
    *,
    host: str,
    port: int,
    request_id: str,
    api_key: str,
    body: bytes,
    content_length: int | None = None,
    chunked: bool = False,
) -> bytes:
    headers = [
        "POST /v1/segment HTTP/1.1",
        f"Host: {host}:{port}",
        "Content-Type: application/json",
        f"X-Request-ID: {request_id}",
        f"X-API-Key: {api_key}",
        "Connection: close",
    ]
    if chunked:
        headers.append("Transfer-Encoding: chunked")
        encoded_body = (
            f"{len(body):X}\r\n".encode("ascii")
            + body
            + b"\r\n0\r\n\r\n"
        )
    else:
        length = len(body) if content_length is None else content_length
        headers.append(f"Content-Length: {length}")
        encoded_body = body
    return "\r\n".join(headers).encode("latin-1") + b"\r\n\r\n" + encoded_body


def validate_case(
    *,
    result: dict[str, Any],
    request_id: str,
    expected_status: int,
    expected_code: str,
    expected_model_version: str,
    sensitive_values: tuple[str, ...],
) -> dict[str, Any]:
    payload = result.get("payload", {})
    headers = result.get("headers", {})
    response_text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    leaked = [
        f"sensitive_value_{index + 1}"
        for index, value in enumerate(sensitive_values)
        if value and value in response_text
    ]
    checks = {
        "status": result.get("status") == expected_status,
        "code": payload.get("code") == expected_code,
        "request_id": payload.get("request_id") == request_id,
        "request_id_header": (
            headers.get("x-request-id") == request_id
        ),
        "model_version_header": (
            headers.get("x-model-version") == expected_model_version
        ),
        "no_sensitive_response": not leaked,
    }
    return {
        "request_id": request_id,
        "http_status": result.get("status"),
        "error_code": payload.get("code"),
        "response_bytes": result.get("body_bytes"),
        "client_latency_ms": result.get("client_latency_ms"),
        "response_connection": headers.get("connection"),
        "sensitive_response_labels": leaked,
        "checks": checks,
        "passed": all(checks.values()),
    }


def wait_for_container_healthy(
    *,
    container_name: str,
    base_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    last_health = "unknown"
    last_payload: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        inspected = docker_inspect(container_name)
        state = inspected.get("State", {})
        if not state.get("Running"):
            raise RuntimeError(
                "container exited before becoming healthy: "
                f"exit_code={state.get('ExitCode')}"
            )
        last_health = str(
            state.get("Health", {}).get("Status", "unavailable")
        )
        try:
            last_payload = get_json(f"{base_url}/health")
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            time.sleep(0.1)
            continue
        if (
            last_health == "healthy"
            and last_payload.get("status") == "ok"
        ):
            return {
                "healthy_time_ms": round(
                    (time.perf_counter() - started) * 1000,
                    3,
                ),
                "docker_health": last_health,
                "health": last_payload,
            }
        time.sleep(0.1)
    raise TimeoutError(
        "container did not become healthy; "
        f"docker_health={last_health}, health={last_payload}"
    )


def build_acceptance_checks(
    *,
    build_exit_code: int,
    unit_test_exit_code: int,
    unit_test_count: int,
    minimum_unit_tests: int,
    cases: list[dict[str, Any]],
    metrics: dict[str, Any],
    configured_user: str,
    runtime_uid: int,
    runtime_gid: int,
    memory_drift_mib: int,
    max_memory_drift_mib: int,
    missing_initial_process_ids: list[int],
    log_issues: dict[str, Any],
    health_after: dict[str, Any],
    stopped_exit_code: int,
) -> dict[str, dict[str, Any]]:
    passed_cases = sum(bool(item.get("passed")) for item in cases)
    expected_metrics = {
        "http_requests_total": 3,
        "http_responses_4xx_total": 3,
        "http_responses_5xx_total": 0,
        "request_body_too_large_total": 2,
        "request_validation_failed_total": 1,
        "http_request_latency_ms_window_samples": 3,
        "requests_received_total": 0,
        "requests_started_total": 0,
        "requests_succeeded_total": 0,
        "gpu_inference_succeeded_total": 0,
        "gpu_inference_failed_total": 0,
        "gpu_inference_in_flight_max": 0,
        "model_loads_total": 0,
        "cuda_oom_total": 0,
        "unexpected_errors_total": 0,
    }
    metric_mismatches = {
        name: {
            "actual": metrics.get(name),
            "expected": expected,
        }
        for name, expected in expected_metrics.items()
        if int(metrics.get(name, 0)) != expected
    }
    return {
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
        "all_http_cases_passed": {
            "actual": passed_cases,
            "expected": len(cases),
            "passed": passed_cases == len(cases),
        },
        "exact_request_limit_metrics": {
            "actual": metric_mismatches,
            "expected": {},
            "passed": not metric_mismatches,
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
        "gpu_memory_drift_mib": {
            "actual": memory_drift_mib,
            "expected": f"abs <= {max_memory_drift_mib}",
            "passed": abs(memory_drift_mib) <= max_memory_drift_mib,
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
        "service_health_after_requests": {
            "actual": health_after.get("status"),
            "expected": "ok",
            "passed": health_after.get("status") == "ok",
        },
        "container_stopped_cleanly": {
            "actual": stopped_exit_code,
            "expected": 0,
            "passed": stopped_exit_code == 0,
        },
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LISA Request Body Limit Verification",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Git commit: `{summary['repo_git_commit']}`",
        f"- Image: `{summary['image_tag']}`",
        f"- Model version: `{summary['model_version']}`",
        f"- Test limit: `{summary['max_request_bytes']} bytes`",
        f"- Container unit tests: `{summary['unit_test_count']}`",
        f"- Container healthy: `{summary['startup']['docker_health']}`",
        "",
        "## HTTP cases",
        "",
        "| Case | HTTP | Code | Latency ms | Pass |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| {case['request_id']} | {case['http_status']} | "
            f"{case['error_code']} | {case['client_latency_ms']:.3f} | "
            f"{'PASS' if case['passed'] else 'FAIL'} |"
        )
    metrics = summary["metrics"]
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- HTTP requests / 4xx / 5xx: "
            f"`{metrics.get('http_requests_total')}` / "
            f"`{metrics.get('http_responses_4xx_total')}` / "
            f"`{metrics.get('http_responses_5xx_total')}`",
            f"- Request body too large: "
            f"`{metrics.get('request_body_too_large_total')}`",
            f"- Request validation failed: "
            f"`{metrics.get('request_validation_failed_total')}`",
            f"- Runtime requests received: "
            f"`{metrics.get('requests_received_total')}`",
            f"- Model loads: `{metrics.get('model_loads_total')}`",
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
        description="Verify LISA request body limits in a real container."
    )
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-version",
        default="lisa13b-request-limit-v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--max-request-bytes", type=int, default=1024)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--client-timeout", type=float, default=10.0)
    parser.add_argument("--minimum-unit-tests", type=int, default=65)
    parser.add_argument("--max-memory-drift-mib", type=int, default=500)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--require-existing-process-substring",
        default="VLLM::EngineCore",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_request_bytes < 8:
        raise ValueError("max request bytes must be at least 8")
    repo_root = Path(__file__).resolve().parents[1]
    dockerfile = args.dockerfile.resolve()
    output_dir = args.output_dir.resolve()
    if not dockerfile.is_file():
        raise FileNotFoundError(f"missing Dockerfile: {dockerfile}")
    validate_docker_ignore_files(repo_root)
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
    memory_before = query_gpu_memory(args.gpu_index)
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
        "host": args.host,
        "port": args.port,
        "max_request_bytes": args.max_request_bytes,
        "eager_load": False,
        "gpu_attached": False,
        "existing_compute_processes_before": processes_before,
        "thresholds": {
            "minimum_unit_tests": args.minimum_unit_tests,
            "max_memory_drift_mib": args.max_memory_drift_mib,
        },
    }
    (output_dir / "runtime_config.json").write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    build_exit_code, build_output = run_logged(
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
    oversized_sentinel = "oversized-body-SENSITIVE-DO-NOT-LOG"
    env_file_path: Path | None = None
    removed = False
    stopped_exit_code = -1
    server_log_path = output_dir / "server.log"
    try:
        environment = {
            "LISA_MODEL_VERSION": args.model_version,
            "LISA_MODEL_PATH": "/models/not-mounted/merged_hf",
            "LISA_VISION_TOWER": "/models/not-mounted/clip",
            "LISA_PRECISION": "bf16",
            "LISA_LOAD_IN_8BIT": "false",
            "LISA_LOAD_IN_4BIT": "false",
            "LISA_GPU_INDEX": "0",
            "LISA_MAX_REQUEST_BYTES": str(args.max_request_bytes),
            "LISA_MAX_CONCURRENCY": "1",
            "LISA_MAX_QUEUE_SIZE": "8",
            "LISA_QUEUE_TIMEOUT_SECONDS": "30",
            "LISA_REQUEST_TIMEOUT_SECONDS": "120",
            "LISA_EAGER_LOAD": "false",
            "LISA_API_KEY": api_key,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        env_file_descriptor, env_file_name = tempfile.mkstemp(
            prefix="lisa-request-limit-",
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
            "--restart",
            "no",
            "--env-file",
            str(env_file_path),
            "--publish",
            f"{args.host}:{args.port}:8000",
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
        startup = wait_for_container_healthy(
            container_name=args.container_name,
            base_url=base_url,
            timeout_seconds=args.startup_timeout,
        )
        inspected = docker_inspect(args.container_name)
        inspected_sanitized = sanitize_container_inspect(inspected)
        (output_dir / "container_inspect.json").write_text(
            json.dumps(
                inspected_sanitized,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
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

        declared_id = "request-limit-content-length"
        declared_raw = build_request(
            host=args.host,
            port=args.port,
            request_id=declared_id,
            api_key=api_key,
            body=b"",
            content_length=args.max_request_bytes + 1,
        )
        declared_result = raw_http_exchange(
            host=args.host,
            port=args.port,
            request=declared_raw,
            timeout_seconds=args.client_timeout,
        )
        declared_case = validate_case(
            result=declared_result,
            request_id=declared_id,
            expected_status=413,
            expected_code="request_too_large",
            expected_model_version=args.model_version,
            sensitive_values=(api_key, oversized_sentinel),
        )

        chunked_id = "request-limit-chunked"
        chunked_body = (
            oversized_sentinel.encode("utf-8")
            + b"x" * (args.max_request_bytes + 1)
        )
        chunked_raw = build_request(
            host=args.host,
            port=args.port,
            request_id=chunked_id,
            api_key=api_key,
            body=chunked_body,
            chunked=True,
        )
        chunked_result = raw_http_exchange(
            host=args.host,
            port=args.port,
            request=chunked_raw,
            timeout_seconds=args.client_timeout,
        )
        chunked_case = validate_case(
            result=chunked_result,
            request_id=chunked_id,
            expected_status=413,
            expected_code="request_too_large",
            expected_model_version=args.model_version,
            sensitive_values=(api_key, oversized_sentinel),
        )

        validation_id = "request-limit-small-validation"
        validation_raw = build_request(
            host=args.host,
            port=args.port,
            request_id=validation_id,
            api_key=api_key,
            body=b"{}",
        )
        validation_result = raw_http_exchange(
            host=args.host,
            port=args.port,
            request=validation_raw,
            timeout_seconds=args.client_timeout,
        )
        validation_case = validate_case(
            result=validation_result,
            request_id=validation_id,
            expected_status=422,
            expected_code="validation_error",
            expected_model_version=args.model_version,
            sensitive_values=(api_key, oversized_sentinel),
        )
        cases = [declared_case, chunked_case, validation_case]
        metrics = get_json(
            f"{base_url}/metrics",
            api_key=api_key,
        )
        health_after = get_json(f"{base_url}/health")
        (output_dir / "request-results.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
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
        memory_after = query_gpu_memory(args.gpu_index)
        processes_after = existing_compute_processes(args.gpu_index)
        initial_process_ids = {item["pid"] for item in processes_before}
        final_process_ids = {item["pid"] for item in processes_after}
        missing_initial_process_ids = sorted(
            initial_process_ids - final_process_ids
        )
        log_issues = find_log_issues(
            server_log_text,
            sensitive_values=[
                api_key,
                oversized_sentinel,
            ],
            private_paths=[str(repo_root)],
        )
        memory_drift_mib = memory_after - memory_before
        checks = build_acceptance_checks(
            build_exit_code=build_exit_code,
            unit_test_exit_code=unit_test_exit_code,
            unit_test_count=unit_test_count,
            minimum_unit_tests=args.minimum_unit_tests,
            cases=cases,
            metrics=metrics,
            configured_user=configured_user,
            runtime_uid=runtime_uid,
            runtime_gid=runtime_gid,
            memory_drift_mib=memory_drift_mib,
            max_memory_drift_mib=args.max_memory_drift_mib,
            missing_initial_process_ids=missing_initial_process_ids,
            log_issues=log_issues,
            health_after=health_after,
            stopped_exit_code=stopped_exit_code,
        )
        summary = {
            **runtime_config,
            "build_output_bytes": len(build_output.encode("utf-8")),
            "unit_test_exit_code": unit_test_exit_code,
            "unit_test_count": unit_test_count,
            "startup": startup,
            "container": {
                "configured_user": configured_user,
                "runtime_uid": runtime_uid,
                "runtime_gid": runtime_gid,
                "stopped_exit_code": stopped_exit_code,
            },
            "cases": cases,
            "metrics": metrics,
            "health_after": health_after,
            "gpu_memory_mib": {
                "before": memory_before,
                "after": memory_after,
                "drift": memory_drift_mib,
            },
            "existing_compute_processes_after": processes_after,
            "missing_initial_process_ids": missing_initial_process_ids,
            "log_issues": log_issues,
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
        print(f"Saved request-limit outputs to: {output_dir}")
        print(
            "Acceptance: "
            + ("PASS" if summary["acceptance"]["passed"] else "FAIL")
        )
        return 0 if summary["acceptance"]["passed"] else 2
    finally:
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
