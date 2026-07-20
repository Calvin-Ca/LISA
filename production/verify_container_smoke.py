from __future__ import annotations

import argparse
import base64
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
    validate_segment_response,
)


def ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        try:
            handle.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"{host}:{port} is already in use") from exc


def validate_docker_ignore_files(repo_root: Path) -> None:
    ignore_paths = (
        repo_root / ".dockerignore",
        repo_root / "production" / "Dockerfile.dockerignore",
    )
    required_rules = {
        "**",
        "!production/",
        "!production/**",
        "!model/",
        "!model/**",
        "!utils/",
        "!utils/**",
        "**/.env",
        "**/.env.*",
    }
    forbidden_rules = {
        "!dataset/**",
        "!artifacts/**",
        "!runs/**",
        "!exp/**",
    }
    rule_sets: list[set[str]] = []
    for ignore_path in ignore_paths:
        if not ignore_path.is_file():
            raise FileNotFoundError(
                f"missing Docker ignore file: {ignore_path}"
            )
        rules = {
            line.strip()
            for line in ignore_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.startswith("#")
        }
        missing_rules = required_rules - rules
        if missing_rules:
            raise ValueError(
                f"{ignore_path} is missing required rules: "
                f"{sorted(missing_rules)}"
            )
        unexpected_rules = forbidden_rules & rules
        if unexpected_rules:
            raise ValueError(
                f"{ignore_path} exposes forbidden trees: "
                f"{sorted(unexpected_rules)}"
            )
        rule_sets.append(rules)
    if rule_sets[0] != rule_sets[1]:
        raise ValueError(
            "root and Dockerfile-specific ignore rules are not aligned"
        )


def run_logged(command: list[str], path: Path) -> tuple[int, str]:
    lines: list[str] = []
    with path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            lines.append(line)
            print(line, end="", flush=True)
        return process.wait(), "".join(lines)


def docker_inspect(container_name: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "container", "inspect", container_name],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError("docker inspect returned an unexpected payload")
    return payload[0]


def container_exists(container_name: str) -> bool:
    result = subprocess.run(
        ["docker", "container", "inspect", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def get_json(
    url: str,
    *,
    api_key: str | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    headers = {"X-API-Key": api_key} if api_key else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_container_ready(
    *,
    container_name: str,
    base_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    ready_ms: float | None = None
    healthy_ms: float | None = None
    last_ready: dict[str, Any] = {}
    last_health = "unknown"
    deadline = started + timeout_seconds
    while time.perf_counter() < deadline:
        inspected = docker_inspect(container_name)
        state = inspected.get("State", {})
        if not state.get("Running"):
            raise RuntimeError(
                "container exited before it became ready: "
                f"status={state.get('Status')}, "
                f"exit_code={state.get('ExitCode')}"
            )
        health = state.get("Health", {})
        last_health = str(health.get("Status", "unavailable"))
        if last_health == "healthy" and healthy_ms is None:
            healthy_ms = (time.perf_counter() - started) * 1000
        try:
            last_ready = get_json(f"{base_url}/ready", timeout=2.0)
            if last_ready.get("status") == "ready" and ready_ms is None:
                ready_ms = (time.perf_counter() - started) * 1000
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            pass
        if ready_ms is not None and healthy_ms is not None:
            return {
                "ready_time_ms": round(ready_ms, 3),
                "healthy_time_ms": round(healthy_ms, 3),
                "ready": last_ready,
                "health_status": last_health,
            }
        time.sleep(0.25)
    raise TimeoutError(
        f"container did not become ready and healthy after "
        f"{timeout_seconds}s; ready={last_ready}, health={last_health}"
    )


def send_smoke_request(
    *,
    url: str,
    api_key: str,
    image_base64: str,
    prompt: str,
    request_id: str,
    expected_model_version: str,
    timeout_seconds: float,
    output_dir: Path,
    cycle: int,
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
            request, timeout=timeout_seconds
        ) as response:
            status = response.status
            raw = response.read()
        payload = json.loads(raw.decode("utf-8"))
        validate_segment_response(payload, request_id)
        if int(payload.get("mask_count", 0)) < 1:
            raise ValueError("segment response did not contain a mask")
        if payload.get("model_version") != expected_model_version:
            raise ValueError("segment response model_version does not match")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
        error = raw.decode("utf-8", errors="replace")[:500]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = (time.perf_counter() - started) * 1000
    mask_files: list[str] = []
    sanitized = dict(payload)
    sanitized_masks: list[dict[str, Any]] = []
    for index, mask in enumerate(payload.get("masks", [])):
        name = f"smoke-cycle-{cycle}-mask-{index}.png"
        try:
            decoded = base64.b64decode(mask["data"], validate=True)
            (output_dir / name).write_bytes(decoded)
            mask_files.append(name)
            sanitized_masks.append(
                {
                    **{key: value for key, value in mask.items() if key != "data"},
                    "data": f"<saved:{name}>",
                    "bytes": len(decoded),
                }
            )
        except Exception as exc:
            error = error or f"failed to save mask {index}: {exc}"
    if payload:
        sanitized["masks"] = sanitized_masks
        (output_dir / f"smoke-cycle-{cycle}-response.json").write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    passed = (
        status == 200
        and not error
        and int(payload.get("mask_count", 0)) >= 1
        and len(mask_files) == int(payload.get("mask_count", 0))
    )
    return {
        "cycle": cycle,
        "request_id": request_id,
        "http_status": status,
        "model_version": payload.get("model_version"),
        "width": int(payload.get("width", 0)),
        "height": int(payload.get("height", 0)),
        "mask_count": int(payload.get("mask_count", 0)),
        "mask_files": mask_files,
        "client_latency_ms": round(latency_ms, 3),
        "server_latency_ms": float(payload.get("latency_ms", 0.0)),
        "response_bytes": len(raw),
        "passed": passed,
        "error": error,
    }


def sanitize_container_inspect(
    inspected: dict[str, Any],
) -> dict[str, Any]:
    state = inspected.get("State", {})
    config = inspected.get("Config", {})
    mounts = inspected.get("Mounts", [])
    return {
        "id": str(inspected.get("Id", ""))[:12],
        "name": str(inspected.get("Name", "")).removeprefix("/"),
        "image": config.get("Image"),
        "configured_user": config.get("User"),
        "state": {
            "status": state.get("Status"),
            "running": state.get("Running"),
            "exit_code": state.get("ExitCode"),
            "health_status": state.get("Health", {}).get("Status"),
        },
        "mounts": [
            {
                "destination": item.get("Destination"),
                "type": item.get("Type"),
                "read_only": not bool(item.get("RW")),
            }
            for item in mounts
        ],
    }


def find_log_issues(
    log_text: str,
    *,
    sensitive_values: list[str],
    private_paths: list[str],
) -> dict[str, list[str] | bool]:
    lowered = log_text.lower()
    sensitive_labels = [
        f"sensitive_value_{index + 1}"
        for index, value in enumerate(sensitive_values)
        if value and value in log_text
    ]
    private_path_labels = [
        f"private_path_{index + 1}"
        for index, value in enumerate(private_paths)
        if value and value in log_text
    ]
    return {
        "cuda_oom": "cuda out of memory" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "error_log": "error:" in lowered,
        "sensitive_value_labels": sensitive_labels,
        "private_path_labels": private_path_labels,
    }


def metric_sum(cycles: list[dict[str, Any]], name: str) -> float:
    return sum(
        float(cycle["metrics_after"].get(name, 0))
        - float(cycle["metrics_before"].get(name, 0))
        for cycle in cycles
    )


def build_acceptance_checks(
    *,
    build_succeeded: bool,
    unit_test_exit_code: int,
    unit_test_count: int,
    minimum_unit_tests: int,
    configured_user: str,
    runtime_uid: int,
    runtime_gid: int,
    mounts_read_only: bool,
    forbidden_image_paths_present: list[str],
    cycles: list[dict[str, Any]],
    gpu_memory_mib: dict[str, int],
    max_peak_memory_mib: int,
    min_remaining_memory_mib: int,
    max_post_stop_drift_mib: int,
    missing_initial_process_ids: list[int],
    log_issues: dict[str, Any],
    stopped_exit_code: int,
) -> dict[str, dict[str, Any]]:
    smoke_passed = sum(
        bool(cycle["smoke"].get("passed")) for cycle in cycles
    )
    healthy_cycles = sum(
        cycle["startup"].get("health_status") == "healthy"
        for cycle in cycles
    )
    ready_cycles = sum(
        cycle["startup"].get("ready", {}).get("status") == "ready"
        for cycle in cycles
    )
    model_loads = sum(
        int(cycle["metrics_after"].get("model_loads_total", 0))
        for cycle in cycles
    )
    checks = {
        "docker_build_succeeded": {
            "actual": build_succeeded,
            "expected": True,
            "passed": build_succeeded,
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
        "runtime_image_excludes_secrets_and_unrelated_files": {
            "actual": forbidden_image_paths_present,
            "expected": [],
            "passed": not forbidden_image_paths_present,
        },
        "both_cycles_healthy": {
            "actual": healthy_cycles,
            "expected": 2,
            "passed": healthy_cycles == 2,
        },
        "both_cycles_ready": {
            "actual": ready_cycles,
            "expected": 2,
            "passed": ready_cycles == 2,
        },
        "service_ready_after_both_smokes": {
            "actual": [
                cycle["ready_after"].get("status") for cycle in cycles
            ],
            "expected": ["ready", "ready"],
            "passed": all(
                cycle["ready_after"].get("status") == "ready"
                for cycle in cycles
            ),
        },
        "both_smoke_requests_passed": {
            "actual": smoke_passed,
            "expected": 2,
            "passed": smoke_passed == 2,
        },
        "model_loaded_once_per_cycle": {
            "actual": model_loads,
            "expected": 2,
            "passed": model_loads == 2,
        },
        "runtime_requests_received": {
            "actual": int(metric_sum(cycles, "requests_received_total")),
            "expected": 2,
            "passed": int(
                metric_sum(cycles, "requests_received_total")
            ) == 2,
        },
        "requests_succeeded": {
            "actual": int(metric_sum(cycles, "requests_succeeded_total")),
            "expected": 2,
            "passed": int(
                metric_sum(cycles, "requests_succeeded_total")
            ) == 2,
        },
        "gpu_inference_succeeded": {
            "actual": int(metric_sum(
                cycles, "gpu_inference_succeeded_total"
            )),
            "expected": 2,
            "passed": int(metric_sum(
                cycles, "gpu_inference_succeeded_total"
            )) == 2,
        },
        "gpu_inference_failed": {
            "actual": int(metric_sum(
                cycles, "gpu_inference_failed_total"
            )),
            "expected": 0,
            "passed": int(metric_sum(
                cycles, "gpu_inference_failed_total"
            )) == 0,
        },
        "gpu_inference_in_flight_max": {
            "actual": max(
                int(cycle["metrics_after"].get(
                    "gpu_inference_in_flight_max", 0
                ))
                for cycle in cycles
            ),
            "expected": 1,
            "passed": max(
                int(cycle["metrics_after"].get(
                    "gpu_inference_in_flight_max", 0
                ))
                for cycle in cycles
            ) == 1,
        },
        "gpu_inference_in_flight_final": {
            "actual": max(
                int(cycle["metrics_after"].get(
                    "gpu_inference_in_flight", 0
                ))
                for cycle in cycles
            ),
            "expected": 0,
            "passed": all(
                int(cycle["metrics_after"].get(
                    "gpu_inference_in_flight", 0
                )) == 0
                for cycle in cycles
            ),
        },
        "queue_and_request_timeouts": {
            "actual": int(
                metric_sum(cycles, "queue_timeout_total")
                + metric_sum(cycles, "requests_timeout_total")
            ),
            "expected": 0,
            "passed": (
                metric_sum(cycles, "queue_timeout_total")
                + metric_sum(cycles, "requests_timeout_total")
            ) == 0,
        },
        "queue_rejections_and_cancellations": {
            "actual": int(
                metric_sum(cycles, "queue_rejected_total")
                + metric_sum(cycles, "queue_cancelled_total")
            ),
            "expected": 0,
            "passed": (
                metric_sum(cycles, "queue_rejected_total")
                + metric_sum(cycles, "queue_cancelled_total")
            ) == 0,
        },
        "cuda_oom_total": {
            "actual": int(metric_sum(cycles, "cuda_oom_total")),
            "expected": 0,
            "passed": int(metric_sum(cycles, "cuda_oom_total")) == 0,
        },
        "unexpected_errors_total": {
            "actual": int(metric_sum(cycles, "unexpected_errors_total")),
            "expected": 0,
            "passed": int(
                metric_sum(cycles, "unexpected_errors_total")
            ) == 0,
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
        "container_logs_no_cuda_oom": {
            "actual": bool(log_issues["cuda_oom"]),
            "expected": False,
            "passed": not bool(log_issues["cuda_oom"]),
        },
        "container_logs_no_traceback_or_error": {
            "actual": {
                "traceback": bool(log_issues["traceback"]),
                "error_log": bool(log_issues["error_log"]),
            },
            "expected": {
                "traceback": False,
                "error_log": False,
            },
            "passed": (
                not bool(log_issues["traceback"])
                and not bool(log_issues["error_log"])
            ),
        },
        "container_logs_no_sensitive_values": {
            "actual": log_issues["sensitive_value_labels"],
            "expected": [],
            "passed": not log_issues["sensitive_value_labels"],
        },
        "container_logs_no_private_host_paths": {
            "actual": log_issues["private_path_labels"],
            "expected": [],
            "passed": not log_issues["private_path_labels"],
        },
        "container_stopped_cleanly": {
            "actual": stopped_exit_code,
            "expected": 0,
            "passed": stopped_exit_code == 0,
        },
    }
    return checks


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LISA Production Container Smoke Verification",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Git commit: `{summary['repo_git_commit']}`",
        f"- Image: `{summary['image_tag']}`",
        f"- Model: `{summary['model_version']}`",
        f"- GPU: `{summary['gpu_name']}`",
        f"- Shared GPU: `{summary['shared_gpu']}`",
        f"- Build time: `{summary['build_time_seconds']:.3f} s`",
        f"- Container unit tests: `{summary['unit_test_count']}`",
        "",
        "## Startup and smoke cycles",
        "",
        "| Cycle | Ready ms | Healthy ms | HTTP | Masks | Client ms | "
        "Server ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cycle in summary["cycles"]:
        lines.append(
            f"| {cycle['cycle']} | "
            f"{cycle['startup']['ready_time_ms']:.3f} | "
            f"{cycle['startup']['healthy_time_ms']:.3f} | "
            f"{cycle['smoke']['http_status']} | "
            f"{cycle['smoke']['mask_count']} | "
            f"{cycle['smoke']['client_latency_ms']:.3f} | "
            f"{cycle['smoke']['server_latency_ms']:.3f} |"
        )
    gpu = summary["gpu_memory_mib"]
    lines.extend(
        [
            "",
            "## GPU memory",
            "",
            f"- Baseline: `{gpu['baseline']} MiB`",
            f"- Peak: `{gpu['peak']} MiB`",
            f"- Remaining at peak: `{gpu['remaining_at_peak']} MiB`",
            f"- After stop: `{gpu['after_stop']} MiB`",
            f"- Post-stop drift: `{gpu['post_stop_drift']} MiB`",
            "",
            "## Container",
            "",
            f"- Configured user: `{summary['container']['configured_user']}`",
            f"- Runtime UID/GID: `{summary['container']['runtime_uid']}` / "
            f"`{summary['container']['runtime_gid']}`",
            f"- Read-only model mounts: `{summary['container']['mounts_read_only']}`",
            f"- Exit code after stop: `{summary['container']['stopped_exit_code']}`",
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
        description="Build and verify the production LISA GPU container."
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
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--startup-timeout", type=float, default=360.0)
    parser.add_argument("--client-timeout", type=float, default=150.0)
    parser.add_argument("--minimum-unit-tests", type=int, default=38)
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
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    vision_tower = (
        vision_model_dir / "snapshots" / args.vision_snapshot
    )

    required_files = [
        (dockerfile, "Dockerfile"),
        (model_artifact / "merged_hf" / "config.json", "model config"),
        (model_artifact / "SHA256SUMS", "model SHA256SUMS"),
        (vision_tower / "config.json", "CLIP config"),
        (
            vision_tower / "preprocessor_config.json",
            "CLIP preprocessor config",
        ),
        (image_path, "smoke image"),
    ]
    for path, description in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"missing {description}: {path}")
    validate_docker_ignore_files(repo_root)
    if not (vision_model_dir / "blobs").is_dir():
        raise FileNotFoundError(
            f"CLIP cache blobs directory is missing: {vision_model_dir / 'blobs'}"
        )
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
        "max_concurrency": 1,
        "max_queue_size": 8,
        "queue_timeout_seconds": 30,
        "request_timeout_seconds": 120,
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

    image_inspect = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", args.image_tag],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]
    configured_user = str(
        image_inspect.get("Config", {}).get("User", "")
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
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    (output_dir / "smoke-request-metadata.json").write_text(
        json.dumps(
            {
                "image": portable_path(image_path, repo_root),
                "prompt": args.prompt,
                "request_ids": [
                    "container-smoke-cycle-1",
                    "container-smoke-cycle-2",
                ],
                "image_base64_saved": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    gpu_csv_path = output_dir / "gpu_metrics.csv"
    server_log_path = output_dir / "server.log"
    monitor: subprocess.Popen | None = None
    monitor_handle = None
    cycles: list[dict[str, Any]] = []
    runtime_uid = -1
    runtime_gid = -1
    mounts_read_only = False
    forbidden_image_paths_present: list[str] = []
    stopped_exit_code = -1
    inspected_sanitized: dict[str, Any] = {}
    removed = False
    env_file_path: Path | None = None
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
            "LISA_MAX_CONCURRENCY": "1",
            "LISA_MAX_QUEUE_SIZE": "8",
            "LISA_QUEUE_TIMEOUT_SECONDS": "30",
            "LISA_REQUEST_TIMEOUT_SECONDS": "120",
            "LISA_EAGER_LOAD": "true",
            "LISA_API_KEY": api_key,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        env_file_descriptor, env_file_name = tempfile.mkstemp(
            prefix="lisa-container-smoke-",
            suffix=".env",
            text=True,
        )
        env_file_path = Path(env_file_name)
        os.fchmod(env_file_descriptor, 0o600)
        with os.fdopen(
            env_file_descriptor, "w", encoding="utf-8"
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
        ]
        docker_run.append(args.image_tag)
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
        forbidden_paths = [
            "/app/production/.env",
            "/app/.env",
            "/app/AGENTS.md",
            "/app/MS.md",
            "/app/dataset",
            "/app/exp",
            "/app/artifacts",
            "/app/runs",
        ]
        forbidden_output = subprocess.run(
            [
                "docker",
                "exec",
                args.container_name,
                "python3",
                "-c",
                (
                    "import json,pathlib;"
                    f"paths={forbidden_paths!r};"
                    "print(json.dumps([p for p in paths "
                    "if pathlib.Path(p).exists()]))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        forbidden_image_paths_present = json.loads(forbidden_output)
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

        for cycle_index in (1, 2):
            if cycle_index == 2:
                subprocess.run(
                    [
                        "docker",
                        "restart",
                        "--timeout",
                        "30",
                        args.container_name,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                startup = wait_for_container_ready(
                    container_name=args.container_name,
                    base_url=base_url,
                    timeout_seconds=args.startup_timeout,
                )
            metrics_before = get_json(
                f"{base_url}/metrics",
                api_key=api_key,
            )
            smoke = send_smoke_request(
                url=f"{base_url}/v1/segment",
                api_key=api_key,
                image_base64=image_base64,
                prompt=args.prompt,
                request_id=f"container-smoke-cycle-{cycle_index}",
                expected_model_version=args.model_version,
                timeout_seconds=args.client_timeout,
                output_dir=output_dir,
                cycle=cycle_index,
            )
            metrics_after = get_json(
                f"{base_url}/metrics",
                api_key=api_key,
            )
            ready_after = get_json(f"{base_url}/ready")
            cycle = {
                "cycle": cycle_index,
                "startup": startup,
                "smoke": smoke,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "ready_after": ready_after,
            }
            cycles.append(cycle)
            (output_dir / f"cycle-{cycle_index}-metrics.json").write_text(
                json.dumps(cycle, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"Container smoke cycle {cycle_index}: "
                f"{'PASS' if smoke['passed'] else 'FAIL'}, "
                f"{smoke['client_latency_ms']} ms",
                flush=True,
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
        inspected_sanitized["state_after_stop"] = {
            "status": stopped_inspect.get("State", {}).get("Status"),
            "running": stopped_inspect.get("State", {}).get("Running"),
            "exit_code": stopped_exit_code,
        }
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
            sensitive_values=[api_key],
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
            build_succeeded=build_exit_code == 0,
            unit_test_exit_code=unit_test_exit_code,
            unit_test_count=unit_test_count,
            minimum_unit_tests=args.minimum_unit_tests,
            configured_user=configured_user,
            runtime_uid=runtime_uid,
            runtime_gid=runtime_gid,
            mounts_read_only=mounts_read_only,
            forbidden_image_paths_present=forbidden_image_paths_present,
            cycles=cycles,
            gpu_memory_mib=gpu_memory_mib,
            max_peak_memory_mib=args.max_peak_memory_mib,
            min_remaining_memory_mib=args.min_remaining_memory_mib,
            max_post_stop_drift_mib=args.max_post_stop_drift_mib,
            missing_initial_process_ids=missing_initial_process_ids,
            log_issues=log_issues,
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
        (output_dir / "container_inspect.json").write_text(
            json.dumps(
                inspected_sanitized,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summary = {
            **runtime_config,
            "gpu_name": gpu_name,
            "shared_gpu": bool(processes_before),
            "build_time_seconds": round(build_time_seconds, 3),
            "image": {
                "id": str(image_inspect.get("Id", "")).removeprefix(
                    "sha256:"
                ),
                "size_bytes": int(image_inspect.get("Size", 0)),
                "configured_user": configured_user,
            },
            "unit_test_exit_code": unit_test_exit_code,
            "unit_test_count": unit_test_count,
            "container": {
                "configured_user": configured_user,
                "runtime_uid": runtime_uid,
                "runtime_gid": runtime_gid,
                "mounts_read_only": mounts_read_only,
                "forbidden_image_paths_present": (
                    forbidden_image_paths_present
                ),
                "stopped_exit_code": stopped_exit_code,
            },
            "cycles": cycles,
            "gpu_memory_mib": gpu_memory_mib,
            "existing_compute_processes_after": processes_after,
            "missing_initial_process_ids": missing_initial_process_ids,
            "log_issues": log_issues,
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
        subprocess.run(
            ["docker", "rm", args.container_name],
            check=True,
            capture_output=True,
            text=True,
        )
        removed = True
        print(f"Saved container outputs to: {output_dir}")
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
