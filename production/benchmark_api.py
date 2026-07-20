from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUEST_FIELDS = [
    "sequence",
    "phase",
    "request_id",
    "success",
    "http_status",
    "client_latency_ms",
    "server_latency_ms",
    "mask_count",
    "response_bytes",
    "error",
]


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_phase(
    rows: list[dict[str, Any]],
    phase: str,
    elapsed_seconds: float,
) -> dict[str, float | int]:
    selected = [row for row in rows if row["phase"] == phase]
    succeeded = [row for row in selected if row["success"]]
    client = [float(row["client_latency_ms"]) for row in succeeded]
    server = [float(row["server_latency_ms"]) for row in succeeded]
    result: dict[str, float | int] = {
        "requests": len(selected),
        "succeeded": len(succeeded),
        "failed": len(selected) - len(succeeded),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "throughput_requests_per_second": round(
            len(succeeded) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            6,
        ),
    }
    if client:
        result.update(
            {
                "client_latency_p50_ms": round(percentile(client, 50), 3),
                "client_latency_p95_ms": round(percentile(client, 95), 3),
                "client_latency_p99_ms": round(percentile(client, 99), 3),
                "server_latency_p50_ms": round(percentile(server, 50), 3),
                "server_latency_p95_ms": round(percentile(server, 95), 3),
                "server_latency_p99_ms": round(percentile(server, 99), 3),
            }
        )
    return result


def validate_segment_response(
    payload: dict[str, Any],
    expected_request_id: str,
) -> None:
    if payload.get("request_id") != expected_request_id:
        raise ValueError("response request_id does not match")
    masks = payload.get("masks")
    if not isinstance(masks, list):
        raise ValueError("response masks is not a list")
    if payload.get("mask_count") != len(masks):
        raise ValueError("response mask_count does not match masks")
    if payload.get("has_segmentation") != bool(masks):
        raise ValueError("response has_segmentation does not match masks")
    if int(payload.get("width", 0)) <= 0 or int(payload.get("height", 0)) <= 0:
        raise ValueError("response image dimensions are invalid")
    for mask in masks:
        if mask.get("format") != "png_base64":
            raise ValueError("response mask format is not png_base64")
        try:
            decoded = base64.b64decode(mask["data"], validate=True)
        except Exception as exc:
            raise ValueError("response mask is not valid base64") from exc
        if not decoded.startswith(PNG_SIGNATURE):
            raise ValueError("response mask is not a PNG")


def query_gpu_memory(gpu_index: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def existing_compute_processes(gpu_index: int) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def ensure_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            handle.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"{host}:{port} is already in use") from exc


def portable_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        pass
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index > 0 and index + 1 < len(parts):
            model_dir = parts[index - 1]
            snapshot = parts[index + 1]
            if model_dir.startswith("models--"):
                model_name = model_dir.removeprefix("models--").replace("--", "/")
                return f"huggingface://{model_name}@{snapshot}"
    return f"external://{path.name}"


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(
    base_url: str,
    server: subprocess.Popen,
    timeout_seconds: float,
) -> float:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    while time.perf_counter() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"Uvicorn exited during startup with code {server.returncode}"
            )
        try:
            payload = get_json(f"{base_url}/ready", timeout=2.0)
            if payload.get("status") == "ready":
                return (time.perf_counter() - started) * 1000
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"service was not ready after {timeout_seconds} seconds")


def send_segment_request(
    *,
    url: str,
    image_base64: str,
    prompt: str,
    request_id: str,
    timeout_seconds: float,
    phase: str,
    sequence: int,
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
    started = time.perf_counter()
    status = 0
    raw = b""
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds
        ) as response:
            status = response.status
            raw = response.read()
        client_latency_ms = (time.perf_counter() - started) * 1000
        payload = json.loads(raw.decode("utf-8"))
        validate_segment_response(payload, request_id)
        return {
            "sequence": sequence,
            "phase": phase,
            "request_id": request_id,
            "success": True,
            "http_status": status,
            "client_latency_ms": round(client_latency_ms, 3),
            "server_latency_ms": float(payload["latency_ms"]),
            "mask_count": int(payload["mask_count"]),
            "response_bytes": len(raw),
            "error": "",
        }
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
        error = raw.decode("utf-8", errors="replace")[:500]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "sequence": sequence,
        "phase": phase,
        "request_id": request_id,
        "success": False,
        "http_status": status,
        "client_latency_ms": round(
            (time.perf_counter() - started) * 1000, 3
        ),
        "server_latency_ms": 0.0,
        "mask_count": 0,
        "response_bytes": len(raw),
        "error": error,
    }


def write_requests_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_peak_gpu_memory(path: Path) -> int:
    values: list[int] = []
    if not path.exists():
        return 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            values.append(int(parts[1]))
        except ValueError:
            continue
    return max(values, default=0)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    measured = summary["phases"]["measured"]
    stability = summary["phases"]["stability"]
    acceptance = summary["acceptance"]
    lines = [
        "# LISA bf16 Production API Performance Baseline",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Model version: `{summary['model_version']}`",
        f"- Model path: `{summary['model_path']}`",
        f"- GPU: `{summary['gpu_name']}`",
        f"- Ready time: `{summary['ready_time_ms']:.3f} ms`",
        "",
        "## GPU memory",
        "",
        "| Metric | MiB |",
        "| --- | ---: |",
        f"| Baseline | {summary['gpu_memory_mib']['baseline']} |",
        f"| Model loaded | {summary['gpu_memory_mib']['loaded']} |",
        f"| After warmup | {summary['gpu_memory_mib']['after_warmup']} |",
        f"| Peak | {summary['gpu_memory_mib']['peak']} |",
        f"| After requests | {summary['gpu_memory_mib']['after_requests']} |",
        f"| Model load delta | {summary['gpu_memory_mib']['model_load_delta']} |",
        f"| Post-warmup drift | {summary['gpu_memory_mib']['post_warmup_drift']} |",
        "",
        "## Measured requests",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Requests | {measured['requests']} |",
        f"| Succeeded | {measured['succeeded']} |",
        f"| Client P50 | {measured.get('client_latency_p50_ms', 0):.3f} ms |",
        f"| Client P95 | {measured.get('client_latency_p95_ms', 0):.3f} ms |",
        f"| Client P99 | {measured.get('client_latency_p99_ms', 0):.3f} ms |",
        f"| Throughput | {measured['throughput_requests_per_second']:.6f} req/s |",
        "",
        "## Stability",
        "",
        f"- Requests: `{stability['requests']}`",
        f"- Succeeded: `{stability['succeeded']}`",
        f"- Failed: `{stability['failed']}`",
        "",
        "## Acceptance",
        "",
    ]
    for name, item in acceptance["checks"].items():
        marker = "PASS" if item["passed"] else "FAIL"
        lines.append(
            f"- {marker} `{name}`: actual `{item['actual']}`, "
            f"limit `{item['limit']}`"
        )
    lines.extend(
        [
            "",
            f"Overall: `{'PASS' if acceptance['passed'] else 'FAIL'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def stop_process(process: subprocess.Popen | None, timeout: float = 30.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the production LISA API on a dedicated GPU."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-version", default="lisa13b-clean030-v1")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--measured-requests", type=int, default=30)
    parser.add_argument("--stability-requests", type=int, default=100)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=150.0)
    parser.add_argument("--max-p95-ms", type=float, default=1500.0)
    parser.add_argument("--max-peak-memory-mib", type=int, default=36864)
    parser.add_argument("--max-memory-drift-mib", type=int, default=500)
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
    active_processes = existing_compute_processes(args.gpu_index)
    if active_processes:
        formatted = "\n".join(f"- {item}" for item in active_processes)
        raise RuntimeError(
            "GPU has existing compute processes; stop them before the "
            f"dedicated benchmark:\n{formatted}"
        )

    baseline_memory = query_gpu_memory(args.gpu_index)
    gpu_csv = output_dir / "gpu_metrics.csv"
    server_log = output_dir / "server.log"
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
        "precision": args.precision,
        "gpu_index": args.gpu_index,
        "host": args.host,
        "port": args.port,
        "warmup_requests": args.warmup_requests,
        "measured_requests": args.measured_requests,
        "stability_requests": args.stability_requests,
        "thresholds": {
            "max_p95_ms": args.max_p95_ms,
            "max_peak_memory_mib": args.max_peak_memory_mib,
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
            "LISA_PRECISION": args.precision,
            "LISA_LOAD_IN_8BIT": "false",
            "LISA_LOAD_IN_4BIT": "false",
            "LISA_GPU_INDEX": "0",
            "LISA_MAX_CONCURRENCY": "1",
            "LISA_REQUEST_TIMEOUT_SECONDS": "120",
            "LISA_EAGER_LOAD": "true",
            "LISA_API_KEY": "",
        }
    )

    monitor: subprocess.Popen | None = None
    server: subprocess.Popen | None = None
    monitor_handle = None
    server_handle = None
    rows: list[dict[str, Any]] = []
    phase_elapsed: dict[str, float] = {}
    try:
        monitor_handle = gpu_csv.open("w", encoding="utf-8")
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
        server_handle = server_log.open("w", encoding="utf-8")
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

        sequence = 0
        after_warmup_memory = loaded_memory
        phases = [
            ("cold", 1),
            ("warmup", args.warmup_requests),
            ("measured", args.measured_requests),
            ("stability", args.stability_requests),
        ]
        for phase, count in phases:
            phase_started = time.perf_counter()
            for index in range(count):
                sequence += 1
                request_id = (
                    f"perf-{phase}-{index + 1:03d}-{int(time.time() * 1000)}"
                )
                row = send_segment_request(
                    url=f"{base_url}/v1/segment",
                    image_base64=image_base64,
                    prompt=args.prompt,
                    request_id=request_id,
                    timeout_seconds=args.request_timeout,
                    phase=phase,
                    sequence=sequence,
                )
                rows.append(row)
                status = "ok" if row["success"] else "failed"
                print(
                    f"{phase} {index + 1}/{count}: {status}, "
                    f"{row['client_latency_ms']} ms",
                    flush=True,
                )
            phase_elapsed[phase] = time.perf_counter() - phase_started
            if phase == "warmup":
                after_warmup_memory = query_gpu_memory(args.gpu_index)

        after_requests_memory = query_gpu_memory(args.gpu_index)
        write_requests_csv(output_dir / "requests.csv", rows)
        time.sleep(0.5)
        stop_process(monitor, timeout=5)
        monitor = None
        monitor_handle.close()
        monitor_handle = None
        peak_memory = read_peak_gpu_memory(gpu_csv)

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
        phase_summaries = {
            phase: summarize_phase(rows, phase, phase_elapsed[phase])
            for phase, _ in phases
        }
        measured_p95 = float(
            phase_summaries["measured"].get("client_latency_p95_ms", 0.0)
        )
        total_failed = sum(not row["success"] for row in rows)
        drift = after_requests_memory - after_warmup_memory
        checks = {
            "all_requests_succeeded": {
                "actual": total_failed,
                "limit": 0,
                "passed": total_failed == 0,
            },
            "measured_client_p95_ms": {
                "actual": measured_p95,
                "limit": args.max_p95_ms,
                "passed": measured_p95 <= args.max_p95_ms,
            },
            "peak_gpu_memory_mib": {
                "actual": peak_memory,
                "limit": args.max_peak_memory_mib,
                "passed": peak_memory <= args.max_peak_memory_mib,
            },
            "post_warmup_memory_drift_mib": {
                "actual": drift,
                "limit": args.max_memory_drift_mib,
                "passed": drift <= args.max_memory_drift_mib,
            },
        }
        summary = {
            **runtime_config,
            "gpu_name": gpu_name,
            "ready_time_ms": round(ready_time_ms, 3),
            "gpu_memory_mib": {
                "baseline": baseline_memory,
                "loaded": loaded_memory,
                "after_warmup": after_warmup_memory,
                "peak": peak_memory,
                "after_requests": after_requests_memory,
                "model_load_delta": loaded_memory - baseline_memory,
                "post_warmup_drift": drift,
            },
            "phases": phase_summaries,
            "total_requests": len(rows),
            "total_failed": total_failed,
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
        print(f"Saved performance outputs to: {output_dir}")
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
