from __future__ import annotations

import math
import re
import threading
from collections import Counter, deque
from typing import Any


PROMETHEUS_NAME_RE = re.compile(r"[^a-zA-Z0-9_:]")

HTTP_COUNTER_NAMES = (
    "http_requests_total",
    "http_responses_2xx_total",
    "http_responses_4xx_total",
    "http_responses_5xx_total",
    "http_responses_other_total",
    "http_requests_in_flight",
    "http_requests_in_flight_max",
    "http_request_latency_ms_total",
    "authentication_failed_total",
    "request_validation_failed_total",
    "prompt_validation_failed_total",
    "image_validation_failed_total",
    "service_errors_total",
    "http_exception_total",
    "unexpected_http_errors_total",
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class RollingWindow:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("rolling window capacity must be positive")
        self._values: deque[float] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._values.append(float(value))

    def snapshot(self, prefix: str) -> dict[str, float | int]:
        with self._lock:
            values = list(self._values)
        return {
            f"{prefix}_window_samples": len(values),
            f"{prefix}_p50": round(percentile(values, 50), 6),
            f"{prefix}_p95": round(percentile(values, 95), 6),
            f"{prefix}_p99": round(percentile(values, 99), 6),
        }


class HttpMetrics:
    def __init__(self, window_size: int):
        self._counters = Counter(
            {name: 0 for name in HTTP_COUNTER_NAMES}
        )
        self._latency_ms = RollingWindow(window_size)
        self._lock = threading.Lock()

    def begin_request(self) -> None:
        with self._lock:
            self._counters["http_requests_total"] += 1
            self._counters["http_requests_in_flight"] += 1
            self._counters["http_requests_in_flight_max"] = max(
                self._counters["http_requests_in_flight_max"],
                self._counters["http_requests_in_flight"],
            )

    def finish_request(self, status_code: int, latency_ms: float) -> None:
        status_class = f"{status_code // 100}xx"
        if status_class not in {"2xx", "4xx", "5xx"}:
            status_class = "other"
        with self._lock:
            self._counters[f"http_responses_{status_class}_total"] += 1
            self._counters["http_requests_in_flight"] -= 1
            self._counters["http_request_latency_ms_total"] += latency_ms
        self._latency_ms.observe(latency_ms)

    def increment(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            counters = dict(self._counters)
        return {
            **counters,
            **self._latency_ms.snapshot("http_request_latency_ms"),
        }


def evaluate_alerts(
    metrics: dict[str, Any],
    *,
    minimum_requests: int,
    max_4xx_rate: float,
    max_5xx_rate: float,
    max_p95_latency_ms: float,
    max_queue_utilization: float,
) -> dict[str, Any]:
    alerts: list[dict[str, Any]] = []

    def add(
        code: str,
        severity: str,
        actual: Any,
        threshold: Any,
    ) -> None:
        alerts.append(
            {
                "code": code,
                "severity": severity,
                "actual": actual,
                "threshold": threshold,
            }
        )

    if not bool(metrics.get("ready")):
        add("model_not_ready", "critical", False, True)
    cuda_oom = int(metrics.get("cuda_oom_total", 0))
    if cuda_oom > 0:
        add("cuda_oom_detected", "critical", cuda_oom, 0)
    unexpected = int(metrics.get("unexpected_errors_total", 0))
    if unexpected > 0:
        add("unexpected_error_detected", "critical", unexpected, 0)
    unavailable = int(
        metrics.get("model_unavailable_rejected_total", 0)
    )
    if unavailable > 0:
        add("model_unavailable_rejection", "critical", unavailable, 0)

    queue_utilization = float(metrics.get("queue_utilization", 0.0))
    if queue_utilization >= max_queue_utilization:
        add(
            "queue_utilization_high",
            "warning",
            round(queue_utilization, 6),
            max_queue_utilization,
        )

    responses_2xx = int(metrics.get("http_responses_2xx_total", 0))
    responses_4xx = int(metrics.get("http_responses_4xx_total", 0))
    responses_5xx = int(metrics.get("http_responses_5xx_total", 0))
    completed = responses_2xx + responses_4xx + responses_5xx
    if completed >= minimum_requests:
        rate_4xx = responses_4xx / completed
        rate_5xx = responses_5xx / completed
        if rate_4xx > max_4xx_rate:
            add(
                "http_4xx_rate_high",
                "warning",
                round(rate_4xx, 6),
                max_4xx_rate,
            )
        if rate_5xx > max_5xx_rate:
            add(
                "http_5xx_rate_high",
                "critical",
                round(rate_5xx, 6),
                max_5xx_rate,
            )
        p95 = float(metrics.get("http_request_latency_ms_p95", 0.0))
        if p95 > max_p95_latency_ms:
            add(
                "http_p95_latency_high",
                "warning",
                round(p95, 3),
                max_p95_latency_ms,
            )

    status = "firing" if alerts else "ok"
    return {
        "status": status,
        "alert_count": len(alerts),
        "alerts": alerts,
    }


def render_prometheus(
    metrics: dict[str, Any],
    *,
    model_version: str,
) -> str:
    escaped_version = (
        model_version.replace("\\", "\\\\").replace('"', '\\"')
    )
    labels = f'model_version="{escaped_version}"'
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, bool):
            numeric_value: int | float = int(value)
        elif isinstance(value, (int, float)):
            numeric_value = value
        else:
            continue
        metric_name = PROMETHEUS_NAME_RE.sub("_", f"lisa_{key}")
        metric_type = "counter" if metric_name.endswith("_total") else "gauge"
        lines.append(f"# TYPE {metric_name} {metric_type}")
        lines.append(f"{metric_name}{{{labels}}} {numeric_value}")
    return "\n".join(lines) + "\n"
