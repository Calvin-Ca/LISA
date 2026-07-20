import unittest

from production.benchmark_api_concurrency import (
    build_acceptance_checks,
    metric_delta,
    summarize_concurrency_phase,
)


def passing_metrics(requests: int):
    return {
        "requests_received_total": requests,
        "requests_started_total": requests,
        "requests_succeeded_total": requests,
        "gpu_inference_succeeded_total": requests,
        "gpu_inference_failed_total": 0,
        "gpu_inference_in_flight_max": 1,
        "gpu_inference_in_flight": 0,
        "masks_returned_total": requests,
        "queue_timeout_total": 0,
        "queue_rejected_total": 0,
        "queue_cancelled_total": 0,
        "requests_timeout_total": 0,
        "unexpected_errors_total": 0,
        "cuda_oom_total": 0,
        "queue_wait_seconds_total": 10.0,
        "gpu_inference_seconds_total": 75.0,
    }


def passing_phases():
    return {
        "measured-c1": {
            "client_latency_p95_ms": 400.0,
            "throughput_requests_per_second": 2.5,
        },
        "measured-c2": {
            "client_latency_p95_ms": 800.0,
            "throughput_requests_per_second": 2.5,
        },
        "measured-c4": {
            "client_latency_p95_ms": 1600.0,
            "throughput_requests_per_second": 2.5,
        },
        "stability-c4": {
            "client_latency_p95_ms": 1650.0,
            "throughput_requests_per_second": 2.4,
        },
    }


def build_checks(
    *,
    metrics_final=None,
    phase_summaries=None,
    gpu_memory_mib=None,
):
    return build_acceptance_checks(
        phase_summaries=phase_summaries or passing_phases(),
        metrics_initial={},
        metrics_final=metrics_final or passing_metrics(195),
        total_requests=195,
        expected_requests=195,
        total_failed=0,
        gpu_memory_mib=gpu_memory_mib
        or {
            "peak": 31740,
            "remaining_at_peak": 9220,
            "post_warmup_drift": 0,
        },
        final_ready={"status": "ready"},
        missing_initial_process_ids=[],
        server_log="Application startup complete.",
        max_p95_ms={
            "measured-c1": 1000.0,
            "measured-c2": 2000.0,
            "measured-c4": 4000.0,
            "stability-c4": 4000.0,
        },
        min_throughput=2.0,
        max_peak_memory_mib=36864,
        min_remaining_memory_mib=4096,
        max_memory_drift_mib=500,
    )


class ApiConcurrencyTest(unittest.TestCase):
    def test_metric_delta_treats_missing_metrics_as_zero(self):
        self.assertEqual(
            metric_delta(
                {},
                {"requests_started_total": 3},
                "requests_started_total",
            ),
            3,
        )

    def test_phase_summary_uses_runtime_metric_deltas(self):
        rows = [
            {
                "phase": "measured-c2",
                "success": True,
                "client_latency_ms": 700.0,
                "server_latency_ms": 690.0,
            },
            {
                "phase": "measured-c2",
                "success": True,
                "client_latency_ms": 800.0,
                "server_latency_ms": 790.0,
            },
        ]
        summary = summarize_concurrency_phase(
            rows=rows,
            phase="measured-c2",
            concurrency=2,
            elapsed_seconds=0.8,
            metrics_before={
                "requests_received_total": 10,
                "requests_started_total": 10,
                "queue_wait_seconds_total": 1.0,
                "gpu_inference_seconds_total": 4.0,
            },
            metrics_after={
                "requests_received_total": 12,
                "requests_started_total": 12,
                "queue_wait_seconds_total": 1.8,
                "gpu_inference_seconds_total": 4.6,
            },
        )
        self.assertEqual(summary["client_concurrency"], 2)
        self.assertEqual(summary["runtime_requests_received"], 2)
        self.assertEqual(summary["average_queue_wait_ms"], 400.0)
        self.assertEqual(summary["average_gpu_inference_ms"], 300.0)
        self.assertEqual(summary["throughput_requests_per_second"], 2.5)

    def test_acceptance_passes_expected_shared_gpu_result(self):
        checks = build_checks()
        self.assertTrue(all(item["passed"] for item in checks.values()))

    def test_acceptance_rejects_hidden_gpu_concurrency_and_queue_failure(self):
        metrics = passing_metrics(195)
        metrics["gpu_inference_in_flight_max"] = 2
        metrics["queue_timeout_total"] = 1
        checks = build_checks(metrics_final=metrics)
        self.assertFalse(
            checks["gpu_inference_in_flight_max"]["passed"]
        )
        self.assertFalse(checks["queue_timeout_total"]["passed"])

    def test_acceptance_rejects_latency_throughput_and_memory_regression(self):
        phases = passing_phases()
        phases["measured-c4"] = {
            "client_latency_p95_ms": 4500.0,
            "throughput_requests_per_second": 1.8,
        }
        checks = build_checks(
            phase_summaries=phases,
            gpu_memory_mib={
                "peak": 37000,
                "remaining_at_peak": 3960,
                "post_warmup_drift": 600,
            },
        )
        self.assertFalse(checks["measured-c4_client_p95_ms"]["passed"])
        self.assertFalse(checks["measured-c4_throughput"]["passed"])
        self.assertFalse(checks["peak_gpu_memory_mib"]["passed"])
        self.assertFalse(checks["remaining_gpu_memory_mib"]["passed"])
        self.assertFalse(
            checks["post_warmup_memory_drift_mib"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
