import unittest

from production.verify_monitoring import (
    build_acceptance_checks,
    parse_prometheus_samples,
    prometheus_json_mismatches,
)


def passing_inputs():
    metrics = {
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
        "http_request_latency_ms_p50": 10.0,
        "http_request_latency_ms_p95": 20.0,
        "http_request_latency_ms_p99": 30.0,
        "queue_wait_ms_p50": 1.0,
        "queue_wait_ms_p95": 2.0,
        "queue_wait_ms_p99": 3.0,
        "gpu_inference_ms_p50": 8.0,
        "gpu_inference_ms_p95": 18.0,
        "gpu_inference_ms_p99": 28.0,
    }
    return {
        "build_exit_code": 0,
        "unit_test_exit_code": 0,
        "unit_test_count": 62,
        "minimum_unit_tests": 62,
        "initial_alerts": {"status": "ok", "alerts": []},
        "firing_alerts": {
            "status": "firing",
            "alerts": [{"code": "http_4xx_rate_high"}],
        },
        "recovered_alerts": {"status": "ok", "alerts": []},
        "valid_requests": [{"passed": True}] * 5,
        "unauthorized_request": {"passed": True},
        "metrics": metrics,
        "prometheus_mismatches": [],
        "monitoring_leaks": [],
        "configured_user": "lisa",
        "runtime_uid": 10001,
        "runtime_gid": 10001,
        "mounts_read_only": True,
        "gpu_memory_mib": {
            "peak": 30000,
            "remaining_at_peak": 10000,
            "post_stop_drift": 0,
        },
        "max_peak_memory_mib": 36864,
        "min_remaining_memory_mib": 4096,
        "max_post_stop_drift_mib": 500,
        "missing_initial_process_ids": [],
        "log_issues": {
            "cuda_oom": False,
            "traceback": False,
            "error_log": False,
            "sensitive_value_labels": [],
            "private_path_labels": [],
        },
        "ready_after": {"status": "ready"},
        "stopped_exit_code": 0,
    }


class MonitoringVerificationTest(unittest.TestCase):
    def test_parse_prometheus_samples_and_compare_json(self):
        text = (
            "# TYPE lisa_ready gauge\n"
            'lisa_ready{model_version="test-v1"} 1\n'
            "# TYPE lisa_http_requests_total counter\n"
            'lisa_http_requests_total{model_version="test-v1"} 6\n'
        )
        samples = parse_prometheus_samples(text)
        self.assertEqual(samples["lisa_ready"], 1.0)
        self.assertEqual(samples["lisa_http_requests_total"], 6.0)
        self.assertEqual(
            prometheus_json_mismatches(
                {"ready": True, "http_requests_total": 6},
                samples,
                ("ready", "http_requests_total"),
            ),
            [],
        )

    def test_acceptance_passes_expected_monitoring_result(self):
        checks = build_acceptance_checks(**passing_inputs())
        self.assertTrue(all(item["passed"] for item in checks.values()))

    def test_acceptance_rejects_alert_metric_and_security_failures(self):
        inputs = passing_inputs()
        inputs["recovered_alerts"] = {
            "status": "firing",
            "alerts": [{"code": "http_4xx_rate_high"}],
        }
        inputs["metrics"]["gpu_inference_in_flight_max"] = 2
        inputs["prometheus_mismatches"] = ["http_requests_total"]
        inputs["monitoring_leaks"] = ["api_key"]
        checks = build_acceptance_checks(**inputs)
        self.assertFalse(checks["high_4xx_alert_recovered"]["passed"])
        self.assertFalse(checks["exact_runtime_metrics"]["passed"])
        self.assertFalse(checks["prometheus_matches_json"]["passed"])
        self.assertFalse(
            checks["monitoring_outputs_no_sensitive_values"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
