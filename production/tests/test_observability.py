import unittest

from production.observability import (
    HttpMetrics,
    RollingWindow,
    evaluate_alerts,
    percentile,
    render_prometheus,
)


class ObservabilityTest(unittest.TestCase):
    def test_percentile_and_rolling_window(self):
        self.assertEqual(percentile([], 95), 0.0)
        self.assertEqual(percentile([1.0], 95), 1.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        window = RollingWindow(3)
        for value in (10, 20, 30, 40):
            window.observe(value)
        snapshot = window.snapshot("latency_ms")
        self.assertEqual(snapshot["latency_ms_window_samples"], 3)
        self.assertEqual(snapshot["latency_ms_p50"], 30.0)

    def test_http_metrics_count_status_and_latency(self):
        metrics = HttpMetrics(window_size=10)
        initial = metrics.snapshot()
        self.assertEqual(initial["http_requests_total"], 0)
        self.assertEqual(initial["http_responses_5xx_total"], 0)
        self.assertEqual(initial["request_body_too_large_total"], 0)
        metrics.begin_request()
        metrics.finish_request(200, 100.0)
        metrics.begin_request()
        metrics.increment("authentication_failed_total")
        metrics.finish_request(401, 20.0)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["http_requests_total"], 2)
        self.assertEqual(snapshot["http_responses_2xx_total"], 1)
        self.assertEqual(snapshot["http_responses_4xx_total"], 1)
        self.assertEqual(snapshot["http_requests_in_flight"], 0)
        self.assertEqual(snapshot["http_request_latency_ms_p50"], 60.0)
        self.assertEqual(snapshot["authentication_failed_total"], 1)

    def test_alerts_fire_for_runtime_and_rate_thresholds(self):
        result = evaluate_alerts(
            {
                "ready": False,
                "cuda_oom_total": 1,
                "queue_utilization": 0.9,
                "http_responses_2xx_total": 8,
                "http_responses_4xx_total": 3,
                "http_responses_5xx_total": 1,
                "http_request_latency_ms_p95": 2500.0,
            },
            minimum_requests=10,
            max_4xx_rate=0.2,
            max_5xx_rate=0.01,
            max_p95_latency_ms=2000.0,
            max_queue_utilization=0.8,
        )
        codes = {item["code"] for item in result["alerts"]}
        self.assertEqual(result["status"], "firing")
        self.assertIn("model_not_ready", codes)
        self.assertIn("cuda_oom_detected", codes)
        self.assertIn("queue_utilization_high", codes)
        self.assertIn("http_4xx_rate_high", codes)
        self.assertIn("http_5xx_rate_high", codes)
        self.assertIn("http_p95_latency_high", codes)

    def test_rate_alerts_wait_for_minimum_sample_count(self):
        result = evaluate_alerts(
            {
                "ready": True,
                "http_responses_5xx_total": 1,
                "http_request_latency_ms_p95": 9999.0,
            },
            minimum_requests=20,
            max_4xx_rate=0.2,
            max_5xx_rate=0.01,
            max_p95_latency_ms=2000.0,
            max_queue_utilization=0.8,
        )
        self.assertEqual(result["status"], "ok")

    def test_prometheus_output_contains_only_numeric_metrics(self):
        output = render_prometheus(
            {
                "ready": True,
                "requests_succeeded_total": 3,
                "queue_utilization": 0.5,
                "private_path": "/private/model",
            },
            model_version='test"v1',
        )
        self.assertIn("# TYPE lisa_requests_succeeded_total counter", output)
        self.assertIn("lisa_ready{model_version=\"test\\\"v1\"} 1", output)
        self.assertIn("lisa_queue_utilization", output)
        self.assertNotIn("/private/model", output)


if __name__ == "__main__":
    unittest.main()
