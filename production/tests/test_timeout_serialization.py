import unittest

from production.verify_timeout_serialization import build_acceptance_checks


def passing_requests():
    return [
        {
            "request_id": f"timeout-guard-{index}",
            "http_status": 504,
            "error_code": "inference_timeout",
            "passed": True,
        }
        for index in (1, 2)
    ]


def passing_metrics():
    return {
        "requests_received_total": 2,
        "requests_started_total": 2,
        "requests_timeout_total": 2,
        "gpu_inference_succeeded_total": 2,
        "gpu_inference_failed_total": 0,
        "gpu_inference_in_flight_max": 1,
        "gpu_inference_in_flight": 0,
        "queue_wait_seconds_total": 0.4,
    }


class TimeoutSerializationTest(unittest.TestCase):
    def test_acceptance_passes_for_serial_execution(self):
        checks = build_acceptance_checks(
            requests=passing_requests(),
            metrics=passing_metrics(),
            jobs_completed=True,
            missing_initial_process_ids=[],
            server_log="Application startup complete.",
        )
        self.assertTrue(all(item["passed"] for item in checks.values()))

    def test_acceptance_rejects_hidden_concurrency(self):
        metrics = passing_metrics()
        metrics["gpu_inference_in_flight_max"] = 2
        checks = build_acceptance_checks(
            requests=passing_requests(),
            metrics=metrics,
            jobs_completed=True,
            missing_initial_process_ids=[],
            server_log="Application startup complete.",
        )
        self.assertFalse(checks["gpu_inference_in_flight_max"]["passed"])

    def test_acceptance_rejects_cuda_oom_and_missing_shared_process(self):
        checks = build_acceptance_checks(
            requests=passing_requests(),
            metrics=passing_metrics(),
            jobs_completed=True,
            missing_initial_process_ids=[123],
            server_log="torch.cuda.OutOfMemoryError: CUDA out of memory",
        )
        self.assertFalse(checks["existing_compute_processes_remained"]["passed"])
        self.assertFalse(checks["no_cuda_oom"]["passed"])


if __name__ == "__main__":
    unittest.main()
