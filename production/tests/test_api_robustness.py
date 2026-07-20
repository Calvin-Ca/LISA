import unittest

from production.verify_api_robustness import build_acceptance_checks


def passing_cases():
    return [
        {
            "case": f"case-{index}",
            "passed": True,
            "sensitive_response": False,
        }
        for index in range(15)
    ]


def passing_final_metrics():
    return {
        "requests_received_total": 5,
        "requests_started_total": 3,
        "requests_succeeded_total": 3,
        "queue_timeout_total": 1,
        "queue_rejected_total": 1,
        "queue_cancelled_total": 1,
        "gpu_inference_succeeded_total": 3,
        "gpu_inference_failed_total": 0,
        "gpu_inference_in_flight_max": 1,
        "gpu_inference_in_flight": 0,
        "masks_returned_total": 3,
        "cuda_oom_total": 0,
        "unexpected_errors_total": 0,
    }


class ApiRobustnessTest(unittest.TestCase):
    def test_acceptance_passes_expected_results(self):
        checks = build_acceptance_checks(
            cases=passing_cases(),
            metrics_after_invalid={},
            metrics_final=passing_final_metrics(),
            final_ready={"status": "ready"},
            leaked_log_labels=[],
            missing_initial_process_ids=[],
            server_log="Application startup complete.",
        )
        self.assertTrue(all(item["passed"] for item in checks.values()))

    def test_acceptance_rejects_invalid_request_entering_gpu(self):
        checks = build_acceptance_checks(
            cases=passing_cases(),
            metrics_after_invalid={"requests_received_total": 1},
            metrics_final=passing_final_metrics(),
            final_ready={"status": "ready"},
            leaked_log_labels=[],
            missing_initial_process_ids=[],
            server_log="Application startup complete.",
        )
        self.assertFalse(
            checks["invalid_requests_did_not_enter_runtime"]["passed"]
        )

    def test_acceptance_rejects_leaks_oom_and_hidden_concurrency(self):
        metrics = passing_final_metrics()
        metrics["gpu_inference_in_flight_max"] = 2
        metrics["cuda_oom_total"] = 1
        cases = passing_cases()
        cases[0]["sensitive_response"] = True
        checks = build_acceptance_checks(
            cases=cases,
            metrics_after_invalid={},
            metrics_final=metrics,
            final_ready={"status": "unavailable"},
            leaked_log_labels=["api_key"],
            missing_initial_process_ids=[123],
            server_log="CUDA out of memory",
        )
        self.assertFalse(checks["gpu_inference_in_flight_max"]["passed"])
        self.assertFalse(checks["cuda_oom_total"]["passed"])
        self.assertFalse(
            checks["sensitive_values_absent_from_responses"]["passed"]
        )
        self.assertFalse(checks["sensitive_values_absent_from_log"]["passed"])
        self.assertFalse(checks["service_ready_after_errors"]["passed"])
        self.assertFalse(
            checks["existing_compute_processes_remained"]["passed"]
        )
        self.assertFalse(checks["no_cuda_oom_in_log"]["passed"])


if __name__ == "__main__":
    unittest.main()
