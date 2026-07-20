import unittest

from production.verify_request_limit import (
    build_acceptance_checks,
    decode_chunked_body,
    parse_http_response,
)


def passing_inputs():
    metrics = {
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
    return {
        "build_exit_code": 0,
        "unit_test_exit_code": 0,
        "unit_test_count": 65,
        "minimum_unit_tests": 65,
        "cases": [{"passed": True}] * 3,
        "metrics": metrics,
        "configured_user": "lisa",
        "runtime_uid": 10001,
        "runtime_gid": 10001,
        "memory_drift_mib": 0,
        "max_memory_drift_mib": 500,
        "missing_initial_process_ids": [],
        "log_issues": {
            "cuda_oom": False,
            "traceback": False,
            "error_log": False,
            "sensitive_value_labels": [],
            "private_path_labels": [],
        },
        "health_after": {"status": "ok"},
        "stopped_exit_code": 0,
    }


class RequestLimitVerificationTest(unittest.TestCase):
    def test_parses_content_length_and_chunked_http_responses(self):
        body = b'{"code":"x"}'
        content_length = (
            b"HTTP/1.1 413 Request Entity Too Large\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"\r\n"
            + body
        )
        parsed = parse_http_response(content_length)
        self.assertEqual(parsed["status"], 413)
        self.assertEqual(parsed["payload"], {"code": "x"})

        chunked = (
            f"{len(body):X}\r\n".encode("ascii")
            + body
            + b"\r\n0\r\n\r\n"
        )
        self.assertEqual(decode_chunked_body(chunked), body)

    def test_acceptance_passes_expected_result(self):
        checks = build_acceptance_checks(**passing_inputs())
        self.assertTrue(all(item["passed"] for item in checks.values()))

    def test_acceptance_rejects_gpu_entry_leak_and_failed_case(self):
        inputs = passing_inputs()
        inputs["cases"][0] = {"passed": False}
        inputs["metrics"]["requests_received_total"] = 1
        inputs["log_issues"]["sensitive_value_labels"] = [
            "sensitive_value_1"
        ]
        checks = build_acceptance_checks(**inputs)
        self.assertFalse(checks["all_http_cases_passed"]["passed"])
        self.assertFalse(checks["exact_request_limit_metrics"]["passed"])
        self.assertFalse(checks["container_logs_clean"]["passed"])


if __name__ == "__main__":
    unittest.main()
