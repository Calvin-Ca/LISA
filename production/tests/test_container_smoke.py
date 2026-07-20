import unittest
from pathlib import Path

from production.verify_container_smoke import (
    build_acceptance_checks,
    find_log_issues,
    sanitize_container_inspect,
)


def passing_cycle(cycle: int):
    return {
        "cycle": cycle,
        "startup": {
            "health_status": "healthy",
            "ready": {"status": "ready"},
        },
        "smoke": {"passed": True},
        "metrics_before": {},
        "metrics_after": {
            "model_loads_total": 1,
            "requests_received_total": 1,
            "requests_succeeded_total": 1,
            "gpu_inference_succeeded_total": 1,
            "gpu_inference_failed_total": 0,
            "gpu_inference_in_flight_max": 1,
            "gpu_inference_in_flight": 0,
            "queue_timeout_total": 0,
            "requests_timeout_total": 0,
            "queue_rejected_total": 0,
            "queue_cancelled_total": 0,
            "cuda_oom_total": 0,
            "unexpected_errors_total": 0,
        },
        "ready_after": {"status": "ready"},
    }


def build_checks(
    *,
    cycles=None,
    log_issues=None,
    gpu_memory_mib=None,
    forbidden_image_paths_present=None,
):
    return build_acceptance_checks(
        build_succeeded=True,
        unit_test_exit_code=0,
        unit_test_count=42,
        minimum_unit_tests=38,
        configured_user="lisa",
        runtime_uid=10001,
        runtime_gid=10001,
        mounts_read_only=True,
        forbidden_image_paths_present=forbidden_image_paths_present or [],
        cycles=cycles or [passing_cycle(1), passing_cycle(2)],
        gpu_memory_mib=gpu_memory_mib
        or {
            "peak": 31770,
            "remaining_at_peak": 9190,
            "post_stop_drift": 0,
        },
        max_peak_memory_mib=36864,
        min_remaining_memory_mib=4096,
        max_post_stop_drift_mib=500,
        missing_initial_process_ids=[],
        log_issues=log_issues
        or {
            "cuda_oom": False,
            "traceback": False,
            "error_log": False,
            "sensitive_value_labels": [],
            "private_path_labels": [],
        },
        stopped_exit_code=0,
    )


class ContainerSmokeTest(unittest.TestCase):
    def test_docker_context_is_allowlisted_and_excludes_env_files(self):
        repo_root = Path(__file__).resolve().parents[2]
        ignore_paths = (
            repo_root / ".dockerignore",
            repo_root / "production" / "Dockerfile.dockerignore",
        )
        for ignore_path in ignore_paths:
            rules = {
                line.strip()
                for line in ignore_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip() and not line.startswith("#")
            }
            with self.subTest(ignore_path=ignore_path):
                self.assertIn("**", rules)
                self.assertIn("!production/**", rules)
                self.assertIn("!model/**", rules)
                self.assertIn("!utils/**", rules)
                self.assertIn("**/.env", rules)
                self.assertIn("**/.env.*", rules)
                self.assertNotIn("!dataset/**", rules)
                self.assertNotIn("!artifacts/**", rules)
                self.assertNotIn("!runs/**", rules)
                self.assertNotIn("!exp/**", rules)

    def test_docker_uses_pinned_production_inference_requirements(self):
        repo_root = Path(__file__).resolve().parents[2]
        dockerfile = (
            repo_root / "production" / "Dockerfile"
        ).read_text(encoding="utf-8")
        requirements = (
            repo_root / "production" / "requirements.txt"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "COPY production/requirements.txt /app/requirements.txt",
            dockerfile,
        )
        for training_dependency in (
            "pycocotools",
            "deepspeed",
            "gradio",
            "ray",
        ):
            self.assertNotIn(training_dependency, requirements.lower())
        package_lines = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.startswith("--")
        ]
        self.assertTrue(package_lines)
        self.assertTrue(all("==" in line for line in package_lines))

    def test_sanitize_inspect_drops_sources_and_environment(self):
        result = sanitize_container_inspect(
            {
                "Id": "abcdef1234567890",
                "Name": "/lisa-test",
                "Config": {
                    "Image": "lisa:test",
                    "User": "lisa",
                    "Env": ["LISA_API_KEY=secret"],
                },
                "State": {
                    "Status": "running",
                    "Running": True,
                    "ExitCode": 0,
                    "Health": {"Status": "healthy"},
                },
                "Mounts": [
                    {
                        "Source": "/private/model",
                        "Destination": "/models/lisa",
                        "Type": "bind",
                        "RW": False,
                    }
                ],
            }
        )
        text = str(result)
        self.assertNotIn("secret", text)
        self.assertNotIn("/private/model", text)
        self.assertEqual(result["id"], "abcdef123456")
        self.assertTrue(result["mounts"][0]["read_only"])

    def test_log_issue_scan_reports_labels_without_echoing_values(self):
        result = find_log_issues(
            "api-secret /home/private/model CUDA out of memory",
            sensitive_values=["api-secret"],
            private_paths=["/home/private/model"],
        )
        self.assertTrue(result["cuda_oom"])
        self.assertEqual(
            result["sensitive_value_labels"],
            ["sensitive_value_1"],
        )
        self.assertEqual(result["private_path_labels"], ["private_path_1"])
        self.assertNotIn("api-secret", str(result))

    def test_acceptance_passes_expected_container_result(self):
        checks = build_checks()
        self.assertTrue(all(item["passed"] for item in checks.values()))

    def test_acceptance_rejects_oom_unready_and_memory_regression(self):
        cycles = [passing_cycle(1), passing_cycle(2)]
        cycles[1]["ready_after"] = {"status": "unavailable"}
        cycles[1]["metrics_after"]["cuda_oom_total"] = 1
        checks = build_checks(
            cycles=cycles,
            log_issues={
                "cuda_oom": True,
                "traceback": True,
                "error_log": True,
                "sensitive_value_labels": ["sensitive_value_1"],
                "private_path_labels": ["private_path_1"],
            },
            gpu_memory_mib={
                "peak": 37000,
                "remaining_at_peak": 3960,
                "post_stop_drift": 600,
            },
            forbidden_image_paths_present=["/app/production/.env"],
        )
        self.assertFalse(
            checks["service_ready_after_both_smokes"]["passed"]
        )
        self.assertFalse(checks["cuda_oom_total"]["passed"])
        self.assertFalse(checks["peak_gpu_memory_mib"]["passed"])
        self.assertFalse(checks["post_stop_memory_drift_mib"]["passed"])
        self.assertFalse(
            checks["container_logs_no_sensitive_values"]["passed"]
        )
        self.assertFalse(
            checks[
                "runtime_image_excludes_secrets_and_unrelated_files"
            ]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
