import base64
import unittest
from pathlib import Path

from production.benchmark_api import (
    PNG_SIGNATURE,
    percentile,
    portable_path,
    summarize_phase,
    validate_segment_response,
)
from production.summarize_api_benchmarks import aggregate_summaries


class BenchmarkApiTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertAlmostEqual(
            percentile([100.0, 200.0, 300.0], 95),
            290.0,
        )

    def test_summarize_phase_excludes_failed_latency(self):
        rows = [
            {
                "phase": "measured",
                "success": True,
                "client_latency_ms": 100.0,
                "server_latency_ms": 80.0,
            },
            {
                "phase": "measured",
                "success": True,
                "client_latency_ms": 200.0,
                "server_latency_ms": 160.0,
            },
            {
                "phase": "measured",
                "success": False,
                "client_latency_ms": 10.0,
                "server_latency_ms": 0.0,
            },
        ]
        summary = summarize_phase(rows, "measured", 2.0)
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["succeeded"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["client_latency_p50_ms"], 150.0)
        self.assertEqual(summary["throughput_requests_per_second"], 1.0)

    def test_validate_segment_response_accepts_png_mask(self):
        encoded = base64.b64encode(PNG_SIGNATURE + b"payload").decode("ascii")
        validate_segment_response(
            {
                "request_id": "request-1",
                "width": 32,
                "height": 16,
                "has_segmentation": True,
                "mask_count": 1,
                "masks": [{"format": "png_base64", "data": encoded}],
            },
            "request-1",
        )

    def test_validate_segment_response_rejects_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "mask_count"):
            validate_segment_response(
                {
                    "request_id": "request-1",
                    "width": 32,
                    "height": 16,
                    "has_segmentation": False,
                    "mask_count": 1,
                    "masks": [],
                },
                "request-1",
            )

    def test_portable_path_removes_private_cache_prefix(self):
        path = Path(
            "/home/user/.cache/huggingface/hub/"
            "models--openai--clip-vit-large-patch14/"
            "snapshots/abc123/config.json"
        ).parent
        self.assertEqual(
            portable_path(path, Path("/workspace/repo")),
            "huggingface://openai/clip-vit-large-patch14@abc123",
        )

    def test_aggregate_summaries_keeps_worst_shared_gpu_metrics(self):
        def make_summary(p95, peak, remaining, passed):
            return {
                "created_at": "2026-07-20T00:00:00+00:00",
                "repo_git_commit": "deadbeef",
                "model_version": "test-v1",
                "model_path": "artifacts/test/merged_hf",
                "shared_gpu": True,
                "ready_time_ms": 1000.0,
                "existing_compute_processes_before": [
                    {
                        "pid": 1,
                        "process_name": "VLLM::EngineCore",
                        "used_memory_mib": 2000,
                    }
                ],
                "gpu_memory_mib": {
                    "baseline": 2000,
                    "peak": peak,
                    "remaining_at_peak": remaining,
                    "peak_increment_over_baseline": peak - 2000,
                    "post_warmup_drift": 10,
                },
                "phases": {
                    "measured": {
                        "client_latency_p50_ms": p95 - 100,
                        "client_latency_p95_ms": p95,
                        "client_latency_p99_ms": p95 + 100,
                        "throughput_requests_per_second": 1.0,
                    }
                },
                "total_requests": 136,
                "total_failed": 0,
                "acceptance": {"passed": passed},
            }

        aggregate = aggregate_summaries(
            [
                make_summary(900.0, 32000, 8960, True),
                make_summary(1100.0, 34000, 6960, True),
                make_summary(1000.0, 33000, 7960, True),
            ]
        )
        self.assertTrue(aggregate["acceptance"]["passed"])
        self.assertEqual(aggregate["measured_client_p95_ms"]["max"], 1100.0)
        self.assertEqual(aggregate["gpu_memory_mib"]["max_peak"], 34000)
        self.assertEqual(
            aggregate["gpu_memory_mib"]["min_remaining_at_peak"],
            6960,
        )
        self.assertEqual(aggregate["total_requests"], 408)


if __name__ == "__main__":
    unittest.main()
