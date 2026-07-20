import asyncio
import threading
import unittest
from dataclasses import replace

from production.backend import SegmentationResult
from production.config import Settings
from production.errors import (
    CudaOutOfMemoryError,
    InferenceQueueFullError,
    InferenceQueueTimeoutError,
    InferenceTimeoutError,
    ModelNotReadyError,
)
from production.runtime import ModelRuntime


def make_settings() -> Settings:
    return Settings(
        model_version="test",
        model_path="/models/test",
        vision_tower="/models/clip",
        precision="bf16",
        load_in_8bit=False,
        load_in_4bit=False,
        gpu_index=0,
        image_size=1024,
        model_max_length=512,
        max_new_tokens=32,
        mask_threshold=0.0,
        max_image_bytes=1024,
        max_image_pixels=1024,
        max_prompt_chars=100,
        max_request_bytes=2048,
        max_concurrency=1,
        max_queue_size=8,
        queue_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
        metrics_window_size=100,
        alert_minimum_requests=20,
        alert_max_4xx_rate=0.2,
        alert_max_5xx_rate=0.01,
        alert_max_p95_latency_ms=2000.0,
        alert_max_queue_utilization=0.8,
        eager_load=False,
        api_key=None,
    )


class FakeBackend:
    def __init__(self, _settings):
        self.loaded = False

    def load(self):
        self.loaded = True

    def segment(self, _image, _prompt):
        return SegmentationResult(
            width=2,
            height=2,
            text="[SEG]",
            masks=["encoded-mask"],
        )


class MaskCountBackend(FakeBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self.calls = 0

    def segment(self, _image, _prompt):
        self.calls += 1
        masks = [] if self.calls == 1 else ["mask-1", "mask-2"]
        return SegmentationResult(
            width=2,
            height=2,
            text="[SEG]",
            masks=masks,
        )


class BlockingBackend(FakeBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self._lock = threading.Lock()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    def segment(self, image, prompt):
        with self._lock:
            self.calls += 1
            call_number = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if call_number == 1:
                self.first_started.set()
                if not self.release_first.wait(timeout=2.0):
                    raise RuntimeError("test did not release first inference")
            return super().segment(image, prompt)
        finally:
            with self._lock:
                self.active -= 1


class OOMBackend(FakeBackend):
    def __init__(self, settings, recovery_succeeds=True):
        super().__init__(settings)
        self.recovery_succeeds = recovery_succeeds
        self.calls = 0
        self.recovery_calls = 0

    def segment(self, image, prompt):
        self.calls += 1
        if self.calls == 1:
            raise CudaOutOfMemoryError(
                "GPU memory was exhausted during inference"
            )
        return super().segment(image, prompt)

    def recover_from_cuda_oom(self):
        self.recovery_calls += 1
        return self.recovery_succeeds


class FailedRecoveryOOMBackend(OOMBackend):
    def __init__(self, settings):
        super().__init__(settings, recovery_succeeds=False)


async def wait_for_thread_event(
    event: threading.Event, timeout: float = 1.0
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for thread event")
        await asyncio.sleep(0.005)


class RuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_lazy_load_and_metrics(self):
        runtime = ModelRuntime(make_settings(), backend_factory=FakeBackend)
        try:
            self.assertFalse(runtime.ready)
            result = await runtime.segment(object(), "segment target")
            self.assertTrue(runtime.ready)
            self.assertEqual(result.masks, ["encoded-mask"])
            metrics = runtime.metrics_snapshot()
            self.assertEqual(metrics["requests_succeeded_total"], 1)
            self.assertEqual(metrics["masks_returned_total"], 1)
            self.assertEqual(metrics["gpu_inference_succeeded_total"], 1)
            self.assertEqual(metrics["queue_wait_ms_window_samples"], 1)
            self.assertEqual(
                metrics["gpu_inference_ms_window_samples"],
                1,
            )
        finally:
            await runtime.shutdown()

    async def test_empty_and_multi_mask_metrics(self):
        runtime = ModelRuntime(
            make_settings(),
            backend_factory=MaskCountBackend,
        )
        try:
            await runtime.segment(object(), "empty")
            await runtime.segment(object(), "multi")
            metrics = runtime.metrics_snapshot()
            self.assertEqual(metrics["empty_mask_responses_total"], 1)
            self.assertEqual(metrics["multi_mask_responses_total"], 1)
            self.assertEqual(metrics["masks_returned_total"], 2)
        finally:
            await runtime.shutdown()

    async def test_timeout_does_not_release_gpu_worker(self):
        settings = replace(
            make_settings(),
            request_timeout_seconds=0.05,
            queue_timeout_seconds=0.5,
        )
        runtime = ModelRuntime(settings, backend_factory=BlockingBackend)
        second = None
        try:
            with self.assertRaises(InferenceTimeoutError):
                await runtime.segment(object(), "first")
            await wait_for_thread_event(runtime.backend.first_started)

            second = asyncio.create_task(
                runtime.segment(object(), "second")
            )
            await asyncio.sleep(0.075)
            self.assertEqual(runtime.backend.calls, 1)
            self.assertEqual(runtime.backend.active, 1)

            runtime.backend.release_first.set()
            result = await asyncio.wait_for(second, timeout=1.0)
            self.assertEqual(result.masks, ["encoded-mask"])
            self.assertEqual(runtime.backend.max_active, 1)

            metrics = runtime.metrics_snapshot()
            self.assertEqual(metrics["requests_timeout_total"], 1)
            self.assertEqual(metrics["gpu_inference_succeeded_total"], 2)
            self.assertEqual(metrics["gpu_inference_in_flight_max"], 1)
        finally:
            runtime.backend.release_first.set()
            if second is not None and not second.done():
                second.cancel()
                await asyncio.gather(second, return_exceptions=True)
            await runtime.shutdown()

    async def test_queued_request_is_cancelled_after_queue_timeout(self):
        settings = replace(
            make_settings(),
            max_queue_size=1,
            queue_timeout_seconds=0.05,
        )
        runtime = ModelRuntime(settings, backend_factory=BlockingBackend)
        first = asyncio.create_task(runtime.segment(object(), "first"))
        try:
            await wait_for_thread_event(runtime.backend.first_started)
            with self.assertRaises(InferenceQueueTimeoutError):
                await runtime.segment(object(), "second")

            runtime.backend.release_first.set()
            await asyncio.wait_for(first, timeout=1.0)
            await asyncio.sleep(0.025)

            self.assertEqual(runtime.backend.calls, 1)
            metrics = runtime.metrics_snapshot()
            self.assertEqual(metrics["queue_timeout_total"], 1)
            self.assertEqual(metrics["queue_cancelled_total"], 1)
        finally:
            runtime.backend.release_first.set()
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
            await runtime.shutdown()

    async def test_full_queue_rejects_request(self):
        settings = replace(
            make_settings(),
            max_queue_size=1,
            queue_timeout_seconds=1.0,
        )
        runtime = ModelRuntime(settings, backend_factory=BlockingBackend)
        first = asyncio.create_task(runtime.segment(object(), "first"))
        second = None
        try:
            await wait_for_thread_event(runtime.backend.first_started)
            second = asyncio.create_task(
                runtime.segment(object(), "second")
            )
            while runtime.metrics_snapshot()["queue_size"] != 1:
                await asyncio.sleep(0.005)

            with self.assertRaises(InferenceQueueFullError):
                await runtime.segment(object(), "third")

            runtime.backend.release_first.set()
            await asyncio.wait_for(first, timeout=1.0)
            await asyncio.wait_for(second, timeout=1.0)
            self.assertEqual(runtime.backend.max_active, 1)
            self.assertEqual(
                runtime.metrics_snapshot()["queue_rejected_total"], 1
            )
        finally:
            runtime.backend.release_first.set()
            tasks = [task for task in (first, second) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await runtime.shutdown()

    async def test_cuda_oom_recovers_without_retrying_failed_request(self):
        runtime = ModelRuntime(make_settings(), backend_factory=OOMBackend)
        try:
            with self.assertRaises(CudaOutOfMemoryError):
                await runtime.segment(object(), "first")

            self.assertTrue(runtime.ready)
            self.assertEqual(runtime.readiness_status, "ready")
            self.assertEqual(runtime.backend.calls, 1)
            self.assertEqual(runtime.backend.recovery_calls, 1)
            self.assertFalse(runtime._workers[0].done())

            result = await runtime.segment(object(), "second")
            self.assertEqual(result.masks, ["encoded-mask"])
            self.assertEqual(runtime.backend.calls, 2)

            metrics = runtime.metrics_snapshot()
            self.assertEqual(metrics["cuda_oom_total"], 1)
            self.assertEqual(
                metrics["cuda_oom_recovery_succeeded_total"], 1
            )
            self.assertEqual(metrics["gpu_inference_failed_total"], 1)
            self.assertEqual(metrics["gpu_inference_succeeded_total"], 1)
        finally:
            await runtime.shutdown()

    async def test_failed_cuda_oom_recovery_marks_model_unavailable(self):
        runtime = ModelRuntime(
            make_settings(),
            backend_factory=FailedRecoveryOOMBackend,
        )
        try:
            with self.assertRaises(CudaOutOfMemoryError):
                await runtime.segment(object(), "first")

            self.assertFalse(runtime.ready)
            self.assertEqual(runtime.readiness_status, "unavailable")
            with self.assertRaises(ModelNotReadyError):
                await runtime.segment(object(), "second")
            self.assertEqual(runtime.backend.calls, 1)

            metrics = runtime.metrics_snapshot()
            self.assertEqual(metrics["cuda_oom_total"], 1)
            self.assertEqual(metrics["cuda_oom_recovery_failed_total"], 1)
            self.assertEqual(
                metrics["model_unavailable_rejected_total"], 1
            )
        finally:
            await runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
