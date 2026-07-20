from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from .backend import LisaBackend, SegmentationResult
from .config import Settings
from .errors import (
    InferenceQueueFullError,
    InferenceQueueTimeoutError,
    InferenceTimeoutError,
    ModelNotReadyError,
)


@dataclass
class _InferenceJob:
    image: Any
    prompt: str
    future: asyncio.Future[SegmentationResult]
    started: asyncio.Event
    enqueued_at: float
    cancelled_before_start: bool = False


def _consume_future_exception(
    future: asyncio.Future[SegmentationResult],
) -> None:
    if future.cancelled():
        return
    future.exception()


class ModelRuntime:
    def __init__(
        self,
        settings: Settings,
        backend_factory: Callable[[Settings], LisaBackend] = LisaBackend,
    ):
        self.settings = settings
        self.backend = backend_factory(settings)
        self._load_lock = threading.Lock()
        self._queue: asyncio.Queue[_InferenceJob] = asyncio.Queue(
            maxsize=settings.max_queue_size
        )
        self._workers: list[asyncio.Task[None]] = []
        self.metrics = Counter()
        self.started_at = time.monotonic()

    @property
    def ready(self) -> bool:
        return self.backend.loaded

    def load(self) -> None:
        if self.ready:
            return
        with self._load_lock:
            if not self.ready:
                self.backend.load()
                self.metrics["model_loads_total"] += 1

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(
                self._gpu_worker(index),
                name=f"lisa-gpu-worker-{index}",
            )
            for index in range(self.settings.max_concurrency)
        ]

    async def shutdown(self) -> None:
        workers, self._workers = self._workers, []
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)

        while True:
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.cancel()
            self._queue.task_done()

    async def _gpu_worker(self, _index: int) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job.cancelled_before_start:
                    self.metrics["queue_cancelled_total"] += 1
                    continue

                self.metrics["requests_started_total"] += 1
                self.metrics["gpu_inference_in_flight"] += 1
                self.metrics["gpu_inference_in_flight_max"] = max(
                    self.metrics["gpu_inference_in_flight_max"],
                    self.metrics["gpu_inference_in_flight"],
                )
                self.metrics["queue_wait_seconds_total"] += (
                    time.monotonic() - job.enqueued_at
                )
                job.started.set()
                inference_started = time.monotonic()
                try:
                    if not self.ready:
                        await asyncio.to_thread(self.load)
                    result = await asyncio.to_thread(
                        self.backend.segment,
                        job.image,
                        job.prompt,
                    )
                except Exception as exc:
                    self.metrics["gpu_inference_failed_total"] += 1
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    self.metrics["gpu_inference_succeeded_total"] += 1
                    if not job.future.done():
                        job.future.set_result(result)
                finally:
                    self.metrics["gpu_inference_seconds_total"] += (
                        time.monotonic() - inference_started
                    )
                    self.metrics["gpu_inference_in_flight"] -= 1
            finally:
                self._queue.task_done()

    async def segment(self, image, prompt: str) -> SegmentationResult:
        await self.start()
        self.metrics["requests_received_total"] += 1

        loop = asyncio.get_running_loop()
        job = _InferenceJob(
            image=image,
            prompt=prompt,
            future=loop.create_future(),
            started=asyncio.Event(),
            enqueued_at=time.monotonic(),
        )
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            self.metrics["queue_rejected_total"] += 1
            raise InferenceQueueFullError(
                "inference queue is full"
            ) from exc

        try:
            await asyncio.wait_for(
                job.started.wait(),
                timeout=self.settings.queue_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if not job.started.is_set():
                job.cancelled_before_start = True
                job.future.cancel()
                self.metrics["queue_timeout_total"] += 1
                raise InferenceQueueTimeoutError(
                    "inference exceeded configured queue timeout"
                ) from exc
        except asyncio.CancelledError:
            if job.started.is_set():
                job.future.add_done_callback(_consume_future_exception)
            else:
                job.cancelled_before_start = True
                job.future.cancel()
            raise

        try:
            result = await asyncio.wait_for(
                asyncio.shield(job.future),
                timeout=self.settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            job.future.add_done_callback(_consume_future_exception)
            self.metrics["requests_timeout_total"] += 1
            raise InferenceTimeoutError(
                "inference exceeded configured timeout"
            ) from exc
        except asyncio.CancelledError:
            job.future.add_done_callback(_consume_future_exception)
            raise
        except Exception:
            self.metrics["requests_failed_total"] += 1
            raise

        self.metrics["requests_succeeded_total"] += 1
        self.metrics["masks_returned_total"] += len(result.masks)
        return result

    def require_ready(self) -> None:
        if not self.ready:
            raise ModelNotReadyError("model has not finished loading")

    def metrics_snapshot(self) -> dict[str, float | int | bool]:
        return {
            "ready": self.ready,
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
            "queue_size": self._queue.qsize(),
            "queue_capacity": self.settings.max_queue_size,
            "gpu_worker_count": len(self._workers),
            **dict(self.metrics),
        }
