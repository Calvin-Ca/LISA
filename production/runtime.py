from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from typing import Callable

from .backend import LisaBackend, SegmentationResult
from .config import Settings
from .errors import InferenceTimeoutError, ModelNotReadyError


class ModelRuntime:
    def __init__(
        self,
        settings: Settings,
        backend_factory: Callable[[Settings], LisaBackend] = LisaBackend,
    ):
        self.settings = settings
        self.backend = backend_factory(settings)
        self._load_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
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

    async def segment(self, image, prompt: str) -> SegmentationResult:
        async with self._semaphore:
            self.metrics["requests_started_total"] += 1
            try:
                if not self.ready:
                    await asyncio.to_thread(self.load)
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.backend.segment, image, prompt),
                    timeout=self.settings.request_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                self.metrics["requests_timeout_total"] += 1
                raise InferenceTimeoutError(
                    "inference exceeded configured timeout"
                ) from exc
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
            **dict(self.metrics),
        }

