import asyncio
import unittest

from production.backend import SegmentationResult
from production.config import Settings
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
        max_concurrency=1,
        request_timeout_seconds=1.0,
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


class RuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_lazy_load_and_metrics(self):
        runtime = ModelRuntime(make_settings(), backend_factory=FakeBackend)
        self.assertFalse(runtime.ready)
        result = await runtime.segment(object(), "segment target")
        self.assertTrue(runtime.ready)
        self.assertEqual(result.masks, ["encoded-mask"])
        metrics = runtime.metrics_snapshot()
        self.assertEqual(metrics["requests_succeeded_total"], 1)
        self.assertEqual(metrics["masks_returned_total"], 1)


if __name__ == "__main__":
    unittest.main()

