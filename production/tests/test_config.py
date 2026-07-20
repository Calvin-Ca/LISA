import os
import unittest
from unittest.mock import patch

from production.config import Settings


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.model_version, "lisa13b-clean030-v1")
        self.assertEqual(settings.precision, "bf16")
        self.assertEqual(settings.max_concurrency, 1)
        self.assertEqual(settings.max_queue_size, 8)
        self.assertEqual(settings.queue_timeout_seconds, 30.0)
        self.assertTrue(settings.eager_load)

    def test_quantization_requires_fp16(self):
        env = {
            "LISA_PRECISION": "bf16",
            "LISA_LOAD_IN_8BIT": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_quantization_modes_are_mutually_exclusive(self):
        env = {
            "LISA_PRECISION": "fp16",
            "LISA_LOAD_IN_8BIT": "true",
            "LISA_LOAD_IN_4BIT": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                Settings.from_env()

    def test_timeouts_must_be_positive(self):
        with patch.dict(
            os.environ,
            {"LISA_QUEUE_TIMEOUT_SECONDS": "0"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()

        with patch.dict(
            os.environ,
            {"LISA_REQUEST_TIMEOUT_SECONDS": "-1"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
