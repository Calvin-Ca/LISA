import unittest

from production.backend import LisaBackend
from production.errors import CudaOutOfMemoryError, InferenceError
from production.tests.test_runtime import make_settings


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self):
        self.empty_cache_calls = 0

    def device(self, _index):
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def empty_cache(self):
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


class BackendErrorTest(unittest.TestCase):
    def make_loaded_backend(self):
        backend = LisaBackend(make_settings())
        backend.loaded = True
        backend._objects["torch"] = FakeTorch()
        return backend

    def test_cuda_oom_has_dedicated_sanitized_error(self):
        backend = self.make_loaded_backend()

        def raise_oom(_image, _prompt):
            raise backend._objects["torch"].cuda.OutOfMemoryError(
                "CUDA out of memory at /private/model/path"
            )

        backend._segment = raise_oom
        with self.assertRaises(CudaOutOfMemoryError) as context:
            backend.segment(object(), "secret prompt")
        self.assertEqual(
            str(context.exception),
            "GPU memory was exhausted during inference",
        )
        self.assertNotIn("/private/model/path", str(context.exception))
        self.assertIsNone(context.exception.__cause__)
        self.assertTrue(context.exception.__suppress_context__)

    def test_generic_backend_error_does_not_expose_internal_details(self):
        backend = self.make_loaded_backend()

        def raise_generic(_image, _prompt):
            raise RuntimeError(
                "private token and /home/user/private/model/path"
            )

        backend._segment = raise_generic
        with self.assertRaises(InferenceError) as context:
            backend.segment(object(), "secret prompt")
        self.assertEqual(str(context.exception), "LISA inference failed")
        self.assertNotIn("private token", str(context.exception))
        self.assertNotIn("/home/user", str(context.exception))
        self.assertIsNone(context.exception.__cause__)
        self.assertTrue(context.exception.__suppress_context__)

    def test_cuda_oom_recovery_clears_cache_once(self):
        backend = self.make_loaded_backend()
        self.assertTrue(backend.recover_from_cuda_oom())
        self.assertEqual(
            backend._objects["torch"].cuda.empty_cache_calls,
            1,
        )


if __name__ == "__main__":
    unittest.main()
