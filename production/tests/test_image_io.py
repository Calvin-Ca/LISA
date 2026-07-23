import base64
import sys
import unittest
from unittest.mock import patch

from production.errors import InvalidRequestError
from production.image_io import (
    PNG_SIGNATURE,
    decode_image_base64,
    decode_image_base64_with_metadata,
    detect_image_format,
    encoded_image_dimensions,
)


def png_header(width: int, height: int) -> bytes:
    return (
        PNG_SIGNATURE
        + b"\x00\x00\x00\x0d"
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def jpeg_header(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + b"\x00\x07"
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
    )


class FakeImage:
    shape = (2, 3, 3)


class FakeCv2:
    IMREAD_COLOR = 1

    @staticmethod
    def imdecode(_raw, _mode):
        return FakeImage()


class FakeNumpy:
    uint8 = "uint8"

    @staticmethod
    def frombuffer(raw, dtype):
        return raw, dtype


class ImageIOTest(unittest.TestCase):
    def test_detects_only_png_and_jpeg(self):
        self.assertEqual(detect_image_format(png_header(3, 2)), "png")
        self.assertEqual(detect_image_format(jpeg_header(3, 2)), "jpeg")
        with self.assertRaises(InvalidRequestError):
            detect_image_format(b"GIF89a")
        with self.assertRaises(InvalidRequestError):
            detect_image_format(b"RIFF\x00\x00\x00\x00WEBP")

    def test_reads_png_and_jpeg_dimensions_before_decode(self):
        self.assertEqual(
            encoded_image_dimensions(png_header(320, 240), "png"),
            (320, 240),
        )
        self.assertEqual(
            encoded_image_dimensions(jpeg_header(640, 480), "jpeg"),
            (640, 480),
        )

    def test_rejects_oversized_header_before_opencv_import(self):
        encoded = base64.b64encode(png_header(1000, 1000)).decode("ascii")
        with patch.dict(sys.modules, {"cv2": None, "numpy": None}):
            with self.assertRaises(InvalidRequestError) as context:
                decode_image_base64(
                    encoded,
                    max_image_bytes=1024,
                    max_image_pixels=100,
                )
        self.assertIn("decoded pixels", str(context.exception))

    def test_rejects_encoded_body_before_base64_decode(self):
        encoded = "A" * 100
        with self.assertRaises(InvalidRequestError) as context:
            decode_image_base64(
                encoded,
                max_image_bytes=3,
                max_image_pixels=100,
            )
        self.assertIn("decoded image exceeds", str(context.exception))

    def test_valid_png_reaches_decoder(self):
        encoded = base64.b64encode(png_header(3, 2)).decode("ascii")
        with patch.dict(
            sys.modules,
            {"cv2": FakeCv2(), "numpy": FakeNumpy()},
        ):
            image = decode_image_base64(
                encoded,
                max_image_bytes=1024,
                max_image_pixels=100,
            )
        self.assertEqual(image.shape, (2, 3, 3))

    def test_decoded_image_metadata_preserves_original_bytes(self):
        raw = png_header(3, 2)
        encoded = base64.b64encode(raw).decode("ascii")
        with patch.dict(
            sys.modules,
            {"cv2": FakeCv2(), "numpy": FakeNumpy()},
        ):
            decoded = decode_image_base64_with_metadata(
                encoded,
                max_image_bytes=1024,
                max_image_pixels=100,
            )
        self.assertIs(decoded.image.__class__, FakeImage)
        self.assertEqual(decoded.raw, raw)
        self.assertEqual(decoded.image_format, "png")
        self.assertEqual((decoded.width, decoded.height), (3, 2))

    def test_invalid_base64_and_corrupt_headers_are_rejected(self):
        with self.assertRaises(InvalidRequestError):
            decode_image_base64(
                "not-base64!",
                max_image_bytes=1024,
                max_image_pixels=100,
            )
        corrupt_png = base64.b64encode(PNG_SIGNATURE).decode("ascii")
        with self.assertRaises(InvalidRequestError):
            decode_image_base64(
                corrupt_png,
                max_image_bytes=1024,
                max_image_pixels=100,
            )


if __name__ == "__main__":
    unittest.main()
