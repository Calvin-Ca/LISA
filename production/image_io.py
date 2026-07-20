from __future__ import annotations

import base64
import binascii

from .errors import InvalidRequestError


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}
JPEG_STANDALONE_MARKERS = {
    0x01,
    0xD8,
    0xD9,
    *range(0xD0, 0xD8),
}


def detect_image_format(raw: bytes) -> str:
    if raw.startswith(PNG_SIGNATURE):
        return "png"
    if raw.startswith(JPEG_SIGNATURE):
        return "jpeg"
    raise InvalidRequestError("only JPEG and PNG images are supported")


def _png_dimensions(raw: bytes) -> tuple[int, int]:
    if len(raw) < 24 or raw[12:16] != b"IHDR":
        raise InvalidRequestError("PNG header is invalid")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return width, height


def _jpeg_dimensions(raw: bytes) -> tuple[int, int]:
    position = 2
    while position < len(raw):
        if raw[position] != 0xFF:
            position += 1
            continue
        while position < len(raw) and raw[position] == 0xFF:
            position += 1
        if position >= len(raw):
            break
        marker = raw[position]
        position += 1
        if marker in JPEG_STANDALONE_MARKERS:
            continue
        if position + 2 > len(raw):
            break
        segment_length = int.from_bytes(raw[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(raw):
            break
        if marker in JPEG_SOF_MARKERS:
            if segment_length < 7:
                break
            height = int.from_bytes(raw[position + 3 : position + 5], "big")
            width = int.from_bytes(raw[position + 5 : position + 7], "big")
            return width, height
        if marker == 0xDA:
            break
        position += segment_length
    raise InvalidRequestError("JPEG header does not contain valid dimensions")


def encoded_image_dimensions(
    raw: bytes, image_format: str
) -> tuple[int, int]:
    if image_format == "png":
        return _png_dimensions(raw)
    if image_format == "jpeg":
        return _jpeg_dimensions(raw)
    raise InvalidRequestError("only JPEG and PNG images are supported")


def _validate_dimensions(
    width: int,
    height: int,
    *,
    max_image_pixels: int,
) -> None:
    if height <= 0 or width <= 0:
        raise InvalidRequestError("image dimensions are invalid")
    if height * width > max_image_pixels:
        raise InvalidRequestError(
            f"image exceeds {max_image_pixels} decoded pixels"
        )


def decode_image_base64(
    encoded: str,
    *,
    max_image_bytes: int,
    max_image_pixels: int,
):
    max_encoded_chars = 4 * ((max_image_bytes + 2) // 3)
    if len(encoded) > max_encoded_chars:
        raise InvalidRequestError(
            f"decoded image exceeds {max_image_bytes} bytes"
        )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError("image_base64 is not valid base64") from exc

    if not raw:
        raise InvalidRequestError("image_base64 is empty")
    if len(raw) > max_image_bytes:
        raise InvalidRequestError(
            f"decoded image exceeds {max_image_bytes} bytes"
        )

    image_format = detect_image_format(raw)
    encoded_width, encoded_height = encoded_image_dimensions(
        raw, image_format
    )
    _validate_dimensions(
        encoded_width,
        encoded_height,
        max_image_pixels=max_image_pixels,
    )

    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidRequestError("decoded JPEG or PNG image is invalid")

    height, width = image.shape[:2]
    _validate_dimensions(
        width,
        height,
        max_image_pixels=max_image_pixels,
    )
    return image


def encode_png_base64(mask) -> str:
    import cv2

    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("failed to encode output mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
