from __future__ import annotations

import base64
import binascii

from .errors import InvalidRequestError


def decode_image_base64(
    encoded: str,
    *,
    max_image_bytes: int,
    max_image_pixels: int,
):
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

    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidRequestError("decoded payload is not a supported image")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise InvalidRequestError("image dimensions are invalid")
    if height * width > max_image_pixels:
        raise InvalidRequestError(
            f"image exceeds {max_image_pixels} decoded pixels"
        )
    return image


def encode_png_base64(mask) -> str:
    import cv2

    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("failed to encode output mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")

