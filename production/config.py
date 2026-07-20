from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _get_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_positive_float(name: str, default: float) -> float:
    value = _get_float(name, default)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _get_ratio(name: str, default: float) -> float:
    value = _get_float(name, default)
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    model_version: str
    model_path: str
    vision_tower: str
    precision: str
    load_in_8bit: bool
    load_in_4bit: bool
    gpu_index: int
    image_size: int
    model_max_length: int
    max_new_tokens: int
    mask_threshold: float
    max_image_bytes: int
    max_image_pixels: int
    max_prompt_chars: int
    max_concurrency: int
    max_queue_size: int
    queue_timeout_seconds: float
    request_timeout_seconds: float
    metrics_window_size: int
    alert_minimum_requests: int
    alert_max_4xx_rate: float
    alert_max_5xx_rate: float
    alert_max_p95_latency_ms: float
    alert_max_queue_utilization: float
    eager_load: bool
    api_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        precision = os.getenv("LISA_PRECISION", "bf16").strip().lower()
        if precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("LISA_PRECISION must be one of: bf16, fp16, fp32")

        load_in_8bit = _get_bool("LISA_LOAD_IN_8BIT", False)
        load_in_4bit = _get_bool("LISA_LOAD_IN_4BIT", False)
        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one quantization mode may be enabled")
        if (load_in_8bit or load_in_4bit) and precision != "fp16":
            raise ValueError("8-bit and 4-bit inference require LISA_PRECISION=fp16")

        api_key = os.getenv("LISA_API_KEY")
        if api_key is not None:
            api_key = api_key.strip() or None

        return cls(
            model_version=os.getenv(
                "LISA_MODEL_VERSION", "lisa13b-clean030-v1"
            ).strip(),
            model_path=os.getenv(
                "LISA_MODEL_PATH",
                "./runs/lisa13b-clean030-lora-v1/merged_hf",
            ).strip(),
            vision_tower=os.getenv(
                "LISA_VISION_TOWER", "./clip-vit-large-patch14"
            ).strip(),
            precision=precision,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            gpu_index=_get_int("LISA_GPU_INDEX", 0, minimum=0),
            image_size=_get_int("LISA_IMAGE_SIZE", 1024),
            model_max_length=_get_int("LISA_MODEL_MAX_LENGTH", 512),
            max_new_tokens=_get_int("LISA_MAX_NEW_TOKENS", 512),
            mask_threshold=_get_float("LISA_MASK_THRESHOLD", 0.0),
            max_image_bytes=_get_int("LISA_MAX_IMAGE_BYTES", 20 * 1024 * 1024),
            max_image_pixels=_get_int("LISA_MAX_IMAGE_PIXELS", 25_000_000),
            max_prompt_chars=_get_int("LISA_MAX_PROMPT_CHARS", 1000),
            max_concurrency=_get_int("LISA_MAX_CONCURRENCY", 1),
            max_queue_size=_get_int("LISA_MAX_QUEUE_SIZE", 8),
            queue_timeout_seconds=_get_positive_float(
                "LISA_QUEUE_TIMEOUT_SECONDS", 30.0
            ),
            request_timeout_seconds=_get_positive_float(
                "LISA_REQUEST_TIMEOUT_SECONDS", 120.0
            ),
            metrics_window_size=_get_int(
                "LISA_METRICS_WINDOW_SIZE", 1000
            ),
            alert_minimum_requests=_get_int(
                "LISA_ALERT_MINIMUM_REQUESTS", 20
            ),
            alert_max_4xx_rate=_get_ratio(
                "LISA_ALERT_MAX_4XX_RATE", 0.2
            ),
            alert_max_5xx_rate=_get_ratio(
                "LISA_ALERT_MAX_5XX_RATE", 0.01
            ),
            alert_max_p95_latency_ms=_get_positive_float(
                "LISA_ALERT_MAX_P95_LATENCY_MS", 2000.0
            ),
            alert_max_queue_utilization=_get_ratio(
                "LISA_ALERT_MAX_QUEUE_UTILIZATION", 0.8
            ),
            eager_load=_get_bool("LISA_EAGER_LOAD", True),
            api_key=api_key,
        )
