from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SegmentRequest(BaseModel):
    image_base64: str = Field(..., min_length=4)
    prompt: str = Field(..., min_length=1)
    request_id: Optional[str] = Field(default=None, max_length=128)


class MaskPayload(BaseModel):
    index: int
    format: str = "png_base64"
    data: str


class SegmentResponse(BaseModel):
    request_id: str
    model_version: str
    width: int
    height: int
    has_segmentation: bool
    mask_count: int
    masks: List[MaskPayload]
    text: str
    latency_ms: float


class ErrorPayload(BaseModel):
    request_id: Optional[str] = None
    code: str
    message: str

