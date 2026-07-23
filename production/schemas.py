from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, root_validator


class SegmentRequest(BaseModel):
    image_base64: str = Field(..., min_length=4)
    prompt: str = Field(..., min_length=1)
    request_id: Optional[str] = Field(default=None, max_length=128)


class MaskPayload(BaseModel):
    index: int
    format: str = "png_base64"
    data: str


class SegmentResponse(BaseModel):
    record_id: Optional[str] = None
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


FeedbackValue = Literal["like", "dislike"]
FeedbackReason = Literal[
    "wrong_object",
    "missing_target",
    "boundary_bad",
    "extra_area",
    "empty_mask",
    "instance_wrong",
    "prompt_mismatch",
    "other",
]


class FeedbackRequest(BaseModel):
    feedback: Optional[FeedbackValue] = None
    reason: Optional[FeedbackReason] = None
    comment: Optional[str] = None

    @root_validator(skip_on_failure=True)
    def validate_reason_and_comment(cls, values):
        feedback = values.get("feedback")
        reason = values.get("reason")
        comment = values.get("comment")
        if isinstance(comment, str):
            comment = comment.strip() or None
            values["comment"] = comment
        if feedback != "dislike" and (reason is not None or comment is not None):
            raise ValueError(
                "feedback reason and comment are only valid for dislike"
            )
        return values


class FeedbackResponse(BaseModel):
    record_id: str
    feedback: Optional[FeedbackValue] = None
    feedback_reason: Optional[FeedbackReason] = None
    feedback_comment: Optional[str] = None
    feedback_at: Optional[str] = None


class StoredMaskPayload(BaseModel):
    index: int
    url: str


class SegmentationRecordPayload(BaseModel):
    record_id: str
    request_id: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    status: Literal["processing", "success", "failed"]
    model_version: str
    prompt: str
    width: Optional[int] = None
    height: Optional[int] = None
    mask_count: int
    image_url: Optional[str] = None
    masks: List[StoredMaskPayload]
    text_response: Optional[str] = None
    latency_ms: Optional[float] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    feedback: Optional[FeedbackValue] = None
    feedback_reason: Optional[FeedbackReason] = None
    feedback_comment: Optional[str] = None
    feedback_at: Optional[str] = None


class SegmentationRecordList(BaseModel):
    total: int
    limit: int
    offset: int
    records: List[SegmentationRecordPayload]
