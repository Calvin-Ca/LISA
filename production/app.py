from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from .config import Settings
from .errors import (
    RecordStorageError,
    RecordsUnavailableError,
    ServiceError,
)
from .image_io import decode_image_base64_with_metadata
from .observability import (
    HttpMetrics,
    evaluate_alerts,
    render_prometheus,
)
from .request_limit import RequestBodyLimitMiddleware
from .records import RecordStore
from .runtime import ModelRuntime
from .schemas import (
    ErrorPayload,
    FeedbackRequest,
    FeedbackResponse,
    MaskPayload,
    SegmentationRecordList,
    SegmentationRecordPayload,
    SegmentRequest,
    SegmentResponse,
)


def create_app(
    settings: Settings | None = None,
    runtime: ModelRuntime | None = None,
    record_store: RecordStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    runtime = runtime or ModelRuntime(settings)
    if record_store is None and settings.records_enabled:
        record_store = RecordStore(settings.records_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if record_store is not None:
            await asyncio.to_thread(record_store.initialize)
        if settings.eager_load:
            runtime.load()
        await runtime.start()
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(
        title="LISA Safety Segmentation API",
        version=settings.model_version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.runtime = runtime
    http_metrics = HttpMetrics(settings.metrics_window_size)
    app.state.http_metrics = http_metrics
    app.state.record_store = record_store
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=settings.max_request_bytes,
        on_rejected=lambda: http_metrics.increment(
            "request_body_too_large_total"
        ),
    )

    def monitoring_snapshot() -> dict[str, float | int | bool]:
        record_metrics: dict[str, float | int | bool]
        if record_store is None:
            record_metrics = {"records_enabled": False}
        else:
            record_metrics = record_store.metrics_snapshot()
        return {
            **runtime.metrics_snapshot(),
            **http_metrics.snapshot(),
            **record_metrics,
        }

    def require_record_store() -> RecordStore:
        if record_store is None:
            raise RecordsUnavailableError(
                "segmentation record storage is disabled"
            )
        return record_store

    def utc_datetime(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    async def mark_record_failed(
        record_id: str,
        *,
        code: str,
        message: str,
    ) -> None:
        store = require_record_store()
        try:
            await asyncio.to_thread(
                store.mark_failed,
                record_id,
                code=code,
                message=message,
            )
        except ServiceError:
            raise RecordStorageError(
                "failed to persist segmentation failure"
            ) from None

    async def authenticate(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        if settings.api_key is None:
            return
        candidate = x_api_key
        if candidate is None and authorization is not None:
            scheme, separator, token = authorization.partition(" ")
            if (
                separator
                and scheme.lower() == "bearer"
                and token
            ):
                candidate = token
        if candidate is None or not secrets.compare_digest(
            candidate, settings.api_key
        ):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.exception_handler(ServiceError)
    async def handle_service_error(
        request: Request, exc: ServiceError
    ) -> JSONResponse:
        stage = getattr(request.state, "validation_stage", "runtime")
        if exc.code == "invalid_request" and stage == "prompt":
            http_metrics.increment("prompt_validation_failed_total")
        elif exc.code == "invalid_request" and stage == "image":
            http_metrics.increment("image_validation_failed_total")
        else:
            http_metrics.increment("service_errors_total")
        request_id = getattr(request.state, "request_id", None)
        payload = ErrorPayload(
            request_id=request_id,
            code=exc.code,
            message=exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.dict(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, _: RequestValidationError
    ) -> JSONResponse:
        http_metrics.increment("request_validation_failed_total")
        payload = ErrorPayload(
            request_id=getattr(request.state, "request_id", None),
            code="validation_error",
            message="request payload is invalid",
        )
        return JSONResponse(status_code=422, content=payload.dict())

    @app.exception_handler(HTTPException)
    async def handle_http_error(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        if exc.status_code == 401:
            http_metrics.increment("authentication_failed_total")
        else:
            http_metrics.increment("http_exception_total")
        payload = ErrorPayload(
            request_id=getattr(request.state, "request_id", None),
            code="unauthorized" if exc.status_code == 401 else "http_error",
            message=str(exc.detail),
        )
        return JSONResponse(status_code=exc.status_code, content=payload.dict())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, _: Exception
    ) -> JSONResponse:
        runtime.metrics["unexpected_errors_total"] += 1
        http_metrics.increment("unexpected_http_errors_total")
        payload = ErrorPayload(
            request_id=getattr(request.state, "request_id", None),
            code="internal_error",
            message="internal server error",
        )
        return JSONResponse(status_code=500, content=payload.dict())

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        tracked = request.url.path == "/v1/segment"
        started = time.perf_counter()
        response = None
        if tracked:
            http_metrics.begin_request()
        request.state.request_id = (
            request.headers.get("x-request-id") or str(uuid.uuid4())
        )[:128]
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request.state.request_id
            response.headers["x-model-version"] = settings.model_version
            return response
        finally:
            if tracked:
                status_code = (
                    response.status_code if response is not None else 500
                )
                http_metrics.finish_request(
                    status_code,
                    (time.perf_counter() - started) * 1000,
                )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        status_code = 200 if runtime.ready else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": runtime.readiness_status,
                "model_version": settings.model_version,
            },
        )

    @app.get("/metrics", dependencies=[Depends(authenticate)])
    async def metrics() -> dict[str, float | int | bool | str]:
        return {
            "model_version": settings.model_version,
            **monitoring_snapshot(),
        }

    @app.get(
        "/metrics/prometheus",
        dependencies=[Depends(authenticate)],
        response_class=PlainTextResponse,
    )
    async def prometheus_metrics() -> str:
        return render_prometheus(
            monitoring_snapshot(),
            model_version=settings.model_version,
        )

    @app.get("/alerts", dependencies=[Depends(authenticate)])
    async def alerts() -> dict:
        return {
            "model_version": settings.model_version,
            **evaluate_alerts(
                monitoring_snapshot(),
                minimum_requests=settings.alert_minimum_requests,
                max_4xx_rate=settings.alert_max_4xx_rate,
                max_5xx_rate=settings.alert_max_5xx_rate,
                max_p95_latency_ms=(
                    settings.alert_max_p95_latency_ms
                ),
                max_queue_utilization=(
                    settings.alert_max_queue_utilization
                ),
            ),
        }

    @app.get(
        "/v1/records",
        response_model=SegmentationRecordList,
        dependencies=[Depends(authenticate)],
    )
    async def list_records(
        status: Literal["processing", "success", "failed"] | None = None,
        feedback: Literal["like", "dislike", "unrated"] | None = None,
        model_version: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        store = require_record_store()
        return await asyncio.to_thread(
            store.list_records,
            status=status,
            feedback=feedback,
            model_version=model_version,
            date_from=utc_datetime(date_from),
            date_to=utc_datetime(date_to),
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/v1/records/{record_id}",
        response_model=SegmentationRecordPayload,
        dependencies=[Depends(authenticate)],
    )
    async def get_record(record_id: str) -> dict:
        store = require_record_store()
        return await asyncio.to_thread(store.get_record, record_id)

    @app.get(
        "/v1/records/{record_id}/image",
        dependencies=[Depends(authenticate)],
        response_class=FileResponse,
    )
    async def get_record_image(record_id: str) -> FileResponse:
        store = require_record_store()
        path, media_type = await asyncio.to_thread(
            store.original_image, record_id
        )
        return FileResponse(path, media_type=media_type)

    @app.get(
        "/v1/records/{record_id}/masks/{mask_index}",
        dependencies=[Depends(authenticate)],
        response_class=FileResponse,
    )
    async def get_record_mask(
        record_id: str,
        mask_index: int,
    ) -> FileResponse:
        store = require_record_store()
        path = await asyncio.to_thread(
            store.mask_image, record_id, mask_index
        )
        return FileResponse(path, media_type="image/png")

    @app.put(
        "/v1/records/{record_id}/feedback",
        response_model=FeedbackResponse,
        dependencies=[Depends(authenticate)],
    )
    async def update_feedback(
        record_id: str,
        payload: FeedbackRequest,
    ) -> dict:
        store = require_record_store()
        comment = payload.comment
        if comment is not None and len(comment) > settings.feedback_comment_max_chars:
            from .errors import InvalidRequestError

            raise InvalidRequestError(
                "feedback comment exceeds "
                f"{settings.feedback_comment_max_chars} characters"
            )
        return await asyncio.to_thread(
            store.update_feedback,
            record_id,
            feedback=payload.feedback,
            reason=payload.reason,
            comment=comment,
        )

    @app.post(
        "/v1/segment",
        response_model=SegmentResponse,
        dependencies=[Depends(authenticate)],
    )
    async def segment(
        payload: SegmentRequest,
        request: Request,
    ) -> SegmentResponse:
        started = time.perf_counter()
        request.state.validation_stage = "prompt"
        request_id = payload.request_id or request.state.request_id
        request.state.request_id = request_id
        prompt = payload.prompt.strip()
        if not prompt:
            from .errors import InvalidRequestError

            raise InvalidRequestError("prompt must not be blank")
        if len(prompt) > settings.max_prompt_chars:
            from .errors import InvalidRequestError

            raise InvalidRequestError(
                f"prompt exceeds {settings.max_prompt_chars} characters"
            )

        request.state.validation_stage = "image"
        decoded = decode_image_base64_with_metadata(
            payload.image_base64,
            max_image_bytes=settings.max_image_bytes,
            max_image_pixels=settings.max_image_pixels,
        )
        record_id: str | None = None
        if record_store is not None:
            record_id = await asyncio.to_thread(
                record_store.create_record,
                request_id=request_id,
                model_version=settings.model_version,
                prompt=prompt,
                image_raw=decoded.raw,
                image_format=decoded.image_format,
                width=decoded.width,
                height=decoded.height,
            )
        request.state.validation_stage = "runtime"
        try:
            result = await runtime.segment(decoded.image, prompt)
        except asyncio.CancelledError:
            if record_id is not None:
                await mark_record_failed(
                    record_id,
                    code="request_cancelled",
                    message="request was cancelled before completion",
                )
            raise
        except ServiceError as exc:
            if record_id is not None:
                await mark_record_failed(
                    record_id,
                    code=exc.code,
                    message=exc.message,
                )
            raise
        except Exception:
            if record_id is not None:
                await mark_record_failed(
                    record_id,
                    code="internal_error",
                    message="internal server error",
                )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        if record_id is not None:
            try:
                await asyncio.to_thread(
                    record_store.complete_record,
                    record_id,
                    masks=result.masks,
                    text_response=result.text,
                    latency_ms=latency_ms,
                )
            except ServiceError as exc:
                try:
                    await mark_record_failed(
                        record_id,
                        code=exc.code,
                        message=exc.message,
                    )
                except ServiceError:
                    pass
                raise
        return SegmentResponse(
            record_id=record_id,
            request_id=request_id,
            model_version=settings.model_version,
            width=result.width,
            height=result.height,
            has_segmentation=bool(result.masks),
            mask_count=len(result.masks),
            masks=[
                MaskPayload(index=index, data=data)
                for index, data in enumerate(result.masks)
            ],
            text=result.text,
            latency_ms=round(latency_ms, 3),
        )

    return app


app = create_app()
