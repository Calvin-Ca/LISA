from __future__ import annotations

import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import Settings
from .errors import ServiceError
from .image_io import decode_image_base64
from .observability import (
    HttpMetrics,
    evaluate_alerts,
    render_prometheus,
)
from .runtime import ModelRuntime
from .schemas import ErrorPayload, MaskPayload, SegmentRequest, SegmentResponse


def create_app(
    settings: Settings | None = None,
    runtime: ModelRuntime | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    runtime = runtime or ModelRuntime(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
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

    def monitoring_snapshot() -> dict[str, float | int | bool]:
        return {
            **runtime.metrics_snapshot(),
            **http_metrics.snapshot(),
        }

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
        image = decode_image_base64(
            payload.image_base64,
            max_image_bytes=settings.max_image_bytes,
            max_image_pixels=settings.max_image_pixels,
        )
        request.state.validation_stage = "runtime"
        result = await runtime.segment(image, prompt)
        latency_ms = (time.perf_counter() - started) * 1000
        return SegmentResponse(
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
