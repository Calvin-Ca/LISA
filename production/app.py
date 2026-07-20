from __future__ import annotations

import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import ServiceError
from .image_io import decode_image_base64
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

    async def authenticate(
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if settings.api_key is None:
            return
        if x_api_key is None or not secrets.compare_digest(
            x_api_key, settings.api_key
        ):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.exception_handler(ServiceError)
    async def handle_service_error(
        request: Request, exc: ServiceError
    ) -> JSONResponse:
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
        payload = ErrorPayload(
            request_id=getattr(request.state, "request_id", None),
            code="internal_error",
            message="internal server error",
        )
        return JSONResponse(status_code=500, content=payload.dict())

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = (
            request.headers.get("x-request-id") or str(uuid.uuid4())
        )[:128]
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        response.headers["x-model-version"] = settings.model_version
        return response

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
            **runtime.metrics_snapshot(),
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

        image = decode_image_base64(
            payload.image_base64,
            max_image_bytes=settings.max_image_bytes,
            max_image_pixels=settings.max_image_pixels,
        )
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
