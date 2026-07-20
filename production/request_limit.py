from __future__ import annotations

import json
from typing import Any, Awaitable, Callable


AsgiMessage = dict[str, Any]
AsgiReceive = Callable[[], Awaitable[AsgiMessage]]
AsgiSend = Callable[[AsgiMessage], Awaitable[None]]
AsgiApp = Callable[[dict[str, Any], AsgiReceive, AsgiSend], Awaitable[None]]


class _RequestBodyTooLarge(Exception):
    pass


def _content_lengths(scope: dict[str, Any]) -> list[int]:
    lengths: list[int] = []
    for name, value in scope.get("headers", []):
        if name.lower() != b"content-length":
            continue
        try:
            parsed = int(value.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            continue
        if parsed >= 0:
            lengths.append(parsed)
    return lengths


def _request_id(scope: dict[str, Any]) -> str | None:
    state = scope.get("state")
    if isinstance(state, dict):
        request_id = state.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id[:128]
    for name, value in scope.get("headers", []):
        if name.lower() == b"x-request-id":
            return value.decode("latin-1")[:128] or None
    return None


class RequestBodyLimitMiddleware:
    def __init__(
        self,
        app: AsgiApp,
        *,
        max_bytes: int,
        on_rejected: Callable[[], None] | None = None,
    ):
        if max_bytes < 1:
            raise ValueError("request body limit must be positive")
        self.app = app
        self.max_bytes = max_bytes
        self.on_rejected = on_rejected

    async def _reject(
        self,
        scope: dict[str, Any],
        send: AsgiSend,
    ) -> None:
        if self.on_rejected is not None:
            self.on_rejected()
        body = json.dumps(
            {
                "request_id": _request_id(scope),
                "code": "request_too_large",
                "message": "request body exceeds configured limit",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        if any(
            length > self.max_bytes
            for length in _content_lengths(scope)
        ):
            await self._reject(scope, send)
            return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> AsgiMessage:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def tracked_send(message: AsgiMessage) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(scope, send)
