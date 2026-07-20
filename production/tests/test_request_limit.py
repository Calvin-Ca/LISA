import asyncio
import json
import unittest

from production.request_limit import RequestBodyLimitMiddleware


def make_scope(*headers):
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/segment",
        "headers": list(headers),
        "state": {},
    }


def run_middleware(middleware, scope, messages):
    sent = []
    queue = list(messages)

    async def receive():
        if queue:
            return queue.pop(0)
        return {
            "type": "http.request",
            "body": b"",
            "more_body": False,
        }

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


class RequestBodyLimitTest(unittest.TestCase):
    def test_rejects_declared_oversized_body_without_reading_it(self):
        app_called = False
        rejected = 0

        async def app(_scope, _receive, _send):
            nonlocal app_called
            app_called = True

        def on_rejected():
            nonlocal rejected
            rejected += 1

        middleware = RequestBodyLimitMiddleware(
            app,
            max_bytes=10,
            on_rejected=on_rejected,
        )
        sent = run_middleware(
            middleware,
            make_scope(
                (b"content-length", b"11"),
                (b"x-request-id", b"oversized-001"),
            ),
            [],
        )

        self.assertFalse(app_called)
        self.assertEqual(rejected, 1)
        self.assertEqual(sent[0]["status"], 413)
        self.assertIn((b"connection", b"close"), sent[0]["headers"])
        payload = json.loads(sent[1]["body"])
        self.assertEqual(payload["request_id"], "oversized-001")
        self.assertEqual(payload["code"], "request_too_large")

    def test_rejects_chunked_body_after_cumulative_limit(self):
        rejected = 0

        async def app(_scope, receive, _send):
            while True:
                message = await receive()
                if not message.get("more_body", False):
                    break

        def on_rejected():
            nonlocal rejected
            rejected += 1

        middleware = RequestBodyLimitMiddleware(
            app,
            max_bytes=10,
            on_rejected=on_rejected,
        )
        sent = run_middleware(
            middleware,
            make_scope((b"transfer-encoding", b"chunked")),
            [
                {
                    "type": "http.request",
                    "body": b"123456",
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": b"78901",
                    "more_body": False,
                },
            ],
        )

        self.assertEqual(rejected, 1)
        self.assertEqual(sent[0]["status"], 413)

    def test_allows_body_equal_to_limit(self):
        received = b""

        async def app(_scope, receive, send):
            nonlocal received
            while True:
                message = await receive()
                received += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )

        middleware = RequestBodyLimitMiddleware(app, max_bytes=10)
        sent = run_middleware(
            middleware,
            make_scope((b"content-length", b"10")),
            [
                {
                    "type": "http.request",
                    "body": b"12345",
                    "more_body": True,
                },
                {
                    "type": "http.request",
                    "body": b"67890",
                    "more_body": False,
                },
            ],
        )

        self.assertEqual(received, b"1234567890")
        self.assertEqual(sent[0]["status"], 204)


if __name__ == "__main__":
    unittest.main()
