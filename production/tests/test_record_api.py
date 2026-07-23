import base64
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    from production.app import create_app
    from production.backend import SegmentationResult
    from production.image_io import PNG_SIGNATURE
    from production.tests.test_runtime import make_settings
except ModuleNotFoundError:
    TestClient = None


def png_bytes(width: int = 3, height: int = 2) -> bytes:
    return (
        PNG_SIGNATURE
        + b"\x00\x00\x00\x0d"
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


class FakeRuntime:
    ready = True
    readiness_status = "ready"

    async def start(self):
        return None

    async def shutdown(self):
        return None

    async def segment(self, _image, _prompt):
        return SegmentationResult(
            width=3,
            height=2,
            text="[SEG]",
            masks=[base64.b64encode(png_bytes()).decode("ascii")],
        )

    def metrics_snapshot(self):
        return {"ready": True}


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are unavailable")
class RecordApiTest(unittest.TestCase):
    def test_segment_record_query_files_and_final_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = replace(
                make_settings(),
                records_enabled=True,
                records_root=str(Path(temporary) / "records"),
            )
            app = create_app(settings=settings, runtime=FakeRuntime())
            decoded = SimpleNamespace(
                image=object(),
                raw=png_bytes(),
                image_format="png",
                width=3,
                height=2,
            )
            with patch(
                "production.app.decode_image_base64_with_metadata",
                return_value=decoded,
            ), TestClient(app) as client:
                response = client.post(
                    "/v1/segment",
                    json={
                        "request_id": "record-api-test",
                        "prompt": "segment the worker",
                        "image_base64": "ignored",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                record_id = response.json()["record_id"]
                self.assertTrue(record_id)

                detail = client.get(f"/v1/records/{record_id}")
                self.assertEqual(detail.status_code, 200, detail.text)
                self.assertEqual(detail.json()["status"], "success")
                self.assertEqual(detail.json()["mask_count"], 1)

                listing = client.get("/v1/records?feedback=unrated")
                self.assertEqual(listing.status_code, 200, listing.text)
                self.assertEqual(listing.json()["total"], 1)

                image = client.get(f"/v1/records/{record_id}/image")
                self.assertEqual(image.status_code, 200)
                self.assertEqual(image.content, png_bytes())
                mask = client.get(f"/v1/records/{record_id}/masks/0")
                self.assertEqual(mask.status_code, 200)
                self.assertEqual(mask.content, png_bytes())

                disliked = client.put(
                    f"/v1/records/{record_id}/feedback",
                    json={
                        "feedback": "dislike",
                        "reason": "boundary_bad",
                    },
                )
                self.assertEqual(disliked.status_code, 200, disliked.text)
                self.assertEqual(
                    disliked.json()["feedback_reason"],
                    "boundary_bad",
                )

                invalid_like = client.put(
                    f"/v1/records/{record_id}/feedback",
                    json={"feedback": "like", "reason": "boundary_bad"},
                )
                self.assertEqual(invalid_like.status_code, 422)

                liked = client.put(
                    f"/v1/records/{record_id}/feedback",
                    json={"feedback": "like"},
                )
                self.assertEqual(liked.status_code, 200, liked.text)
                self.assertEqual(liked.json()["feedback"], "like")

                cancelled = client.put(
                    f"/v1/records/{record_id}/feedback",
                    json={"feedback": None},
                )
                self.assertEqual(cancelled.status_code, 200, cancelled.text)
                self.assertIsNone(cancelled.json()["feedback"])

                metrics = client.get("/metrics")
                self.assertEqual(metrics.status_code, 200, metrics.text)
                self.assertEqual(metrics.json()["records_success_total"], 1)


if __name__ == "__main__":
    unittest.main()
