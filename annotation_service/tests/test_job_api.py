import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from annotation_service.app import create_app
from annotation_service.config import Settings
from annotation_service.storage import AnnotationStore


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


def make_settings(**overrides) -> Settings:
    values = {
        "service_version": "test-v1",
        "api_key": None,
        "cors_origins": (),
        "cors_allow_credentials": False,
        "max_request_bytes": 4096,
        "max_image_bytes": 2048,
        "max_image_pixels": 1000,
        "max_metadata_chars": 100,
        "max_queued_jobs": 100,
        "docs_enabled": True,
        "storage_enabled": False,
        "storage_root": "./annotation-data",
    }
    values.update(overrides)
    return Settings(**values)


class JobApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = AnnotationStore(
            Path(self.temporary.name) / "annotation-data"
        )
        self.client_context = TestClient(
            create_app(make_settings(), storage=self.store)
        )
        self.client = self.client_context.__enter__()
        self.asset_id = self.upload_asset()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def upload_asset(self) -> str:
        response = self.client.post(
            "/v1/annotation/assets",
            files={
                "file": (
                    "sample.png",
                    png_bytes(),
                    "image/png",
                )
            },
            data={"group_id": "site01:video03"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["asset_id"]

    def request_payload(self, **overrides) -> dict:
        payload = {
            "asset_ids": [self.asset_id],
            "requested_categories": [
                "helmet_missing",
                "equipment_proximity",
            ],
            "pipeline_version": "grounded-qwen-v1",
            "options": {
                "generate_masks": True,
                "enrich_prompts": True,
                "prompt_count": 6,
            },
        }
        payload.update(overrides)
        return payload

    def create_job(
        self,
        *,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ):
        return self.client.post(
            "/v1/annotation/jobs",
            json=payload or self.request_payload(),
            headers=headers or {},
        )

    def test_create_and_query_job(self):
        response = self.create_job()
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        self.assertEqual(job["status"], "queued")
        self.assertIsNone(job["stage"])
        self.assertEqual(
            job["progress"],
            {
                "total_assets": 1,
                "completed_assets": 0,
                "generated_tasks": 0,
            },
        )
        self.assertEqual(
            set(job["stages"]),
            {
                "grounding_dino",
                "hazard_rules",
                "sam",
                "qwen_facts",
                "qwen_prompts",
                "build_review_tasks",
            },
        )
        self.assertNotIn("claimed_by", job)
        self.assertNotIn("asset_ids", job)

        detail = self.client.get(
            f"/v1/annotation/jobs/{job['job_id']}"
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json(), job)

    def test_detection_only_job_option_is_accepted(self):
        payload = self.request_payload()
        payload["options"] = {
            "generate_masks": False,
            "enrich_prompts": False,
            "prompt_count": 6,
            "stop_after": "grounding_dino",
        }
        response = self.create_job(payload=payload)
        self.assertEqual(response.status_code, 202, response.text)

        invalid = dict(payload)
        invalid["options"] = {
            **payload["options"],
            "stop_after": "not_a_stage",
        }
        rejected = self.create_job(payload=invalid)
        self.assertEqual(rejected.status_code, 422)

        inconsistent = dict(payload)
        inconsistent["options"] = {
            **payload["options"],
            "generate_masks": True,
        }
        rejected = self.create_job(payload=inconsistent)
        self.assertEqual(rejected.status_code, 422)

        hazard_options = {
            **payload["options"],
            "stop_after": "hazard_rules",
        }
        hazard_payload = self.request_payload(options=hazard_options)
        accepted = self.create_job(payload=hazard_payload)
        self.assertEqual(accepted.status_code, 202, accepted.text)

    def test_query_job_detections(self):
        response = self.create_job()
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["job_id"]

        empty = self.client.get(
            f"/v1/annotation/jobs/{job_id}/detections"
        )
        self.assertEqual(empty.status_code, 200, empty.text)
        self.assertEqual(
            empty.json(),
            {"job_id": job_id, "items": [], "total": 0},
        )

        saved = self.store.add_detection(
            job_id=job_id,
            asset_id=self.asset_id,
            entity="person",
            box_xyxy=[0.25, 0.5, 2.75, 1.75],
            box_score=0.91,
            phrase_score=0.83,
            metadata={"caption": "person ."},
        )

        result = self.client.get(
            f"/v1/annotation/jobs/{job_id}/detections"
        )
        self.assertEqual(result.status_code, 200, result.text)
        payload = result.json()
        self.assertEqual(payload["job_id"], job_id)
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["items"][0]["asset_id"], self.asset_id)
        self.assertEqual(
            payload["items"][0]["detection_id"],
            saved["detection_id"],
        )
        self.assertEqual(
            payload["items"][0]["box_xyxy"],
            [0.25, 0.5, 2.75, 1.75],
        )
        self.assertEqual(
            payload["items"][0]["metadata"],
            {"caption": "person ."},
        )

        filtered = self.client.get(
            f"/v1/annotation/jobs/{job_id}/detections",
            params={"asset_id": self.asset_id},
        )
        self.assertEqual(filtered.status_code, 200, filtered.text)
        self.assertEqual(filtered.json(), payload)

        unknown_asset = self.client.get(
            f"/v1/annotation/jobs/{job_id}/detections",
            params={"asset_id": "ast_missing"},
        )
        self.assertEqual(unknown_asset.status_code, 404)

        unknown_job = self.client.get(
            "/v1/annotation/jobs/job_missing/detections"
        )
        self.assertEqual(unknown_job.status_code, 404)

    def test_query_job_hazard_candidates(self):
        payload = self.request_payload()
        payload["options"] = {
            "generate_masks": False,
            "enrich_prompts": False,
            "prompt_count": 6,
            "stop_after": "hazard_rules",
        }
        created = self.create_job(payload=payload)
        self.assertEqual(created.status_code, 202, created.text)
        job_id = created.json()["job_id"]
        claimed = self.store.claim_next_job(
            worker_id="test-worker",
            lease_seconds=30,
            required_stop_after="hazard_rules",
        )
        self.assertEqual(claimed["job_id"], job_id)
        detection_result = self.store.replace_detections(
            job_id=job_id,
            asset_id=self.asset_id,
            detections=[
                {
                    "entity": "person",
                    "box_xyxy": [0.25, 0.1, 2.75, 1.9],
                    "box_score": 0.91,
                    "phrase_score": 0.83,
                    "metadata": {},
                }
            ],
            worker_id="test-worker",
        )
        saved = self.store.replace_hazard_candidates(
            job_id=job_id,
            asset_id=self.asset_id,
            candidates=[
                {
                    "category": "helmet_missing",
                    "target_entity": "person",
                    "target_detection_ids": [
                        detection_result[0]["detection_id"]
                    ],
                    "box_xyxy": [0.25, 0.1, 2.75, 1.9],
                    "confidence": 0.55,
                    "rule_id": "ppe.helmet_absent_in_head_region",
                    "rule_version": "construction-hazard-rules-v1",
                    "evidence": ["no helmet detection matched"],
                    "metadata": {
                        "requires_visual_verification": True
                    },
                }
            ],
            worker_id="test-worker",
        )

        response = self.client.get(
            f"/v1/annotation/jobs/{job_id}/hazard-candidates"
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["total"], 1)
        self.assertEqual(
            result["items"][0]["hazard_id"],
            saved[0]["hazard_id"],
        )
        self.assertEqual(
            result["items"][0]["target_detection_ids"],
            [detection_result[0]["detection_id"]],
        )

        unknown_asset = self.client.get(
            f"/v1/annotation/jobs/{job_id}/hazard-candidates",
            params={"asset_id": "ast_missing"},
        )
        self.assertEqual(unknown_asset.status_code, 404)

    def test_idempotent_retry_and_conflict(self):
        headers = {"Idempotency-Key": "frontend-job-001"}
        first = self.create_job(headers=headers)
        retry = self.create_job(headers=headers)
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(retry.status_code, 202, retry.text)
        self.assertEqual(
            retry.json()["job_id"],
            first.json()["job_id"],
        )

        conflict = self.create_job(
            payload=self.request_payload(
                pipeline_version="grounded-qwen-v2"
            ),
            headers=headers,
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(
            conflict.json()["code"],
            "idempotency_conflict",
        )

    def test_missing_asset_and_invalid_request(self):
        missing = self.create_job(
            payload=self.request_payload(asset_ids=["ast_missing"])
        )
        self.assertEqual(missing.status_code, 404, missing.text)
        self.assertEqual(missing.json()["code"], "not_found")

        duplicate = self.create_job(
            payload=self.request_payload(
                asset_ids=[self.asset_id, self.asset_id]
            )
        )
        self.assertEqual(duplicate.status_code, 422)

        blank_version = self.create_job(
            payload=self.request_payload(pipeline_version="   ")
        )
        self.assertEqual(blank_version.status_code, 422)

    def test_queue_limit_and_idempotent_retry_when_full(self):
        limited_store = AnnotationStore(
            Path(self.temporary.name) / "limited-data"
        )
        app = create_app(
            make_settings(max_queued_jobs=1),
            storage=limited_store,
        )
        with TestClient(app) as client:
            asset = limited_store.create_asset(
                image_bytes=png_bytes(),
                media_type="image/png",
                width=3,
                height=2,
                group_id="site01:video04",
            )
            payload = self.request_payload(
                asset_ids=[asset["asset_id"]]
            )
            headers = {"Idempotency-Key": "limited-job-001"}
            first = client.post(
                "/v1/annotation/jobs",
                json=payload,
                headers=headers,
            )
            retry = client.post(
                "/v1/annotation/jobs",
                json=payload,
                headers=headers,
            )
            full = client.post(
                "/v1/annotation/jobs",
                json=payload,
            )

        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(retry.status_code, 202, retry.text)
        self.assertEqual(full.status_code, 429, full.text)
        self.assertEqual(full.json()["code"], "queue_full")

    def test_authentication_and_disabled_storage(self):
        auth_store = AnnotationStore(
            Path(self.temporary.name) / "auth-data"
        )
        with TestClient(
            create_app(
                make_settings(api_key="job-secret"),
                storage=auth_store,
            )
        ) as client:
            unauthorized = client.get(
                "/v1/annotation/jobs/job_missing"
            )
            unauthorized_detections = client.get(
                "/v1/annotation/jobs/job_missing/detections"
            )
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized_detections.status_code, 401)

        with TestClient(create_app(make_settings())) as client:
            unavailable = client.post(
                "/v1/annotation/jobs",
                json=self.request_payload(),
            )
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(
            unavailable.json()["code"],
            "downstream_unavailable",
        )


if __name__ == "__main__":
    unittest.main()
