import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation_service.errors import AnnotationValidationError
from annotation_service.storage import AnnotationStore
from annotation_service.review_task_builder import build_review_tasks
from annotation_service.worker.grounding_dino import (
    GroundingDINODetection,
    build_caption,
    entities_for_categories,
    normalized_cxcywh_to_xyxy,
)
from annotation_service.worker.runner import GroundingDINOJobWorker
from annotation_service.worker.settings import GroundingDINOWorkerSettings


class FakePredictor:
    model_version = "fake-grounding-dino-v1"
    prompt_version = "fake-prompts-v1"

    def __init__(self, *, fail_on_call: int | None = None):
        self.fail_on_call = fail_on_call
        self.calls = 0

    def predict(
        self,
        *,
        image_path: Path,
        width: int,
        height: int,
        categories,
    ):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("synthetic detector failure")
        return [
            GroundingDINODetection(
                entity="person",
                box_xyxy=(1.0, 1.0, width - 1.0, height - 1.0),
                box_score=0.91,
                phrase_score=0.87,
                metadata={
                    "model_version": self.model_version,
                    "prompt_version": self.prompt_version,
                    "image_name": image_path.name,
                    "categories": list(categories),
                },
            )
        ]


class GroundingDINOHelpersTest(unittest.TestCase):
    def test_category_entities_are_ordered_and_deduplicated(self):
        entities = entities_for_categories(
            ["helmet_missing", "equipment_proximity"]
        )
        self.assertEqual(entities[0:2], ("person", "helmet"))
        self.assertEqual(entities.count("person"), 1)
        self.assertIn("excavator", entities)
        self.assertEqual(
            build_caption(("person", "helmet")),
            "person . helmet .",
        )

    def test_normalized_box_conversion_clamps_to_image(self):
        self.assertEqual(
            normalized_cxcywh_to_xyxy(
                (0.5, 0.5, 0.4, 0.6),
                width=100,
                height=50,
            ),
            (30.0, 10.0, 70.0, 40.0),
        )
        self.assertEqual(
            normalized_cxcywh_to_xyxy(
                (0.0, 0.0, 0.5, 0.5),
                width=100,
                height=50,
            ),
            (0.0, 0.0, 25.0, 12.5),
        )
        self.assertIsNone(
            normalized_cxcywh_to_xyxy(
                (2.0, 2.0, 0.1, 0.1),
                width=100,
                height=50,
            )
        )


class GroundingDINOWorkerSettingsTest(unittest.TestCase):
    def test_paths_are_resolved_from_model_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "GroundingDINO"
            bert = root / "weights" / "bert-base-uncased"
            config = (
                root
                / "groundingdino"
                / "config"
                / "GroundingDINO_SwinT_OGC.py"
            )
            checkpoint = root / "weights" / "groundingdino_swint_ogc.pth"
            config.parent.mkdir(parents=True)
            bert.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            config.write_text("text_encoder_type = 'bert-base-uncased'")
            checkpoint.write_bytes(b"checkpoint")
            for filename in (
                "config.json",
                "model.safetensors",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.txt",
            ):
                (bert / filename).write_bytes(b"x")
            environment = {
                "ANNOTATION_STORAGE_ROOT": str(
                    Path(temporary) / "annotation-data"
                ),
                "ANNOTATION_GROUNDING_DINO_ROOT": str(root),
                "ANNOTATION_WORKER_ID": "worker-test",
            }
            with patch.dict(os.environ, environment, clear=True):
                settings = GroundingDINOWorkerSettings.from_env()
            settings.validate_model_files()

            self.assertEqual(settings.config_path, config.resolve())
            self.assertEqual(settings.checkpoint_path, checkpoint.resolve())
            self.assertEqual(settings.bert_path, bert.resolve())
            self.assertEqual(settings.box_threshold, 0.35)

    def test_heartbeat_must_be_shorter_than_lease(self):
        environment = {
            "ANNOTATION_STORAGE_ROOT": "/tmp/annotation-data",
            "ANNOTATION_GROUNDING_DINO_ROOT": "/tmp/GroundingDINO",
            "ANNOTATION_WORKER_LEASE_SECONDS": "30",
            "ANNOTATION_WORKER_HEARTBEAT_SECONDS": "30",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ValueError):
                GroundingDINOWorkerSettings.from_env()


class GroundingDINOJobWorkerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = AnnotationStore(
            Path(self.temporary.name) / "annotation-data"
        )
        self.store.initialize()

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def create_asset(self, content: bytes) -> dict:
        return self.store.create_asset(
            image_bytes=content,
            media_type="image/png",
            width=20,
            height=10,
            group_id="site01:camera01",
        )

    def create_job(
        self,
        asset_ids: list[str],
        *,
        detection_only: bool = True,
        stop_after: str = "grounding_dino",
    ) -> dict:
        options = (
            {
                "generate_masks": False,
                "enrich_prompts": False,
                "stop_after": stop_after,
            }
            if detection_only
            else {}
        )
        return self.store.create_job(
            asset_ids=asset_ids,
            requested_categories=[
                "helmet_missing",
                "equipment_proximity",
            ],
            pipeline_version="grounding-dino-v1",
            options=options,
        )

    def make_worker(self, predictor: FakePredictor):
        return GroundingDINOJobWorker(
            store=self.store,
            predictor=predictor,
            worker_id="test-gpu-worker",
            lease_seconds=30,
            heartbeat_seconds=5,
            poll_seconds=0.1,
        )

    def test_worker_processes_detection_only_job(self):
        asset = self.create_asset(b"first")
        full_job = self.create_job(
            [asset["asset_id"]],
            detection_only=False,
        )
        detection_job = self.create_job([asset["asset_id"]])
        worker = self.make_worker(FakePredictor())

        self.assertTrue(worker.run_once())

        untouched = self.store.get_job(full_job["job_id"])
        completed = self.store.get_job(detection_job["job_id"])
        detections = self.store.list_detections(
            job_id=detection_job["job_id"],
            asset_id=asset["asset_id"],
        )
        self.assertEqual(untouched["status"], "queued")
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["progress"]["completed_assets"], 1)
        self.assertEqual(
            completed["stages"]["grounding_dino"]["status"],
            "succeeded",
        )
        self.assertEqual(completed["stages"]["sam"]["status"], "skipped")
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["entity"], "person")
        self.assertEqual(
            detections[0]["metadata"]["model_version"],
            "fake-grounding-dino-v1",
        )

    def test_worker_records_partial_asset_failure(self):
        first = self.create_asset(b"first")
        second = self.create_asset(b"second")
        job = self.create_job(
            [first["asset_id"], second["asset_id"]]
        )
        worker = self.make_worker(FakePredictor(fail_on_call=2))

        self.assertTrue(worker.run_once())

        completed = self.store.get_job(job["job_id"])
        self.assertEqual(completed["status"], "partial_failed")
        self.assertEqual(len(completed["errors"]), 1)
        self.assertEqual(
            completed["errors"][0]["asset_id"],
            second["asset_id"],
        )
        self.assertEqual(
            len(
                self.store.list_detections(
                    job_id=job["job_id"],
                    asset_id=first["asset_id"],
                )
            ),
            1,
        )
        self.assertEqual(
            self.store.list_detections(
                job_id=job["job_id"],
                asset_id=second["asset_id"],
            ),
            [],
        )

    def test_worker_runs_hazard_rules_and_persists_candidates(self):
        asset = self.create_asset(b"hazard")
        job = self.create_job(
            [asset["asset_id"]],
            stop_after="hazard_rules",
        )
        worker = self.make_worker(FakePredictor())

        self.assertTrue(worker.run_once())

        completed = self.store.get_job(job["job_id"])
        candidates = self.store.list_job_hazard_candidates(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
        )
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["stage"], "hazard_rules")
        self.assertEqual(
            completed["stages"]["grounding_dino"]["status"],
            "succeeded",
        )
        self.assertEqual(
            completed["stages"]["hazard_rules"]["status"],
            "succeeded",
        )
        self.assertEqual(completed["stages"]["sam"]["status"], "skipped")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["category"], "helmet_missing")
        self.assertEqual(candidates[0]["target_entity"], "person")
        self.assertEqual(
            candidates[0]["rule_version"],
            "construction-hazard-rules-v1",
        )
        job_completed_at = completed["completed_at"]

        first = build_review_tasks(
            self.store,
            job_id=job["job_id"],
        )
        second = build_review_tasks(
            self.store,
            job_id=job["job_id"],
        )
        self.assertEqual(first["created_count"], 1)
        self.assertEqual(second["created_count"], 0)
        self.assertEqual(second["existing_count"], 1)
        self.assertEqual(first["task_ids"], second["task_ids"])
        materialized = self.store.get_job(job["job_id"])
        self.assertEqual(materialized["completed_at"], job_completed_at)
        self.assertEqual(
            materialized["stages"]["build_review_tasks"]["status"],
            "succeeded",
        )
        self.assertEqual(materialized["progress"]["generated_tasks"], 1)
        task = self.store.get_task(first["task_ids"][0])
        self.assertEqual(task["status"], "generated")
        self.assertEqual(task["annotation"]["shapes"], [])
        self.assertEqual(task["annotation"]["prompts"], [])
        self.assertEqual(
            task["source_hazard"]["hazard_id"],
            candidates[0]["hazard_id"],
        )
        self.assertEqual(
            task["provenance"]["hazard_candidate_id"],
            candidates[0]["hazard_id"],
        )
        self.assertTrue(
            any("人工" in warning for warning in task["warnings"])
        )
        with self.assertRaises(AnnotationValidationError):
            self.store.submit_task(
                task["task_id"],
                expected_version=1,
                annotator_id="annotator-1",
                primary_result="mask_missing",
            )

    def test_detection_replacement_is_idempotent_and_lease_guarded(self):
        asset = self.create_asset(b"first")
        job = self.create_job([asset["asset_id"]])
        claimed = self.store.claim_next_job(
            worker_id="test-gpu-worker",
            lease_seconds=30,
            required_stop_after="grounding_dino",
        )
        self.assertIsNotNone(claimed)
        payload = [
            {
                "entity": "person",
                "box_xyxy": [1, 1, 19, 9],
                "box_score": 0.9,
                "phrase_score": 0.8,
                "metadata": {"model": "fake"},
            }
        ]

        first = self.store.replace_detections(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
            detections=payload,
            worker_id="test-gpu-worker",
        )
        second = self.store.replace_detections(
            job_id=job["job_id"],
            asset_id=asset["asset_id"],
            detections=payload,
            worker_id="test-gpu-worker",
        )

        self.assertEqual(first[0]["detection_id"], second[0]["detection_id"])
        self.assertEqual(
            len(
                self.store.list_detections(
                    job_id=job["job_id"],
                    asset_id=asset["asset_id"],
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
