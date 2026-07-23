import base64
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from production.errors import (
    RecordConflictError,
    RecordNotFoundError,
    RecordStorageError,
)
from production.image_io import PNG_SIGNATURE
from production.records import RecordStore


def png_bytes(width: int = 3, height: int = 2) -> bytes:
    return (
        PNG_SIGNATURE
        + b"\x00\x00\x00\x0d"
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


class RecordStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "records"
        self.store = RecordStore(self.root)
        self.store.initialize()

    def tearDown(self):
        self.temporary.cleanup()

    def create_record(self, request_id: str = "request-1") -> str:
        return self.store.create_record(
            request_id=request_id,
            model_version="test-v1",
            prompt="segment the worker",
            image_raw=png_bytes(),
            image_format="png",
            width=3,
            height=2,
        )

    def complete_record(self, record_id: str, mask_count: int = 1) -> None:
        encoded = base64.b64encode(png_bytes()).decode("ascii")
        self.store.complete_record(
            record_id,
            masks=[encoded for _ in range(mask_count)],
            text_response="[SEG]",
            latency_ms=12.3456,
        )

    def test_successful_record_persists_image_masks_and_metrics(self):
        record_id = self.create_record()
        processing = self.store.get_record(record_id)
        self.assertEqual(processing["status"], "processing")
        self.assertIsNotNone(processing["image_url"])

        self.complete_record(record_id, mask_count=2)
        record = self.store.get_record(record_id)
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["mask_count"], 2)
        self.assertEqual(record["text_response"], "[SEG]")
        self.assertEqual(record["latency_ms"], 12.346)
        self.assertEqual([mask["index"] for mask in record["masks"]], [0, 1])

        image_path, media_type = self.store.original_image(record_id)
        self.assertEqual(image_path.read_bytes(), png_bytes())
        self.assertEqual(media_type, "image/png")
        self.assertEqual(self.store.mask_image(record_id, 1).read_bytes(), png_bytes())

        metrics = self.store.metrics_snapshot()
        self.assertEqual(metrics["records_created_total"], 1)
        self.assertEqual(metrics["records_success_total"], 1)
        self.assertEqual(metrics["feedback_unrated_records"], 1)
        self.assertGreater(metrics["record_images_bytes_total"], 0)
        self.assertGreater(metrics["record_masks_bytes_total"], 0)

    def test_empty_mask_result_is_a_successful_record(self):
        record_id = self.create_record()
        self.complete_record(record_id, mask_count=0)
        record = self.store.get_record(record_id)
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["mask_count"], 0)
        self.assertEqual(record["masks"], [])

    def test_mask_dimensions_must_match_original_image(self):
        record_id = self.create_record()
        encoded = base64.b64encode(png_bytes(2, 2)).decode("ascii")
        with self.assertRaises(RecordStorageError):
            self.store.complete_record(
                record_id,
                masks=[encoded],
                text_response="[SEG]",
                latency_ms=1.0,
            )

    def test_failed_and_interrupted_records_are_retained(self):
        failed_id = self.create_record("failed")
        self.store.mark_failed(
            failed_id,
            code="inference_failed",
            message="LISA inference failed",
        )
        failed = self.store.get_record(failed_id)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_code"], "inference_failed")

        interrupted_id = self.create_record("interrupted")
        restarted = RecordStore(self.root)
        restarted.initialize()
        interrupted = restarted.get_record(interrupted_id)
        self.assertEqual(interrupted["status"], "failed")
        self.assertEqual(interrupted["error_code"], "interrupted")

    def test_feedback_overwrites_final_state_and_reason_is_optional(self):
        record_id = self.create_record()
        self.complete_record(record_id)

        liked = self.store.update_feedback(
            record_id,
            feedback="like",
            reason=None,
            comment=None,
        )
        self.assertEqual(liked["feedback"], "like")

        disliked = self.store.update_feedback(
            record_id,
            feedback="dislike",
            reason=None,
            comment=None,
        )
        self.assertEqual(disliked["feedback"], "dislike")
        self.assertIsNone(disliked["feedback_reason"])

        explained = self.store.update_feedback(
            record_id,
            feedback="dislike",
            reason="boundary_bad",
            comment="mask crosses the edge",
        )
        self.assertEqual(explained["feedback_reason"], "boundary_bad")

        cancelled = self.store.update_feedback(
            record_id,
            feedback=None,
            reason=None,
            comment=None,
        )
        self.assertIsNone(cancelled["feedback"])
        record = self.store.get_record(record_id)
        self.assertIsNone(record["feedback_reason"])
        self.assertIsNone(record["feedback_comment"])

    def test_database_rejects_reason_without_dislike(self):
        record_id = self.create_record()
        with self.assertRaises(sqlite3.IntegrityError):
            with self.store._connect() as connection:
                connection.execute(
                    """
                    UPDATE segmentation_records
                    SET feedback_reason = 'boundary_bad'
                    WHERE record_id = ?
                    """,
                    (record_id,),
                )

    def test_failed_record_cannot_be_rated(self):
        record_id = self.create_record()
        self.store.mark_failed(
            record_id,
            code="inference_failed",
            message="LISA inference failed",
        )
        with self.assertRaises(RecordConflictError):
            self.store.update_feedback(
                record_id,
                feedback="dislike",
                reason="empty_mask",
                comment=None,
            )

    def test_list_filters_and_missing_files(self):
        liked_id = self.create_record("liked")
        self.complete_record(liked_id)
        self.store.update_feedback(
            liked_id,
            feedback="like",
            reason=None,
            comment=None,
        )
        unrated_id = self.create_record("unrated")
        self.complete_record(unrated_id)

        liked = self.store.list_records(feedback="like")
        self.assertEqual(liked["total"], 1)
        self.assertEqual(liked["records"][0]["record_id"], liked_id)
        unrated = self.store.list_records(feedback="unrated")
        self.assertEqual(unrated["total"], 1)
        self.assertEqual(unrated["records"][0]["record_id"], unrated_id)

        with self.assertRaises(RecordNotFoundError):
            self.store.get_record("missing")
        with self.assertRaises(RecordNotFoundError):
            self.store.mask_image(liked_id, 99)

    def test_concurrent_record_writes_are_serialized_safely(self):
        def create_and_complete(index: int) -> str:
            record_id = self.create_record(f"concurrent-{index}")
            self.complete_record(record_id)
            return record_id

        with ThreadPoolExecutor(max_workers=4) as executor:
            record_ids = list(executor.map(create_and_complete, range(12)))
        self.assertEqual(len(set(record_ids)), 12)
        records = self.store.list_records(status="success", limit=20)
        self.assertEqual(records["total"], 12)

    def test_storage_root_rejects_filesystem_root(self):
        with self.assertRaises(ValueError):
            RecordStore(Path(self.root.anchor))
if __name__ == "__main__":
    unittest.main()
