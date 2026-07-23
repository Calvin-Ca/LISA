from __future__ import annotations

import base64
import binascii
import hashlib
import os
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import (
    RecordConflictError,
    RecordNotFoundError,
    RecordStorageError,
)
from .image_io import PNG_SIGNATURE, encoded_image_dimensions


SCHEMA = """
CREATE TABLE IF NOT EXISTS segmentation_records (
    record_id TEXT PRIMARY KEY,
    request_id TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL
        CHECK (status IN ('processing', 'success', 'failed')),
    model_version TEXT NOT NULL,
    prompt TEXT NOT NULL,
    original_image_path TEXT,
    original_image_sha256 TEXT,
    original_image_size_bytes INTEGER,
    image_media_type TEXT,
    width INTEGER,
    height INTEGER,
    mask_count INTEGER NOT NULL DEFAULT 0,
    text_response TEXT,
    latency_ms REAL,
    error_code TEXT,
    error_message TEXT,
    feedback TEXT
        CHECK (feedback IS NULL OR feedback IN ('like', 'dislike')),
    feedback_reason TEXT,
    feedback_comment TEXT,
    feedback_at TEXT,
    CHECK (
        feedback IS 'dislike'
        OR (feedback_reason IS NULL AND feedback_comment IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS segmentation_masks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL,
    mask_index INTEGER NOT NULL,
    mask_path TEXT NOT NULL,
    mask_sha256 TEXT NOT NULL,
    mask_size_bytes INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    UNIQUE (record_id, mask_index),
    FOREIGN KEY (record_id)
        REFERENCES segmentation_records(record_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_records_created_at
ON segmentation_records(created_at);
CREATE INDEX IF NOT EXISTS idx_records_status
ON segmentation_records(status);
CREATE INDEX IF NOT EXISTS idx_records_feedback
ON segmentation_records(feedback);
CREATE INDEX IF NOT EXISTS idx_records_request_id
ON segmentation_records(request_id);
"""


VALID_FEEDBACK = {"like", "dislike"}
VALID_FEEDBACK_REASONS = {
    "wrong_object",
    "missing_target",
    "boundary_bad",
    "extra_area",
    "empty_mask",
    "instance_wrong",
    "prompt_mismatch",
    "other",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RecordStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if self.root == Path(self.root.anchor):
            raise ValueError("record storage root must not be a filesystem root")
        self.db_path = self.root / "records.db"
        self.images_root = self.root / "images"
        self.masks_root = self.root / "masks"
        self.tmp_root = self.root / "tmp"
        self._lock = threading.RLock()
        self._initialized = False
        self._metrics = Counter({"record_storage_errors_total": 0})

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            try:
                for path in (
                    self.root,
                    self.images_root,
                    self.masks_root,
                    self.tmp_root,
                ):
                    path.mkdir(parents=True, exist_ok=True)
                with self._connect() as connection:
                    connection.executescript(SCHEMA)
                    connection.execute(
                        """
                        UPDATE segmentation_records
                        SET status = 'failed',
                            completed_at = ?,
                            error_code = 'interrupted',
                            error_message = 'request was interrupted before completion'
                        WHERE status = 'processing'
                        """,
                        (utc_now(),),
                    )
                self._initialized = True
            except Exception as exc:
                self._storage_failed()
                raise RecordStorageError(
                    "segmentation record storage is unavailable"
                ) from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _storage_failed(self) -> None:
        self._metrics["record_storage_errors_total"] += 1

    def _relative_date_path(self, category: str, name: str) -> Path:
        now = datetime.now(timezone.utc)
        return Path(category) / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}" / name

    def _resolve_relative(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise RecordStorageError("stored record path is invalid") from exc
        return candidate

    def _atomic_write(self, relative: Path, data: bytes) -> None:
        destination = self._resolve_relative(relative.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.tmp_root / f"{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def create_record(
        self,
        *,
        request_id: str,
        model_version: str,
        prompt: str,
        image_raw: bytes,
        image_format: str,
        width: int,
        height: int,
    ) -> str:
        self._ensure_initialized()
        record_id = str(uuid.uuid4())
        extension = "jpg" if image_format == "jpeg" else "png"
        relative = self._relative_date_path(
            "images", f"{record_id}.{extension}"
        )
        media_type = "image/jpeg" if image_format == "jpeg" else "image/png"
        created_at = utc_now()
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        """
                        INSERT INTO segmentation_records (
                            record_id, request_id, created_at, status,
                            model_version, prompt, width, height,
                            image_media_type
                        ) VALUES (?, ?, ?, 'processing', ?, ?, ?, ?, ?)
                        """,
                        (
                            record_id,
                            request_id,
                            created_at,
                            model_version,
                            prompt,
                            width,
                            height,
                            media_type,
                        ),
                    )
                    connection.execute("COMMIT")
                self._atomic_write(relative, image_raw)
                with self._connect() as connection:
                    connection.execute(
                        """
                        UPDATE segmentation_records
                        SET original_image_path = ?,
                            original_image_sha256 = ?,
                            original_image_size_bytes = ?
                        WHERE record_id = ?
                        """,
                        (
                            relative.as_posix(),
                            sha256_bytes(image_raw),
                            len(image_raw),
                            record_id,
                        ),
                    )
            except Exception as exc:
                self._storage_failed()
                try:
                    self.mark_failed(
                        record_id,
                        code="record_storage_error",
                        message="failed to save the original image",
                    )
                except Exception:
                    pass
                raise RecordStorageError(
                    "failed to create segmentation record"
                ) from exc
        return record_id

    def complete_record(
        self,
        record_id: str,
        *,
        masks: list[str],
        text_response: str,
        latency_ms: float,
    ) -> None:
        self._ensure_initialized()
        prepared: list[tuple[int, Path, bytes, int, int]] = []
        written: list[Path] = []
        try:
            for index, encoded in enumerate(masks):
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise RecordStorageError(
                        "model returned an invalid mask encoding"
                    ) from exc
                if not raw.startswith(PNG_SIGNATURE):
                    raise RecordStorageError("model returned a non-PNG mask")
                width, height = encoded_image_dimensions(raw, "png")
                relative = self._relative_date_path(
                    "masks", f"{record_id}-{index}.png"
                )
                prepared.append((index, relative, raw, width, height))

            with self._lock:
                with self._connect() as connection:
                    expected = connection.execute(
                        """
                        SELECT status, width, height
                        FROM segmentation_records WHERE record_id = ?
                        """,
                        (record_id,),
                    ).fetchone()
                if expected is None:
                    raise RecordNotFoundError(
                        "segmentation record was not found"
                    )
                if expected["status"] != "processing":
                    raise RecordConflictError(
                        "segmentation record is no longer processing"
                    )
                expected_size = (expected["width"], expected["height"])
                for _, _, _, width, height in prepared:
                    if (width, height) != expected_size:
                        raise RecordStorageError(
                            "model returned a mask with unexpected dimensions"
                        )
                for _, relative, raw, _, _ in prepared:
                    self._atomic_write(relative, raw)
                    written.append(relative)
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    row = connection.execute(
                        "SELECT status FROM segmentation_records WHERE record_id = ?",
                        (record_id,),
                    ).fetchone()
                    if row is None:
                        raise RecordNotFoundError("segmentation record was not found")
                    if row["status"] != "processing":
                        raise RecordConflictError(
                            "segmentation record is no longer processing"
                        )
                    for index, relative, raw, width, height in prepared:
                        connection.execute(
                            """
                            INSERT INTO segmentation_masks (
                                record_id, mask_index, mask_path, mask_sha256,
                                mask_size_bytes, width, height
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                record_id,
                                index,
                                relative.as_posix(),
                                sha256_bytes(raw),
                                len(raw),
                                width,
                                height,
                            ),
                        )
                    connection.execute(
                        """
                        UPDATE segmentation_records
                        SET status = 'success', completed_at = ?, mask_count = ?,
                            text_response = ?, latency_ms = ?,
                            error_code = NULL, error_message = NULL
                        WHERE record_id = ?
                        """,
                        (
                            utc_now(),
                            len(prepared),
                            text_response,
                            round(float(latency_ms), 3),
                            record_id,
                        ),
                    )
                    connection.execute("COMMIT")
        except (RecordNotFoundError, RecordConflictError):
            for relative in written:
                self._resolve_relative(relative.as_posix()).unlink(missing_ok=True)
            raise
        except Exception as exc:
            self._storage_failed()
            for relative in written:
                self._resolve_relative(relative.as_posix()).unlink(missing_ok=True)
            if isinstance(exc, RecordStorageError):
                raise
            raise RecordStorageError(
                "failed to save segmentation result"
            ) from exc

    def mark_failed(self, record_id: str, *, code: str, message: str) -> None:
        self._ensure_initialized()
        with self._lock:
            try:
                with self._connect() as connection:
                    cursor = connection.execute(
                        """
                        UPDATE segmentation_records
                        SET status = 'failed', completed_at = ?,
                            error_code = ?, error_message = ?
                        WHERE record_id = ? AND status = 'processing'
                        """,
                        (utc_now(), code, message, record_id),
                    )
                    if cursor.rowcount == 0:
                        row = connection.execute(
                            "SELECT 1 FROM segmentation_records WHERE record_id = ?",
                            (record_id,),
                        ).fetchone()
                        if row is None:
                            raise RecordNotFoundError(
                                "segmentation record was not found"
                            )
            except RecordNotFoundError:
                raise
            except Exception as exc:
                self._storage_failed()
                raise RecordStorageError(
                    "failed to update segmentation record"
                ) from exc

    def _record_payload(self, row: sqlite3.Row, masks: list[sqlite3.Row]) -> dict[str, Any]:
        record_id = row["record_id"]
        return {
            "record_id": record_id,
            "request_id": row["request_id"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "status": row["status"],
            "model_version": row["model_version"],
            "prompt": row["prompt"],
            "width": row["width"],
            "height": row["height"],
            "mask_count": row["mask_count"],
            "image_url": (
                f"/v1/records/{record_id}/image"
                if row["original_image_path"]
                else None
            ),
            "masks": [
                {
                    "index": mask["mask_index"],
                    "url": (
                        f"/v1/records/{record_id}/masks/"
                        f"{mask['mask_index']}"
                    ),
                }
                for mask in masks
            ],
            "text_response": row["text_response"],
            "latency_ms": row["latency_ms"],
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "feedback": row["feedback"],
            "feedback_reason": row["feedback_reason"],
            "feedback_comment": row["feedback_comment"],
            "feedback_at": row["feedback_at"],
        }

    def get_record(self, record_id: str) -> dict[str, Any]:
        self._ensure_initialized()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM segmentation_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError("segmentation record was not found")
            masks = connection.execute(
                """
                SELECT * FROM segmentation_masks
                WHERE record_id = ? ORDER BY mask_index
                """,
                (record_id,),
            ).fetchall()
        return self._record_payload(row, masks)

    def list_records(
        self,
        *,
        status: str | None = None,
        feedback: str | None = None,
        model_version: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        clauses: list[str] = []
        parameters: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if feedback == "unrated":
            clauses.append("feedback IS NULL")
        elif feedback is not None:
            clauses.append("feedback = ?")
            parameters.append(feedback)
        if model_version is not None:
            clauses.append("model_version = ?")
            parameters.append(model_version)
        if date_from is not None:
            clauses.append("created_at >= ?")
            parameters.append(date_from)
        if date_to is not None:
            clauses.append("created_at <= ?")
            parameters.append(date_to)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock, self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM segmentation_records{where}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM segmentation_records{where}
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()
            records = []
            for row in rows:
                masks = connection.execute(
                    """
                    SELECT * FROM segmentation_masks
                    WHERE record_id = ? ORDER BY mask_index
                    """,
                    (row["record_id"],),
                ).fetchall()
                records.append(self._record_payload(row, masks))
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "records": records,
        }

    def update_feedback(
        self,
        record_id: str,
        *,
        feedback: str | None,
        reason: str | None,
        comment: str | None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if feedback is not None and feedback not in VALID_FEEDBACK:
            raise ValueError("invalid feedback")
        if reason is not None and reason not in VALID_FEEDBACK_REASONS:
            raise ValueError("invalid feedback reason")
        if feedback != "dislike":
            reason = None
            comment = None
        feedback_at = utc_now() if feedback is not None else None
        with self._lock:
            try:
                with self._connect() as connection:
                    row = connection.execute(
                        """
                        SELECT status FROM segmentation_records
                        WHERE record_id = ?
                        """,
                        (record_id,),
                    ).fetchone()
                    if row is None:
                        raise RecordNotFoundError(
                            "segmentation record was not found"
                        )
                    if row["status"] != "success":
                        raise RecordConflictError(
                            "only successful segmentation records can be rated"
                        )
                    connection.execute(
                        """
                        UPDATE segmentation_records
                        SET feedback = ?, feedback_reason = ?,
                            feedback_comment = ?, feedback_at = ?
                        WHERE record_id = ?
                        """,
                        (feedback, reason, comment, feedback_at, record_id),
                    )
            except (RecordNotFoundError, RecordConflictError):
                raise
            except Exception as exc:
                self._storage_failed()
                raise RecordStorageError(
                    "failed to save segmentation feedback"
                ) from exc
        return {
            "record_id": record_id,
            "feedback": feedback,
            "feedback_reason": reason,
            "feedback_comment": comment,
            "feedback_at": feedback_at,
        }

    def original_image(self, record_id: str) -> tuple[Path, str]:
        self._ensure_initialized()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT original_image_path, image_media_type
                FROM segmentation_records WHERE record_id = ?
                """,
                (record_id,),
            ).fetchone()
        if row is None or not row["original_image_path"]:
            raise RecordNotFoundError("record image was not found")
        path = self._resolve_relative(row["original_image_path"])
        if not path.is_file():
            raise RecordNotFoundError("record image was not found")
        return path, row["image_media_type"]

    def mask_image(self, record_id: str, mask_index: int) -> Path:
        self._ensure_initialized()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT mask_path FROM segmentation_masks
                WHERE record_id = ? AND mask_index = ?
                """,
                (record_id, mask_index),
            ).fetchone()
        if row is None:
            raise RecordNotFoundError("record mask was not found")
        path = self._resolve_relative(row["mask_path"])
        if not path.is_file():
            raise RecordNotFoundError("record mask was not found")
        return path

    def metrics_snapshot(self) -> dict[str, int | bool]:
        self._ensure_initialized()
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT
                        COUNT(*) AS created,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                        SUM(CASE WHEN feedback = 'like' THEN 1 ELSE 0 END) AS likes,
                        SUM(CASE WHEN feedback = 'dislike' THEN 1 ELSE 0 END) AS dislikes,
                        SUM(CASE WHEN feedback IS NULL THEN 1 ELSE 0 END) AS unrated,
                        COALESCE(SUM(original_image_size_bytes), 0) AS image_bytes
                    FROM segmentation_records
                    """
                ).fetchone()
                mask_bytes = int(
                    connection.execute(
                        """
                        SELECT COALESCE(SUM(mask_size_bytes), 0)
                        FROM segmentation_masks
                        """
                    ).fetchone()[0]
                )
        except Exception:
            self._storage_failed()
            return {
                "records_enabled": True,
                "records_healthy": False,
                "record_storage_errors_total": int(
                    self._metrics["record_storage_errors_total"]
                ),
            }
        return {
            "records_enabled": True,
            "records_healthy": True,
            "records_created_total": int(row["created"] or 0),
            "records_success_total": int(row["success"] or 0),
            "records_failed_total": int(row["failed"] or 0),
            "record_storage_errors_total": int(
                self._metrics["record_storage_errors_total"]
            ),
            "record_images_bytes_total": int(row["image_bytes"] or 0),
            "record_masks_bytes_total": mask_bytes,
            "feedback_like_records": int(row["likes"] or 0),
            "feedback_dislike_records": int(row["dislikes"] or 0),
            "feedback_unrated_records": int(row["unrated"] or 0),
        }
