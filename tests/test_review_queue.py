"""Tests for the human review queue manager."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.review.queue_manager import CorrectionSubmission, ReviewQueueManager
from config import ReviewConfig


@pytest.fixture
def queue(tmp_path):
    cfg = ReviewConfig(
        db_path=tmp_path / "test_queue.db",
        crops_dir=tmp_path / "crops",
    )
    return ReviewQueueManager(cfg=cfg)


class TestReviewQueueManager:

    def test_enqueue_and_fetch(self, queue):
        item_id = queue.enqueue(
            document_id="DOC-001",
            field_name="name_field",
            system_value="Joh? Smith",
            ocr_confidence=0.55,
            routing_reason="low_confidence",
        )
        assert item_id is not None

        pending = queue.fetch_pending()
        assert len(pending) == 1
        assert pending[0].field_name == "name_field"
        assert pending[0].system_value == "Joh? Smith"

    def test_pending_count(self, queue):
        assert queue.pending_count() == 0
        queue.enqueue("DOC-001", "name_field", "Test", 0.4, "low_confidence")
        queue.enqueue("DOC-001", "id_field",   "A1",   0.3, "low_confidence")
        assert queue.pending_count() == 2

    def test_submit_correction_changes_status(self, queue):
        item_id = queue.enqueue("DOC-001", "name_field", "Joh?", 0.5, "low_confidence")

        success = queue.submit_correction(CorrectionSubmission(
            item_id=item_id,
            corrected_value="John",
            reviewer_id="rev_01",
        ))
        assert success is True

        # Should no longer appear in pending
        pending = queue.fetch_pending()
        assert len(pending) == 0

    def test_submit_correction_unknown_id(self, queue):
        success = queue.submit_correction(CorrectionSubmission(
            item_id="nonexistent-id",
            corrected_value="anything",
            reviewer_id="rev_01",
        ))
        assert success is False

    def test_enqueue_with_crop_saves_file(self, queue):
        crop = np.full((40, 120), 200, dtype=np.uint8)
        item_id = queue.enqueue(
            "DOC-001", "name_field", "Smith", 0.4, "low_confidence",
            crop_image=crop,
        )
        pending = queue.fetch_pending()
        assert pending[0].crop_path is not None
        assert Path(pending[0].crop_path).exists()

    def test_export_corrections(self, queue, tmp_path):
        item_id = queue.enqueue("DOC-001", "name_field", "Joh?", 0.5, "low_confidence")
        queue.submit_correction(CorrectionSubmission(
            item_id=item_id, corrected_value="John", reviewer_id="rev_01"
        ))

        export_path = tmp_path / "export.jsonl"
        count = queue.export_corrections(export_path)
        assert count == 1
        assert export_path.exists()

        import json
        lines = export_path.read_text(encoding="utf-8").strip().splitlines()
        record = json.loads(lines[0])
        assert record["corrected_value"] == "John"
        assert record["field_name"] == "name_field"

    def test_fetch_by_document(self, queue):
        queue.enqueue("DOC-001", "name_field", "A", 0.4, "low_confidence")
        queue.enqueue("DOC-001", "id_field",   "B", 0.3, "low_confidence")
        queue.enqueue("DOC-002", "name_field", "C", 0.5, "low_confidence")

        items_001 = queue.fetch_by_document("DOC-001")
        assert len(items_001) == 2

        items_002 = queue.fetch_by_document("DOC-002")
        assert len(items_002) == 1

    def test_escalate_sets_correct_status(self, queue):
        item_id = queue.enqueue("DOC-001", "name_field", "???", 0.1, "low_confidence")
        queue.submit_correction(CorrectionSubmission(
            item_id=item_id,
            corrected_value="",
            reviewer_id="rev_01",
            escalate=True,
        ))
        items = queue.fetch_by_document("DOC-001")
        assert items[0].status == "escalated"
