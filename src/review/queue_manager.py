"""
Human review queue — SQLite-backed.

When the confidence scorer decides a field needs human review, this module:
  1. Saves the image crop to disk.
  2. Inserts a queue record into SQLite.

Reviewers call fetch_pending() to get items, then submit_correction() to
record the verified value. Corrections are stored and can be exported for
model fine-tuning.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger
from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings, ReviewConfig


# ─────────────────────────────────────────────────────────────────────────────
#  ORM model
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class ReviewItem(Base):
    __tablename__ = "review_queue"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id     = Column(String, nullable=False, index=True)
    field_name      = Column(String, nullable=False)
    system_value    = Column(Text,   nullable=False)
    ocr_confidence  = Column(Float,  nullable=False)
    routing_reason  = Column(String, nullable=False)
    crop_path       = Column(String, nullable=True)
    status          = Column(String, nullable=False, default="pending")  # pending | corrected | escalated
    corrected_value = Column(Text,   nullable=True)
    reviewer_id     = Column(String, nullable=True)
    reviewed_at     = Column(DateTime, nullable=True)
    created_at      = Column(DateTime, nullable=False, default=datetime.utcnow)
    metadata_json   = Column(Text, nullable=True)   # arbitrary extra context


# ─────────────────────────────────────────────────────────────────────────────
#  Public dataclasses (no ORM coupling outside this module)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueueRecord:
    id: str
    document_id: str
    field_name: str
    system_value: str
    ocr_confidence: float
    routing_reason: str
    crop_path: Optional[str]
    status: str
    created_at: str


@dataclass
class CorrectionSubmission:
    item_id: str
    corrected_value: str
    reviewer_id: str
    escalate: bool = False   # True if reviewer cannot read the field


# ─────────────────────────────────────────────────────────────────────────────
#  Manager
# ─────────────────────────────────────────────────────────────────────────────

class ReviewQueueManager:

    def __init__(self, cfg: Optional[ReviewConfig] = None):
        self.cfg = cfg or settings.review
        self.cfg.crops_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.db_path.parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(
            f"sqlite:///{self.cfg.db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine)
        logger.info(f"Review queue DB: {self.cfg.db_path}")

    # ── Enqueue ───────────────────────────────────────────────────────────────

    def enqueue(
        self,
        document_id: str,
        field_name: str,
        system_value: str,
        ocr_confidence: float,
        routing_reason: str,
        crop_image: Optional[np.ndarray] = None,
        extra_metadata: Optional[dict] = None,
    ) -> str:
        """Add one field to the review queue. Returns the new item ID."""
        item_id = str(uuid.uuid4())
        crop_path = None

        if crop_image is not None:
            crop_path = self._save_crop(item_id, field_name, crop_image)

        with self._Session() as session:
            session.add(ReviewItem(
                id=item_id,
                document_id=document_id,
                field_name=field_name,
                system_value=system_value,
                ocr_confidence=ocr_confidence,
                routing_reason=routing_reason,
                crop_path=str(crop_path) if crop_path else None,
                metadata_json=json.dumps(extra_metadata) if extra_metadata else None,
            ))
            session.commit()

        logger.info(
            f"Enqueued review item {item_id} | doc={document_id} field={field_name}"
        )
        return item_id

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch_pending(self, limit: int = 50) -> List[QueueRecord]:
        """Return the oldest *limit* pending review items."""
        with self._Session() as session:
            rows = (
                session.query(ReviewItem)
                .filter(ReviewItem.status == "pending")
                .order_by(ReviewItem.created_at.asc())
                .limit(limit)
                .all()
            )
            return [self._to_record(r) for r in rows]

    def fetch_by_document(self, document_id: str) -> List[QueueRecord]:
        with self._Session() as session:
            rows = (
                session.query(ReviewItem)
                .filter(ReviewItem.document_id == document_id)
                .all()
            )
            return [self._to_record(r) for r in rows]

    def pending_count(self) -> int:
        with self._Session() as session:
            return session.query(ReviewItem).filter(
                ReviewItem.status == "pending"
            ).count()

    # ── Correct ───────────────────────────────────────────────────────────────

    def submit_correction(self, submission: CorrectionSubmission) -> bool:
        """Record a reviewer's correction. Returns True if item was found."""
        with self._Session() as session:
            item = session.get(ReviewItem, submission.item_id)
            if item is None:
                logger.warning(f"Review item not found: {submission.item_id}")
                return False

            item.corrected_value = submission.corrected_value
            item.reviewer_id     = submission.reviewer_id
            item.reviewed_at     = datetime.utcnow()
            item.status          = "escalated" if submission.escalate else "corrected"
            session.commit()

        logger.info(
            f"Correction submitted | item={submission.item_id} "
            f"value='{submission.corrected_value}' reviewer={submission.reviewer_id}"
        )
        return True

    # ── Training data export ──────────────────────────────────────────────────

    def export_corrections(self, output_path: Path) -> int:
        """
        Export all corrected items as a JSONL file for model fine-tuning.
        Returns number of records exported.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._Session() as session:
            rows = session.query(ReviewItem).filter(
                ReviewItem.status == "corrected"
            ).all()

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                record = {
                    "document_id":     row.document_id,
                    "field_name":      row.field_name,
                    "system_value":    row.system_value,
                    "corrected_value": row.corrected_value,
                    "ocr_confidence":  row.ocr_confidence,
                    "crop_path":       row.crop_path,
                    "reviewed_at":     str(row.reviewed_at),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"Exported {count} corrections to {output_path}")
        return count

    # ── Private ───────────────────────────────────────────────────────────────

    def _save_crop(
        self, item_id: str, field_name: str, image: np.ndarray
    ) -> Path:
        filename = f"{item_id}_{field_name}.png"
        path = self.cfg.crops_dir / filename
        if len(image.shape) == 3:
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(path), image)
        return path

    @staticmethod
    def _to_record(row: ReviewItem) -> QueueRecord:
        return QueueRecord(
            id=row.id,
            document_id=row.document_id,
            field_name=row.field_name,
            system_value=row.system_value,
            ocr_confidence=row.ocr_confidence,
            routing_reason=row.routing_reason,
            crop_path=row.crop_path,
            status=row.status,
            created_at=str(row.created_at),
        )
