"""
History Manager — saves ALL inputs and outputs professionally.

Each document gets a session (doc_id). Every operation on that document
(OCR, summary, translation, Q&A, key info, voice) is saved as a record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, desc
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ROOT = Path(__file__).parent.parent.parent


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    """One record = one operation performed on a document."""
    __tablename__ = "document_history"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(String, nullable=False, index=True)
    doc_id      = Column(String, nullable=False, index=True)
    op_type     = Column(String, nullable=False)   # ocr | summary | translation | keyinfo | qa | voice
    lang        = Column(String, default="auto")
    extra       = Column(String, nullable=True)    # JSON: target_lang, question, etc.
    content     = Column(Text,   nullable=False)   # the actual text output
    word_count  = Column(Integer, default=0)
    created_at  = Column(DateTime, default=datetime.utcnow)

    @property
    def extra_data(self) -> dict:
        try:
            return json.loads(self.extra or "{}")
        except Exception:
            return {}


@dataclass
class HistoryEntry:
    id:         int
    doc_id:     str
    op_type:    str
    lang:       str
    preview:    str
    word_count: int
    created_at: str
    full_text:  str
    extra:      dict


# Icons for each operation type
OP_ICONS = {
    "ocr":         "📖",
    "summary":     "🤖",
    "translation": "🌐",
    "keyinfo":     "🔑",
    "qa":          "❓",
    "voice":       "🎤",
    "handwriting": "✍️",
}

OP_LABELS = {
    "ocr":         "Read Handwriting",
    "summary":     "Summary",
    "translation": "Translation",
    "keyinfo":     "Key Info",
    "qa":          "Q&A",
    "voice":       "Voice",
    "handwriting": "Handwriting",
}


class HistoryManager:
    MAX_PER_USER = 200   # keep last 200 records per user

    def __init__(self):
        db_path = ROOT / "data" / "ocr_history.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine)
        logger.info(f"History DB: {db_path}")

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        user_id: str,
        doc_id: str,
        content: str,
        lang: str = "auto",
        op_type: str = "ocr",
        extra: dict | None = None,
    ) -> None:
        """Save any operation to history."""
        with self._Session() as session:
            session.add(DocumentRecord(
                user_id    = str(user_id),
                doc_id     = doc_id,
                op_type    = op_type,
                lang       = lang,
                extra      = json.dumps(extra or {}),
                content    = content,
                word_count = len(content.split()),
            ))
            session.commit()

            # Trim old records
            records = (
                session.query(DocumentRecord)
                .filter(DocumentRecord.user_id == str(user_id))
                .order_by(desc(DocumentRecord.created_at))
                .all()
            )
            if len(records) > self.MAX_PER_USER:
                for old in records[self.MAX_PER_USER:]:
                    session.delete(old)
                session.commit()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_user_history(self, user_id: str, limit: int = 10) -> list[HistoryEntry]:
        """Get latest records across all operation types."""
        with self._Session() as session:
            rows = (
                session.query(DocumentRecord)
                .filter(DocumentRecord.user_id == str(user_id))
                .order_by(desc(DocumentRecord.created_at))
                .limit(limit)
                .all()
            )
            return [self._to_entry(r) for r in rows]

    def get_document_history(self, user_id: str, doc_id: str) -> list[HistoryEntry]:
        """Get ALL operations performed on one document."""
        with self._Session() as session:
            rows = (
                session.query(DocumentRecord)
                .filter(
                    DocumentRecord.user_id == str(user_id),
                    DocumentRecord.doc_id  == doc_id,
                )
                .order_by(DocumentRecord.created_at.asc())
                .all()
            )
            return [self._to_entry(r) for r in rows]

    def get_by_doc_id(self, doc_id: str) -> HistoryEntry | None:
        """Get the OCR text for a document (used by button callbacks)."""
        with self._Session() as session:
            row = (
                session.query(DocumentRecord)
                .filter(
                    DocumentRecord.doc_id  == doc_id,
                    DocumentRecord.op_type == "ocr",
                )
                .first()
            )
            if not row:
                # Try any type for this doc_id
                row = session.query(DocumentRecord).filter(
                    DocumentRecord.doc_id == doc_id
                ).first()
            return self._to_entry(row) if row else None

    def get_by_id(self, record_id: int) -> HistoryEntry | None:
        with self._Session() as session:
            row = session.get(DocumentRecord, record_id)
            return self._to_entry(row) if row else None

    def search_history(self, user_id: str, keyword: str, limit: int = 10) -> list[HistoryEntry]:
        with self._Session() as session:
            rows = (
                session.query(DocumentRecord)
                .filter(
                    DocumentRecord.user_id == str(user_id),
                    DocumentRecord.content.ilike(f"%{keyword}%"),
                )
                .order_by(desc(DocumentRecord.created_at))
                .limit(limit)
                .all()
            )
            return [self._to_entry(r) for r in rows]

    def filter_by_lang(self, user_id: str, lang: str, limit: int = 20) -> list[HistoryEntry]:
        with self._Session() as session:
            rows = (
                session.query(DocumentRecord)
                .filter(
                    DocumentRecord.user_id == str(user_id),
                    DocumentRecord.lang    == lang,
                )
                .order_by(desc(DocumentRecord.created_at))
                .limit(limit)
                .all()
            )
            return [self._to_entry(r) for r in rows]

    def get_user_stats(self, user_id: str) -> dict:
        with self._Session() as session:
            rows = (
                session.query(DocumentRecord)
                .filter(DocumentRecord.user_id == str(user_id))
                .all()
            )
            docs = len(set(r.doc_id for r in rows))
            return {
                "processed": docs,
                "operations": len(rows),
                "words": sum(r.word_count for r in rows if r.op_type == "ocr"),
            }

    def clear_user_history(self, user_id: str) -> int:
        with self._Session() as session:
            count = session.query(DocumentRecord).filter(
                DocumentRecord.user_id == str(user_id)
            ).delete()
            session.commit()
            return count

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_entry(self, r: DocumentRecord) -> HistoryEntry:
        preview = r.content[:80].replace("\n", " ")
        if len(r.content) > 80:
            preview += "..."
        return HistoryEntry(
            id=r.id, doc_id=r.doc_id, op_type=r.op_type,
            lang=r.lang, preview=preview, word_count=r.word_count,
            created_at=str(r.created_at)[:16], full_text=r.content,
            extra=r.extra_data,
        )
