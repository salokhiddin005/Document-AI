"""
OCR History Manager — stores and retrieves past OCR results per user.
SQLite-backed, max 50 records per user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ROOT = Path(__file__).parent.parent.parent


class Base(DeclarativeBase):
    pass


class HistoryRecord(Base):
    __tablename__ = "ocr_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(String, nullable=False, index=True)
    doc_id     = Column(String, nullable=False)
    full_text  = Column(Text, nullable=False)
    lang       = Column(String, default="en")
    word_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


@dataclass
class HistoryEntry:
    id:         int
    doc_id:     str
    preview:    str      # first 80 chars
    lang:       str
    word_count: int
    created_at: str
    full_text:  str


class HistoryManager:
    MAX_PER_USER = 50

    def __init__(self):
        db_path = ROOT / "data" / "ocr_history.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(engine)
        self._Session = sessionmaker(bind=engine)

    def save(self, user_id: str, doc_id: str, full_text: str, lang: str = "en") -> None:
        with self._Session() as session:
            session.add(HistoryRecord(
                user_id=str(user_id),
                doc_id=doc_id,
                full_text=full_text,
                lang=lang,
                word_count=len(full_text.split()),
            ))
            session.commit()

            # Keep only latest MAX_PER_USER records per user
            records = (
                session.query(HistoryRecord)
                .filter(HistoryRecord.user_id == str(user_id))
                .order_by(HistoryRecord.created_at.desc())
                .all()
            )
            if len(records) > self.MAX_PER_USER:
                for old in records[self.MAX_PER_USER:]:
                    session.delete(old)
                session.commit()

    def get_user_history(self, user_id: str, limit: int = 10) -> list[HistoryEntry]:
        with self._Session() as session:
            records = (
                session.query(HistoryRecord)
                .filter(HistoryRecord.user_id == str(user_id))
                .order_by(HistoryRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                HistoryEntry(
                    id=r.id,
                    doc_id=r.doc_id,
                    preview=r.full_text[:80].replace("\n", " ") + ("..." if len(r.full_text) > 80 else ""),
                    lang=r.lang,
                    word_count=r.word_count,
                    created_at=str(r.created_at)[:16],
                    full_text=r.full_text,
                )
                for r in records
            ]

    def get_by_id(self, record_id: int) -> HistoryEntry | None:
        with self._Session() as session:
            r = session.get(HistoryRecord, record_id)
            if not r:
                return None
            return HistoryEntry(
                id=r.id, doc_id=r.doc_id,
                preview=r.full_text[:80], lang=r.lang,
                word_count=r.word_count, created_at=str(r.created_at)[:16],
                full_text=r.full_text,
            )

    def get_by_doc_id(self, doc_id: str) -> HistoryEntry | None:
        """Look up a record by its document ID (e.g. 'DOC-ABCD1234')."""
        with self._Session() as session:
            r = (
                session.query(HistoryRecord)
                .filter(HistoryRecord.doc_id == doc_id)
                .first()
            )
            if not r:
                return None
            return HistoryEntry(
                id=r.id, doc_id=r.doc_id,
                preview=r.full_text[:80], lang=r.lang,
                word_count=r.word_count, created_at=str(r.created_at)[:16],
                full_text=r.full_text,
            )

    def get_user_stats(self, user_id: str) -> dict:
        """Compute persistent stats from the database."""
        with self._Session() as session:
            records = (
                session.query(HistoryRecord)
                .filter(HistoryRecord.user_id == str(user_id))
                .all()
            )
            return {
                "processed":  len(records),
                "words":      sum(r.word_count for r in records),
            }

    def search_history(self, user_id: str, keyword: str, limit: int = 10) -> list[HistoryEntry]:
        """Search user's history by keyword in text content."""
        with self._Session() as session:
            rows = (
                session.query(HistoryRecord)
                .filter(
                    HistoryRecord.user_id == str(user_id),
                    HistoryRecord.full_text.ilike(f"%{keyword}%"),
                )
                .order_by(HistoryRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [self._to_entry(r) for r in rows]

    def filter_by_lang(self, user_id: str, lang: str, limit: int = 20) -> list[HistoryEntry]:
        """Filter history by language."""
        with self._Session() as session:
            rows = (
                session.query(HistoryRecord)
                .filter(
                    HistoryRecord.user_id == str(user_id),
                    HistoryRecord.lang == lang,
                )
                .order_by(HistoryRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [self._to_entry(r) for r in rows]

    def _to_entry(self, r: HistoryRecord) -> HistoryEntry:
        return HistoryEntry(
            id=r.id, doc_id=r.doc_id,
            preview=(r.full_text[:80] + "...") if len(r.full_text) > 80 else r.full_text,
            lang=r.lang, word_count=r.word_count,
            created_at=str(r.created_at)[:16], full_text=r.full_text,
        )

    def clear_user_history(self, user_id: str) -> int:
        with self._Session() as session:
            count = session.query(HistoryRecord).filter(
                HistoryRecord.user_id == str(user_id)
            ).delete()
            session.commit()
            return count
