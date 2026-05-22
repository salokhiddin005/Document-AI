"""
Document processing pipeline — free-text mode.

Works with ANY handwritten document regardless of structure.
No field labels required.

Output:
  - Full extracted text (reading order)
  - Line-by-line breakdown with confidence
  - Paragraph grouping
  - Barcode value (if present)
  - Low-confidence lines routed to human review
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger

from config import settings
from src.confidence.scorer import RoutingDecision
from src.ocr.barcode_reader import BarcodeReader, BarcodeResult
from src.ocr.page_reader import PageReadResult, PageReader, TextLine
from src.ocr.postprocessor import correct as ocr_correct
from src.ocr.text_extractor import TextExtractor
from src.preprocessing.image_processor import ImageProcessor, PreprocessingResult
from src.review.queue_manager import ReviewQueueManager


# ─────────────────────────────────────────────────────────────────────────────
#  Output types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LineOutput:
    text: str
    confidence: float
    needs_review: bool
    bbox: List[int]


@dataclass
class ParagraphOutput:
    text: str
    confidence: float


@dataclass
class DocumentResult:
    document_id: str
    status: str                          # "processed" | "low_quality" | "error"
    full_text: str                       # entire page as plain text
    lines: List[LineOutput]
    paragraphs: List[ParagraphOutput]
    barcode: Optional[str]               # decoded barcode value
    barcode_confidence: float
    overall_confidence: float
    word_count: int
    requires_human_review: bool
    low_confidence_lines: List[LineOutput]
    processing_time_ms: int
    image_quality: float
    skew_angle: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        from datetime import datetime

        # ── Document metadata ─────────────────────────────────────────────────
        doc = {
            "id":           self.document_id,
            "status":       self.status,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.error:
            doc["error"] = self.error

        # ── Quality metrics ───────────────────────────────────────────────────
        quality = {
            "image_score":        round(self.image_quality, 2),
            "ocr_confidence":     round(self.overall_confidence, 2),
            "skew_corrected_deg": round(self.skew_angle, 2),
            "processing_time_ms": self.processing_time_ms,
        }

        # ── Content summary ───────────────────────────────────────────────────
        content = {
            "word_count":      self.word_count,
            "line_count":      len(self.lines),
            "paragraph_count": len(self.paragraphs),
            # full_text as a list of strings — one entry per line, easy to read
            "full_text": self.full_text.split("\n") if self.full_text else [],
        }

        # ── Lines — compact bbox on one line ──────────────────────────────────
        lines = [
            {
                "line":       i,
                "text":       l.text,
                "confidence": round(l.confidence, 2),
                "bbox":       l.bbox,          # already a list [x1,y1,x2,y2]
                **({"needs_review": True} if l.needs_review else {}),
            }
            for i, l in enumerate(self.lines, start=1)
        ]

        # ── Paragraphs — list of text lines per paragraph ─────────────────────
        paragraphs = [
            {
                "paragraph":  i,
                "confidence": round(p.confidence, 2),
                "text":       p.text.split("\n"),
            }
            for i, p in enumerate(self.paragraphs, start=1)
        ]

        # ── Review — only included when there are flagged lines ───────────────
        review: dict = {"required": self.requires_human_review}
        if self.low_confidence_lines:
            review["flagged_lines"] = [
                {
                    "line":       l.text,
                    "confidence": round(l.confidence, 2),
                    "bbox":       l.bbox,
                }
                for l in self.low_confidence_lines
            ]

        # ── Barcode — only included when detected ─────────────────────────────
        result: dict = {
            "document":   doc,
            "quality":    quality,
            "content":    content,
            "lines":      lines,
            "paragraphs": paragraphs,
            "review":     review,
        }
        if self.barcode:
            result["barcode"] = {
                "value":      self.barcode,
                "confidence": round(self.barcode_confidence, 2),
            }

        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DocumentPipeline:
    """End-to-end pipeline for any handwritten document."""

    def __init__(self):
        self.preprocessor = ImageProcessor()
        self.extractor    = TextExtractor()
        self.page_reader  = PageReader()
        self.barcode      = BarcodeReader()
        self.queue        = ReviewQueueManager()

    # ── Public ────────────────────────────────────────────────────────────────

    def process(
        self,
        source: Union[str, Path, np.ndarray],
        document_id: Optional[str] = None,
    ) -> DocumentResult:
        document_id = document_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
        t_start     = time.time()
        logger.info(f"Pipeline start | doc_id={document_id}")

        # ── Stage 1: Preprocessing ────────────────────────────────────────────
        prep: PreprocessingResult = self.preprocessor.process(source)
        if not prep.success:
            return self._error_result(document_id, prep.error or "preprocessing_failed", t_start)

        low_quality = prep.quality_score < settings.preprocessing.min_quality_score
        if low_quality:
            logger.warning(f"Low image quality: {prep.quality_score:.2f}")

        # ── Stage 2: Full-page OCR (single pass — no per-field calls) ─────────
        full_page = self.extractor.extract_full_page(prep.processed_image)

        if not full_page.blocks:
            logger.warning("No text detected on page")

        # ── Stage 3: Reading-order assembly ───────────────────────────────────
        page: PageReadResult = self.page_reader.read(full_page.blocks)

        # ── Stage 4: Barcode ──────────────────────────────────────────────────
        barcode_result: BarcodeResult = self.barcode.decode(prep.binary_image)

        # ── Stage 5: Route low-confidence lines to human review ───────────────
        for line in page.low_confidence_lines:
            # Find the image crop for this line
            crop = ImageProcessor.crop_region(
                prep.processed_image, tuple(line.bbox)
            )
            self.queue.enqueue(
                document_id=document_id,
                field_name="text_line",
                system_value=line.text,
                ocr_confidence=line.confidence,
                routing_reason="low_ocr_confidence",
                crop_image=crop,
            )

        # ── Stage 6: Build output ─────────────────────────────────────────────
        elapsed_ms = int((time.time() - t_start) * 1000)

        line_outputs = [
            LineOutput(
                text=l.text,
                confidence=l.confidence,
                needs_review=l.needs_review,
                bbox=list(l.bbox),
            )
            for l in page.lines
        ]

        para_outputs = [
            ParagraphOutput(text=p.text, confidence=p.confidence)
            for p in page.paragraphs
        ]

        low_conf_outputs = [
            LineOutput(
                text=l.text,
                confidence=l.confidence,
                needs_review=True,
                bbox=list(l.bbox),
            )
            for l in page.low_confidence_lines
        ]

        status = "low_quality" if low_quality else "processed"

        # Drop garbled first line: if first line confidence << average,
        # it's likely a partial/cut-off line at the top of the image.
        if page.lines and len(page.lines) > 2:
            first_conf = page.lines[0].confidence
            avg_conf   = page.overall_confidence
            if first_conf < avg_conf * 0.60:   # first line much worse than average
                logger.warning(
                    f"Dropping low-confidence first line "
                    f"({first_conf:.2f} vs avg {avg_conf:.2f}): "
                    f"'{page.lines[0].text[:50]}'"
                )
                clean_lines = page.lines[1:]
                raw_text    = "\n".join(l.text for l in clean_lines)
            else:
                raw_text = page.full_text
        else:
            raw_text = page.full_text

        # Apply language-aware post-correction
        lang      = self.extractor._lang if hasattr(self.extractor, '_lang') else "en"
        corrected = ocr_correct(raw_text, lang=lang)

        result = DocumentResult(
            document_id=document_id,
            status=status,
            full_text=corrected,
            lines=line_outputs,
            paragraphs=para_outputs,
            barcode=barcode_result.value if barcode_result.detected else None,
            barcode_confidence=barcode_result.confidence,
            overall_confidence=page.overall_confidence,
            word_count=page.word_count,
            requires_human_review=len(page.low_confidence_lines) > 0,
            low_confidence_lines=low_conf_outputs,
            processing_time_ms=elapsed_ms,
            image_quality=prep.quality_score,
            skew_angle=prep.skew_angle,
        )

        logger.info(
            f"Pipeline complete | doc_id={document_id} words={page.word_count} "
            f"lines={len(page.lines)} conf={page.overall_confidence:.2f} "
            f"review={len(page.low_confidence_lines)} time={elapsed_ms}ms"
        )
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _error_result(doc_id: str, error: str, t_start: float) -> DocumentResult:
        elapsed = int((time.time() - t_start) * 1000)
        return DocumentResult(
            document_id=doc_id, status="error",
            full_text="", lines=[], paragraphs=[],
            barcode=None, barcode_confidence=0.0,
            overall_confidence=0.0, word_count=0,
            requires_human_review=True, low_confidence_lines=[],
            processing_time_ms=elapsed, image_quality=0.0,
            skew_angle=0.0, error=error,
        )
