"""
Free-text page reader.

Converts raw OCR blocks (from PaddleOCR) into clean, reading-order text.

Steps:
  1. Sort blocks by vertical position (top → bottom)
  2. Merge blocks on the same visual line (left → right)
  3. Group lines into paragraphs (gap-based)
  4. Score each line's confidence
  5. Flag low-confidence lines for human review
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.ocr.text_extractor import TextBlock


# ─────────────────────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextLine:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    needs_review: bool = False


@dataclass
class Paragraph:
    lines: List[TextLine]
    text: str          # joined lines
    confidence: float  # average


@dataclass
class PageReadResult:
    lines: List[TextLine]
    paragraphs: List[Paragraph]
    full_text: str                  # entire page as plain text
    overall_confidence: float
    low_confidence_lines: List[TextLine]
    word_count: int


# ─────────────────────────────────────────────────────────────────────────────
#  Reader
# ─────────────────────────────────────────────────────────────────────────────

class PageReader:
    """
    Converts a flat list of OCR text blocks into structured page content.
    Works with ANY document type — no field labels required.
    """

    def __init__(
        self,
        line_merge_threshold: float = 0.6,   # fraction of avg line height for same-line merging
        paragraph_gap_threshold: float = 1.8, # fraction of avg line height to start new paragraph
        review_threshold: float = 0.70,       # lines below this go to human review
    ):
        self.line_merge_threshold     = line_merge_threshold
        self.paragraph_gap_threshold  = paragraph_gap_threshold
        self.review_threshold         = review_threshold

    # ── Public ────────────────────────────────────────────────────────────────

    def read(self, blocks: List[TextBlock]) -> PageReadResult:
        if not blocks:
            return PageReadResult(
                lines=[], paragraphs=[], full_text="",
                overall_confidence=0.0, low_confidence_lines=[], word_count=0,
            )

        avg_line_height = self._avg_line_height(blocks)
        lines           = self._merge_into_lines(blocks, avg_line_height)
        paragraphs      = self._group_into_paragraphs(lines, avg_line_height)
        full_text       = "\n\n".join(p.text for p in paragraphs)
        low_conf        = [l for l in lines if l.needs_review]
        overall_conf    = round(
            sum(l.confidence for l in lines) / len(lines), 3
        ) if lines else 0.0
        word_count      = len(full_text.split())

        return PageReadResult(
            lines=lines,
            paragraphs=paragraphs,
            full_text=full_text,
            overall_confidence=overall_conf,
            low_confidence_lines=low_conf,
            word_count=word_count,
        )

    # ── Step 1: sort blocks top → bottom ─────────────────────────────────────

    def _sorted_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    # ── Step 2: merge blocks on the same visual line ──────────────────────────

    def _merge_into_lines(
        self, blocks: List[TextBlock], avg_height: float
    ) -> List[TextLine]:
        sorted_blocks = self._sorted_blocks(blocks)
        threshold     = avg_height * self.line_merge_threshold

        lines: List[List[TextBlock]] = []
        current_group: List[TextBlock] = []

        for block in sorted_blocks:
            _, by1, _, by2 = block.bbox
            block_cy = (by1 + by2) / 2

            if not current_group:
                current_group.append(block)
                continue

            # Compare with the vertical centre of the current group
            group_ys = [(b.bbox[1] + b.bbox[3]) / 2 for b in current_group]
            group_cy = sum(group_ys) / len(group_ys)

            if abs(block_cy - group_cy) <= threshold:
                current_group.append(block)
            else:
                lines.append(current_group)
                current_group = [block]

        if current_group:
            lines.append(current_group)

        # Within each group: sort left → right, then build TextLine
        result: List[TextLine] = []
        for group in lines:
            group_sorted = sorted(group, key=lambda b: b.bbox[0])
            text  = " ".join(b.text for b in group_sorted)
            conf  = round(sum(b.confidence for b in group_sorted) / len(group_sorted), 3)
            x1    = min(b.bbox[0] for b in group_sorted)
            y1    = min(b.bbox[1] for b in group_sorted)
            x2    = max(b.bbox[2] for b in group_sorted)
            y2    = max(b.bbox[3] for b in group_sorted)
            result.append(TextLine(
                text=text, confidence=conf, bbox=(x1, y1, x2, y2),
                needs_review=(conf < self.review_threshold),
            ))

        return result

    # ── Step 3: group lines into paragraphs ───────────────────────────────────

    def _group_into_paragraphs(
        self, lines: List[TextLine], avg_height: float
    ) -> List[Paragraph]:
        if not lines:
            return []

        gap_threshold = avg_height * self.paragraph_gap_threshold
        paragraphs: List[List[TextLine]] = []
        current: List[TextLine] = [lines[0]]

        for prev, curr in zip(lines, lines[1:]):
            gap = curr.bbox[1] - prev.bbox[3]   # top of curr − bottom of prev
            if gap > gap_threshold:
                paragraphs.append(current)
                current = []
            current.append(curr)

        if current:
            paragraphs.append(current)

        result: List[Paragraph] = []
        for group in paragraphs:
            text = "\n".join(l.text for l in group)
            conf = round(sum(l.confidence for l in group) / len(group), 3)
            result.append(Paragraph(lines=group, text=text, confidence=conf))

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _avg_line_height(blocks: List[TextBlock]) -> float:
        heights = [b.bbox[3] - b.bbox[1] for b in blocks]
        return sum(heights) / len(heights) if heights else 20.0
