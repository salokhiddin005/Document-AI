"""
Document layout / region detector.

Strategy: keyword-anchor detection.
  1. Run a fast full-image OCR pass to find all text blocks.
  2. Identify "label" blocks (e.g. "Full Name:", "ID Number:").
  3. The VALUE region is the area to the right of / below that label.
  4. Special zone: detect the barcode area by high-frequency gradient analysis.

This approach works without a trained object-detection model and can bootstrap
a labelled dataset for a future YOLO/LayoutLM detector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────────────────────

BBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)


@dataclass
class DocumentRegion:
    label: str          # semantic name, e.g. "name_field"
    bbox: BBox          # (x1, y1, x2, y2) in pixels
    confidence: float   # detector confidence 0–1
    source: str         # "keyword_anchor" | "barcode_gradient" | "fallback"


@dataclass
class LayoutResult:
    regions: List[DocumentRegion]
    image_shape: Tuple[int, int]   # (H, W)

    def get(self, label: str) -> Optional[DocumentRegion]:
        for r in self.regions:
            if r.label == label:
                return r
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Field keyword configuration
#  Maps a semantic field name → list of label strings to look for in the OCR.
#  Add Korean labels here as you extend the system.
# ─────────────────────────────────────────────────────────────────────────────

FIELD_KEYWORDS: Dict[str, List[str]] = {
    "name_field":     ["full name", "name", "이름", "성명"],
    "id_field":       ["id number", "id", "등록번호", "번호"],
    "category_field": ["category", "type", "분류", "종류"],
    "address_field":  ["address", "주소"],
    "phone_field":    ["phone", "tel", "전화"],
    "notes_field":    ["notes", "note", "remark", "비고", "메모"],
    "date_field":     ["date", "날짜", "일자"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Detector
# ─────────────────────────────────────────────────────────────────────────────

class RegionDetector:
    """
    Detects semantic field regions in a document image.

    Requires an OCR result (list of (bbox, text, confidence) tuples from
    PaddleOCR) plus the preprocessed grayscale image for barcode zone detection.
    """

    def __init__(self, padding: int = 6):
        # Extra pixels added around each detected value region
        self.padding = padding

    # ── Public ────────────────────────────────────────────────────────────────

    def detect(
        self,
        gray_image: np.ndarray,
        ocr_blocks: List[Tuple[List, str, float]],  # raw PaddleOCR output
    ) -> LayoutResult:
        h, w = gray_image.shape[:2]

        normalised = self._normalise_blocks(ocr_blocks, w, h)
        regions: List[DocumentRegion] = []

        # 1. Keyword-anchor field detection
        for field_label, keywords in FIELD_KEYWORDS.items():
            region = self._find_field_region(normalised, keywords, w, h)
            if region:
                region.label = field_label
                regions.append(region)

        # 2. Barcode zone via gradient analysis
        barcode_region = self._find_barcode_zone(gray_image)
        if barcode_region:
            regions.append(barcode_region)

        # 3. Fallback: if critical fields not found, use full-image strips
        found_labels = {r.label for r in regions}
        for field_label in FIELD_KEYWORDS:
            if field_label not in found_labels:
                regions.append(self._fallback_region(field_label, w, h))

        logger.info(f"Layout detection: {len(regions)} regions found")
        return LayoutResult(regions=regions, image_shape=(h, w))

    # ── Keyword anchor matching ───────────────────────────────────────────────

    def _normalise_blocks(
        self,
        ocr_blocks: List,
        img_w: int,
        img_h: int,
    ) -> List[Dict]:
        """Convert raw PaddleOCR output to normalised dicts."""
        result = []
        for block in ocr_blocks:
            # PaddleOCR format: [[box_points], (text, score)]
            if len(block) == 2:
                box_points, (text, score) = block
            else:
                continue

            pts = np.array(box_points, dtype=np.int32)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            result.append({
                "text": text.strip(),
                "text_lower": text.strip().lower(),
                "bbox": (x1, y1, x2, y2),
                "score": float(score),
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
            })
        return result

    def _find_field_region(
        self,
        blocks: List[Dict],
        keywords: List[str],
        img_w: int,
        img_h: int,
    ) -> Optional[DocumentRegion]:
        """
        Find the value region associated with a label keyword.

        Strategy:
          - Find the block whose text best matches a keyword.
          - The value region is the bounding box of all blocks that are:
              * to the right of the label on the same row, OR
              * immediately below the label row.
        """
        label_block = self._best_keyword_match(blocks, keywords)
        if label_block is None:
            return None

        lx1, ly1, lx2, ly2 = label_block["bbox"]
        row_center = (ly1 + ly2) // 2
        row_tolerance = (ly2 - ly1) * 1.2   # vertical proximity threshold

        # Collect value blocks: right of label on same row, or row below
        value_blocks = []
        for b in blocks:
            bx1, by1, bx2, by2 = b["bbox"]
            b_center_y = (by1 + by2) // 2
            is_same_row = abs(b_center_y - row_center) < row_tolerance
            is_right_of_label = bx1 > lx2
            is_below = by1 > ly2 and (by1 - ly2) < row_tolerance * 2

            if b is label_block:
                continue
            if (is_same_row and is_right_of_label) or is_below:
                value_blocks.append(b)

        if not value_blocks:
            return None

        # Merge all value block boxes
        all_x1 = min(b["bbox"][0] for b in value_blocks)
        all_y1 = min(b["bbox"][1] for b in value_blocks)
        all_x2 = max(b["bbox"][2] for b in value_blocks)
        all_y2 = max(b["bbox"][3] for b in value_blocks)

        # Add padding, clamp to image
        p = self.padding
        bbox: BBox = (
            max(0, all_x1 - p),
            max(0, all_y1 - p),
            min(img_w, all_x2 + p),
            min(img_h, all_y2 + p),
        )

        avg_score = sum(b["score"] for b in value_blocks) / len(value_blocks)

        return DocumentRegion(
            label="",          # caller sets this
            bbox=bbox,
            confidence=round(avg_score, 3),
            source="keyword_anchor",
        )

    def _best_keyword_match(
        self, blocks: List[Dict], keywords: List[str]
    ) -> Optional[Dict]:
        """Find the OCR block whose text best matches any keyword."""
        best_block = None
        best_score = 0.0

        for b in blocks:
            text = b["text_lower"].rstrip(":").strip()
            for kw in keywords:
                if kw in text or text in kw:
                    # Longer matches are more specific
                    score = len(kw) / max(len(text), 1)
                    if score > best_score:
                        best_score = score
                        best_block = b

        return best_block if best_score > 0.3 else None

    # ── Barcode zone detection ────────────────────────────────────────────────

    def _find_barcode_zone(self, gray: np.ndarray) -> Optional[DocumentRegion]:
        """
        Barcodes have very high horizontal frequency (alternating black/white bars).
        Detect by computing horizontal gradient magnitude and finding the region
        with the highest density of high-frequency transitions.
        """
        h, w = gray.shape

        # Horizontal Sobel gradient
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        magnitude = np.abs(sobel_x)

        # Threshold to find high-gradient pixels
        _, mask = cv2.threshold(
            magnitude.astype(np.uint8), 50, 255, cv2.THRESH_BINARY
        )

        # Morphological close to merge nearby bar regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Pick the contour with the highest aspect ratio (wider than tall = barcode)
        best = None
        best_score = 0.0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < 2000:
                continue
            aspect = cw / max(ch, 1)
            if aspect > 2.0 and aspect > best_score:
                best_score = aspect
                best = (x, y, x + cw, y + ch)

        if best is None:
            return None

        return DocumentRegion(
            label="barcode",
            bbox=best,
            confidence=min(best_score / 10.0, 0.99),
            source="barcode_gradient",
        )

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback_region(self, label: str, img_w: int, img_h: int) -> DocumentRegion:
        """Return the full image as the region when detection failed."""
        return DocumentRegion(
            label=label,
            bbox=(0, 0, img_w, img_h),
            confidence=0.0,
            source="fallback",
        )
