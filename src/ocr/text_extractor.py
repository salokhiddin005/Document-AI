"""
Handwritten text recognition using PaddleOCR 3.x.

PaddleOCR 3.x API changes vs 2.x:
  - Method:  .ocr(cls=...)  →  .predict(input)
  - Result:  [[box, (text,score)], ...]
             →  {"dt_polys": [...], "rec_texts": [...], "rec_scores": [...]}
  - Init:    use_gpu / use_angle_cls / det_db_thresh / rec_batch_num
             →  device / use_textline_orientation / text_det_thresh /
                text_recognition_batch_size / enable_mkldnn
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config import settings, OCRConfig

# ─────────────────────────────────────────────────────────────────────────────
#  Supported languages shown in the Telegram bot
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en":     "🇬🇧 English",
    "uz":     "🇺🇿 Uzbek",
    "korean": "🇰🇷 Korean",
    "ru":     "🇷🇺 Russian",
    "fr":     "🇫🇷 French",
    "de":     "🇩🇪 German",
    "es":     "🇪🇸 Spanish",
    "tr":     "🇹🇷 Turkish",
    "ar":     "🇸🇦 Arabic",
    "pl":     "🇵🇱 Polish",
    "it":     "🇮🇹 Italian",
    "pt":     "🇵🇹 Portuguese",
    "nl":     "🇳🇱 Dutch",
    "ja":     "🇯🇵 Japanese",
    "ch":     "🇨🇳 Chinese",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    """One detected text region in the document."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)


@dataclass
class FieldExtractionResult:
    """OCR result for a single field."""
    field_name: str
    raw_text: str
    confidence: float
    char_confidences: List[float]
    lines: List[str]


@dataclass
class FullPageOCRResult:
    """OCR result for a complete document image."""
    blocks: List[TextBlock]
    # Normalised to legacy [[box_points, (text, score)]] format so the
    # layout detector works without changes.
    raw_paddle_output: list


# ─────────────────────────────────────────────────────────────────────────────
#  Extractor
# ─────────────────────────────────────────────────────────────────────────────

class TextExtractor:
    """
    PaddleOCR 3.x wrapper.
    Supports dynamic language switching — one OCR instance cached per language.
    """

    def __init__(self, cfg: Optional[OCRConfig] = None):
        self.cfg        = cfg or settings.ocr
        self._ocr_cache: Dict[str, object] = {}   # lang → PaddleOCR instance
        self._ocr       = None                     # current active instance
        self._lang      = self.cfg.language[0] if self.cfg.language else "en"

    def set_language(self, lang: str) -> None:
        """Switch OCR language. Loads model if not cached yet."""
        if lang != self._lang:
            self._lang = lang
            self._ocr  = None   # force re-init with new lang
            logger.info(f"OCR language switched to: {lang}")

    # ── Public ────────────────────────────────────────────────────────────────

    def extract_full_page(self, image: np.ndarray) -> FullPageOCRResult:
        """Run OCR on the entire document. Returns all text blocks."""
        ocr = self._get_ocr()
        raw = ocr.predict(self._to_rgb(image))

        blocks: List[TextBlock] = []
        normalised: list = []   # legacy format for RegionDetector

        if not raw or raw[0] is None:
            logger.warning("PaddleOCR returned no results for full-page scan")
            return FullPageOCRResult(blocks=[], raw_paddle_output=[])

        page = raw[0]   # OCRResult dict-like object
        polys  = page.get("dt_polys",   [])
        texts  = page.get("rec_texts",  [])
        scores = page.get("rec_scores", [])

        for poly, text, score in zip(polys, texts, scores):
            text  = str(text).strip()
            score = float(score)
            if not text:
                continue

            pts = np.array(poly, dtype=np.int32)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())

            blocks.append(TextBlock(text=text, confidence=round(score, 4), bbox=(x1, y1, x2, y2)))

            # Normalise to legacy format: [box_points_list, (text, score)]
            box_points = pts.tolist()
            normalised.append([box_points, (text, score)])

        logger.info(f"Full-page OCR: {len(blocks)} text blocks detected")
        return FullPageOCRResult(blocks=blocks, raw_paddle_output=normalised)

    def extract_field(self, crop: np.ndarray, field_name: str) -> FieldExtractionResult:
        """Run OCR on a cropped field region."""
        if crop is None or crop.size == 0:
            return self._empty_result(field_name)

        ocr = self._get_ocr()
        raw = ocr.predict(self._to_rgb(crop))

        if not raw or raw[0] is None:
            logger.debug(f"No text detected in field: {field_name}")
            return self._empty_result(field_name)

        page   = raw[0]
        texts  = page.get("rec_texts",  [])
        scores = page.get("rec_scores", [])

        lines: List[str]  = []
        confidences: List[float] = []

        for text, score in zip(texts, scores):
            text = str(text).strip()
            if text:
                lines.append(text)
                confidences.append(float(score))

        if not lines:
            return self._empty_result(field_name)

        raw_text = " ".join(lines)
        avg_conf = sum(confidences) / len(confidences)

        logger.debug(f"Field '{field_name}': text='{raw_text}' conf={avg_conf:.3f}")

        return FieldExtractionResult(
            field_name=field_name,
            raw_text=raw_text,
            confidence=round(avg_conf, 4),
            char_confidences=confidences,
            lines=lines,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_ocr(self):
        lang = self._lang
        # Return cached instance for this language if available
        if lang in self._ocr_cache:
            self._ocr = self._ocr_cache[lang]
            return self._ocr

        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    lang=lang,
                    device=self.cfg.device,
                    use_textline_orientation=self.cfg.use_textline_orientation,
                    text_det_thresh=self.cfg.text_det_thresh,
                    text_det_box_thresh=self.cfg.text_det_box_thresh,
                    text_recognition_batch_size=self.cfg.text_recognition_batch_size,
                    enable_mkldnn=self.cfg.enable_mkldnn,
                )
                self._ocr_cache[lang] = self._ocr
                logger.info(f"PaddleOCR 3.x initialised (lang={lang}, device={self.cfg.device})")
            except ImportError:
                raise RuntimeError(
                    "PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle"
                )
        return self._ocr

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """PaddleOCR 3.x requires 3-channel RGB input — convert grayscale if needed."""
        import cv2
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    @staticmethod
    def _empty_result(field_name: str) -> FieldExtractionResult:
        return FieldExtractionResult(
            field_name=field_name,
            raw_text="",
            confidence=0.0,
            char_confidences=[],
            lines=[],
        )
