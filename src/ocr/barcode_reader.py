"""
Barcode detection and decoding.

Decoder priority:
  1. pyzbar  — best format coverage (CODE_128, EAN_13, QR, etc.)
  2. OpenCV  — automatic fallback if pyzbar DLL is missing on Windows
               (cv2.barcode.BarcodeDetector from opencv-contrib)

Both decoders retry with contrast enhancement and small rotations
if the first pass fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config import settings, BarcodeConfig


# ─────────────────────────────────────────────────────────────────────────────
#  Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BarcodeResult:
    detected: bool
    value: Optional[str]
    barcode_format: Optional[str]
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]
    attempts: int


# ─────────────────────────────────────────────────────────────────────────────
#  Reader
# ─────────────────────────────────────────────────────────────────────────────

class BarcodeReader:

    def __init__(self, cfg: Optional[BarcodeConfig] = None):
        self.cfg = cfg or settings.barcode
        self._use_pyzbar = self._pyzbar_available()
        if self._use_pyzbar:
            logger.info("Barcode backend: pyzbar")
        else:
            logger.info("Barcode backend: OpenCV (pyzbar DLL not found on this system)")

    # ── Public ────────────────────────────────────────────────────────────────

    def decode(self, region: np.ndarray) -> BarcodeResult:
        if region is None or region.size == 0:
            return self._no_result(0)

        gray     = self._to_gray(region)
        enhanced = self._enhance(gray)

        for attempt, image in enumerate([gray, enhanced], start=1):
            result = self._decode_once(image, attempt)
            if result:
                return result

            # Retry with small rotations on the enhanced image
            if attempt == 2:
                for i in range(1, self.cfg.scan_attempts + 1):
                    rotated = self._rotate(enhanced, i * self.cfg.rotation_step)
                    result  = self._decode_once(rotated, attempt + i)
                    if result:
                        return result

        logger.debug("Barcode: no decode after all attempts")
        return self._no_result(self.cfg.scan_attempts + 2)

    def decode_from_full_image(self, image: np.ndarray) -> BarcodeResult:
        return self.decode(image)

    # ── Dispatch to available backend ─────────────────────────────────────────

    def _decode_once(self, gray: np.ndarray, attempt: int) -> Optional[BarcodeResult]:
        if self._use_pyzbar:
            return self._try_pyzbar(gray, attempt)
        return self._try_opencv(gray, attempt)

    # ── pyzbar backend ────────────────────────────────────────────────────────

    def _try_pyzbar(self, gray: np.ndarray, attempt: int) -> Optional[BarcodeResult]:
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            objects = pyzbar_decode(gray)
        except Exception as exc:
            logger.warning(f"pyzbar error: {exc} — switching to OpenCV backend")
            self._use_pyzbar = False
            return self._try_opencv(gray, attempt)

        if not objects:
            return None

        best  = max(objects, key=lambda d: d.rect.width * d.rect.height)
        value = best.data.decode("utf-8", errors="replace").strip()
        if not value:
            return None

        r    = best.rect
        bbox = (r.left, r.top, r.left + r.width, r.top + r.height)
        conf = round(max(0.99 - (attempt - 1) * 0.08, 0.70), 2)

        logger.info(f"Barcode (pyzbar): '{value}' format={best.type} attempt={attempt}")
        return BarcodeResult(
            detected=True, value=value, barcode_format=best.type,
            confidence=conf, bbox=bbox, attempts=attempt,
        )

    # ── OpenCV backend ────────────────────────────────────────────────────────

    def _try_opencv(self, gray: np.ndarray, attempt: int) -> Optional[BarcodeResult]:
        try:
            detector = cv2.barcode.BarcodeDetector()
            ok, decoded_info, decoded_type, points = detector.detectAndDecodeWithType(gray)
        except AttributeError:
            # cv2.barcode not available in this build
            return None
        except Exception as exc:
            logger.debug(f"OpenCV barcode error: {exc}")
            return None

        if not ok or not decoded_info:
            return None

        # decoded_info and decoded_type are tuples of results
        for info, fmt, pts in zip(decoded_info, decoded_type, points or [None]*len(decoded_info)):
            value = str(info).strip()
            if not value:
                continue

            bbox = None
            if pts is not None:
                pts = pts.astype(int)
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                bbox = (x1, y1, x2, y2)

            conf = round(max(0.95 - (attempt - 1) * 0.08, 0.65), 2)
            logger.info(f"Barcode (OpenCV): '{value}' format={fmt} attempt={attempt}")
            return BarcodeResult(
                detected=True, value=value, barcode_format=str(fmt),
                confidence=conf, bbox=bbox, attempts=attempt,
            )

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enhance(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        sharp   = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _no_result(attempts: int) -> BarcodeResult:
        return BarcodeResult(detected=False, value=None, barcode_format=None,
                             confidence=0.0, bbox=None, attempts=attempts)

    @staticmethod
    def _pyzbar_available() -> bool:
        """Check if pyzbar and its native DLL load cleanly."""
        try:
            from pyzbar.pyzbar import decode  # noqa: F401
            return True
        except Exception:
            return False
