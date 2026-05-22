"""
Image preprocessing pipeline.

Converts a raw scanned/photographed document into a clean, normalised image
ready for layout detection and OCR.

Steps (in order):
  1. Load & validate
  2. Convert to grayscale
  3. Deskew (straighten tilted scans)
  4. Denoise
  5. Contrast enhancement (CLAHE)
  6. Binarisation (adaptive threshold)
  7. Quality scoring
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from config import settings, PreprocessingConfig


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessingResult:
    success: bool
    original_shape: Tuple[int, int, int]   # (H, W, C)
    processed_image: Optional[np.ndarray]  # cleaned RGB image
    gray_image: Optional[np.ndarray]       # grayscale version (used by OCR)
    binary_image: Optional[np.ndarray]     # binarised version (used by barcode)
    skew_angle: float                      # degrees corrected
    quality_score: float                   # 0.0 – 1.0
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Processor
# ─────────────────────────────────────────────────────────────────────────────

class ImageProcessor:
    """Stateless image preprocessing.  All methods are pure functions of the input."""

    def __init__(self, cfg: Optional[PreprocessingConfig] = None):
        self.cfg = cfg or settings.preprocessing

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, source: Union[str, Path, np.ndarray]) -> PreprocessingResult:
        """Run the full preprocessing pipeline on *source* (file path or numpy array)."""
        image = self._load(source)
        if image is None:
            return PreprocessingResult(
                success=False, original_shape=(0, 0, 0),
                processed_image=None, gray_image=None, binary_image=None,
                skew_angle=0.0, quality_score=0.0,
                error="Could not load image",
            )

        original_shape = image.shape

        gray      = self._to_gray(image)
        angle     = self._detect_skew(gray)
        gray      = self._rotate(gray, angle)
        image     = self._rotate(image, angle)
        gray      = self._denoise(gray)
        gray_clahe = self._enhance_contrast(gray)
        binary    = self._binarise(gray_clahe)
        quality   = self._quality_score(gray_clahe)

        if quality < self.cfg.min_quality_score:
            logger.warning(f"Low image quality score: {quality:.2f}")

        logger.info(
            f"Preprocessing done | skew={angle:.1f}° | quality={quality:.2f} | "
            f"shape={original_shape}"
        )

        return PreprocessingResult(
            success=True,
            original_shape=original_shape,
            processed_image=image,
            gray_image=gray_clahe,
            binary_image=binary,
            skew_angle=angle,
            quality_score=quality,
        )

    # ── Internal steps ────────────────────────────────────────────────────────

    def _load(self, source: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        if isinstance(source, np.ndarray):
            return source.copy()
        path = Path(source)
        if not path.exists():
            logger.error(f"Image not found: {path}")
            return None
        img = cv2.imread(str(path))
        if img is None:
            logger.error(f"OpenCV could not decode: {path}")
            return None
        # Always work in RGB internally
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _detect_skew(self, gray: np.ndarray) -> float:
        """Estimate document rotation angle via Hough line transform."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return 0.0

        angles = []
        for line in lines[:50]:  # use top 50 lines only
            rho, theta = line[0]
            angle = math.degrees(theta) - 90
            if abs(angle) < self.cfg.max_skew_angle:
                angles.append(angle)

        if not angles:
            return 0.0

        # Use median — robust against outlier lines
        skew = float(np.median(angles))
        logger.debug(f"Detected skew: {skew:.2f}°")
        return skew

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 0.1:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Use white background fill (255) for document images
        fill = 255 if len(image.shape) == 2 else (255, 255, 255)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )
        return rotated

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            gray,
            h=self.cfg.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE — improves local contrast without blowing out bright regions."""
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg.contrast_clip_limit,
            tileGridSize=(self.cfg.contrast_tile_size, self.cfg.contrast_tile_size),
        )
        return clahe.apply(gray)

    def _binarise(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive threshold — handles uneven illumination across the scan."""
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

    def _quality_score(self, gray: np.ndarray) -> float:
        """
        Heuristic quality score in [0, 1].
        Combines:
          - Laplacian variance (sharpness)
          - Contrast (std of pixel values)
        """
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 500.0, 1.0)   # normalised to ~[0,1]
        contrast  = float(gray.std()) / 128.0
        contrast  = min(contrast, 1.0)
        return round((sharpness * 0.6 + contrast * 0.4), 3)

    # ── Utility helpers (used by other modules) ───────────────────────────────

    @staticmethod
    def crop_region(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop image to bbox (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    @staticmethod
    def save(image: np.ndarray, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(image.shape) == 3:
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(path), image)
