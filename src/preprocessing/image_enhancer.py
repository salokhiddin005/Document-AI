"""
Image Enhancer — improves low-quality photos before OCR.

Pipeline:
  1. Upscale small images (2x if width < 1000px)
  2. Denoise
  3. Sharpen
  4. Adaptive contrast (CLAHE)
  5. Deskew

Result: cleaner image → better Gemini Vision accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from loguru import logger


def enhance_for_ocr(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Enhance an image for better OCR accuracy.
    Accepts file path or numpy array (BGR or RGB).
    Returns enhanced numpy array (RGB).
    """
    # Load
    if isinstance(image, np.ndarray):
        img = image.copy()
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Cannot read image: {image}")

    h, w = img.shape[:2]
    logger.debug(f"Enhancing image: {w}x{h}px")

    # 1. Upscale if too small (Gemini works best at 1000+ px wide)
    if w < 1000:
        scale = 1000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Upscaled to {img.shape[1]}x{img.shape[0]}px")

    # 2. Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=8,
                                     templateWindowSize=7,
                                     searchWindowSize=21)

    # 4. Sharpen using unsharp mask
    blurred  = cv2.GaussianBlur(gray, (0, 0), 2)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # 5. CLAHE — adaptive contrast enhancement
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(sharpened)

    # 6. Deskew
    enhanced = _deskew(enhanced)

    # Convert back to RGB for Gemini
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    logger.info(f"Enhancement done: {result.shape[1]}x{result.shape[0]}px")
    return result


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct tilt using Hough line transform."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

    if lines is None:
        return gray

    angles = []
    for line in lines[:30]:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 20:
            angles.append(angle)

    if not angles:
        return gray

    skew = float(np.median(angles))
    if abs(skew) < 0.3:
        return gray

    h, w    = gray.shape
    M       = cv2.getRotationMatrix2D((w // 2, h // 2), skew, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=255)
    logger.debug(f"Deskewed: {skew:.2f}°")
    return rotated
