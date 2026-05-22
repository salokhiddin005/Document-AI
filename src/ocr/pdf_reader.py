"""
PDF Reader — converts PDF pages to images for Gemini Vision OCR.
Uses PyMuPDF (fitz). Each page rendered at 200 DPI.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from loguru import logger


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> List[np.ndarray]:
    """
    Convert each page of a PDF to an RGB numpy array.
    Returns list of images (one per page).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")

    doc = fitz.open(str(pdf_path))
    images: List[np.ndarray] = []
    zoom   = dpi / 72.0   # 72 DPI is PDF default
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page  = doc[page_num]
        pix   = page.get_pixmap(matrix=matrix, alpha=False)
        img   = np.frombuffer(pix.samples, dtype=np.uint8)
        img   = img.reshape(pix.height, pix.width, 3)
        images.append(img)
        logger.debug(f"PDF page {page_num+1}: {pix.width}x{pix.height}px")

    doc.close()
    logger.info(f"PDF: {len(images)} pages extracted from {pdf_path.name}")
    return images
