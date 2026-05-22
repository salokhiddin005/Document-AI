"""
PDF Exporter — converts handwriting PNG pages into a proper A4 PDF.
Uses fpdf2. Each HandwritingResult image becomes one PDF page.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List, Union

from PIL import Image as PILImage
from loguru import logger


def export_to_pdf(
    images: Union["HandwritingResult", List["HandwritingResult"]],
    output_path: Path,
) -> Path:
    """
    Convert one or more HandwritingResult images into a single A4 PDF.
    Returns the path to the saved PDF.
    """
    from fpdf import FPDF

    if not isinstance(images, list):
        images = [images]

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(False)

    for hw in images:
        pdf.add_page()
        img = hw.image

        # Save image to temp bytes buffer
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name

        # A4 = 210 x 297 mm — fill entire page
        pdf.image(tmp_path, x=0, y=0, w=210, h=297)
        Path(tmp_path).unlink(missing_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    logger.info(f"PDF exported: {output_path} ({len(images)} pages)")
    return output_path
