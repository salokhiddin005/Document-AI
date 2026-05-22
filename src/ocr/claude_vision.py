"""
Claude Vision OCR — high-accuracy text extraction from images.

Uses Anthropic's Claude claude-haiku-4-5 model to extract text from document images.
Dramatically more accurate than PaddleOCR for:
  - Uzbek, Korean, and other non-Latin languages
  - Mixed-font documents
  - Browser screenshots
  - Documents with special characters (apostrophes, dashes)

Requires ANTHROPIC_API_KEY in .env file.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger


def _image_to_base64(source: Union[str, Path, np.ndarray]) -> tuple[str, str]:
    """Convert image to base64 string. Returns (base64_data, media_type)."""
    import cv2
    from PIL import Image
    import io

    if isinstance(source, np.ndarray):
        img = source
    else:
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Cannot read image: {source}")

    # Encode as JPEG
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
    return b64, "image/jpeg"


def extract_text_claude(
    image: Union[str, Path, np.ndarray],
    lang_hint: str = "uz",
    api_key: Optional[str] = None,
) -> tuple[str, float]:
    """
    Extract text from an image using Claude Vision.

    Returns:
        (extracted_text, confidence)
        confidence is always 0.97 for Claude (it's either correct or says "unclear")
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic not installed. Run: pip install anthropic")

    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found in .env file.\n"
            "Add: ANTHROPIC_API_KEY=your_key_here"
        )

    client = anthropic.Anthropic(api_key=key)

    b64_data, media_type = _image_to_base64(image)

    lang_names = {
        "uz": "Uzbek", "en": "English", "ru": "Russian",
        "korean": "Korean", "ko": "Korean",
        "fr": "French", "de": "German", "ar": "Arabic",
        "pl": "Polish", "tr": "Turkish",
    }
    lang_name = lang_names.get(lang_hint.lower(), lang_hint)

    prompt = (
        f"Extract ALL text from this document image. "
        f"The text is in {lang_name}. "
        f"Output ONLY the extracted text, exactly as it appears. "
        f"Preserve paragraph breaks with blank lines. "
        f"Do NOT add any explanation, commentary, or formatting. "
        f"Do NOT translate. "
        f"If a word is unclear, write your best guess."
    )

    logger.info(f"Claude Vision OCR | lang={lang_name}")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    logger.info(f"Claude Vision: {len(text)} chars extracted")
    return text, 0.97
