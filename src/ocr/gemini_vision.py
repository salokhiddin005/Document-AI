"""
Gemini Vision OCR -- free high-accuracy text extraction.
Uses Google Gemini 2.0 Flash (completely free tier).
Free limits: 15 requests/minute, 1,500/day -- no credit card needed.
Get your free API key from: aistudio.google.com
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger


def extract_text_gemini(
    image: Union[str, Path, np.ndarray],
    lang_hint: str = "uz",
) -> tuple[str, float]:
    """Extract text from image using Gemini Vision (free)."""

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in .env\n"
            "Get free key from: aistudio.google.com"
        )

    # Load image as PIL
    import cv2
    from PIL import Image as PILImage

    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = cv2.imread(str(image))
        if arr is None:
            raise ValueError(f"Cannot read image: {image}")

    rgb     = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

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
        f"Do NOT add any explanation or commentary. "
        f"Do NOT translate. Reproduce every word exactly."
    )

    # Model priority: fastest/cheapest first
    MODELS = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
    ]

    from google import genai
    import time

    client = genai.Client(api_key=api_key)
    last_error = None

    for model_name in MODELS:
        logger.info(f"Gemini Vision | lang={lang_name} | model={model_name}")
        for attempt in range(2):   # retry once on rate limit
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[pil_img, prompt],
                )
                text = response.text.strip()
                logger.info(f"Gemini: {len(text.split())} words | model={model_name}")
                return text, 0.97

            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0:
                        logger.warning(f"Rate limit on {model_name}, waiting 30s...")
                        time.sleep(30)
                        continue
                    else:
                        last_error = e
                        break   # try next model
                elif "404" in err or "not found" in err.lower():
                    last_error = e
                    break   # model not available, try next
                else:
                    raise   # unexpected error, propagate

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}\n"
        f"Make sure your API key is from aistudio.google.com"
    )
