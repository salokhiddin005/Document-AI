"""
Gemini Vision OCR -- supports ANY language automatically.
Uses Google Gemini 2.0 Flash (free tier).
Free: 1,500 requests/day -- no credit card needed.
Get key: aistudio.google.com
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger


def extract_text_gemini(
    image: Union[str, Path, np.ndarray],
    lang_hint: str = "auto",
) -> tuple[str, float]:
    """
    Extract text from image using Gemini Vision.
    Supports ANY language — Gemini auto-detects if lang_hint is 'auto'.
    """

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found.\n"
            "Get free key from: aistudio.google.com"
        )

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

    # Build prompt — auto-detect OR specific language
    lang_hint_lower = lang_hint.lower()
    if lang_hint_lower in ("auto", "", "any"):
        prompt = (
            "Extract ALL text from this document image. "
            "The text may be in ANY language — detect and read it exactly as written. "
            "Output ONLY the extracted text, preserving the original language. "
            "Preserve paragraph structure with blank lines between paragraphs. "
            "Do NOT translate, summarize, or add any commentary. "
            "If a word is unclear, write your best guess."
        )
        log_lang = "auto"
    else:
        lang_names = {
            "uz": "Uzbek",        "en": "English",    "ru": "Russian",
            "ko": "Korean",       "korean": "Korean",  "fr": "French",
            "de": "German",       "ar": "Arabic",      "pl": "Polish",
            "tr": "Turkish",      "zh": "Chinese",     "ja": "Japanese",
            "es": "Spanish",      "it": "Italian",     "pt": "Portuguese",
            "nl": "Dutch",        "hi": "Hindi",       "fa": "Persian",
            "vi": "Vietnamese",   "th": "Thai",        "uk": "Ukrainian",
        }
        lang_name = lang_names.get(lang_hint_lower, lang_hint)
        prompt = (
            f"Extract ALL text from this document image. "
            f"The text is in {lang_name}. "
            f"Output ONLY the extracted text, exactly as it appears. "
            f"Preserve paragraph breaks with blank lines. "
            f"Do NOT add any explanation or commentary. "
            f"Do NOT translate. Reproduce every word exactly."
        )
        log_lang = lang_name

    MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]

    from google import genai
    client     = genai.Client(api_key=api_key)
    last_error = None

    for model_name in MODELS:
        logger.info(f"Gemini Vision | lang={log_lang} | model={model_name}")
        for attempt in range(2):
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
                        break
                elif "404" in err or "not found" in err.lower():
                    last_error = e
                    break
                else:
                    raise

    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}\n"
        f"Check your API key at aistudio.google.com"
    )
