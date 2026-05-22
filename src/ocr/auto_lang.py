"""
Auto Language Detection — detects document language using Gemini Vision.
Called before OCR to automatically select the right language model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger

LANG_LABELS = {
    "uz": "🇺🇿 Uzbek",    "en": "🇬🇧 English",  "ru": "🇷🇺 Russian",
    "ko": "🇰🇷 Korean",   "fr": "🇫🇷 French",    "de": "🇩🇪 German",
    "ar": "🇸🇦 Arabic",   "tr": "🇹🇷 Turkish",   "pl": "🇵🇱 Polish",
    "es": "🇪🇸 Spanish",  "it": "🇮🇹 Italian",   "pt": "🇵🇹 Portuguese",
    "zh": "🇨🇳 Chinese",  "ja": "🇯🇵 Japanese",
}


def detect_language(image: Union[str, Path, np.ndarray]) -> str:
    """
    Detect the primary language in a document image.
    Returns a language code like 'uz', 'en', 'ru', etc.
    Falls back to 'en' on any error.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return "en"

    try:
        import cv2
        from PIL import Image as PILImage
        from google import genai

        if isinstance(image, np.ndarray):
            arr = image
        else:
            arr = cv2.imread(str(image))
            if arr is None:
                return "en"

        rgb     = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        client  = genai.Client(api_key=api_key)
        prompt  = (
            "What language is the text in this document image written in? "
            "Reply with ONLY the ISO 639-1 two-letter language code "
            "(e.g. 'uz' for Uzbek, 'en' for English, 'ru' for Russian, "
            "'ko' for Korean, 'ar' for Arabic). "
            "Nothing else — just the code."
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[pil_img, prompt],
        )
        lang = response.text.strip().lower()[:5].split()[0]

        # Map some variations
        mapping = {"korean": "ko", "chinese": "zh", "japanese": "ja",
                   "arabic": "ar", "uzbek": "uz", "russian": "ru"}
        lang = mapping.get(lang, lang)

        if lang in LANG_LABELS:
            logger.info(f"Auto-detected language: {lang} ({LANG_LABELS[lang]})")
            return lang

        logger.warning(f"Unknown lang code '{lang}', defaulting to 'en'")
        return "en"

    except Exception as exc:
        logger.warning(f"Language detection failed: {exc}")
        return "en"
