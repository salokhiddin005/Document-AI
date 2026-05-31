"""
Translator - translate extracted text to any language using Gemini.
"""

from __future__ import annotations

import os
import time
from loguru import logger

LANGUAGES = {
    "en": "English",    "uz": "Uzbek",      "ru": "Russian",
    "ko": "Korean",     "fr": "French",     "de": "German",
    "es": "Spanish",    "ar": "Arabic",     "zh": "Chinese",
    "ja": "Japanese",   "tr": "Turkish",    "pl": "Polish",
    "it": "Italian",    "pt": "Portuguese", "hi": "Hindi",
    "uk": "Ukrainian",  "vi": "Vietnamese", "th": "Thai",
}

LANG_FLAGS = {
    "en": "🇬🇧", "uz": "🇺🇿", "ru": "🇷🇺", "ko": "🇰🇷",
    "fr": "🇫🇷", "de": "🇩🇪", "es": "🇪🇸", "ar": "🇸🇦",
    "zh": "🇨🇳", "ja": "🇯🇵", "tr": "🇹🇷", "pl": "🇵🇱",
    "it": "🇮🇹", "pt": "🇵🇹", "hi": "🇮🇳", "uk": "🇺🇦",
    "vi": "🇻🇳", "th": "🇹🇭",
}


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using Gemini."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")

    lang_name = LANGUAGES.get(target_lang.lower(), target_lang)

    prompt = (
        f"Translate the following text to {lang_name}. "
        f"Output ONLY the translated text, nothing else. "
        f"Preserve paragraph breaks. Do not add any explanation.\n\n"
        f"{text[:5000]}"
    )

    from google import genai
    client = genai.Client(api_key=api_key)
    models = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]

    for model in models:
        try:
            logger.info(f"Translating to {lang_name} using {model}")
            response = client.models.generate_content(model=model, contents=[prompt])
            result = response.text.strip()
            logger.info(f"Translation done: {len(result.split())} words")
            return result
        except Exception as e:
            if "429" in str(e):
                time.sleep(15)
                continue
            raise

    raise RuntimeError("Translation failed — all models unavailable")
