"""Auto Language Detector — uses Groq vision, rotates API keys on rate limit."""

from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
from loguru import logger
from src.ocr.api_key_manager import run_with_key_rotation

LANG_LABELS = {
    "uz":"🇺🇿 Uzbek","en":"🇬🇧 English","ru":"🇷🇺 Russian","ko":"🇰🇷 Korean",
    "fr":"🇫🇷 French","de":"🇩🇪 German","ar":"🇸🇦 Arabic","pl":"🇵🇱 Polish",
    "tr":"🇹🇷 Turkish","zh":"🇨🇳 Chinese","ja":"🇯🇵 Japanese","es":"🇪🇸 Spanish",
}


def _detect_with_key(api_key: str, b64: str) -> str:
    from groq import Groq
    client   = Groq(api_key=api_key)
    prompt   = (
        "What language is the text in this image? "
        "Reply with ONLY the ISO 639-1 two-letter code (e.g. 'uz', 'en', 'ru'). Nothing else."
    )
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=10,
    )
    lang = response.choices[0].message.content.strip().lower()[:5].split()[0]
    mapping = {"korean":"ko","chinese":"zh","japanese":"ja","arabic":"ar","uzbek":"uz","russian":"ru"}
    return mapping.get(lang, lang)


def detect_language(image: Union[str, Path, np.ndarray]) -> str:
    try:
        from src.ocr.gemini_vision import _img_to_b64
        b64  = _img_to_b64(image)
        lang = run_with_key_rotation(_detect_with_key, b64)
        if lang in LANG_LABELS:
            logger.info(f"Auto-detected language: {lang}")
            return lang
        return "en"
    except Exception as exc:
        logger.warning(f"Language detection failed: {exc}")
        return "en"
