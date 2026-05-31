"""Auto Language Detector — Gemini first, Groq as fallback."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Union
import numpy as np
from loguru import logger
from src.ocr.api_key_manager import run_with_fallback

LANG_LABELS = {
    "uz":"🇺🇿 Uzbek","en":"🇬🇧 English","ru":"🇷🇺 Russian","ko":"🇰🇷 Korean",
    "fr":"🇫🇷 French","de":"🇩🇪 German","ar":"🇸🇦 Arabic","pl":"🇵🇱 Polish",
    "tr":"🇹🇷 Turkish","zh":"🇨🇳 Chinese","ja":"🇯🇵 Japanese","es":"🇪🇸 Spanish",
}

PROMPT = (
    "What language is the text in this image? "
    "Reply with ONLY the ISO 639-1 two-letter code (e.g. 'uz','en','ru'). Nothing else."
)


def _gemini_detect(api_key: str, b64: str) -> str:
    from google import genai
    from PIL import Image as PILImage
    import io, base64
    client  = genai.Client(api_key=api_key)
    img     = PILImage.open(io.BytesIO(base64.b64decode(b64)))
    r       = client.models.generate_content(model="gemini-2.0-flash-lite", contents=[img, PROMPT])
    return r.text.strip().lower()[:5].split()[0]


def _groq_detect(api_key: str, b64: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    r = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        max_tokens=10,
    )
    return r.choices[0].message.content.strip().lower()[:5].split()[0]


def detect_language(image: Union[str, Path, np.ndarray]) -> str:
    try:
        from src.ocr.gemini_vision import _img_to_b64
        b64  = _img_to_b64(image)
        mapping = {"korean":"ko","chinese":"zh","japanese":"ja","arabic":"ar","uzbek":"uz","russian":"ru"}
        lang = run_with_fallback(_gemini_detect, _groq_detect, b64)
        lang = mapping.get(lang, lang)
        if lang in LANG_LABELS:
            logger.info(f"Auto-detected: {lang}")
            return lang
        return "en"
    except Exception as exc:
        logger.warning(f"Language detection failed: {exc}")
        return "en"
