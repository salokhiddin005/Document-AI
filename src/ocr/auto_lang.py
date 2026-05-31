"""Auto Language Detector — rotates API keys on rate limit."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Union
import numpy as np
from loguru import logger
from src.ocr.api_key_manager import run_with_key_rotation

LANG_LABELS = {
    "uz":"🇺🇿 Uzbek","en":"🇬🇧 English","ru":"🇷🇺 Russian","ko":"🇰🇷 Korean",
    "fr":"🇫🇷 French","de":"🇩🇪 German","ar":"🇸🇦 Arabic","pl":"🇵🇱 Polish",
    "tr":"🇹🇷 Turkish","zh":"🇨🇳 Chinese","ja":"🇯🇵 Japanese","es":"🇪🇸 Spanish",
    "it":"🇮🇹 Italian","pt":"🇵🇹 Portuguese","hi":"🇮🇳 Hindi",
}


def _detect_with_key(api_key: str, pil_img) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    prompt = (
        "What language is the text in this document image? "
        "Reply with ONLY the ISO 639-1 two-letter code (e.g. 'uz', 'en', 'ru'). Nothing else."
    )
    try:
        r    = client.models.generate_content(model="gemini-2.0-flash-lite", contents=[pil_img, prompt])
        lang = r.text.strip().lower()[:5].split()[0]
        mapping = {"korean":"ko","chinese":"zh","japanese":"ja","arabic":"ar","uzbek":"uz","russian":"ru"}
        return mapping.get(lang, lang)
    except Exception as e:
        if "429" in str(e): raise
        return "en"


def detect_language(image: Union[str, Path, np.ndarray]) -> str:
    try:
        import cv2
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            arr = image
        else:
            arr = cv2.imread(str(image))
            if arr is None: return "en"
        pil_img = PILImage.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        lang = run_with_key_rotation(_detect_with_key, pil_img)
        if lang in LANG_LABELS:
            logger.info(f"Auto-detected: {lang}")
            return lang
        return "en"
    except Exception as exc:
        logger.warning(f"Language detection failed: {exc}")
        return "en"
