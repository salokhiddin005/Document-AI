"""
Gemini Vision OCR — supports any language, rotates API keys on rate limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger

from src.ocr.api_key_manager import run_with_key_rotation

MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]


def _ocr_with_key(api_key: str, pil_img, prompt: str) -> tuple[str, float]:
    import time
    from google import genai
    client     = genai.Client(api_key=api_key)
    last_error = None
    for model in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model, contents=[pil_img, prompt]
                )
                text = response.text.strip()
                logger.info(f"Gemini OCR: {len(text.split())} words | {model}")
                return text, 0.97
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0:
                        time.sleep(15); continue
                    else:
                        last_error = e; break
                elif "404" in err or "not found" in err.lower():
                    last_error = e; break
                else:
                    raise
    raise Exception(f"429 All models rate-limited: {last_error}")


def extract_text_gemini(
    image: Union[str, Path, np.ndarray],
    lang_hint: str = "auto",
) -> tuple[str, float]:
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

    lang_lower = lang_hint.lower()
    if lang_lower in ("auto", "", "any"):
        prompt = (
            "Extract ALL text from this document image. "
            "The text may be in ANY language — detect and read it exactly as written. "
            "Output ONLY the extracted text, preserving the original language. "
            "Preserve paragraph structure. "
            "Do NOT translate, summarize, or add commentary. "
            "If a word is unclear, write your best guess."
        )
    else:
        lang_names = {
            "uz":"Uzbek","en":"English","ru":"Russian","ko":"Korean",
            "fr":"French","de":"German","ar":"Arabic","pl":"Polish",
            "tr":"Turkish","zh":"Chinese","ja":"Japanese","es":"Spanish",
            "it":"Italian","pt":"Portuguese","nl":"Dutch","hi":"Hindi",
            "fa":"Persian","vi":"Vietnamese","th":"Thai","uk":"Ukrainian",
        }
        lang_name = lang_names.get(lang_lower, lang_hint)
        prompt = (
            f"Extract ALL text from this document image. "
            f"The text is in {lang_name}. "
            f"Output ONLY the extracted text, exactly as it appears. "
            f"Preserve paragraph breaks. Do NOT translate or add commentary."
        )

    return run_with_key_rotation(_ocr_with_key, pil_img, prompt)
