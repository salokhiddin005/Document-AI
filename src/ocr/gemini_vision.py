"""
Groq Vision OCR — reads text from images using Groq's Llama Vision.
Supports any language. Rotates API keys on rate limit.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger

from src.ocr.api_key_manager import run_with_key_rotation

VISION_MODELS = [
    "llama-3.2-90b-vision-preview",   # best accuracy
    "llama-3.2-11b-vision-preview",    # fallback
]


def _img_to_b64(image: Union[str, Path, np.ndarray]) -> str:
    import cv2
    from PIL import Image as PILImage
    import io

    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = cv2.imread(str(image))
        if arr is None:
            raise ValueError(f"Cannot read image: {image}")

    rgb     = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)
    buf     = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _ocr_with_key(api_key: str, b64_image: str, prompt: str) -> tuple[str, float]:
    import time
    from groq import Groq
    client = Groq(api_key=api_key)

    for model in VISION_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=4096,
            )
            text = response.choices[0].message.content.strip()
            logger.info(f"Groq OCR: {len(text.split())} words | {model}")
            return text, 0.97
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                raise   # let key manager handle rotation
            elif "model_not_active" in err or "not found" in err.lower():
                logger.warning(f"Model {model} unavailable, trying next")
                continue
            else:
                raise
    raise RuntimeError("All vision models unavailable")


def extract_text_gemini(
    image: Union[str, Path, np.ndarray],
    lang_hint: str = "auto",
) -> tuple[str, float]:
    b64 = _img_to_b64(image)

    lang_lower = lang_hint.lower()
    if lang_lower in ("auto", "", "any"):
        prompt = (
            "Extract ALL text from this document image. "
            "Detect the language automatically and read every word exactly as written. "
            "Output ONLY the extracted text, preserving the original language and paragraph structure. "
            "Do NOT translate, summarize, or add any commentary."
        )
    else:
        lang_names = {
            "uz":"Uzbek","en":"English","ru":"Russian","ko":"Korean",
            "fr":"French","de":"German","ar":"Arabic","pl":"Polish",
            "tr":"Turkish","zh":"Chinese","ja":"Japanese","es":"Spanish",
            "it":"Italian","pt":"Portuguese","nl":"Dutch","hi":"Hindi",
        }
        lang_name = lang_names.get(lang_lower, lang_hint)
        prompt = (
            f"Extract ALL text from this document image. "
            f"The text is in {lang_name}. "
            f"Output ONLY the extracted text exactly as it appears. "
            f"Do NOT translate or add any commentary."
        )

    return run_with_key_rotation(_ocr_with_key, b64, prompt)
