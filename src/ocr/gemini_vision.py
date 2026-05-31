"""
OCR Vision — Gemini first, Groq Llama Vision as fallback.
"""

from __future__ import annotations

import base64, io
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger
from src.ocr.api_key_manager import run_with_fallback


def _img_to_b64(image: Union[str, Path, np.ndarray]) -> str:
    import cv2
    from PIL import Image as PILImage

    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = cv2.imread(str(image))
        if arr is None:
            raise ValueError(f"Cannot read image: {image}")

    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    PILImage.fromarray(rgb).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gemini_ocr(api_key: str, b64: str, prompt: str) -> tuple[str, float]:
    import time
    from google import genai
    from PIL import Image as PILImage
    import io, base64

    client  = genai.Client(api_key=api_key)
    img_buf = io.BytesIO(base64.b64decode(b64))
    pil_img = PILImage.open(img_buf)

    for model in ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]:
        for attempt in range(2):
            try:
                r    = client.models.generate_content(model=model, contents=[pil_img, prompt])
                text = r.text.strip()
                logger.info(f"Gemini OCR: {len(text.split())} words | {model}")
                return text, 0.97
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0: time.sleep(10); continue
                    else: break
                elif "404" in err: break
                else: raise
    raise Exception("429 Gemini rate limited")


def _groq_ocr(api_key: str, b64: str, prompt: str) -> tuple[str, float]:
    from groq import Groq
    client = Groq(api_key=api_key)

    for model in ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]:
        try:
            r    = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=4096,
            )
            text = r.choices[0].message.content.strip()
            logger.info(f"Groq OCR: {len(text.split())} words | {model}")
            return text, 0.95
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower(): raise
            elif "model_not_active" in err or "not found" in err.lower(): continue
            else: raise
    raise Exception("429 Groq rate limited")


def extract_text_gemini(
    image: Union[str, Path, np.ndarray],
    lang_hint: str = "auto",
) -> tuple[str, float]:
    b64        = _img_to_b64(image)
    lang_lower = lang_hint.lower()

    if lang_lower in ("auto", "", "any"):
        prompt = (
            "Extract ALL text from this document image. "
            "Detect the language automatically. "
            "Output ONLY the extracted text preserving the original language and paragraphs. "
            "Do NOT translate or add commentary."
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
            f"Extract ALL text from this document image. The text is in {lang_name}. "
            f"Output ONLY the extracted text exactly as it appears. Do NOT translate."
        )

    return run_with_fallback(_gemini_ocr, _groq_ocr, b64, prompt)
