"""
Key Info Extractor - extract dates, names, amounts, phones, emails from text.
"""

from __future__ import annotations

import os
import time
from loguru import logger


def extract_key_info(text: str) -> str:
    """Extract key structured information from document text."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")

    prompt = (
        "Extract all important information from the text below. "
        "Look for and list:\n"
        "📅 Dates & Times\n"
        "👤 Names of people\n"
        "🏢 Organization/Company names\n"
        "📍 Addresses & Locations\n"
        "📞 Phone numbers\n"
        "📧 Email addresses\n"
        "💰 Amounts, prices, numbers\n"
        "🔑 Key terms or important phrases\n\n"
        "Format each category clearly. If nothing found for a category, skip it. "
        "Output in the same language as the text.\n\n"
        f"TEXT:\n{text[:5000]}"
    )

    from google import genai
    client = genai.Client(api_key=api_key)
    models = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]

    for model in models:
        try:
            logger.info(f"Key info extraction using {model}")
            response = client.models.generate_content(model=model, contents=[prompt])
            return response.text.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(15)
                continue
            raise

    raise RuntimeError("Key info extraction failed")
