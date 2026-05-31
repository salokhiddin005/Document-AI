"""Key Info Extractor — rotates API keys on rate limit."""

from __future__ import annotations
import time
from src.ocr.api_key_manager import run_with_key_rotation

MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]


def _extract_with_key(api_key: str, prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    for model in MODELS:
        for attempt in range(2):
            try:
                r = client.models.generate_content(model=model, contents=[prompt])
                return r.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0: time.sleep(15); continue
                    else: break
                elif "404" in err: break
                else: raise
    raise Exception("429 All models rate-limited")


def extract_key_info(text: str) -> str:
    prompt = (
        "Extract all important information from the text below. Look for:\n"
        "📅 Dates & Times\n👤 Names of people\n🏢 Organizations\n"
        "📍 Addresses & Locations\n📞 Phone numbers\n📧 Emails\n"
        "💰 Amounts & prices\n🔑 Key terms\n\n"
        "Format each category clearly. Skip empty categories. "
        "Output in the same language as the text.\n\n"
        f"TEXT:\n{text[:5000]}"
    )
    return run_with_key_rotation(_extract_with_key, prompt)
