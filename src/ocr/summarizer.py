"""Summarizer — rotates API keys on rate limit."""

from __future__ import annotations
import os, time
from loguru import logger
from src.ocr.api_key_manager import run_with_key_rotation

MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]


def _summarize_with_key(api_key: str, prompt: str) -> str:
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
    raise Exception(f"429 All models rate-limited")


def summarize_text(text: str, lang_hint: str = "en") -> str:
    if len(text.split()) < 30:
        return "✏️  Text too short to summarize."
    lang_names = {"uz":"Uzbek","en":"English","ru":"Russian","ko":"Korean",
                  "fr":"French","de":"German","ar":"Arabic","pl":"Polish"}
    lang_name = lang_names.get(lang_hint.lower(), "English")
    prompt = (
        f"The following text is in {lang_name}. "
        f"Write a concise summary in {lang_name}:\n"
        f"• Main topic (1 sentence)\n• Key points (3-5 bullets)\n"
        f"• Important numbers/dates if any\n\nText:\n{text[:4000]}"
    )
    try:
        return run_with_key_rotation(_summarize_with_key, prompt)
    except RuntimeError:
        return "⏳  Summary unavailable right now — all API keys are busy. Try again in a minute!"
