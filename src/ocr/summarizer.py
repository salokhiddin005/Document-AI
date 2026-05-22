"""
AI Document Summarizer — generates a concise summary using Gemini.
Tries multiple models with retry on rate limit (429).
"""

from __future__ import annotations

import os
import time
from loguru import logger

MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]


def summarize_text(text: str, lang_hint: str = "en") -> str:
    """
    Generate a short summary of *text* using Gemini.
    Returns the summary string.
    Automatically retries with next model on 429 rate limit.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    if len(text.split()) < 30:
        return "✏️  Text is too short to summarize."

    lang_names = {
        "uz": "Uzbek", "en": "English", "ru": "Russian",
        "ko": "Korean", "fr": "French", "de": "German",
        "ar": "Arabic", "tr": "Turkish", "pl": "Polish",
    }
    lang_name = lang_names.get(lang_hint.lower(), "English")

    prompt = (
        f"The following text is in {lang_name}. "
        f"Write a concise summary in {lang_name} covering:\n"
        f"• Main topic (1 sentence)\n"
        f"• Key points (3-5 bullet points)\n"
        f"• Important numbers/dates if any\n\n"
        f"Text:\n{text[:4000]}"
    )

    from google import genai

    client    = genai.Client(api_key=api_key)
    last_error = None

    for model in MODELS:
        for attempt in range(2):
            try:
                logger.debug(f"Summarizing with {model} (attempt {attempt+1})")
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt],
                )
                summary = response.text.strip()
                logger.info(f"Summary generated ({len(summary)} chars) using {model}")
                return summary

            except Exception as exc:
                err = str(exc)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0:
                        wait = 30
                        logger.warning(f"Rate limit on {model}, waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    else:
                        last_error = exc
                        logger.warning(f"Rate limit persists on {model}, trying next model")
                        break
                else:
                    logger.error(f"Summarization failed: {exc}")
                    raise RuntimeError(f"Summary failed: {exc}")

    # All models failed — return a graceful fallback
    logger.warning(f"All models rate-limited, skipping summary. Last error: {last_error}")
    return "⏳  Summary not available right now (API quota reached). Try again in a minute!"
