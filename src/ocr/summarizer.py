"""Summarizer — uses Groq Llama, rotates API keys on rate limit."""

from __future__ import annotations
from loguru import logger
from src.ocr.api_key_manager import run_with_key_rotation

TEXT_MODEL = "llama-3.3-70b-versatile"


def _run_with_key(api_key: str, prompt: str) -> str:
    from groq import Groq
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def summarize_text(text: str, lang_hint: str = "en") -> str:
    if len(text.split()) < 30:
        return "✏️  Text too short to summarize."
    lang_names = {"uz":"Uzbek","en":"English","ru":"Russian","ko":"Korean",
                  "fr":"French","de":"German","ar":"Arabic","pl":"Polish","tr":"Turkish"}
    lang_name  = lang_names.get(lang_hint.lower(), "English")
    prompt = (
        f"The following text is in {lang_name}. "
        f"Write a concise summary in {lang_name}:\n"
        f"• Main topic (1 sentence)\n• Key points (3-5 bullets)\n"
        f"• Important numbers/dates if any\n\nText:\n{text[:4000]}"
    )
    try:
        return run_with_key_rotation(_run_with_key, prompt)
    except RuntimeError:
        return "⏳  Summary unavailable — API rate limit. Try again in a minute!"
