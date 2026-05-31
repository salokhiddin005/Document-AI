"""Summarizer — auto-detects document language, summarizes in same language."""

from __future__ import annotations
from loguru import logger
from src.ocr.api_key_manager import run_with_fallback


def _gemini_run(api_key: str, prompt: str) -> str:
    import time
    from google import genai
    client = genai.Client(api_key=api_key)
    for model in ["gemini-2.0-flash-lite", "gemini-2.0-flash"]:
        for attempt in range(2):
            try:
                r = client.models.generate_content(model=model, contents=[prompt])
                return r.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0: time.sleep(10); continue
                    else: break
                elif "404" in err: break
                else: raise
    raise Exception("429 Gemini rate limited")


def _groq_run(api_key: str, prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return r.choices[0].message.content.strip()


def summarize_text(text: str, lang_hint: str = "en") -> str:
    if len(text.split()) < 30:
        return "✏️  Text too short to summarize."

    # KEY FIX: Don't force a language — let the AI detect it and respond in
    # the same language as the document. This handles Korean, Uzbek, Arabic, etc.
    prompt = (
        "Read the following text and provide a concise summary "
        "IN THE SAME LANGUAGE as the text. Do NOT translate it to English.\n\n"
        "Include:\n"
        "• Main topic (1 sentence)\n"
        "• Key points (3-5 bullet points)\n"
        "• Important numbers/dates if any\n\n"
        f"Text:\n{text[:4000]}"
    )

    try:
        return run_with_fallback(_gemini_run, _groq_run, prompt)
    except RuntimeError:
        return "⏳  Summary unavailable — all APIs busy. Try again in a minute!"
