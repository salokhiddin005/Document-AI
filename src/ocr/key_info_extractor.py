"""Key Info Extractor — uses Groq Llama, rotates API keys on rate limit."""

from __future__ import annotations
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
    return run_with_key_rotation(_run_with_key, prompt)
