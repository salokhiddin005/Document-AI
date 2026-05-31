"""Translator — uses Groq Llama, rotates API keys on rate limit."""

from __future__ import annotations
from src.ocr.api_key_manager import run_with_key_rotation

TEXT_MODEL = "llama-3.3-70b-versatile"

LANGUAGES = {
    "en":"English","uz":"Uzbek","ru":"Russian","ko":"Korean",
    "fr":"French","de":"German","es":"Spanish","ar":"Arabic",
    "zh":"Chinese","ja":"Japanese","tr":"Turkish","pl":"Polish",
    "it":"Italian","pt":"Portuguese","hi":"Hindi","uk":"Ukrainian",
    "vi":"Vietnamese","th":"Thai",
}
LANG_FLAGS = {
    "en":"🇬🇧","uz":"🇺🇿","ru":"🇷🇺","ko":"🇰🇷","fr":"🇫🇷","de":"🇩🇪",
    "es":"🇪🇸","ar":"🇸🇦","zh":"🇨🇳","ja":"🇯🇵","tr":"🇹🇷","pl":"🇵🇱",
    "it":"🇮🇹","pt":"🇵🇹","hi":"🇮🇳","uk":"🇺🇦","vi":"🇻🇳","th":"🇹🇭",
}


def _run_with_key(api_key: str, prompt: str) -> str:
    from groq import Groq
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def translate_text(text: str, target_lang: str) -> str:
    lang_name = LANGUAGES.get(target_lang.lower(), target_lang)
    prompt    = (
        f"Translate the following text to {lang_name}. "
        f"Output ONLY the translated text. Preserve paragraph breaks.\n\n{text[:5000]}"
    )
    return run_with_key_rotation(_run_with_key, prompt)
