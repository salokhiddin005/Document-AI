"""Translator — rotates API keys on rate limit."""

from __future__ import annotations
import time
from loguru import logger
from src.ocr.api_key_manager import run_with_key_rotation

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
MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]


def _translate_with_key(api_key: str, prompt: str) -> str:
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


def translate_text(text: str, target_lang: str) -> str:
    lang_name = LANGUAGES.get(target_lang.lower(), target_lang)
    prompt = (
        f"Translate the following text to {lang_name}. "
        f"Output ONLY the translated text, nothing else. "
        f"Preserve paragraph breaks.\n\n{text[:5000]}"
    )
    return run_with_key_rotation(_translate_with_key, prompt)
