"""Voice Transcriber — rotates API keys on rate limit."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Union
from loguru import logger
from src.ocr.api_key_manager import run_with_key_rotation

MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]

MIME_MAP = {
    ".ogg":".ogg","audio/ogg":"audio/ogg",".mp3":"audio/mpeg",
    ".wav":"audio/wav",".m4a":"audio/mp4",".flac":"audio/flac",".opus":"audio/ogg",
}


def _transcribe_with_key(api_key: str, audio_bytes: bytes, mime_type: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    prompt = (
        "Transcribe this voice message accurately. "
        "Output ONLY the transcribed text, nothing else. "
        "If unclear, write your best guess."
    )
    for model in MODELS:
        for attempt in range(2):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type), prompt],
                )
                return r.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0: time.sleep(20); continue
                    else: break
                elif "404" in err: break
                else: raise
    raise Exception("429 All models rate-limited")


def transcribe_voice(audio_path: Union[str, Path]) -> str:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    suffix    = audio_path.suffix.lower()
    mime_type = {".ogg":"audio/ogg",".mp3":"audio/mpeg",".wav":"audio/wav",
                 ".m4a":"audio/mp4",".flac":"audio/flac",".opus":"audio/ogg"}.get(suffix,"audio/ogg")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    logger.info(f"Transcribing {audio_path.name} ({len(audio_bytes)} bytes)")
    return run_with_key_rotation(_transcribe_with_key, audio_bytes, mime_type)
