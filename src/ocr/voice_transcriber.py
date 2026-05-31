"""Voice Transcriber — Gemini first, Groq Whisper as fallback."""

from __future__ import annotations
import time
from pathlib import Path
from typing import Union
from loguru import logger
from src.ocr.api_key_manager import run_with_fallback

MIME_MAP = {".ogg":"audio/ogg",".mp3":"audio/mpeg",".wav":"audio/wav",
            ".m4a":"audio/mp4",".flac":"audio/flac",".opus":"audio/ogg"}


def _gemini_transcribe(api_key: str, audio_path: Path) -> str:
    from google import genai
    from google.genai import types
    client     = genai.Client(api_key=api_key)
    suffix     = audio_path.suffix.lower()
    mime_type  = MIME_MAP.get(suffix, "audio/ogg")
    prompt     = "Transcribe this voice message. Output ONLY the transcribed text."

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    for model in ["gemini-2.0-flash-lite", "gemini-2.0-flash"]:
        for attempt in range(2):
            try:
                r    = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type), prompt],
                )
                text = r.text.strip()
                logger.info(f"Gemini Whisper: {len(text.split())} words | {model}")
                return text
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0: time.sleep(15); continue
                    else: break
                elif "404" in err: break
                else: raise
    raise Exception("429 Gemini rate limited")


def _groq_transcribe(api_key: str, audio_path: Path) -> str:
    from groq import Groq
    client    = Groq(api_key=api_key)
    suffix    = audio_path.suffix.lower()
    mime_type = MIME_MAP.get(suffix, "audio/ogg")

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(audio_path.name, f.read(), mime_type),
            model="whisper-large-v3-turbo",
            response_format="text",
        )
    text = result.strip() if isinstance(result, str) else result.text.strip()
    logger.info(f"Groq Whisper: {len(text.split())} words")
    return text


def transcribe_voice(audio_path: Union[str, Path]) -> str:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    return run_with_fallback(_gemini_transcribe, _groq_transcribe, audio_path)
