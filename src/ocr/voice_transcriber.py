"""
Voice Transcriber — transcribes voice messages using Gemini.
Tries multiple models with retry on rate limit (429).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union

from loguru import logger

MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]


def transcribe_voice(audio_path: Union[str, Path]) -> str:
    """
    Transcribe a voice/audio file to text using Gemini.
    Automatically retries with next model on 429 rate limit.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    suffix   = audio_path.suffix.lower()
    mime_map = {
        ".ogg": "audio/ogg", ".mp3": "audio/mpeg",
        ".wav": "audio/wav", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".opus": "audio/ogg",
    }
    mime_type = mime_map.get(suffix, "audio/ogg")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    prompt = (
        "Transcribe this voice message accurately. "
        "Output ONLY the transcribed text, nothing else. "
        "If the speech is unclear, write your best guess."
    )

    from google import genai
    from google.genai import types

    client     = genai.Client(api_key=api_key)
    last_error = None

    for model in MODELS:
        logger.info(f"Transcribing with {model}")
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
                text = response.text.strip()
                logger.info(f"Transcribed: {len(text.split())} words using {model}")
                return text

            except Exception as exc:
                err = str(exc)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0:
                        wait = 40
                        logger.warning(f"Rate limit on {model}, waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    else:
                        last_error = exc
                        logger.warning(f"Rate limit persists on {model}, trying next model")
                        break
                elif "404" in err or "not found" in err.lower():
                    last_error = exc
                    break
                else:
                    raise RuntimeError(f"Voice transcription failed: {exc}")

    # All models rate-limited — friendly message
    raise RuntimeError(
        "⏳ API is busy right now (rate limit reached).\n"
        "Please wait 1 minute and try again."
    )
