"""
Voice Transcriber — transcribes voice messages using Gemini.
Accepts OGG/MP3/WAV audio and returns text.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union

from loguru import logger


def transcribe_voice(audio_path: Union[str, Path]) -> str:
    """
    Transcribe a voice/audio file to text using Gemini.
    Returns the transcribed text string.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        from google import genai
        import time

        client = genai.Client(api_key=api_key)

        # Upload audio file
        logger.info(f"Uploading audio: {audio_path.name} ({audio_path.stat().st_size} bytes)")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Determine MIME type
        suffix = audio_path.suffix.lower()
        mime_map = {
            ".ogg": "audio/ogg",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
        }
        mime_type = mime_map.get(suffix, "audio/ogg")

        # Use inline data for small files (< 20MB)
        from google.genai import types

        prompt = (
            "Transcribe this voice message accurately. "
            "Output ONLY the transcribed text, nothing else. "
            "If the speech is unclear, write your best guess."
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                prompt,
            ],
        )

        text = response.text.strip()
        logger.info(f"Transcribed: {len(text.split())} words")
        return text

    except Exception as exc:
        logger.exception(f"Transcription failed: {exc}")
        raise RuntimeError(f"Voice transcription failed: {exc}")
