"""Voice Transcriber — uses Groq Whisper, rotates API keys on rate limit."""

from __future__ import annotations

from pathlib import Path
from typing import Union
from loguru import logger
from src.ocr.api_key_manager import run_with_key_rotation

WHISPER_MODEL = "whisper-large-v3-turbo"


def _transcribe_with_key(api_key: str, audio_path: Path) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)

    suffix    = audio_path.suffix.lower()
    mime_map  = {".ogg":"audio/ogg",".mp3":"audio/mpeg",".wav":"audio/wav",
                 ".m4a":"audio/mp4",".flac":"audio/flac",".opus":"audio/ogg"}
    mime_type = mime_map.get(suffix, "audio/ogg")

    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(audio_path.name, f.read(), mime_type),
            model=WHISPER_MODEL,
            response_format="text",
        )

    text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
    logger.info(f"Groq Whisper: {len(text.split())} words")
    return text


def transcribe_voice(audio_path: Union[str, Path]) -> str:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    logger.info(f"Transcribing {audio_path.name}")
    return run_with_key_rotation(_transcribe_with_key, audio_path)
