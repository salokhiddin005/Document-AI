"""
Voice Transcriber.

Groq Whisper is primary (purpose-built for audio, much more reliable).
Gemini is fallback if Groq fails.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union
from loguru import logger
from src.ocr.api_key_manager import get_groq_keys, get_gemini_keys, is_rate_limit

MIME_MAP = {
    ".ogg": "audio/ogg",  ".opus": "audio/ogg",
    ".mp3": "audio/mpeg", ".wav":  "audio/wav",
    ".m4a": "audio/mp4",  ".flac": "audio/flac",
}


def _try_groq(api_key: str, audio_path: Path) -> str:
    """Groq Whisper — best for audio transcription."""
    from groq import Groq

    suffix    = audio_path.suffix.lower()
    mime_type = MIME_MAP.get(suffix, "audio/ogg")
    client    = Groq(api_key=api_key)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    result = client.audio.transcriptions.create(
        file=(audio_path.name, audio_bytes, mime_type),
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    text = result.strip() if isinstance(result, str) else result.text.strip()
    logger.info(f"Groq Whisper: {len(text.split())} words")
    return text


def _try_gemini(api_key: str, audio_path: Path) -> str:
    """Gemini audio fallback — uses File API for reliable upload."""
    from google import genai

    suffix    = audio_path.suffix.lower()
    mime_type = MIME_MAP.get(suffix, "audio/ogg")
    client    = genai.Client(api_key=api_key)

    # Upload via File API (more reliable than inline bytes)
    uploaded = client.files.upload(
        file=str(audio_path),
        config={"mime_type": mime_type},
    )
    prompt   = "Transcribe this voice message. Output ONLY the transcribed text."

    for model in ["gemini-2.0-flash", "gemini-2.5-flash"]:
        try:
            r    = client.models.generate_content(model=model, contents=[uploaded, prompt])
            text = r.text.strip()
            logger.info(f"Gemini transcribe: {len(text.split())} words | {model}")
            # Clean up uploaded file
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
            return text
        except Exception as e:
            if "404" in str(e):
                continue
            raise

    raise RuntimeError("Gemini transcription failed on all models")


def transcribe_voice(audio_path: Union[str, Path]) -> str:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    logger.info(f"Transcribing: {audio_path.name} ({audio_path.stat().st_size} bytes)")

    # ── Try Groq Whisper (primary for audio) ──────────────────────────────────
    groq_keys = get_groq_keys()
    for i, key in enumerate(groq_keys, 1):
        for attempt in range(2):
            try:
                return _try_groq(key, audio_path)
            except Exception as exc:
                err = str(exc)
                if is_rate_limit(err):
                    if attempt == 0:
                        logger.warning(f"Groq Key {i} rate limited, waiting 10s...")
                        time.sleep(10)
                        continue
                    else:
                        logger.warning(f"Groq Key {i} still limited, trying next key")
                        break
                else:
                    logger.error(f"Groq Key {i} error: {err}")
                    break   # non-rate-limit error — try next key

    # ── Groq failed → try Gemini ──────────────────────────────────────────────
    gemini_keys = get_gemini_keys()
    if gemini_keys:
        logger.info("Groq unavailable — trying Gemini for audio")
        for i, key in enumerate(gemini_keys, 1):
            for attempt in range(2):
                try:
                    return _try_gemini(key, audio_path)
                except Exception as exc:
                    err = str(exc)
                    if is_rate_limit(err):
                        if attempt == 0:
                            time.sleep(15); continue
                        else:
                            break
                    else:
                        logger.error(f"Gemini Key {i} error: {err}")
                        break

    raise RuntimeError(
        "⏳ Voice transcription unavailable right now.\n"
        "All API keys are busy. Please wait 1 minute and try again."
    )
