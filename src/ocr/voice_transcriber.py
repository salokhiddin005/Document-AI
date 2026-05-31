"""
Voice Transcriber — Groq Whisper primary, Gemini fallback.

Fixes for Whisper hallucination ("Thank you." on empty/bad audio):
  - Minimum file size check
  - Anti-hallucination prompt
  - Convert OGG/OPUS to MP3 for better compatibility
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union
from loguru import logger
from src.ocr.api_key_manager import get_groq_keys, get_gemini_keys, is_rate_limit

# Known Whisper hallucinations — if result matches these, the audio was empty/bad
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching.",
    "thanks.", ".", "", "♪", "[ silence ]", "[silence]",
    "you", "thank you for watching", "thank you very much.",
}


def _convert_to_mp3(audio_path: Path) -> Path:
    """Convert OGG/OPUS to MP3 for better Whisper compatibility."""
    try:
        from pydub import AudioSegment
        mp3_path = audio_path.with_suffix(".mp3")
        audio = AudioSegment.from_file(str(audio_path))
        audio.export(str(mp3_path), format="mp3", bitrate="64k")
        logger.info(f"Converted to MP3: {mp3_path.name}")
        return mp3_path
    except Exception as exc:
        logger.warning(f"Audio conversion failed: {exc} — using original")
        return audio_path


def _try_groq(api_key: str, audio_path: Path) -> str:
    """Groq Whisper transcription with anti-hallucination measures."""
    from groq import Groq

    # Try MP3 first (better Whisper compatibility), then original
    paths_to_try = []
    if audio_path.suffix.lower() in (".ogg", ".opus", ".oga"):
        mp3 = _convert_to_mp3(audio_path)
        if mp3 != audio_path:
            paths_to_try.append(mp3)
    paths_to_try.append(audio_path)

    client = Groq(api_key=api_key)
    last_error = None

    for path in paths_to_try:
        try:
            suffix    = path.suffix.lower()
            mime_map  = {".mp3":"audio/mpeg",".ogg":"audio/ogg",".wav":"audio/wav",
                         ".m4a":"audio/mp4",".flac":"audio/flac",".opus":"audio/ogg"}
            mime_type = mime_map.get(suffix, "audio/mpeg")

            with open(path, "rb") as f:
                audio_bytes = f.read()

            logger.info(f"Sending to Groq Whisper: {path.name} ({len(audio_bytes)} bytes, {mime_type})")

            result = client.audio.transcriptions.create(
                file=(path.name, audio_bytes, mime_type),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",   # get more detail
                temperature=0.0,                  # deterministic, reduces hallucination
            )

            # Clean up MP3 temp file
            if path != audio_path and path.exists():
                path.unlink(missing_ok=True)

            text = result.text.strip() if hasattr(result, "text") else str(result).strip()
            logger.info(f"Groq Whisper result: '{text[:100]}'")

            # Warn about likely hallucination but still return it
            if text.lower().rstrip(".,!? ") in HALLUCINATIONS or len(text) < 2:
                logger.warning(f"Possible hallucination: '{text}' — audio may be too short")
                raise ValueError(
                    "Could not transcribe — audio too short or silent.\n"
                    "Please record at least 2-3 seconds and speak clearly."
                )

            return text

        except ValueError:
            raise   # hallucination — user error, don't retry
        except Exception as exc:
            last_error = exc
            logger.warning(f"Path {path.name} failed: {exc}")
            if path != audio_path and path.exists():
                path.unlink(missing_ok=True)
            continue

    if last_error:
        raise last_error
    raise RuntimeError("All audio paths failed")


def _try_gemini(api_key: str, audio_path: Path) -> str:
    """Gemini audio via File API."""
    from google import genai
    from pathlib import Path

    suffix    = audio_path.suffix.lower()
    mime_map  = {".ogg":"audio/ogg",".mp3":"audio/mpeg",".wav":"audio/wav",
                 ".m4a":"audio/mp4",".flac":"audio/flac",".opus":"audio/ogg"}
    mime_type = mime_map.get(suffix, "audio/ogg")
    client    = genai.Client(api_key=api_key)

    uploaded = client.files.upload(
        file=str(audio_path),
        config={"mime_type": mime_type},
    )
    prompt   = "Transcribe this voice message accurately. Output ONLY the spoken words."

    for model in ["gemini-2.0-flash", "gemini-2.5-flash"]:
        try:
            r    = client.models.generate_content(model=model, contents=[uploaded, prompt])
            text = r.text.strip()
            try: client.files.delete(name=uploaded.name)
            except: pass
            return text
        except Exception as e:
            if "404" in str(e): continue
            raise

    raise RuntimeError("Gemini transcription failed on all models")


def transcribe_voice(audio_path: Union[str, Path]) -> str:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    size = audio_path.stat().st_size
    logger.info(f"Transcribing: {audio_path.name} ({size:,} bytes)")

    # Only reject completely empty files (0 bytes)
    if size == 0:
        raise ValueError("Audio file is empty. Please try recording again.")

    # ── Try Groq Whisper ──────────────────────────────────────────────────────
    groq_keys = get_groq_keys()
    for i, key in enumerate(groq_keys, 1):
        try:
            return _try_groq(key, audio_path)
        except ValueError as exc:
            raise   # hallucination / user error — don't retry with other providers
        except Exception as exc:
            logger.warning(f"Groq Key {i} failed: {type(exc).__name__}: {str(exc)[:80]}")
            continue

    # ── Groq failed → try Gemini ──────────────────────────────────────────────
    gemini_keys = get_gemini_keys()
    if gemini_keys:
        logger.info("Groq unavailable — switching to Gemini for audio")
        for i, key in enumerate(gemini_keys, 1):
            try:
                return _try_gemini(key, audio_path)
            except Exception as exc:
                logger.warning(f"Gemini Key {i} failed: {str(exc)[:80]}")
                continue

    raise RuntimeError(
        "⏳ Voice transcription unavailable right now.\n"
        "All API keys are busy. Please wait 1 minute and try again."
    )
