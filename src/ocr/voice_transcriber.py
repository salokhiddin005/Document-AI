"""
Voice Transcriber — Groq Whisper primary, Gemini fallback.
Uses FFmpeg to convert Telegram OGG/OPUS to WAV before transcription.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Union
from loguru import logger
from src.ocr.api_key_manager import get_groq_keys, get_gemini_keys, is_rate_limit


def _convert_to_wav(audio_path: Path) -> Path:
    """
    Convert any audio to WAV using FFmpeg.
    WAV is universally supported by all Whisper implementations.
    Returns WAV path, or original path if FFmpeg not available.
    """
    wav_path = audio_path.with_suffix(".wav")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",           # overwrite output
                "-i", str(audio_path),    # input file
                "-ar", "16000",           # 16kHz sample rate (Whisper optimal)
                "-ac", "1",               # mono channel
                "-f", "wav",              # WAV format
                str(wav_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0:
            logger.info(f"FFmpeg converted: {audio_path.name} → {wav_path.name} ({wav_path.stat().st_size} bytes)")
            return wav_path
        else:
            logger.warning(f"FFmpeg failed (code {result.returncode}): {result.stderr.decode()[:200]}")
            return audio_path
    except FileNotFoundError:
        logger.warning("FFmpeg not found — sending original audio to Whisper")
        return audio_path
    except Exception as exc:
        logger.warning(f"FFmpeg error: {exc}")
        return audio_path


def _try_groq(api_key: str, audio_path: Path) -> str:
    """Groq Whisper transcription."""
    from groq import Groq

    # Convert to WAV first for best compatibility
    wav_path    = _convert_to_wav(audio_path)
    use_path    = wav_path if wav_path != audio_path else audio_path
    mime_type   = "audio/wav" if use_path.suffix == ".wav" else "audio/ogg"

    client = Groq(api_key=api_key)

    try:
        with open(use_path, "rb") as f:
            audio_bytes = f.read()

        logger.info(f"Sending to Groq Whisper: {use_path.name} ({len(audio_bytes):,} bytes, {mime_type})")

        result = client.audio.transcriptions.create(
            file=(use_path.name, audio_bytes, mime_type),
            model="whisper-large-v3",     # full model, not turbo — more accurate
            response_format="text",
            temperature=0.0,              # deterministic
        )

        text = result.strip() if isinstance(result, str) else result.text.strip()
        logger.info(f"Groq Whisper result: '{text[:100]}'")

        if not text:
            raise ValueError("Whisper returned empty result — please try again.")

        return text

    finally:
        # Clean up WAV temp file
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def _try_gemini(api_key: str, audio_path: Path) -> str:
    """Gemini audio via File API as fallback."""
    from google import genai

    wav_path  = _convert_to_wav(audio_path)
    use_path  = wav_path if wav_path != audio_path else audio_path
    mime_type = "audio/wav" if use_path.suffix == ".wav" else "audio/ogg"
    client    = genai.Client(api_key=api_key)

    try:
        uploaded = client.files.upload(file=str(use_path), config={"mime_type": mime_type})
        prompt   = "Transcribe this audio accurately. Output ONLY the spoken words, nothing else."

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

        raise RuntimeError("Gemini transcription failed")
    finally:
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def transcribe_voice(audio_path: Union[str, Path]) -> str:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    size = audio_path.stat().st_size
    logger.info(f"Transcribing: {audio_path.name} ({size:,} bytes)")

    if size == 0:
        raise ValueError("Audio file is empty. Please try recording again.")

    # ── Try Groq Whisper (primary) ────────────────────────────────────────────
    for i, key in enumerate(get_groq_keys(), 1):
        try:
            return _try_groq(key, audio_path)
        except ValueError:
            raise  # user-facing error, don't retry
        except Exception as exc:
            logger.warning(f"Groq Key {i} failed: {str(exc)[:100]}")
            continue

    # ── Groq failed → try Gemini (fallback) ──────────────────────────────────
    if get_gemini_keys():
        logger.info("Groq unavailable — switching to Gemini")
        for i, key in enumerate(get_gemini_keys(), 1):
            try:
                return _try_gemini(key, audio_path)
            except Exception as exc:
                logger.warning(f"Gemini Key {i} failed: {str(exc)[:100]}")
                continue

    raise RuntimeError(
        "Voice transcription unavailable right now.\n"
        "Please wait 1 minute and try again."
    )
