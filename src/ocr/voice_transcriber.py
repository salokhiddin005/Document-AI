"""
Voice Transcriber — Groq Whisper with FFmpeg conversion + language hint.
The #1 fix for hallucination: tell Whisper what language to expect.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional, Union
from loguru import logger
from src.ocr.api_key_manager import get_groq_keys, get_gemini_keys

# Groq Whisper language codes (ISO 639-1)
GROQ_LANGS = {
    "uz", "en", "ru", "ko", "fr", "de", "es", "ar",
    "zh", "ja", "tr", "pl", "it", "pt", "hi", "uk", "vi", "th",
}


def _convert_to_wav(audio_path: Path) -> Path:
    """Convert audio to 16kHz mono WAV using FFmpeg."""
    wav_path = audio_path.with_suffix(".wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-f", "wav", str(wav_path)],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 100:
            logger.info(f"FFmpeg OK: {wav_path.name} ({wav_path.stat().st_size:,} bytes)")
            return wav_path
        logger.warning(f"FFmpeg failed: {result.stderr.decode()[:150]}")
    except FileNotFoundError:
        logger.warning("FFmpeg not found — sending original to Whisper")
    except Exception as exc:
        logger.warning(f"FFmpeg error: {exc}")
    return audio_path


def _groq_transcribe(api_key: str, audio_path: Path, language: Optional[str] = None) -> str:
    from groq import Groq

    # Convert to WAV for best compatibility
    wav_path  = _convert_to_wav(audio_path)
    use_path  = wav_path if wav_path != audio_path else audio_path
    mime_type = "audio/wav" if use_path.suffix == ".wav" else "audio/ogg"

    client = Groq(api_key=api_key)

    try:
        with open(use_path, "rb") as f:
            audio_bytes = f.read()

        logger.info(f"Groq Whisper: {use_path.name} ({len(audio_bytes):,} bytes), lang={language}")

        # KEY FIX: Always specify language — prevents wrong auto-detection
        kwargs = dict(
            file=(use_path.name, audio_bytes, mime_type),
            model="whisper-large-v3",
            response_format="text",
            temperature=0.0,
        )
        if language and language in GROQ_LANGS:
            kwargs["language"] = language
            logger.info(f"Whisper language set to: {language}")

        result  = client.audio.transcriptions.create(**kwargs)
        text    = result.strip() if isinstance(result, str) else result.text.strip()
        logger.info(f"Whisper result: '{text[:150]}'")

        if not text:
            raise ValueError("Empty transcription — please speak louder or try again.")
        return text

    finally:
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def _gemini_transcribe(api_key: str, audio_path: Path, language: Optional[str] = None) -> str:
    from google import genai

    wav_path  = _convert_to_wav(audio_path)
    use_path  = wav_path if wav_path != audio_path else audio_path
    mime_type = "audio/wav" if use_path.suffix == ".wav" else "audio/ogg"
    client    = genai.Client(api_key=api_key)

    try:
        uploaded = client.files.upload(file=str(use_path), config={"mime_type": mime_type})
        lang_hint = f" The speaker is speaking in {language}." if language else ""
        prompt    = f"Transcribe this audio accurately. Output ONLY the spoken words.{lang_hint}"

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
        raise RuntimeError("All Gemini models failed")
    finally:
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


def transcribe_voice(audio_path: Union[str, Path], language: Optional[str] = None) -> str:
    """
    Transcribe audio file to text.
    language: ISO 639-1 code (e.g. 'uz', 'en', 'ru') — strongly recommended!
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    size = audio_path.stat().st_size
    logger.info(f"Transcribing: {audio_path.name} ({size:,} bytes), language={language}")

    if size == 0:
        raise ValueError("Audio file is empty. Please try recording again.")

    # ── Try Groq Whisper ──────────────────────────────────────────────────────
    for i, key in enumerate(get_groq_keys(), 1):
        try:
            return _groq_transcribe(key, audio_path, language)
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(f"Groq Key {i} failed: {str(exc)[:100]}")
            continue

    # ── Fallback: Gemini ──────────────────────────────────────────────────────
    if get_gemini_keys():
        logger.info("Groq unavailable — switching to Gemini")
        for i, key in enumerate(get_gemini_keys(), 1):
            try:
                return _gemini_transcribe(key, audio_path, language)
            except Exception as exc:
                logger.warning(f"Gemini Key {i} failed: {str(exc)[:100]}")
                continue

    raise RuntimeError("Voice transcription unavailable. Please wait 1 minute and try again.")
