"""
Voice Transcriber — Multiple providers, tries each until one works.

Priority:
  1. AssemblyAI    — handles OGG/OPUS natively, most reliable (ASSEMBLYAI_API_KEY)
  2. Groq Whisper  — fast, free (GROQ_API_KEY)
  3. Gemini        — fallback (GEMINI_API_KEY)
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Union
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  AssemblyAI — handles ANY audio format including Telegram OGG/OPUS
# ─────────────────────────────────────────────────────────────────────────────

def _assemblyai_transcribe(
    audio_path: Path,
    language: Optional[str] = None,
    telegram_url: Optional[str] = None,
) -> str:
    """Transcribe via AssemblyAI — uses Telegram URL directly (most reliable)."""
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not configured")

    import assemblyai as aai
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(
        language_code=language if language and language not in ("auto", "any") else None,
        language_detection=not bool(language and language not in ("auto", "any")),
    )

    # Prefer Telegram URL — AssemblyAI downloads it directly, no format issues
    source = telegram_url if telegram_url else str(audio_path)
    size   = audio_path.stat().st_size if audio_path.exists() else 0

    logger.info(f"AssemblyAI: source={'URL' if telegram_url else 'file'} ({size:,} bytes) lang={language}")
    transcriber = aai.Transcriber()
    transcript  = transcriber.transcribe(source, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI error: {transcript.error}")

    text = (transcript.text or "").strip()
    logger.info(f"AssemblyAI result: '{text[:150]}'")

    if not text:
        raise ValueError("No speech detected — please speak clearly into the microphone.")
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  FFmpeg conversion (best-effort — works if ffmpeg installed)
# ─────────────────────────────────────────────────────────────────────────────

def _to_wav(audio_path: Path) -> Path:
    wav_path = audio_path.with_suffix(".wav")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", str(wav_path)],
            capture_output=True, timeout=30,
        )
        if r.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 100:
            logger.info(f"FFmpeg: converted to WAV ({wav_path.stat().st_size:,} bytes)")
            return wav_path
    except Exception as exc:
        logger.warning(f"FFmpeg unavailable: {exc}")
    return audio_path


# ─────────────────────────────────────────────────────────────────────────────
#  Groq Whisper
# ─────────────────────────────────────────────────────────────────────────────

def _groq_transcribe(api_key: str, audio_path: Path, language: Optional[str] = None) -> str:
    from groq import Groq

    wav_path  = _to_wav(audio_path)
    use_path  = wav_path if wav_path != audio_path else audio_path
    mime_type = "audio/wav" if use_path.suffix == ".wav" else "audio/ogg"

    client = Groq(api_key=api_key)
    try:
        with open(use_path, "rb") as f:
            audio_bytes = f.read()

        kwargs = dict(
            file=(use_path.name, audio_bytes, mime_type),
            model="whisper-large-v3",
            response_format="text",
            temperature=0.0,
        )
        if language and language not in ("auto", "any", ""):
            kwargs["language"] = language

        logger.info(f"Groq Whisper: {use_path.name} ({len(audio_bytes):,}b) lang={language}")
        result = client.audio.transcriptions.create(**kwargs)
        text   = (result if isinstance(result, str) else result.text).strip()
        logger.info(f"Groq result: '{text[:100]}'")

        if not text:
            raise ValueError("Empty result from Whisper.")
        return text
    finally:
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Gemini fallback
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_transcribe(api_key: str, audio_path: Path, language: Optional[str] = None) -> str:
    from google import genai

    wav_path  = _to_wav(audio_path)
    use_path  = wav_path if wav_path != audio_path else audio_path
    mime_type = "audio/wav" if use_path.suffix == ".wav" else "audio/ogg"
    client    = genai.Client(api_key=api_key)

    try:
        uploaded = client.files.upload(file=str(use_path), config={"mime_type": mime_type})
        lang_str = f" Speak language: {language}." if language else ""
        prompt   = f"Transcribe this audio. Output ONLY the spoken words.{lang_str}"

        for model in ["gemini-2.0-flash", "gemini-2.5-flash"]:
            try:
                r = client.models.generate_content(model=model, contents=[uploaded, prompt])
                try: client.files.delete(name=uploaded.name)
                except: pass
                return r.text.strip()
            except Exception as e:
                if "404" in str(e): continue
                raise
        raise RuntimeError("All Gemini models failed")
    finally:
        if wav_path != audio_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_voice(
    audio_path: Union[str, Path],
    language: Optional[str] = None,
    telegram_url: Optional[str] = None,
) -> str:
    audio_path = Path(audio_path)
    size = audio_path.stat().st_size if audio_path.exists() else 0
    logger.info(f"Transcribing: {audio_path.name} ({size:,} bytes) lang={language}")

    if size == 0 and not telegram_url:
        raise ValueError("Audio file is empty. Please try again.")

    from src.ocr.api_key_manager import get_groq_keys, get_gemini_keys

    # ── 1. AssemblyAI — uses Telegram URL directly, most reliable ─────────────
    if os.getenv("ASSEMBLYAI_API_KEY", "").strip():
        try:
            return _assemblyai_transcribe(audio_path, language, telegram_url)
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(f"AssemblyAI failed: {exc} — trying Groq")

    # ── 2. Groq Whisper ───────────────────────────────────────────────────────
    for i, key in enumerate(get_groq_keys(), 1):
        try:
            return _groq_transcribe(key, audio_path, language)
        except ValueError:
            raise
        except Exception as exc:
            logger.warning(f"Groq Key {i} failed: {str(exc)[:100]}")

    # ── 3. Gemini ─────────────────────────────────────────────────────────────
    for i, key in enumerate(get_gemini_keys(), 1):
        try:
            return _gemini_transcribe(key, audio_path, language)
        except Exception as exc:
            logger.warning(f"Gemini Key {i} failed: {str(exc)[:100]}")

    raise RuntimeError("All transcription services failed. Please wait 1 minute and try again.")
