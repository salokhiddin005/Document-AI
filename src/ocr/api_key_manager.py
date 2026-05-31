"""
AI Provider Manager — Gemini first, Groq as fallback.

Priority:
  1. GEMINI_API_KEY, GEMINI_API_KEY_2  (Gemini keys, tried in order)
  2. GROQ_API_KEY, GROQ_API_KEY_2      (Groq fallback if ALL Gemini keys fail)

Switches to Groq on ANY Gemini failure — not just rate limits.
"""

from __future__ import annotations

import os
import time
from loguru import logger


def get_gemini_keys() -> list[str]:
    return [os.getenv(v, "").strip() for v in
            ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
            if os.getenv(v, "").strip()]


def get_groq_keys() -> list[str]:
    return [os.getenv(v, "").strip() for v in
            ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
            if os.getenv(v, "").strip()]


def is_rate_limit(err: str) -> bool:
    return any(x in err for x in [
        "429", "RESOURCE_EXHAUSTED", "rate_limit",
        "RateLimitError", "quota", "too many requests",
    ])


def run_with_fallback(gemini_func, groq_func, *args, **kwargs):
    """
    Try ALL Gemini keys first. If any Gemini key fails for ANY reason,
    log it and move on. After all Gemini keys are exhausted, switch to Groq.
    """
    gemini_keys = get_gemini_keys()
    groq_keys   = get_groq_keys()

    if not gemini_keys and not groq_keys:
        raise RuntimeError(
            "No API keys configured!\n"
            "Add GEMINI_API_KEY and/or GROQ_API_KEY in Railway → Variables."
        )

    # ── Try Gemini keys ───────────────────────────────────────────────────────
    for i, key in enumerate(gemini_keys, 1):
        try:
            result = gemini_func(key, *args, **kwargs)
            logger.debug(f"✅ Gemini Key {i} succeeded")
            return result
        except Exception as exc:
            err = str(exc)
            if is_rate_limit(err):
                logger.warning(f"⏳ Gemini Key {i} rate limited — trying next")
            else:
                logger.warning(f"⚠️  Gemini Key {i} failed ({type(exc).__name__}: {err[:80]}) — trying next")
            # Always continue to next key regardless of error type

    # ── All Gemini keys failed → switch to Groq ───────────────────────────────
    if groq_keys:
        logger.info("🔁 All Gemini keys failed — switching to Groq automatically")

        for i, key in enumerate(groq_keys, 1):
            try:
                result = groq_func(key, *args, **kwargs)
                logger.info(f"✅ Groq Key {i} succeeded")
                return result
            except Exception as exc:
                err = str(exc)
                if is_rate_limit(err):
                    logger.warning(f"⏳ Groq Key {i} rate limited — trying next")
                else:
                    logger.warning(f"⚠️  Groq Key {i} failed ({type(exc).__name__}: {err[:80]}) — trying next")
                # Always continue to next key

    # ── Everything failed ─────────────────────────────────────────────────────
    providers = []
    if gemini_keys: providers.append(f"{len(gemini_keys)} Gemini key(s)")
    if groq_keys:   providers.append(f"{len(groq_keys)} Groq key(s)")

    raise RuntimeError(
        f"All providers failed ({', '.join(providers)}).\n"
        f"Please wait 1 minute and try again.\n"
        f"💡 Add more keys: GEMINI_API_KEY_2, GROQ_API_KEY_2 in Railway Variables."
    )
