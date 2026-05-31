"""
AI Provider Manager — Gemini first, Groq as fallback.

Priority order:
  1. GEMINI_API_KEY    (primary)
  2. GEMINI_API_KEY_2  (second Gemini key)
  3. GROQ_API_KEY      (Groq fallback)
  4. GROQ_API_KEY_2    (second Groq key)

When Gemini hits rate limit → automatically switches to Groq.
"""

from __future__ import annotations

import os
import time
from loguru import logger


def get_gemini_keys() -> list[str]:
    keys = []
    for var in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]:
        k = os.getenv(var, "").strip()
        if k:
            keys.append(k)
    return keys


def get_groq_keys() -> list[str]:
    keys = []
    for var in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]:
        k = os.getenv(var, "").strip()
        if k:
            keys.append(k)
    return keys


def is_rate_limit(err: str) -> bool:
    return any(x in err for x in ["429", "RESOURCE_EXHAUSTED", "rate_limit", "RateLimitError", "quota"])


def run_with_fallback(gemini_func, groq_func, *args, **kwargs):
    """
    Try Gemini keys first (in order), then Groq keys as fallback.
    gemini_func(api_key, *args) and groq_func(api_key, *args) must have same signature.
    """
    gemini_keys = get_gemini_keys()
    groq_keys   = get_groq_keys()

    if not gemini_keys and not groq_keys:
        raise RuntimeError(
            "No API keys found!\n"
            "Add GEMINI_API_KEY or GROQ_API_KEY in Railway → Variables."
        )

    # ── Try all Gemini keys ───────────────────────────────────────────────────
    for i, key in enumerate(gemini_keys, 1):
        for attempt in range(2):
            try:
                result = gemini_func(key, *args, **kwargs)
                logger.debug(f"✅ Gemini Key {i} succeeded")
                return result
            except Exception as exc:
                err = str(exc)
                if is_rate_limit(err):
                    if attempt == 0:
                        logger.warning(f"⏳ Gemini Key {i} rate limited, waiting 15s...")
                        time.sleep(15)
                        continue
                    else:
                        logger.warning(f"🔄 Gemini Key {i} still limited, trying next...")
                        break
                else:
                    raise   # not a rate limit — real error

    # ── Gemini exhausted → fall back to Groq ─────────────────────────────────
    if groq_keys:
        logger.info("🔁 Gemini rate limited — switching to Groq")
        for i, key in enumerate(groq_keys, 1):
            for attempt in range(2):
                try:
                    result = groq_func(key, *args, **kwargs)
                    logger.info(f"✅ Groq Key {i} succeeded (Gemini was rate limited)")
                    return result
                except Exception as exc:
                    err = str(exc)
                    if is_rate_limit(err):
                        if attempt == 0:
                            logger.warning(f"⏳ Groq Key {i} rate limited, waiting 10s...")
                            time.sleep(10)
                            continue
                        else:
                            logger.warning(f"🔄 Groq Key {i} still limited, trying next...")
                            break
                    else:
                        raise

    raise RuntimeError(
        "⏳ All API keys are rate limited right now.\n"
        "Please wait 1-2 minutes and try again.\n"
        "💡 Tip: Add more keys (GEMINI_API_KEY_2, GROQ_API_KEY_2) in Railway Variables."
    )
