"""
API Key Manager — rotates between multiple Groq API keys on rate limit.

Add keys to Railway Variables:
  GROQ_API_KEY    = first key
  GROQ_API_KEY_2  = second key
  GROQ_API_KEY_3  = third key (optional)
"""

from __future__ import annotations

import os
import time
from loguru import logger


def get_all_keys() -> list[str]:
    keys = []
    for var in ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]:
        k = os.getenv(var, "").strip()
        if k:
            keys.append(k)
    return keys


def run_with_key_rotation(func, *args, **kwargs):
    """
    Run func(api_key, *args, **kwargs) rotating through all Groq API keys.
    Switches key automatically on 429 rate limit.
    """
    keys = get_all_keys()
    if not keys:
        raise RuntimeError(
            "No Groq API key found!\n"
            "Add GROQ_API_KEY in Railway → Variables tab.\n"
            "Get free key from: console.groq.com"
        )

    last_error = None
    for key_idx, api_key in enumerate(keys, 1):
        key_label = f"Key {key_idx}"
        for attempt in range(2):
            try:
                logger.debug(f"Trying Groq {key_label} (attempt {attempt+1})")
                result = func(api_key, *args, **kwargs)
                if key_idx > 1:
                    logger.info(f"Success using Groq {key_label}")
                return result
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate_limit" in err.lower() or "RateLimitError" in type(exc).__name__:
                    if attempt == 0:
                        logger.warning(f"Rate limit on Groq {key_label}, waiting 10s...")
                        time.sleep(10)
                        continue
                    else:
                        last_error = exc
                        logger.warning(
                            f"Rate limit persists on Groq {key_label}, "
                            f"{'switching to next key' if key_idx < len(keys) else 'all keys exhausted'}"
                        )
                        break
                else:
                    raise

    raise RuntimeError(
        f"All {len(keys)} Groq API key(s) are rate-limited.\n"
        f"Please wait 1 minute and try again.\n"
        f"Or add more keys: GROQ_API_KEY_2, GROQ_API_KEY_3 in Railway Variables."
    )
