"""
Gemini API Key Manager — rotates between multiple API keys on rate limit.

Add keys to Railway Variables:
  GEMINI_API_KEY    = first key
  GEMINI_API_KEY_2  = second key
  GEMINI_API_KEY_3  = third key (optional)

The manager tries each key in order. When one hits 429, it switches to the next.
"""

from __future__ import annotations

import os
import time
from loguru import logger


def get_all_keys() -> list[str]:
    """Collect all configured Gemini API keys."""
    keys = []
    for var in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]:
        k = os.getenv(var, "").strip()
        if k:
            keys.append(k)
    return keys


def run_with_key_rotation(func, *args, **kwargs):
    """
    Run func(api_key, *args, **kwargs) rotating through all available API keys.
    Switches key automatically on 429 rate limit.
    Retries each key once before moving on.
    """
    keys = get_all_keys()
    if not keys:
        raise RuntimeError(
            "No Gemini API key found!\n"
            "Add GEMINI_API_KEY in Railway → Variables tab."
        )

    last_error = None

    for key_idx, api_key in enumerate(keys, 1):
        key_label = f"Key {key_idx}"
        for attempt in range(2):   # retry each key once on 429
            try:
                logger.debug(f"Trying Gemini {key_label} (attempt {attempt+1})")
                result = func(api_key, *args, **kwargs)
                if key_idx > 1:
                    logger.info(f"Success using {key_label} after key rotation")
                return result

            except Exception as exc:
                err = str(exc)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0:
                        wait = 20
                        logger.warning(
                            f"Rate limit on {key_label}, waiting {wait}s..."
                        )
                        time.sleep(wait)
                        continue
                    else:
                        last_error = exc
                        logger.warning(
                            f"Rate limit persists on {key_label}, "
                            f"{'switching to next key' if key_idx < len(keys) else 'all keys exhausted'}"
                        )
                        break   # try next key
                else:
                    raise   # not a rate limit error — propagate immediately

    raise RuntimeError(
        f"All {len(keys)} API key(s) are rate-limited.\n"
        f"Please wait 1 minute and try again.\n"
        f"Or add more keys: GEMINI_API_KEY_2, GEMINI_API_KEY_3 in Railway Variables."
    )
