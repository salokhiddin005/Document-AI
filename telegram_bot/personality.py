"""
Bot personality — fun messages, stickers, and emoji-rich responses.
"""

from __future__ import annotations

import random
from telegram import Update
from telegram.ext import ContextTypes


# ─────────────────────────────────────────────────────────────────────────────
#  Sticker file IDs (from popular public Telegram packs)
# ─────────────────────────────────────────────────────────────────────────────

STICKERS = {
    "hello":    "CAACAgIAAxkBAAIBZWX0AAGa4j_WfGQpDJiUFZ_uSoVoAAIfBAAC4c6hS_nFbfGpZlFpNAQ",
    "thinking": "CAACAgIAAxkBAAIBZ2X0AAGn1rTGVuiAqeYkJYmrTz3kAAIvBQACKp-hS5iJeYCmFONjNAQ",
    "success":  "CAACAgIAAxkBAAIBaWX0AAGqfDk9IiHPEwnPjRyZ4sTpAAITBQACKp-hS3pxkP1c7bFBNAQ",
    "writing":  "CAACAgIAAxkBAAIBa2X0AAGtZ3_p7_yPWyHmPaL3O8tRAAIPBQACKp-hS8rB2xJBBX9HNAQ",
    "magic":    "CAACAgIAAxkBAAIBbWX0AAGv7_QfMfBq7FIe5WLqwZkgAAIRBQACKp-hS1kfDQkFrp9yNAQ",
    "done":     "CAACAgIAAxkBAAIBb2X0AAGyHuH7tHIKrPfvk_sxCfRnAAIVBQACKp-hS-hFUomHY9JiNAQ",
    "error":    "CAACAgIAAxkBAAIBcWX0AAG1q7SvYdHK1-Yz8OqrVjJjAAIXBQACKp-hS5PXLE4Dm_xgNAQ",
    "pdf":      "CAACAgIAAxkBAAIBc2X0AAG4L4pJbQiXDn5WMxvjgPDQAAIZBQACKp-hS-kSBN5jDFcBNAQ",
}

# ─────────────────────────────────────────────────────────────────────────────
#  Fun processing step messages
# ─────────────────────────────────────────────────────────────────────────────

OCR_STEPS = [
    "🔍  Putting on my reading glasses...",
    "🧠  Activating AI brain cells...",
    "👀  Reading every single pixel...",
    "🤓  Deciphering the handwriting...",
    "⚡  Almost there, magic happening...",
    "✨  Polishing the results...",
]

HW_STEPS = [
    "✍️  Picking up the pen...",
    "📝  Writing line by line...",
    "🖊️  Adding personal touches...",
    "🎨  Making it look beautiful...",
    "✨  Final flourishes...",
    "🎉  Almost done!",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Random fun messages
# ─────────────────────────────────────────────────────────────────────────────

GREETINGS = [
    "Hey there, superstar! 🌟",
    "Welcome back, genius! 🧠",
    "Hello, document wizard! 🪄",
    "Great to see you! 🎉",
    "Ready to make magic? ✨",
]

OCR_COMPLETE = [
    "📖  Reading complete! Look what I found! 🎯",
    "✅  Done! Every word captured! 🏆",
    "🎉  Nailed it! Here's what your document says:",
    "⚡  Lightning fast reading complete! 📚",
    "🤓  I read it all! Check this out:",
]

HW_COMPLETE = [
    "✍️  Your handwriting is ready! Looks beautiful! 🌟",
    "🎨  Masterpiece created! 🖼️",
    "✨  Magic done! Your handwritten page is here!",
    "📝  Written with love! Here it is! 💙",
    "🏆  Perfect handwriting delivered!",
]

SUMMARY_INTROS = [
    "🤖  Here's what your document is about:",
    "💡  TL;DR — The quick version:",
    "🧠  Smart summary coming right up:",
    "⚡  Here's the gist:",
    "📋  Quick summary for you:",
]

ERROR_MESSAGES = [
    "😅  Oops! Something went sideways.",
    "🙈  Hmm, that didn't work as expected.",
    "😬  Small hiccup! Let's try again.",
    "🤔  Something went wrong, but we'll fix it!",
]

TIPS = [
    "💡 <b>Pro tip:</b> Send images as <b>File</b> for best quality!",
    "💡 <b>Pro tip:</b> Use /lang to set your document language!",
    "💡 <b>Pro tip:</b> Try different /style for unique handwriting!",
    "💡 <b>Pro tip:</b> /history shows your last 10 results!",
    "💡 <b>Pro tip:</b> PDF files work too — just send them!",
]


def random_greeting() -> str:
    return random.choice(GREETINGS)

def random_ocr_complete() -> str:
    return random.choice(OCR_COMPLETE)

def random_hw_complete() -> str:
    return random.choice(HW_COMPLETE)

def random_summary_intro() -> str:
    return random.choice(SUMMARY_INTROS)

def random_error() -> str:
    return random.choice(ERROR_MESSAGES)

def random_tip() -> str:
    return random.choice(TIPS)

def random_ocr_step(step: int) -> str:
    idx = min(step, len(OCR_STEPS) - 1)
    return OCR_STEPS[idx]

def random_hw_step(step: int) -> str:
    idx = min(step, len(HW_STEPS) - 1)
    return HW_STEPS[idx]


async def send_sticker(msg, sticker_key: str) -> None:
    """Try to send a sticker — silently skip if it fails."""
    try:
        file_id = STICKERS.get(sticker_key)
        if file_id:
            await msg.reply_sticker(sticker=file_id)
    except Exception:
        pass  # sticker not critical — never crash for it
