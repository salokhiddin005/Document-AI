"""
Formats DocumentResult into professional Telegram messages.
Includes confidence bars, annotated images, and clean result cards.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from src.pipeline.document_pipeline import DocumentResult

MAX_MSG_LEN = 4096


# ─────────────────────────────────────────────────────────────────────────────
#  Visual helpers
# ─────────────────────────────────────────────────────────────────────────────

def confidence_bar(value: float, width: int = 10) -> str:
    """
    Convert 0-1 float to a visual bar.
    0.96 → ██████████ 96%  (filled)
    0.60 → ██████░░░░ 60%
    """
    filled = round(value * width)
    empty  = width - filled
    bar    = "█" * filled + "░" * empty
    pct    = round(value * 100)
    return f"{bar} {pct}%"


def quality_emoji(score: float) -> str:
    if score >= 0.80: return "🟢"
    if score >= 0.50: return "🟡"
    return "🔴"


def review_badge(required: bool) -> str:
    return "⚠️ Some lines need review" if required else "✅ All lines auto-accepted"


# ─────────────────────────────────────────────────────────────────────────────
#  Main result message
# ─────────────────────────────────────────────────────────────────────────────

def format_result(result: "DocumentResult", lang: str = "en") -> str:
    from src.ocr.text_extractor import SUPPORTED_LANGUAGES
    q          = quality_emoji(result.image_quality)
    lang_label = SUPPORTED_LANGUAGES.get(lang, lang)

    msg = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📄  <b>DOCUMENT AI — RESULTS</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "📊  <b>Quality Report</b>",
        f"  OCR Confidence  {confidence_bar(result.overall_confidence)}",
        f"  Image Quality {q} {confidence_bar(result.image_quality)}",
        f"  Language        {lang_label}",
        f"  Words detected   <code>{result.word_count}</code>",
        f"  Lines detected   <code>{len(result.lines)}</code>",
        f"  Paragraphs       <code>{len(result.paragraphs)}</code>",
        f"  Processing time  <code>{result.processing_time_ms / 1000:.1f}s</code>",
        f"  Document ID      <code>{result.document_id}</code>",
    ]

    if result.barcode:
        msg += [
            "",
            "🔳  <b>Barcode Detected</b>",
            f"  Value      <code>{result.barcode}</code>",
            f"  Confidence {confidence_bar(result.barcode_confidence)}",
        ]

    # ── Extracted text block ──────────────────────────────────────────────────
    msg += [
        "",
        "📝  <b>Extracted Text</b>",
        "<pre>",
    ]
    for line in result.lines:
        prefix = "⚠ " if line.needs_review else ""
        msg.append(f"{prefix}{line.text}")
    msg.append("</pre>")

    # ── Per-line confidence breakdown ─────────────────────────────────────────
    msg += [
        "",
        "📈  <b>Line Confidence</b>",
    ]
    for i, line in enumerate(result.lines, 1):
        icon    = "🔴" if line.needs_review else ("🟡" if line.confidence < 0.90 else "🟢")
        display = line.text[:35] + "…" if len(line.text) > 35 else line.text
        msg.append(f"  {icon} L{i:02d} <code>{line.confidence:.0%}</code>  {display}")

    # ── Review summary ────────────────────────────────────────────────────────
    msg += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  {review_badge(result.requires_human_review)}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if result.low_confidence_lines:
        msg += ["", "⚠️  <b>Lines Sent to Review Queue</b>"]
        for lc in result.low_confidence_lines:
            display = lc.text[:40] + "…" if len(lc.text) > 40 else lc.text
            msg.append(f"  • <code>{display}</code>  ({lc.confidence:.0%})")

    full = "\n".join(msg)
    if len(full) > MAX_MSG_LEN:
        full = full[: MAX_MSG_LEN - 80] + "\n\n<i>…truncated. See attached JSON.</i>"
    return full


# ─────────────────────────────────────────────────────────────────────────────
#  Annotated image — draws coloured bounding boxes on the original
# ─────────────────────────────────────────────────────────────────────────────

def build_annotated_image(
    result: "DocumentResult",
    original_image: np.ndarray,
    out_dir: Path,
) -> Path:
    """
    Draw coloured bounding boxes around each detected line.
    Green  = high confidence (≥ 85%)
    Yellow = medium (70–85%)
    Red    = low / needs review (< 70%)
    Returns path to the saved annotated image.
    """
    img = original_image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for i, line in enumerate(result.lines, 1):
        x1, y1, x2, y2 = line.bbox

        # Choose colour by confidence
        if line.confidence >= 0.85:
            colour = (34, 197, 94)    # green
        elif line.confidence >= 0.70:
            colour = (234, 179, 8)    # yellow
        else:
            colour = (239, 68, 68)    # red

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        # Label background
        label     = f"L{i:02d} {line.confidence:.0%}"
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness  = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        lx, ly = x1, max(y1 - 4, th + 4)
        cv2.rectangle(img, (lx, ly - th - 4), (lx + tw + 6, ly + 2), colour, -1)
        cv2.putText(img, label, (lx + 3, ly - 1), font, font_scale, (255, 255, 255), thickness)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.document_id}_annotated.jpg"
    cv2.imwrite(str(path), img)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  File builders
# ─────────────────────────────────────────────────────────────────────────────

def build_json_file(result: "DocumentResult", out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.document_id}.json"
    from main import _to_clean_json
    with open(path, "w", encoding="utf-8") as f:
        f.write(_to_clean_json(result.to_dict()))
    return path


def build_txt_file(result: "DocumentResult", out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.document_id}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Document ID  : {result.document_id}\n")
        f.write(f"Confidence   : {result.overall_confidence:.0%}\n")
        f.write(f"Words        : {result.word_count}\n")
        f.write(f"Lines        : {len(result.lines)}\n")
        if result.barcode:
            f.write(f"Barcode      : {result.barcode}\n")
        f.write("\n" + "─" * 40 + "\n\n")
        f.write(result.full_text)
        f.write("\n\n" + "─" * 40 + "\n")
        f.write("\nLINE BREAKDOWN\n")
        for i, line in enumerate(result.lines, 1):
            flag = " [REVIEW]" if line.needs_review else ""
            f.write(f"L{i:02d} [{line.confidence:.0%}]{flag}  {line.text}\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Error / status messages
# ─────────────────────────────────────────────────────────────────────────────

def format_error(error: str) -> str:
    err_lower = error.lower()
    is_rate_limit = any(x in err_lower for x in [
        "429", "rate", "quota", "exhausted", "all providers failed", "busy"
    ])

    if is_rate_limit:
        return (
            "⏳  <b>API Rate Limit Reached</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Both Gemini and Groq API keys are temporarily busy.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔧  <b>How to fix:</b>\n\n"
            "  ⏱️  <b>Option 1 — Wait 1 minute</b>\n"
            "       Free API limits reset every 60 seconds\n\n"
            "  🔑  <b>Option 2 — Add more API keys</b>\n"
            "       Go to Railway → Variables → Add:\n"
            "       <code>GEMINI_API_KEY_2</code> = new Gemini key\n"
            "       <code>GROQ_API_KEY_2</code> = new Groq key\n\n"
            "  🌐  Get free keys:\n"
            "       Gemini: <b>aistudio.google.com</b>\n"
            "       Groq:   <b>console.groq.com</b>"
        )

    # Generic error with context-aware tips
    err_short = error[:200]
    if any(x in err_lower for x in ["voice", "audio", "transcri", "whisper"]):
        tips = (
            "🎙️  Speak clearly and loudly\n"
            "  🔇  Record in a quiet place\n"
            "  ⏱️  Send at least 3-5 seconds of speech"
        )
    elif any(x in err_lower for x in ["image", "photo", "file", "download"]):
        tips = (
            "📎  Send as <b>File</b> for better quality\n"
            "  💡  Use good lighting, no shadows\n"
            "  📐  Hold camera flat above the page"
        )
    else:
        tips = (
            "⏳  Wait 1 minute and try again\n"
            "  🔄  Use /start to reset if needed"
        )

    return (
        "😬  <b>Something went wrong</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<code>{err_short}</code>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💡  {tips}"
    )


def format_low_quality() -> str:
    return (
        "⚠️  <b>Low Image Quality Detected</b>\n\n"
        "Results may be less accurate.\n\n"
        "💡 <b>For better results:</b>\n"
        "  • 💡 Use good lighting\n"
        "  • 📐 Hold camera flat above the page\n"
        "  • 🎯 Make sure text is in sharp focus\n"
        "  • 📎 Send as <b>File</b> instead of photo\n"
        "     (Attach → File → pick image)"
    )
