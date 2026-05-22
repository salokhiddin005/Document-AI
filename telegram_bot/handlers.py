"""
Telegram handlers — child-friendly, guided, simple UX.
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ocr.auto_lang import detect_language, LANG_LABELS
from src.ocr.gemini_vision import extract_text_gemini
from src.ocr.history_manager import HistoryManager
from src.ocr.md_exporter import export_to_md, export_text_to_pdf
from src.ocr.pdf_reader import pdf_to_images
from src.ocr.summarizer import summarize_text
from src.ocr.text_extractor import SUPPORTED_LANGUAGES
from src.ocr.voice_transcriber import transcribe_voice
from src.ocr.word_exporter import export_to_docx
from src.preprocessing.image_enhancer import enhance_for_ocr
from src.text_to_handwriting.pdf_exporter import export_to_pdf
from src.text_to_handwriting.renderer import (
    HANDWRITING_STYLES, INK_COLORS, PAPER_STYLES, HandwritingRenderer
)
from telegram_bot.formatter import (
    build_json_file,
    build_txt_file,
    format_error,
)
from telegram_bot.personality import (
    random_summary_intro, send_sticker,
)

# ── Shared instances ──────────────────────────────────────────────────────────
_history: HistoryManager | None = None

TMP_DIR = ROOT / "data" / "telegram_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

_user_stats: dict = defaultdict(lambda: {"processed": 0, "words": 0, "time_ms": 0})

MODE_OCR    = "ocr"
MODE_HTR    = "htr"
MODE_IMG_HW = "img_to_hw"


def get_history() -> HistoryManager:
    global _history
    if _history is None:
        _history = HistoryManager()
    return _history


def get_renderer(context=None) -> HandwritingRenderer:
    if context is not None:
        style = context.user_data.get("hw_style", "casual")
        color = context.user_data.get("hw_color", "blue")
        paper = context.user_data.get("hw_paper", "lined")
    else:
        style, color, paper = "casual", "blue", "lined"
    return HandwritingRenderer(style=style, ink_color=color, paper_style=paper)


def _track_msg(context, message_id: int) -> None:
    ids = context.chat_data.setdefault("all_msg_ids", [])
    if message_id not in ids:
        ids.append(message_id)
    if len(ids) > 1000:
        context.chat_data["all_msg_ids"] = ids[-1000:]


def _track_user_msg(context, message_id: int) -> None:
    _track_msg(context, message_id)


def _resolve_text(doc_id: str, context) -> str:
    files     = (context.bot_data.get("files") or {}).get(doc_id)
    full_text = (files or {}).get("full_text", "")
    if full_text:
        return full_text
    entry = get_history().get_by_doc_id(doc_id)
    if entry:
        return entry.full_text
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Menus — simple, visual, child-friendly
# ─────────────────────────────────────────────────────────────────────────────

def _main_menu(context=None) -> InlineKeyboardMarkup:
    mode  = context.user_data.get("mode", "") if context else ""
    color = context.user_data.get("hw_color", "blue") if context else "blue"
    color_icons = {"blue":"🔵","black":"⚫","red":"🔴","green":"🟢","purple":"🟣"}

    def active(m): return "▶ " if mode == m else ""

    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{active(MODE_OCR)}📖  Read Handwriting",        callback_data="mode:ocr")],
        [InlineKeyboardButton(f"{active(MODE_HTR)}✍️  Type → Handwriting",      callback_data="mode:htr")],
        [InlineKeyboardButton(f"{active(MODE_IMG_HW)}🔄  Image → Handwriting",  callback_data="mode:img_to_hw")],
        [
            InlineKeyboardButton("🎨 Style",                        callback_data="select_style"),
            InlineKeyboardButton(f"{color_icons.get(color,'🔵')} Color", callback_data="select_color"),
            InlineKeyboardButton("📄 Paper",                        callback_data="select_paper"),
        ],
        [
            InlineKeyboardButton("🌐 Language",   callback_data="select_lang"),
            InlineKeyboardButton("📚 My History", callback_data="history"),
        ],
    ])


def _what_next_menu(doc_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🤖 Summarize",        callback_data=f"summarize:{doc_id}"),
            InlineKeyboardButton("✍️ Make Handwritten",  callback_data=f"to_hw:{doc_id}"),
        ],
        [
            InlineKeyboardButton("📝 Text file",  callback_data=f"txt:{doc_id}"),
            InlineKeyboardButton("📃 Word file",  callback_data=f"word:{doc_id}"),
            InlineKeyboardButton("📄 JSON file",  callback_data=f"json:{doc_id}"),
        ],
        [InlineKeyboardButton("🏠 Main Menu",   callback_data="menu")],
    ])


def _style_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✒️ Neat — Clean & tidy",       callback_data="style:neat")],
        [InlineKeyboardButton("🖊️ Casual — Normal handwriting", callback_data="style:casual")],
        [InlineKeyboardButton("📝 Student — School style",    callback_data="style:student")],
        [InlineKeyboardButton("👨‍⚕️ Doctor — Messy style",      callback_data="style:doctor")],
        [InlineKeyboardButton("🏠 Back",                       callback_data="menu")],
    ])


def _color_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔵 Blue (default)", callback_data="color:blue"),
         InlineKeyboardButton("⚫ Black",          callback_data="color:black")],
        [InlineKeyboardButton("🔴 Red",            callback_data="color:red"),
         InlineKeyboardButton("🟢 Green",          callback_data="color:green")],
        [InlineKeyboardButton("🟣 Purple",         callback_data="color:purple"),
         InlineKeyboardButton("🏠 Back",           callback_data="menu")],
    ])


def _paper_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Lined (default)", callback_data="paper:lined"),
         InlineKeyboardButton("⬜ Blank",            callback_data="paper:blank")],
        [InlineKeyboardButton("🔲 Grid",             callback_data="paper:grid"),
         InlineKeyboardButton("📓 Notebook",         callback_data="paper:notebook")],
        [InlineKeyboardButton("🏠 Back",             callback_data="menu")],
    ])


def _lang_keyboard() -> InlineKeyboardMarkup:
    items = list(SUPPORTED_LANGUAGES.items())
    rows  = []
    for i in range(0, len(items), 2):
        row = [InlineKeyboardButton(label, callback_data=f"lang:{code}")
               for code, label in items[i:i+2]]
        rows.append(row)
    rows.append([InlineKeyboardButton("🏠 Back", callback_data="menu")])
    return InlineKeyboardMarkup(rows)


def _back_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]])


# ─────────────────────────────────────────────────────────────────────────────
#  Welcome
# ─────────────────────────────────────────────────────────────────────────────

def _welcome(name: str, context=None) -> str:
    lang = context.user_data.get("ocr_lang", "en") if context else "en"
    lang_label = SUPPORTED_LANGUAGES.get(lang, "🇬🇧 English")
    mode = context.user_data.get("mode", "") if context else ""
    mode_names = {MODE_OCR: "📖 Read Handwriting",
                  MODE_HTR: "✍️ Type → Handwriting",
                  MODE_IMG_HW: "🔄 Image → Handwriting"}
    active_line = f"\n✅  Active: <b>{mode_names.get(mode,'')}</b>" if mode else ""

    return (
        f"👋  <b>Hi {name}! I am Document AI 🤖</b>\n\n"
        f"I can help you with 3 things:\n\n"
        f"  📖  <b>Read Handwriting</b>\n"
        f"       Take a photo → I read all the text\n\n"
        f"  ✍️  <b>Type → Handwriting</b>\n"
        f"       Type any text → I write it by hand\n\n"
        f"  🔄  <b>Image → Handwriting</b>\n"
        f"       Send a photo → I rewrite it by hand\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌐  Language: <b>{lang_label}</b>{active_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"👇  <b>Pick what you want to do:</b>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Commands
# ─────────────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    name = update.effective_user.first_name or "there"
    await send_sticker(update.message, "hello")
    m = await update.message.reply_html(_welcome(name, context), reply_markup=_main_menu(context))
    _track_msg(context, m.message_id)


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    m = await update.message.reply_html(
        "👇  <b>Choose what you want to do:</b>",
        reply_markup=_main_menu(context),
    )
    _track_msg(context, m.message_id)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    m = await update.message.reply_html(
        "📖  <b>How to use Document AI</b>\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "  <b>Step 1:</b> Tap a button to choose what you want\n\n"
        "  <b>📖 Read Handwriting</b>\n"
        "  → Tap it, then send a photo of any handwritten page\n"
        "  → I will read every word and give you the text!\n\n"
        "  <b>✍️ Type → Handwriting</b>\n"
        "  → Tap it, then type any text\n"
        "  → I will write it as if a human wrote it by hand!\n\n"
        "  <b>🔄 Image → Handwriting</b>\n"
        "  → Tap it, then send any document photo\n"
        "  → I will rewrite it in handwriting!\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "  📎  <b>Tip:</b> Send photos as <b>File</b> for best quality!\n"
        "  🎤  You can also send <b>voice messages</b>!\n"
        "  📄  PDF files are supported too!\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  /start  — Main menu\n"
        "  /history — See past results\n"
        "  /clear  — Clear chat\n"
        "  /mystats — Your statistics",
        reply_markup=_back_menu(),
    )
    _track_msg(context, m.message_id)


async def cmd_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    lang = context.user_data.get("ocr_lang", "en")
    m = await update.message.reply_html(
        f"🌐  <b>Choose the language of your document</b>\n\n"
        f"Current: <b>{SUPPORTED_LANGUAGES.get(lang, lang)}</b>\n\n"
        f"This helps me read the text more accurately!",
        reply_markup=_lang_keyboard(),
    )
    _track_msg(context, m.message_id)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    m = await update.message.reply_html(
        "🖥  <b>System Status</b>\n\n"
        "  ✅  Bot is online\n"
        "  ✅  Gemini AI reading engine ready\n"
        "  ✅  Handwriting engine ready\n"
        "  ✅  All systems operational",
        reply_markup=_back_menu(),
    )
    _track_msg(context, m.message_id)


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    user    = update.effective_user
    entries = get_history().get_user_history(str(user.id), limit=10)
    if not entries:
        m = await update.message.reply_html(
            "📚  <b>No history yet!</b>\n\n"
            "Tap <b>📖 Read Handwriting</b> and send me a photo.\n"
            "Your results will be saved here automatically! 🎯",
            reply_markup=_back_menu(),
        )
        _track_msg(context, m.message_id)
        return

    lines = ["📚  <b>Your Saved Results</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]
    for e in entries:
        lines.append(
            f"  🔖 <code>{e.id}</code>  🌐 {e.lang}  📝 {e.word_count} words  🕐 {e.created_at}\n"
            f"  💬 <i>{e.preview}</i>\n"
        )
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("💡 Get full text: /history_get &lt;id&gt;  •  🗑️ Clear chat: /clear")
    m = await update.message.reply_html("\n".join(lines), reply_markup=_back_menu())
    _track_msg(context, m.message_id)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    bot     = context.bot

    all_ids = list(context.chat_data.get("all_msg_ids", []))
    all_ids.append(update.message.message_id)

    deleted     = 0
    not_deleted = 0
    for mid in set(all_ids):
        try:
            await bot.delete_message(chat_id=chat_id, message_id=mid)
            deleted += 1
        except Exception:
            not_deleted += 1

    context.chat_data["all_msg_ids"] = []

    if not_deleted > 0:
        sep = await bot.send_message(
            chat_id=chat_id,
            text=(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "              🧹  Chat Cleared  🧹\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"<i>{not_deleted} old messages could not be deleted\n"
                f"(Telegram only allows deleting messages from the last 48 hours)</i>\n\n"
                "💾  Your saved results are still in /history"
            ),
            parse_mode="HTML",
        )
        _track_msg(context, sep.message_id)
    else:
        clean = await bot.send_message(
            chat_id=chat_id,
            text="🧹  Cleared!  Tap /start to begin.",
        )
        _track_msg(context, clean.message_id)


async def cmd_history_get(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    args = context.args
    if not args or not args[0].isdigit():
        m = await update.message.reply_text("Usage: /history_get <number>"); _track_msg(context, m.message_id); return
    entry = get_history().get_by_id(int(args[0]))
    if not entry:
        m = await update.message.reply_text("❌  Not found. Check /history for valid numbers."); _track_msg(context, m.message_id); return
    m = await update.message.reply_html(
        f"📄  <b>Result #{entry.id}</b>  [{entry.lang}]\n\n<pre>{entry.full_text[:3000]}</pre>"
    )
    _track_msg(context, m.message_id)


async def cmd_style(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    style = context.user_data.get("hw_style", "casual")
    m = await update.message.reply_html(
        f"🎨  <b>Handwriting Style</b>\n\nCurrent: <b>{style}</b>\n\nChoose a style:",
        reply_markup=_style_keyboard(),
    )
    _track_msg(context, m.message_id)


async def cmd_mystats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_msg(context, update.message.message_id)
    uid      = update.effective_user.id
    db_stats = get_history().get_user_stats(str(uid))
    mem      = _user_stats[uid]
    avg      = (mem["time_ms"] / mem["processed"] / 1000) if mem["processed"] > 0 else 0

    m = await update.message.reply_html(
        f"📊  <b>Your Stats</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🗂️  Documents processed : <code>{db_stats['processed']}</code>\n"
        f"  📝  Total words read    : <code>{db_stats['words']:,}</code>\n"
        f"  ⚡  Avg speed          : <code>{avg:.1f}s</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        reply_markup=_back_menu(),
    )
    _track_msg(context, m.message_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Callback handler
# ─────────────────────────────────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    name = update.effective_user.first_name or "there"

    if data == "menu":
        m = await query.message.reply_html(_welcome(name, context), reply_markup=_main_menu(context))
        _track_msg(context, m.message_id); return

    if data == "status_inline":
        pending = get_pipeline().queue.pending_count()
        m = await query.message.reply_html(f"✅  Bot online  •  Queue: {pending} items", reply_markup=_back_menu())
        _track_msg(context, m.message_id); return

    if data in ("mode:ocr", "mode:htr", "mode:img_to_hw"):
        mode = data.split(":", 1)[1]
        context.user_data["mode"] = mode
        instructions = {
            MODE_OCR:    ("📖  <b>Ready to Read!</b>\n\n📸  Now send me a photo of any handwritten page\n\nI'll read every single word! 🔍", "📸 Send a photo or image file now"),
            MODE_HTR:    ("✍️  <b>Ready to Write!</b>\n\n⌨️  Now type any text below\n\nI'll convert it to beautiful handwriting! 🎨", "⌨️ Type your text now"),
            MODE_IMG_HW: ("🔄  <b>Ready to Convert!</b>\n\n📸  Now send me any document photo\n\nI'll rewrite it in handwriting! ✍️", "📸 Send a photo or image file now"),
        }
        title, hint = instructions[mode]
        m = await query.message.reply_html(
            f"{title}\n\n<i>💡 {hint}</i>",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]]),
        )
        _track_msg(context, m.message_id); return

    if data == "select_style":
        m = await query.message.reply_html("🎨  <b>Choose Handwriting Style</b>", reply_markup=_style_keyboard())
        _track_msg(context, m.message_id); return

    if data == "select_color":
        m = await query.message.reply_html("🖋️  <b>Choose Ink Color</b>", reply_markup=_color_keyboard())
        _track_msg(context, m.message_id); return

    if data == "select_paper":
        m = await query.message.reply_html("📄  <b>Choose Paper Style</b>", reply_markup=_paper_keyboard())
        _track_msg(context, m.message_id); return

    if data == "select_lang":
        m = await query.message.reply_html("🌐  <b>Choose Document Language</b>\n\nThis helps me read more accurately!", reply_markup=_lang_keyboard())
        _track_msg(context, m.message_id); return

    if data.startswith("style:"):
        val = data.split(":", 1)[1]
        context.user_data["hw_style"] = val
        m = await query.message.reply_html(f"✅  Style set to <b>{val}</b>!", reply_markup=_main_menu(context))
        _track_msg(context, m.message_id); return

    if data.startswith("color:"):
        val = data.split(":", 1)[1]
        context.user_data["hw_color"] = val
        m = await query.message.reply_html(f"✅  Ink color set to <b>{val}</b>!", reply_markup=_main_menu(context))
        _track_msg(context, m.message_id); return

    if data.startswith("paper:"):
        val = data.split(":", 1)[1]
        context.user_data["hw_paper"] = val
        m = await query.message.reply_html(f"✅  Paper set to <b>{val}</b>!", reply_markup=_main_menu(context))
        _track_msg(context, m.message_id); return

    if data.startswith("lang:"):
        lang = data.split(":", 1)[1]
        context.user_data["ocr_lang"] = lang
        lang_label = SUPPORTED_LANGUAGES.get(lang, lang)
        get_pipeline().extractor.set_language(lang)
        m = await query.message.reply_html(
            f"✅  Language set to <b>{lang_label}</b>!\n\nNow send me a photo and I'll read it in {lang_label}.",
            reply_markup=_main_menu(context),
        )
        _track_msg(context, m.message_id); return

    if data == "history":
        entries = get_history().get_user_history(str(query.from_user.id), limit=8)
        if not entries:
            m = await query.message.reply_html("📚  No history yet. Process a document first!", reply_markup=_back_menu())
            _track_msg(context, m.message_id); return
        lines = ["📚  <b>Your Saved Results</b>\n"]
        for e in entries:
            lines.append(f"  🔖 <code>{e.id}</code>  {e.word_count}w  [{e.lang}]  {e.created_at}\n  <i>{e.preview}</i>\n")
        m = await query.message.reply_html("\n".join(lines), reply_markup=_back_menu())
        _track_msg(context, m.message_id); return

    if data.startswith("summarize:"):
        doc_id    = data.split(":", 1)[1]
        full_text = _resolve_text(doc_id, context)
        lang      = context.user_data.get("ocr_lang", "en")
        if not full_text:
            m = await query.message.reply_text("❌  Text not found. Please send the image again."); _track_msg(context, m.message_id); return
        thinking = await query.message.reply_html("🤖  <b>Creating summary...</b>\n\n<i>This takes a few seconds...</i>")
        _track_msg(context, thinking.message_id)
        try:
            loop    = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, lambda: summarize_text(full_text, lang_hint=lang))
            await thinking.delete()
            m = await query.message.reply_html(f"{random_summary_intro()}\n\n{summary}", reply_markup=_back_menu())
            _track_msg(context, m.message_id)
        except Exception as exc:
            await thinking.delete()
            m = await query.message.reply_html(format_error(str(exc))); _track_msg(context, m.message_id)
        return

    if data.startswith("word:"):
        doc_id    = data.split(":", 1)[1]
        full_text = _resolve_text(doc_id, context)
        if not full_text:
            m = await query.message.reply_text("❌  Text not found. Please send the image again."); _track_msg(context, m.message_id); return
        try:
            docx_path = TMP_DIR / f"{doc_id}.docx"
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: export_to_docx(full_text, docx_path, lang=context.user_data.get("ocr_lang","en"))
            )
            await context.bot.send_chat_action(chat_id=query.message.chat_id, action=ChatAction.UPLOAD_DOCUMENT)
            with open(docx_path, "rb") as f:
                m = await query.message.reply_document(document=f, filename=f"{doc_id}.docx", caption="📃  Word file (.docx) — open in Microsoft Word!")
            _track_msg(context, m.message_id)
            docx_path.unlink(missing_ok=True)
        except Exception as exc:
            m = await query.message.reply_html(format_error(str(exc))); _track_msg(context, m.message_id)
        return

    if data.startswith("to_hw:"):
        doc_id    = data.split(":", 1)[1]
        full_text = _resolve_text(doc_id, context)
        if not full_text.strip():
            m = await query.message.reply_text("❌  Text not found. Please send the image again."); _track_msg(context, m.message_id); return
        converting = await query.message.reply_html("✍️  <b>Converting to handwriting...</b>")
        _track_msg(context, converting.message_id)
        try:
            hw = await asyncio.get_event_loop().run_in_executor(None, lambda: get_renderer(context).render_a4_pages(full_text))
            await converting.delete()
            await _send_hw_result(query.message, context, hw, f"hw_{doc_id}.png")
        except Exception as exc:
            await converting.delete()
            m = await query.message.reply_html(format_error(str(exc))); _track_msg(context, m.message_id)
        return

    if ":" in data and data.split(":")[0] in ("json", "txt"):
        kind, doc_id = data.split(":", 1)
        files = (context.bot_data.get("files") or {}).get(doc_id)
        if not files:
            m = await query.message.reply_text("❌  File not found. Please process the image again."); _track_msg(context, m.message_id); return
        path = Path(files.get(kind, ""))
        if not path.exists():
            m = await query.message.reply_text("❌  File expired. Please process the image again."); _track_msg(context, m.message_id); return
        await context.bot.send_chat_action(chat_id=query.message.chat_id, action=ChatAction.UPLOAD_DOCUMENT)
        captions = {"json": "📄  JSON file (structured data)", "txt": "📝  Text file (.txt)"}
        with open(path, "rb") as f:
            m = await query.message.reply_document(document=f, filename=f"{doc_id}.{kind}", caption=captions.get(kind, ""))
        _track_msg(context, m.message_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Progress animation
# ─────────────────────────────────────────────────────────────────────────────

STEPS = [
    "🔍  Looking at your image...",
    "🧠  Thinking really hard...",
    "👀  Reading every word...",
    "✍️  Writing down what I see...",
    "✨  Almost done...",
    "🎉  Finishing up!",
]


async def _animate_progress(progress_msg, stop_event: asyncio.Event) -> None:
    total = len(STEPS)
    step  = 0
    while not stop_event.is_set():
        label  = STEPS[step % len(STEPS)]
        filled = min(step + 1, total - 1)
        bar    = "█" * round((filled / total) * 10) + "░" * (10 - round((filled / total) * 10))
        pct    = round((filled / total) * 100)
        try:
            await progress_msg.edit_text(
                f"⏳  <b>Working on it...</b>\n\n  {bar}  {pct}%\n\n  {label}",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
        step += 1
        try:
            await asyncio.wait_for(asyncio.shield(stop_event.wait()), timeout=4.0)
        except asyncio.TimeoutError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Send handwriting result
# ─────────────────────────────────────────────────────────────────────────────

async def _send_hw_result(msg, context, hw_result_or_list, filename: str) -> None:
    if isinstance(hw_result_or_list, list):
        hw = hw_result_or_list[0]
    else:
        hw = hw_result_or_list

    stem      = Path(filename).stem
    jpg_out   = TMP_DIR / f"{stem}.jpg"
    png_out   = TMP_DIR / f"{stem}.png"
    pdf_out   = TMP_DIR / f"{stem}.pdf"

    hw.image.save(str(jpg_out), "JPEG", quality=93, optimize=True)
    hw.save(png_out)

    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.UPLOAD_PHOTO)

    with open(jpg_out, "rb") as f:
        m = await msg.reply_photo(
            photo=f,
            caption=(
                f"🎨  <b>Here's your handwriting!</b>\n\n"
                f"  ✍️  Lines: <code>{hw.line_count}</code>  •  "
                f"🔤 Characters: <code>{hw.char_count:,}</code>"
            ),
            parse_mode=ParseMode.HTML,
        )
    _track_msg(context, m.message_id)

    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.UPLOAD_DOCUMENT)
    try:
        await asyncio.get_event_loop().run_in_executor(None, lambda: export_to_pdf(hw, pdf_out))
        with open(pdf_out, "rb") as f:
            m = await msg.reply_document(document=f, filename="handwriting.pdf", caption="📎  PDF file — ready to print!")
        _track_msg(context, m.message_id)
        pdf_out.unlink(missing_ok=True)
    except Exception:
        with open(png_out, "rb") as f:
            m = await msg.reply_document(document=f, filename="handwriting.png", caption="📎  Image file")
        _track_msg(context, m.message_id)

    jpg_out.unlink(missing_ok=True)
    png_out.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Image handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_user_msg(context, update.message.message_id)
    mode = context.user_data.get("mode")
    if not mode:
        m = await update.message.reply_html(
            "👋  First, tell me what you want to do!\n\n"
            "👇  Tap one of these buttons:",
            reply_markup=_main_menu(context),
        )
        _track_msg(context, m.message_id); return

    photo = update.message.photo[-1]
    if mode == MODE_IMG_HW:
        await _process_img_to_hw(update, context, file_id=photo.file_id)
    else:
        await _process_ocr(update, context, file_id=photo.file_id)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_user_msg(context, update.message.message_id)
    doc  = update.message.document
    mime = doc.mime_type or ""

    if mime == "application/pdf":
        await _process_pdf(update, context, file_id=doc.file_id); return

    if mime.startswith("image/"):
        mode = context.user_data.get("mode")
        if not mode:
            m = await update.message.reply_html(
                "👋  First, tell me what you want to do!\n\n👇  Tap one of these:",
                reply_markup=_main_menu(context),
            )
            _track_msg(context, m.message_id); return
        if mode == MODE_IMG_HW:
            await _process_img_to_hw(update, context, file_id=doc.file_id)
        else:
            await _process_ocr(update, context, file_id=doc.file_id)
        return

    m = await update.message.reply_html(
        "❓  I can read:\n  📸  Photos & images\n  📄  PDF files\n\nPlease send one of those!"
    )
    _track_msg(context, m.message_id)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_user_msg(context, update.message.message_id)
    voice = update.message.voice or update.message.audio
    if not voice:
        m = await update.message.reply_text("❓  No audio found."); _track_msg(context, m.message_id); return

    proc = await update.message.reply_html("🎤  <b>Listening to your voice...</b>")
    _track_msg(context, proc.message_id)
    try:
        tg_file  = await context.bot.get_file(voice.file_id)
        tmp_path = TMP_DIR / f"{voice.file_id}.ogg"
        await tg_file.download_to_drive(str(tmp_path))
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, lambda: transcribe_voice(tmp_path))
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        await proc.delete()
        m = await update.message.reply_html(format_error(str(exc))); _track_msg(context, m.message_id); return

    await proc.edit_text(
        f"🎤  <b>I heard you!</b>\n\n<i>{text[:200]}{'...' if len(text)>200 else ''}</i>\n\n✍️  Now converting to handwriting...",
        parse_mode=ParseMode.HTML,
    )
    try:
        hw_pages = await asyncio.get_event_loop().run_in_executor(
            None, lambda: get_renderer(context).render_a4_pages(text)
        )
        await proc.delete()
        await _send_hw_result(update.message, context, hw_pages, f"voice_{update.effective_user.id}.png")
        m = await update.message.reply_html("✅  Done!", reply_markup=_main_menu(context))
        _track_msg(context, m.message_id)
    except Exception as exc:
        await proc.delete()
        m = await update.message.reply_html(format_error(str(exc))); _track_msg(context, m.message_id)


async def _process_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE, file_id: str) -> None:
    msg  = update.message
    mode = context.user_data.get("mode", MODE_OCR)
    proc = await msg.reply_html("📄  <b>Opening your PDF...</b>")
    _track_msg(context, proc.message_id)
    try:
        tg_file  = await context.bot.get_file(file_id)
        pdf_path = TMP_DIR / f"{file_id}.pdf"
        await tg_file.download_to_drive(str(pdf_path))
        images = await asyncio.get_event_loop().run_in_executor(None, lambda: pdf_to_images(pdf_path))
        pdf_path.unlink(missing_ok=True)
    except Exception as exc:
        await proc.delete()
        m = await msg.reply_html(format_error(f"Could not open PDF: {exc}")); _track_msg(context, m.message_id); return

    await proc.edit_text(f"📄  <b>PDF has {len(images)} pages</b>\n\n🔍  Reading all pages...", parse_mode=ParseMode.HTML)
    lang = context.user_data.get("ocr_lang", "en")
    parts = []
    for i, img in enumerate(images, 1):
        try:
            text, _ = await asyncio.get_event_loop().run_in_executor(None, lambda im=img: extract_text_gemini(im, lang_hint=lang))
            parts.append(f"[Page {i}]\n{text}")
        except Exception:
            parts.append(f"[Page {i} — could not read]")

    full_text = "\n\n".join(parts)
    await proc.delete()

    import uuid as _u
    doc_id = f"PDF-{_u.uuid4().hex[:8].upper()}"
    get_history().save(str(update.effective_user.id), doc_id, full_text, lang)

    if mode == MODE_IMG_HW:
        hw = await asyncio.get_event_loop().run_in_executor(None, lambda: get_renderer(context).render_a4_pages(full_text))
        await _send_hw_result(msg, context, hw, f"pdf_{doc_id}.png")
    else:
        preview = full_text[:1500] + ("..." if len(full_text) > 1500 else "")
        m = await msg.reply_html(
            f"✅  <b>PDF Read!</b>\n\n"
            f"📄  Pages: <code>{len(images)}</code>  •  Words: <code>{len(full_text.split())}</code>\n\n"
            f"<pre>{preview}</pre>",
            reply_markup=_what_next_menu(doc_id),
        )
        _track_msg(context, m.message_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Mode A — OCR (Read Handwriting)
# ─────────────────────────────────────────────────────────────────────────────

async def _process_ocr(update: Update, context: ContextTypes.DEFAULT_TYPE, file_id: str) -> None:
    msg  = update.message
    user = update.effective_user

    try:
        tg_file  = await context.bot.get_file(file_id)
        tmp_path = TMP_DIR / f"{file_id}.jpg"
        await tg_file.download_to_drive(str(tmp_path))
    except Exception as exc:
        m = await msg.reply_html(format_error(f"Could not download: {exc}")); _track_msg(context, m.message_id); return

    # Enhance image
    try:
        enhanced = await asyncio.get_event_loop().run_in_executor(None, lambda: enhance_for_ocr(tmp_path))
        enh_path = TMP_DIR / f"{file_id}_enhanced.jpg"
        cv2.imwrite(str(enh_path), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        tmp_path.unlink(missing_ok=True)
        tmp_path = enh_path
    except Exception as exc:
        logger.warning(f"Enhancement skipped: {exc}")

    prog = await msg.reply_html("⏳  <b>Working on it...</b>\n\n  ░░░░░░░░░░  0%\n\n  🔍  Looking at your image...")
    _track_msg(context, prog.message_id)
    stop  = asyncio.Event()
    anim  = asyncio.create_task(_animate_progress(prog, stop))

    try:
        loop = asyncio.get_event_loop()
        lang = context.user_data.get("ocr_lang", "auto")
        full_text, conf = await loop.run_in_executor(None, lambda: extract_text_gemini(tmp_path, lang_hint=lang))

        import uuid as _uuid
        _ft     = full_text
        _conf   = conf
        _doc_id = f"DOC-{_uuid.uuid4().hex[:8].upper()}"
        _wc     = len(_ft.split())
        _lines  = [type('L', (), {'text': l, 'confidence': _conf, 'needs_review': False, 'bbox': [0,0,0,0]})()
                   for l in _ft.split('\n') if l.strip()]

        class _Result:
            document_id = _doc_id
            status = "processed"
            full_text = _ft
            overall_confidence = _conf
            word_count = _wc
            requires_human_review = False
            low_confidence_lines = []
            barcode = None
            barcode_confidence = 0.0
            image_quality = 0.97
            processing_time_ms = 0
            skew_angle = 0.0
            error = None
            lines = _lines
            paragraphs = []
            def to_dict(self):
                return {"document_id": self.document_id, "full_text": self.full_text,
                        "word_count": self.word_count, "overall_confidence": self.overall_confidence}

        result = _Result()
    except Exception as exc:
        stop.set(); anim.cancel()
        await prog.delete()
        m = await msg.reply_html(format_error(str(exc))); _track_msg(context, m.message_id)
        tmp_path.unlink(missing_ok=True); return
    finally:
        stop.set(); anim.cancel()

    await prog.delete()
    tmp_path.unlink(missing_ok=True)

    # ── Show result ───────────────────────────────────────────────────────────
    lang_label = SUPPORTED_LANGUAGES.get(lang, lang)
    m = await msg.reply_html(
        f"✅  <b>Done! I read your document!</b>\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  🌐  Language  : <b>{lang_label}</b>\n"
        f"  📝  Words     : <code>{result.word_count}</code>\n"
        f"  🎯  Accuracy  : <code>{result.overall_confidence:.0%}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"<pre>{result.full_text[:2000]}{'...' if len(result.full_text)>2000 else ''}</pre>\n\n"
        f"👇  <b>What do you want to do with this?</b>",
        reply_markup=_what_next_menu(result.document_id),
    )
    _track_msg(context, m.message_id)

    # ── Auto-send files ───────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    for path, ext, caption in [
        (TMP_DIR / f"{result.document_id}.md",  "md",  "📝  Text file (.md)"),
        (TMP_DIR / f"{result.document_id}_text.pdf", "pdf", "📄  PDF file"),
    ]:
        try:
            if ext == "md":
                await loop.run_in_executor(None, lambda p=path: export_to_md(result.full_text, p))
            else:
                await loop.run_in_executor(None, lambda p=path: export_text_to_pdf(result.full_text, p))
            with open(path, "rb") as f:
                m = await msg.reply_document(document=f, filename=f"document.{ext}", caption=caption)
            _track_msg(context, m.message_id)
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"{ext} export failed: {exc}")

    # ── Auto summary ──────────────────────────────────────────────────────────
    try:
        summary = await loop.run_in_executor(None, lambda: summarize_text(result.full_text, lang_hint=lang))
        m = await msg.reply_html(f"🤖  <b>Quick Summary</b>\n\n{summary}")
        _track_msg(context, m.message_id)
    except Exception as exc:
        logger.warning(f"Summary failed: {exc}")

    # ── Cache & save ──────────────────────────────────────────────────────────
    if context.bot_data.get("files") is None:
        context.bot_data["files"] = {}
    context.bot_data["files"][result.document_id] = {
        "json": str(build_json_file(result, TMP_DIR)),
        "txt":  str(build_txt_file(result, TMP_DIR)),
        "full_text": result.full_text,
    }
    get_history().save(str(user.id), result.document_id, result.full_text, lang)
    _user_stats[user.id]["processed"] += 1
    _user_stats[user.id]["words"]     += result.word_count


# ─────────────────────────────────────────────────────────────────────────────
#  Mode B — Text → Handwriting
# ─────────────────────────────────────────────────────────────────────────────

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _track_user_msg(context, update.message.message_id)
    mode = context.user_data.get("mode")

    if not mode or mode != MODE_HTR:
        m = await update.message.reply_html(
            "👋  Want me to convert your text to handwriting?\n\n"
            "Tap ✍️ <b>Type → Handwriting</b> first!",
            reply_markup=_main_menu(context),
        )
        _track_msg(context, m.message_id); return

    text = update.message.text.strip()
    if len(text) > 10000:
        m = await update.message.reply_html(f"⚠️  Too long! Max 10,000 characters. Yours: {len(text):,}"); _track_msg(context, m.message_id); return

    proc = await update.message.reply_html(f"✍️  <b>Writing it by hand...</b>\n\n📝  {len(text):,} characters")
    _track_msg(context, proc.message_id)
    try:
        hw_pages = await asyncio.get_event_loop().run_in_executor(None, lambda: get_renderer(context).render_a4_pages(text))
        await proc.delete()
        await _send_hw_result(update.message, context, hw_pages, f"hw_{update.effective_user.id}.png")
        m = await update.message.reply_html("✅  <b>Done!</b>  What's next?", reply_markup=_main_menu(context))
        _track_msg(context, m.message_id)
    except Exception as exc:
        await proc.delete()
        m = await update.message.reply_html(format_error(str(exc))); _track_msg(context, m.message_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Mode C — Image → Handwriting
# ─────────────────────────────────────────────────────────────────────────────

async def _process_img_to_hw(update: Update, context: ContextTypes.DEFAULT_TYPE, file_id: str) -> None:
    msg  = update.message
    user = update.effective_user

    try:
        tg_file  = await context.bot.get_file(file_id)
        tmp_path = TMP_DIR / f"{file_id}.jpg"
        await tg_file.download_to_drive(str(tmp_path))
    except Exception as exc:
        m = await msg.reply_html(format_error(f"Download failed: {exc}")); _track_msg(context, m.message_id); return

    # Enhance
    try:
        enhanced = await asyncio.get_event_loop().run_in_executor(None, lambda: enhance_for_ocr(tmp_path))
        enh_path = TMP_DIR / f"{file_id}_enh.jpg"
        cv2.imwrite(str(enh_path), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        tmp_path.unlink(missing_ok=True)
        tmp_path = enh_path
    except Exception as exc:
        logger.warning(f"Enhancement skipped: {exc}")

    prog = await msg.reply_html("⏳  <b>Step 1 of 2 — Reading your image...</b>")
    _track_msg(context, prog.message_id)
    stop = asyncio.Event()
    anim = asyncio.create_task(_animate_progress(prog, stop))

    try:
        loop = asyncio.get_event_loop()
        lang = context.user_data.get("ocr_lang", "")
        if not lang:
            lang = await loop.run_in_executor(None, lambda: detect_language(tmp_path))
            context.user_data["ocr_lang"] = lang
        full_text, conf = await loop.run_in_executor(None, lambda: extract_text_gemini(tmp_path, lang_hint=lang))
    except Exception as exc:
        stop.set(); anim.cancel()
        await prog.delete()
        m = await msg.reply_html(format_error(str(exc))); _track_msg(context, m.message_id)
        tmp_path.unlink(missing_ok=True); return
    finally:
        stop.set(); anim.cancel()

    tmp_path.unlink(missing_ok=True)
    if not full_text.strip():
        await prog.delete()
        m = await msg.reply_html("⚠️  I couldn't find any text in this image. Try a clearer photo!"); _track_msg(context, m.message_id); return

    await prog.edit_text(
        f"⏳  <b>Step 2 of 2 — Writing by hand...</b>\n\n  ██████████  Reading done ✓\n\n  📝  {len(full_text.split())} words found",
        parse_mode=ParseMode.HTML,
    )

    try:
        hw_pages = await asyncio.get_event_loop().run_in_executor(None, lambda: get_renderer(context).render_a4_pages(full_text))
        await prog.delete()
        await _send_hw_result(msg, context, hw_pages, f"hw_img_{user.id}.png")
        m = await msg.reply_html("✅  <b>Done!</b>  What's next?", reply_markup=_main_menu(context))
        _track_msg(context, m.message_id)
    except Exception as exc:
        await prog.delete()
        m = await msg.reply_html(format_error(str(exc))); _track_msg(context, m.message_id)

    _user_stats[user.id]["processed"] += 1
    _user_stats[user.id]["words"]     += len(full_text.split())


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback
# ─────────────────────────────────────────────────────────────────────────────

async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        _track_user_msg(context, update.message.message_id)
        m = await update.message.reply_html(
            "🤔  I'm not sure what to do with that!\n\n👇  Pick what you want:",
            reply_markup=_main_menu(context),
        )
        _track_msg(context, m.message_id)
