"""
Telegram Bot — Document AI.

Run:
    python telegram_bot/bot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from loguru import logger
from telegram import BotCommand
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    PicklePersistence,   # ← persists chat_data across restarts
    filters,
)

from telegram_bot.handlers import (
    cmd_clear,
    cmd_drive,
    cmd_help,
    cmd_history,
    cmd_history_get,
    cmd_lang,
    cmd_library,
    cmd_mode,
    cmd_mystats,
    cmd_start,
    cmd_status,
    cmd_style,
    handle_callback,
    handle_document,
    handle_photo,
    handle_text,
    handle_unknown,
    handle_voice,
)

load_dotenv(ROOT / ".env")   # works locally; on Railway uses env vars directly
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
PROXY_URL = os.getenv("PROXY_URL", "").strip() or None
IS_CLOUD  = os.getenv("RAILWAY_ENVIRONMENT") is not None   # True when on Railway

if not BOT_TOKEN:
    logger.error("BOT_TOKEN not set! Add it in Railway → Variables tab.")
    sys.exit(1)

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(log_dir / "telegram_bot.log", rotation="10 MB", retention="30 days", level="DEBUG")

PERSISTENCE_FILE = ROOT / "data" / "bot_persistence.pkl"
PERSISTENCE_FILE.parent.mkdir(parents=True, exist_ok=True)


async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start",    "Main menu — all modes"),
        BotCommand("mode",     "Switch conversion mode"),
        BotCommand("library",  "Browse all your documents"),
        BotCommand("history",  "Recent OCR results"),
        BotCommand("drive",    "Google Drive setup & status"),
        BotCommand("lang",     "Change OCR language"),
        BotCommand("style",    "Handwriting style"),
        BotCommand("clear",    "Clear chat"),
        BotCommand("mystats",  "Your usage statistics"),
        BotCommand("help",     "How to use the bot"),
    ])
    info = await application.bot.get_me()
    logger.info(f"Bot ready: @{info.username} ({info.full_name})")


def main() -> None:
    logger.info("Starting Document AI Telegram Bot...")

    # PicklePersistence saves chat_data to disk
    # → message IDs tracked by /clear survive bot restarts
    persistence = PicklePersistence(filepath=str(PERSISTENCE_FILE))

    builder = (
        Application.builder()
        .token(BOT_TOKEN)
        .persistence(persistence)
        .post_init(post_init)
        .connect_timeout(30)
        .read_timeout(60)
        .write_timeout(60)
        .pool_timeout(30)
    )
    if PROXY_URL:
        builder = builder.proxy(PROXY_URL)

    app = builder.build()

    # ── Commands ──────────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("mode",        cmd_mode))
    app.add_handler(CommandHandler("lang",        cmd_lang))
    app.add_handler(CommandHandler("style",       cmd_style))
    app.add_handler(CommandHandler("history",     cmd_history))
    app.add_handler(CommandHandler("history_get", cmd_history_get))
    app.add_handler(CommandHandler("library",     cmd_library))
    app.add_handler(CommandHandler("drive",       cmd_drive))
    app.add_handler(CommandHandler("clear",       cmd_clear))
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("status",      cmd_status))
    app.add_handler(CommandHandler("mystats",     cmd_mystats))

    # ── Image / document / voice ──────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.PHOTO,                    handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL,             handle_document))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO,   handle_voice))

    # ── Text ──────────────────────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,  handle_text))

    # ── Callbacks ────────────────────────────────────────────────────────────
    app.add_handler(CallbackQueryHandler(handle_callback))

    # ── Fallback ─────────────────────────────────────────────────────────────
    app.add_handler(MessageHandler(filters.ALL, handle_unknown))

    logger.info("Bot running. Press Ctrl+C to stop.")
    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
