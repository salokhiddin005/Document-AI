# Document AI

A production-grade document intelligence platform for extracting text from scanned and handwritten documents. Supports 20+ languages, barcode detection, confidence-based routing to human review, and multiple interfaces: CLI, REST API, Telegram bot, and admin dashboard.

## Features

- **Handwriting OCR** — extracts text from scanned or photographed documents without predefined field structure
- **Multi-language** — 20+ languages with automatic detection (English, Russian, Uzbek, Korean, Arabic, Chinese, and more)
- **Barcode decoding** — detects and decodes barcodes embedded in documents
- **Confidence routing** — auto-accepts high-confidence results, routes uncertain ones to human review
- **Text-to-handwriting** — converts plain text to realistic handwritten PDF output
- **Voice transcription** — audio-to-text via Google Gemini API
- **Text summarization** — summarizes extracted content via Gemini
- **Continuous learning** — human corrections exported as JSONL for model fine-tuning

## Interfaces

| Interface | Description |
|-----------|-------------|
| CLI | Single-document processing and review queue management |
| REST API | HTTP endpoints for programmatic access (FastAPI, port 8000) |
| Telegram Bot | User-friendly document upload and results via Telegram |
| Admin Dashboard | Web UI for managing the human review queue (port 8001) |

## Tech Stack

- **Python 3.12**
- **PaddleOCR** — handwriting and multi-language text recognition
- **OpenCV / Pillow / NumPy** — image preprocessing
- **FastAPI + Uvicorn** — REST API
- **python-telegram-bot** — Telegram integration
- **SQLAlchemy + SQLite** — review queue persistence
- **fpdf2 / PyMuPDF** — PDF generation and reading
- **Google Gemini API** — voice transcription and summarization
- **pyzbar** — barcode decoding
- **Docker + Railway** — cloud deployment

## Project Structure

```
Document-AI/
├── main.py                    # CLI entry point
├── config.py                  # Centralized configuration (reads configs/settings.yaml)
├── configs/settings.yaml      # All thresholds and tuning parameters
├── src/
│   ├── pipeline/              # Main document processing orchestrator
│   ├── preprocessing/         # Image loading, deskewing, denoising, contrast
│   ├── ocr/                   # Text extraction, layout assembly, barcode reading
│   ├── confidence/            # Scoring engine and routing decisions
│   ├── review/                # Human review queue (SQLite-backed)
│   └── text_to_handwriting/   # Text-to-handwriting renderer and PDF export
├── api/
│   ├── main.py                # FastAPI REST endpoints
│   └── dashboard.py           # Admin web dashboard
├── telegram_bot/
│   ├── bot.py                 # Bot initialization
│   └── handlers.py            # Message/photo/voice handlers
├── tests/                     # Unit tests
├── Dockerfile                 # Production container image
├── Procfile                   # Railway worker declaration
└── requirements.txt           # Full local dependencies
```

## Installation

**Prerequisites:** Python 3.12, pip

```bash
git clone <repo-url>
cd Document-AI

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```env
BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_google_gemini_api_key
```

## Usage

### CLI

Process a single document:

```bash
python main.py process path/to/image.jpg
python main.py process path/to/image.jpg --output results.json
```

Manage the human review queue:

```bash
python main.py review list
python main.py review export --output corrections.jsonl
```

### REST API

Start the API server:

```bash
python main.py serve
# or with hot-reload for development:
python main.py serve --reload
```

Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/process` | Upload image, receive extracted JSON |
| GET | `/review/queue` | List pending review items |
| POST | `/review/{id}/correct` | Submit a human correction |
| GET | `/review/export` | Export corrections for retraining |

### Telegram Bot

```bash
python telegram_bot/bot.py
```

Send photos, documents, or voice messages to the bot. It returns extracted text with confidence scores and can summarize or export results.

### Admin Dashboard

```bash
python api/dashboard.py
```

Opens a web UI at `http://localhost:8001` for reviewing and correcting low-confidence extractions.

## Confidence Routing

Every extraction is scored and routed to one of three outcomes:

| Decision | Threshold | Action |
|----------|-----------|--------|
| `AUTO_ACCEPT` | ≥ 0.85 | Returned immediately |
| `SPOT_CHECK` | 0.70 – 0.85 | Flagged for optional review |
| `HUMAN_REVIEW` | < 0.70 | Queued for mandatory correction |

Thresholds are configurable in `configs/settings.yaml`.

## Docker

Build and run the container:

```bash
docker build -t document-ai .
docker run -e BOT_TOKEN=<token> -e GEMINI_API_KEY=<key> document-ai
```

## Deployment (Railway)

The project is pre-configured for Railway:

- `Dockerfile` — Python 3.12 slim image with all system dependencies
- `Procfile` — declares the Telegram bot as the worker process
- `railway.json` — auto-restart on failure (max 10 retries)
- `requirements_cloud.txt` — cloud-optimized dependencies (headless OpenCV, pinned versions)

Set `BOT_TOKEN` and `GEMINI_API_KEY` as Railway environment variables before deploying.

## Configuration

All runtime parameters live in `configs/settings.yaml`. Key sections:

- **preprocessing** — DPI requirement (300), max skew angle (45°), denoising strength
- **ocr** — language list, device (cpu/gpu), batch size
- **confidence** — auto-accept and spot-check thresholds
- **review** — queue database path, image crop storage
- **api** — host, port, CORS settings

## Testing

```bash
pytest tests/
```

Test modules cover confidence scoring, image preprocessing, barcode detection, and review queue persistence.
