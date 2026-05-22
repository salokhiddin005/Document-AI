"""
CLI entry point.

Usage examples:
  # Process a single image and print results
  python main.py process path/to/form.jpg

  # Process and save results to JSON
  python main.py process path/to/form.jpg --output results.json

  # Start the REST API server
  python main.py serve

  # Show pending review queue items
  python main.py review list

  # Export corrections for retraining
  python main.py review export --out data/feedback/corrections.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
#  Clean JSON serialiser
#  Produces indented JSON but keeps short lists (bbox, text lines) on one line.
# ─────────────────────────────────────────────────────────────────────────────

def _to_clean_json(obj: dict) -> str:
    """
    Serialize *obj* to a human-friendly JSON string:
      - Indented with 2 spaces for readability
      - bbox arrays stay on a single line: [26, 0, 664, 51]
      - Short string arrays (text lines) stay on a single line
      - Numbers rounded: no scientific notation, max 2 decimal places
    """
    raw = json.dumps(obj, ensure_ascii=False, indent=2)

    # Collapse bbox arrays: [  \n    26,\n    0,\n    664,\n    51\n  ] → [26, 0, 664, 51]
    raw = re.sub(
        r'\[\s*(\-?\d+),\s*(\-?\d+),\s*(\-?\d+),\s*(\-?\d+)\s*\]',
        r'[\1, \2, \3, \4]',
        raw,
    )

    # Collapse short string arrays (text lines inside content/paragraphs)
    # Matches ["line one", "line two"] style arrays of strings
    def collapse_string_array(m: re.Match) -> str:
        inner = m.group(1)
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', inner)
        if sum(len(s) for s in items) > 120:
            return m.group(0)   # too long — leave expanded
        joined = ", ".join(f'"{s}"' for s in items)
        return f'[{joined}]'

    raw = re.sub(
        r'\[\s*("(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*)\s*\]',
        collapse_string_array,
        raw,
    )

    return raw

# ─────────────────────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.add(log_dir / "document_ai.log", rotation="10 MB", retention="30 days", level="DEBUG")


# ─────────────────────────────────────────────────────────────────────────────
#  Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_process(args):
    from src.pipeline.document_pipeline import DocumentPipeline

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"File not found: {image_path}")
        sys.exit(1)

    logger.info(f"Processing: {image_path}")
    pipeline = DocumentPipeline()
    result   = pipeline.process(image_path, document_id=args.doc_id)
    output   = result.to_dict()

    # ── Pretty-print results ──────────────────────────────────────────────────
    W = 65
    print("\n" + "═" * W)
    print(f"  Document ID  : {result.document_id}")
    print(f"  Status       : {result.status}")
    print(f"  Image quality: {result.image_quality:.2f}/1.00")
    print(f"  Confidence   : {result.overall_confidence:.2f}/1.00")
    print(f"  Words found  : {result.word_count}")
    print(f"  Lines found  : {len(result.lines)}")
    print(f"  Paragraphs   : {len(result.paragraphs)}")
    print(f"  Time         : {result.processing_time_ms} ms")
    if result.barcode:
        print(f"  Barcode      : {result.barcode}  (conf={result.barcode_confidence:.2f})")
    print("═" * W)

    # Full extracted text
    print("\n  EXTRACTED TEXT")
    print("  " + "─" * (W - 2))
    for para in result.paragraphs:
        for line in para.text.split("\n"):
            print(f"  {line}")
        print()

    # Line-by-line confidence breakdown
    print("  LINE CONFIDENCE BREAKDOWN")
    print("  " + "─" * (W - 2))
    for i, line in enumerate(result.lines, 1):
        icon = "!" if line.needs_review else "✓"
        # Truncate long lines for display
        display = line.text if len(line.text) <= 50 else line.text[:47] + "..."
        print(f"  [{icon}] L{i:02d}  conf={line.confidence:.2f}  {display}")

    # Review summary
    print("\n  REVIEW SUMMARY")
    print("  " + "─" * (W - 2))
    if result.low_confidence_lines:
        print(f"  ⚠  {len(result.low_confidence_lines)} line(s) sent to human review queue")
        for line in result.low_confidence_lines:
            display = line.text if len(line.text) <= 45 else line.text[:42] + "..."
            print(f"     conf={line.confidence:.2f}  → '{display}'")
    else:
        print(f"  ✓  All lines accepted automatically")
    print("═" * W + "\n")

    # ── Save JSON output ──────────────────────────────────────────────────────
    # Always saves automatically. --output lets you choose the path.
    # Default: outputs/<document_id>.json
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{result.document_id}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(_to_clean_json(output))
        print(f"  JSON saved → {out_path.resolve()}\n")
        logger.info(f"Results saved to: {out_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")


def cmd_serve(args):
    import uvicorn
    from config import settings
    logger.info(f"Starting API server on {settings.api.host}:{settings.api.port}")
    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_review_list(args):
    from src.review.queue_manager import ReviewQueueManager
    queue = ReviewQueueManager()
    items = queue.fetch_pending(limit=args.limit)

    if not items:
        print("\n  Review queue is empty.\n")
        return

    print(f"\n  Pending review items ({len(items)}):")
    print("  " + "─" * 70)
    for item in items:
        print(
            f"  [{item.id[:8]}] doc={item.document_id} "
            f"field={item.field_name:<18} "
            f"value={item.system_value!r:<25} "
            f"conf={item.ocr_confidence:.2f} "
            f"reason={item.routing_reason}"
        )
    print()


def cmd_review_export(args):
    from src.review.queue_manager import ReviewQueueManager
    queue = ReviewQueueManager()
    count = queue.export_corrections(Path(args.out))
    logger.info(f"Exported {count} corrections to {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-ai",
        description="Custom OCR & Document AI System for Handwritten Records",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # process
    p_proc = sub.add_parser("process", help="Process a document image")
    p_proc.add_argument("image", help="Path to image file")
    p_proc.add_argument("--doc-id", default=None, help="Optional document ID")
    p_proc.add_argument("--output", "-o", default=None, help="Save JSON output to file")

    # serve
    p_serve = sub.add_parser("serve", help="Start the REST API server")
    p_serve.add_argument("--reload", action="store_true", help="Enable hot-reload (dev mode)")

    # review
    p_rev = sub.add_parser("review", help="Review queue management")
    rev_sub = p_rev.add_subparsers(dest="review_command", required=True)

    r_list = rev_sub.add_parser("list", help="List pending review items")
    r_list.add_argument("--limit", type=int, default=20, help="Max items to show")

    r_export = rev_sub.add_parser("export", help="Export corrections for retraining")
    r_export.add_argument("--out", default="data/feedback/corrections.jsonl")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "review":
        if args.review_command == "list":
            cmd_review_list(args)
        elif args.review_command == "export":
            cmd_review_export(args)


if __name__ == "__main__":
    main()
