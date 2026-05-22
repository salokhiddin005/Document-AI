"""
FastAPI application — HTTP interface to the document pipeline.

Endpoints:
  POST /process              Upload an image → get extracted JSON
  GET  /review/queue         List pending review items
  GET  /review/queue/count   Count pending items
  POST /review/{id}/correct  Submit a human correction
  POST /review/export        Export corrections as JSONL for retraining
  GET  /health               Health check
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from src.pipeline.document_pipeline import DocumentPipeline, DocumentResult
from src.review.queue_manager import CorrectionSubmission, QueueRecord, ReviewQueueManager


# ─────────────────────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document AI — OCR & Extraction API",
    description=(
        "Extracts structured data from handwritten documents. "
        "Supports Korean, English, and barcode decoding. "
        "Routes low-confidence results to human review."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared instances (initialised once at startup)
_pipeline: Optional[DocumentPipeline] = None
_queue:    Optional[ReviewQueueManager] = None


@app.on_event("startup")
async def startup():
    global _pipeline, _queue
    logger.info("Initialising pipeline components…")
    _pipeline = DocumentPipeline()
    _queue    = _pipeline.queue
    logger.info("API ready")


def get_pipeline() -> DocumentPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return _pipeline

def get_queue() -> ReviewQueueManager:
    if _queue is None:
        raise HTTPException(status_code=503, detail="Queue not initialised")
    return _queue


# ─────────────────────────────────────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class CorrectionRequest(BaseModel):
    corrected_value: str = Field(..., description="The correct field value as verified by the reviewer")
    reviewer_id: str     = Field(..., description="Reviewer identifier (username or employee ID)")
    escalate: bool       = Field(False, description="True if the field cannot be read at all")


class ProcessResponse(BaseModel):
    document_id: str
    status: str
    full_text: str
    word_count: int
    overall_confidence: float
    requires_human_review: bool
    barcode: Optional[str]
    processing_time_ms: int
    image_quality: float


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pending_review_items": get_queue().pending_count()}


@app.post("/process", response_model=ProcessResponse, summary="Process a document image")
async def process_document(
    file: UploadFile = File(..., description="Scanned/photographed document (JPEG, PNG, PDF)"),
    document_id: Optional[str] = Form(None, description="Optional document ID; auto-generated if omitted"),
):
    """
    Upload a scanned or photographed handwritten document.
    Returns extracted field values, confidence scores, and routing decisions.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/tiff", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(pil_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    try:
        result: DocumentResult = get_pipeline().process(image_array, document_id=document_id)
    except Exception as exc:
        logger.exception(f"Pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    return JSONResponse(content=result.to_dict())


@app.get("/review/queue", summary="List pending review items")
async def list_review_queue(limit: int = 50) -> List[dict]:
    """Returns the oldest *limit* pending review items."""
    items = get_queue().fetch_pending(limit=limit)
    return [
        {
            "id":             item.id,
            "document_id":    item.document_id,
            "field_name":     item.field_name,
            "system_value":   item.system_value,
            "ocr_confidence": item.ocr_confidence,
            "routing_reason": item.routing_reason,
            "crop_path":      item.crop_path,
            "status":         item.status,
            "created_at":     item.created_at,
        }
        for item in items
    ]


@app.get("/review/queue/count", summary="Count pending review items")
async def review_count():
    return {"pending": get_queue().pending_count()}


@app.post("/review/{item_id}/correct", summary="Submit a human correction")
async def submit_correction(item_id: str, body: CorrectionRequest):
    """Record a reviewer's verified correction for a queued field."""
    submission = CorrectionSubmission(
        item_id=item_id,
        corrected_value=body.corrected_value,
        reviewer_id=body.reviewer_id,
        escalate=body.escalate,
    )
    success = get_queue().submit_correction(submission)
    if not success:
        raise HTTPException(status_code=404, detail=f"Review item not found: {item_id}")
    return {"status": "ok", "item_id": item_id}


@app.post("/review/export", summary="Export corrections for model retraining")
async def export_corrections(output_filename: str = "corrections_export.jsonl"):
    """Export all corrected items as a JSONL file for fine-tuning."""
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w"
    ) as tmp:
        tmp_path = Path(tmp.name)

    count = get_queue().export_corrections(tmp_path)
    return FileResponse(
        path=tmp_path,
        filename=output_filename,
        media_type="application/json",
        headers={"X-Record-Count": str(count)},
    )


@app.get("/review/{document_id}/items", summary="Get all review items for a document")
async def document_review_items(document_id: str):
    items = get_queue().fetch_by_document(document_id)
    return [
        {
            "id":             i.id,
            "field_name":     i.field_name,
            "system_value":   i.system_value,
            "ocr_confidence": i.ocr_confidence,
            "status":         i.status,
        }
        for i in items
    ]
