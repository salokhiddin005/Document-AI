"""
Q&A Engine - answer questions about document content using Gemini.
"""

from __future__ import annotations

import os
import time
from loguru import logger


def answer_question(document_text: str, question: str) -> str:
    """Answer a question about the document using Gemini."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")

    if not document_text.strip():
        return "No document text available to answer from."

    prompt = (
        f"You are a helpful assistant. Answer the following question based ONLY "
        f"on the document text provided below. If the answer is not in the document, "
        f"say 'This information is not in the document.'\n\n"
        f"DOCUMENT:\n{document_text[:6000]}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    from google import genai
    client = genai.Client(api_key=api_key)
    models = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]

    for model in models:
        try:
            logger.info(f"Q&A using {model}")
            response = client.models.generate_content(model=model, contents=[prompt])
            return response.text.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(15)
                continue
            raise

    raise RuntimeError("Q&A failed — all models unavailable")
