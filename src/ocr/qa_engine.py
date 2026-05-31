"""Q&A Engine — uses Groq Llama, rotates API keys on rate limit."""

from __future__ import annotations
from src.ocr.api_key_manager import run_with_key_rotation

TEXT_MODEL = "llama-3.3-70b-versatile"


def _run_with_key(api_key: str, prompt: str) -> str:
    from groq import Groq
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def answer_question(document_text: str, question: str) -> str:
    if not document_text.strip():
        return "No document text available."
    prompt = (
        f"Answer this question based ONLY on the document below. "
        f"If the answer is not in the document, say so clearly.\n\n"
        f"DOCUMENT:\n{document_text[:6000]}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    return run_with_key_rotation(_run_with_key, prompt)
