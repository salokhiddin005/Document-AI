"""Q&A Engine — Gemini first, Groq as fallback."""

from __future__ import annotations
import time
from src.ocr.api_key_manager import run_with_fallback


def _gemini_run(api_key: str, prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    for model in ["gemini-2.0-flash-lite", "gemini-2.0-flash"]:
        for attempt in range(2):
            try:
                r = client.models.generate_content(model=model, contents=[prompt])
                return r.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    if attempt == 0: time.sleep(10); continue
                    else: break
                elif "404" in err: break
                else: raise
    raise Exception("429 Gemini rate limited")


def _groq_run(api_key: str, prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return r.choices[0].message.content.strip()


def answer_question(document_text: str, question: str) -> str:
    if not document_text.strip():
        return "No document text available."
    prompt = (
        f"Answer this question based ONLY on the document below. "
        f"Answer in the SAME LANGUAGE as the question. "
        f"If the answer is not in the document, say so.\n\n"
        f"DOCUMENT:\n{document_text[:6000]}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    return run_with_fallback(_gemini_run, _groq_run, prompt)
