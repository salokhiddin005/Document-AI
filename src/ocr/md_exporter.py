"""
Markdown (.md) Exporter — saves OCR text as clean Markdown file.
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger


def export_to_md(
    text: str,
    output_path: Path,
    title: str = "",
) -> Path:
    """Export OCR text as a Markdown (.md) file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if title:
        lines.append(f"# {title}\n")

    # Convert double newlines to paragraph breaks
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        cleaned = para.replace("\n", " ").strip()
        if cleaned:
            lines.append(cleaned + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Markdown saved: {output_path}")
    return output_path


def export_text_to_pdf(text: str, output_path: Path, title: str = "") -> Path:
    """Export OCR text as a clean readable PDF."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(25, 20, 25)

    # Title
    if title:
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, title, align="C")
        pdf.ln(6)

    # Body text
    pdf.set_font("Helvetica", size=11)

    paragraphs = text.split("\n\n")
    for para in paragraphs:
        cleaned = para.replace("\n", " ").strip()
        if cleaned:
            pdf.multi_cell(0, 6, cleaned)
            pdf.ln(4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    logger.info(f"Text PDF saved: {output_path}")
    return output_path
