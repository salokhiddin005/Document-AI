"""
Text → Handwriting renderer.

Style: writebyhand.in reference
  - ONE page always — font shrinks to fit all content
  - Proper cursive handwriting font (Kalam → Dancing Script → Segoe Print fallback)
  - Clean white paper with very subtle ruled lines
  - Date + Page label in top-right corner
  - Dark blue cursive ink
"""

from __future__ import annotations

import random
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from loguru import logger
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ─────────────────────────────────────────────────────────────────────────────
#  Font management — download cursive fonts on first use
# ─────────────────────────────────────────────────────────────────────────────

FONTS_DIR = Path(__file__).parent.parent.parent / "fonts"

_DOWNLOAD_FONTS = [
    ("Kalam-Regular.ttf",
     "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf"),
    ("Kalam-Bold.ttf",
     "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Bold.ttf"),
]

_WINDOWS_FALLBACKS = [
    Path("C:/Windows/Fonts/Inkfree.ttf"),
    Path("C:/Windows/Fonts/segoepr.ttf"),
    Path("C:/Windows/Fonts/comic.ttf"),
]


def _ensure_fonts() -> None:
    """Download cursive fonts if not already present."""
    FONTS_DIR.mkdir(exist_ok=True)
    for fname, url in _DOWNLOAD_FONTS:
        path = FONTS_DIR / fname
        if not path.exists():
            try:
                logger.info(f"Downloading font: {fname}")
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                path.write_bytes(r.content)
                logger.info(f"Font saved: {path}")
            except Exception as exc:
                logger.warning(f"Font download failed ({fname}): {exc}")


def _load_font(size: int) -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    _ensure_fonts()

    # Priority: Kalam (natural cursive look) → Windows fallbacks
    candidates = [
        FONTS_DIR / "Kalam-Regular.ttf",
        Path("C:/Windows/Fonts/Inkfree.ttf"),
        Path("C:/Windows/Fonts/segoepr.ttf"),
        Path("C:/Windows/Fonts/comic.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]

    bold_candidates = [
        FONTS_DIR / "Kalam-Bold.ttf",
        Path("C:/Windows/Fonts/segoeprb.ttf"),
        Path("C:/Windows/Fonts/comicbd.ttf"),
    ]

    regular = None
    for p in candidates:
        if p.exists():
            try:
                regular = ImageFont.truetype(str(p), size)
                logger.info(f"Font: {p.name} @ {size}px")
                break
            except Exception:
                continue

    bold = None
    for p in bold_candidates:
        if p.exists():
            try:
                bold = ImageFont.truetype(str(p), size)
                break
            except Exception:
                continue

    if regular is None:
        regular = ImageFont.load_default()
        logger.warning("Using PIL default font")

    return regular, (bold or regular)


# ─────────────────────────────────────────────────────────────────────────────
#  Style / Color / Paper presets
# ─────────────────────────────────────────────────────────────────────────────

HANDWRITING_STYLES = {
    "neat":    {"font_size": 42, "jitter": 0.08, "angle": 0.15, "line_sp": 1.80},
    "casual":  {"font_size": 44, "jitter": 0.20, "angle": 0.35, "line_sp": 1.85},
    "student": {"font_size": 46, "jitter": 0.28, "angle": 0.45, "line_sp": 1.90},
    "doctor":  {"font_size": 38, "jitter": 0.40, "angle": 0.60, "line_sp": 1.70},
}

INK_COLORS = {
    "blue":   (25,  35, 130),
    "black":  (20,  20,  20),
    "red":    (160, 20,  20),
    "green":  (20, 100,  40),
    "purple": (80,  20, 120),
}

PAPER_STYLES = {
    "lined":    "lined",
    "blank":    "blank",
    "grid":     "grid",
    "notebook": "notebook",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Result
# ─────────────────────────────────────────────────────────────────────────────

class HandwritingResult:
    def __init__(self, image: Image.Image, line_count: int, char_count: int):
        self.image      = image
        self.line_count = line_count
        self.char_count = char_count

    def to_numpy(self) -> np.ndarray:
        return np.array(self.image)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(str(path), "PNG", dpi=(150, 150))


# ─────────────────────────────────────────────────────────────────────────────
#  Renderer
# ─────────────────────────────────────────────────────────────────────────────

class HandwritingRenderer:
    """
    A4 handwriting renderer with style, ink color, and paper type support.
    """

    A4_W       = 1240
    A4_H       = 1754
    MARGIN_X   = 90
    MARGIN_TOP = 100
    MARGIN_BOT = 80
    SHADOW     = 5
    PARA_EXTRA = 0.55
    INDENT_SPACES = 0

    FONT_SIZES   = [46, 42, 38, 36]
    MIN_FONT_PX  = 36

    def __init__(
        self,
        style:       str = "casual",
        ink_color:   str = "blue",
        paper_style: str = "lined",
    ):
        preset        = HANDWRITING_STYLES.get(style, HANDWRITING_STYLES["casual"])
        self.LINE_SP  = preset["line_sp"]
        self.JITTER   = preset["jitter"]
        self._angle_max = preset["angle"]
        base_size     = preset["font_size"]
        self.FONT_SIZES = [base_size, base_size - 4, base_size - 8,
                           max(36, base_size - 10)]
        self.INK         = INK_COLORS.get(ink_color, INK_COLORS["blue"])
        self.paper_style = PAPER_STYLES.get(paper_style, "lined")

    # ── Public ────────────────────────────────────────────────────────────────

    def render_a4_single(self, text: str, title: Optional[str] = None) -> HandwritingResult:
        """
        Render all text onto exactly ONE A4 page.
        Finds the largest font where everything fits.
        """
        text_w = self.A4_W - 2 * self.MARGIN_X
        text_h = self.A4_H - self.MARGIN_TOP - self.MARGIN_BOT

        chosen_font       = None
        chosen_bold       = None
        chosen_font_size  = self.FONT_SIZES[-1]
        chosen_lines      = []
        chosen_lh         = 20

        for font_size in self.FONT_SIZES:
            font, bold = _load_font(font_size)
            lh         = int(font_size * self.LINE_SP)
            para_gap   = int(lh * self.PARA_EXTRA)

            # Pixel-based wrapping — accurate for any variable-width font
            segments = self._parse(text, title, font, text_w)
            needed   = self._measure_height(segments, lh, para_gap)

            if needed <= text_h:
                chosen_font      = font
                chosen_bold      = bold
                chosen_font_size = font_size
                chosen_lines     = segments
                chosen_lh        = lh
                break

        # Fallback: use smallest font even if overflowing
        if chosen_font is None:
            font_size              = self.FONT_SIZES[-1]
            chosen_font, chosen_bold = _load_font(font_size)
            chosen_font_size       = font_size
            chosen_lh              = int(font_size * self.LINE_SP)
            chosen_lines           = self._parse(text, title, chosen_font, text_w)

        para_gap = int(chosen_lh * self.PARA_EXTRA)

        logger.info(
            f"Render: font={chosen_font_size}px | "
            f"lines={sum(1 for s in chosen_lines if not s[2])} | "
            f"size={self.A4_W + self.SHADOW*2}×{self.A4_H + self.SHADOW*2}"
        )

        return self._render_page(
            chosen_lines, chosen_font, chosen_bold,
            chosen_font_size, chosen_lh, para_gap,
        )

    # Aliases for backward compatibility
    def render(self, text, title=None):               return self.render_a4_single(text, title)
    def render_pages(self, text, title=None):         return [self.render_a4_single(text, title)]
    def render_single_page(self, text, title=None):   return self.render_a4_single(text, title)
    def render_a4_pages(self, text, title=None):      return [self.render_a4_single(text, title)]

    # ── Page builder ──────────────────────────────────────────────────────────

    def _draw_justified(
        self,
        draw:     ImageDraw.Draw,
        img:      Image.Image,
        text:     str,
        font:     ImageFont.FreeTypeFont,
        x:        int,
        y:        int,
        max_px:   int,
        angle:    float,
        colour:   Tuple[int, int, int],
        is_last:  bool,
        rng:      random.Random,
    ) -> None:
        """
        Draw one line of text with full justification:
        - Non-last lines: stretch word spacing to fill max_px
        - Last line of paragraph: left-aligned
        """
        words = text.split()
        if not words:
            return

        # Last line of paragraph or single word → left-align, no stretch
        if is_last or len(words) == 1:
            self._draw_line(draw, img, text, font, x, y, angle, colour)
            return

        dd         = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        word_widths = [dd.textlength(w, font=font) for w in words]
        total_text  = sum(word_widths)
        total_gap   = max_px - total_text
        gap_per     = total_gap / (len(words) - 1)

        curr_x = x
        for i, (word, ww) in enumerate(zip(words, word_widths)):
            # Small per-word jitter (natural handwriting variation)
            j = int(rng.gauss(0, 1.5))
            self._draw_line(draw, img, word, font, int(curr_x), y + j, angle, colour)
            curr_x += ww + gap_per

    def _render_page(
        self,
        segments: list,
        font:      ImageFont.FreeTypeFont,
        font_bold: ImageFont.FreeTypeFont,
        font_size: int,
        lh:        int,
        para_gap:  int,
    ) -> HandwritingResult:

        shadow   = self.SHADOW
        canvas_w = self.A4_W + shadow * 2
        canvas_h = self.A4_H + shadow * 2

        # Grey desktop background + drop shadow
        canvas = Image.new("RGB", (canvas_w, canvas_h), (168, 168, 168))
        canvas.paste(
            Image.new("RGB", (self.A4_W, self.A4_H), (138, 138, 138)),
            (shadow * 2, shadow * 2),
        )

        # Paper background — depends on paper_style
        paper = self._make_paper_bg()
        draw  = ImageDraw.Draw(paper)
        self._draw_paper_lines(draw, lh)

        # No date/page label — clean page

        # Text — justified with indentation and centered title
        rng        = random.Random()
        y          = self.MARGIN_TOP
        char_count = 0
        line_count = 0
        text_w     = self.A4_W - 2 * self.MARGIN_X

        # Indent width
        dd         = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        indent_px  = int(dd.textlength("m" * self.INDENT_SPACES, font=font))

        # A line is "last in paragraph" when its immediate next segment is a
        # paragraph break (or it's the final segment). These lines are NOT justified.
        last_flags: list[bool] = []
        for i, seg in enumerate(segments):
            if seg[2]:   # is_break placeholder
                last_flags.append(False)
                continue
            next_seg = segments[i + 1] if i + 1 < len(segments) else None
            last_flags.append(next_seg is None or next_seg[2])

        for seg, is_last in zip(segments, last_flags):
            seg_text, is_title, is_break, is_first = seg[0], seg[1], seg[2], seg[3]

            if is_break:
                y += para_gap
                continue
            if y + lh > self.A4_H - self.MARGIN_BOT:
                break   # stop at page boundary — excess text not rendered

            f      = font_bold if is_title else font
            jitter = int(rng.gauss(0, font_size * self.JITTER * 0.1))
            angle  = rng.gauss(0, 0.25)
            colour = self._vary_ink(rng)

            if is_title:
                # Center the title
                tw = dd.textlength(seg_text, font=f)
                cx = self.MARGIN_X + (text_w - tw) // 2
                self._draw_line(draw, paper, seg_text, f,
                                int(cx), y + jitter, angle, colour)
            elif is_first:
                # Indented first line of paragraph — left-aligned (not justified)
                self._draw_line(draw, paper, seg_text, f,
                                self.MARGIN_X + indent_px, y + jitter, angle, colour)
            else:
                # Justified body line
                self._draw_justified(
                    draw, paper, seg_text, f,
                    self.MARGIN_X, y + jitter,
                    text_w, angle, colour,
                    is_last=is_last, rng=rng,
                )

            char_count += len(seg_text)
            line_count += 1
            y += lh

        paper = paper.filter(ImageFilter.GaussianBlur(radius=0.4))
        canvas.paste(paper, (shadow, shadow))

        return HandwritingResult(canvas, line_count=line_count, char_count=char_count)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pixel_wrap(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_px: int,
    ) -> List[str]:
        """
        Wrap a paragraph into lines that fit within max_px.
        Measures actual rendered pixel width — not character count.
        This ensures text reaches the right margin for any font.
        """
        dd    = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        words = text.split()
        lines = []
        current: List[str] = []

        for word in words:
            candidate = " ".join(current + [word])
            if dd.textlength(candidate, font=font) <= max_px:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                    current = [word]
                else:
                    # Single word wider than max — force it on its own line
                    lines.append(word)
                    current = []

        if current:
            lines.append(" ".join(current))

        return lines if lines else [text]

    INDENT_SPACES = 0   # no indent — text starts at left margin

    def _parse(
        self,
        text: str,
        title: Optional[str],
        font: ImageFont.FreeTypeFont,
        max_px: int,
    ) -> list[tuple[str, bool, bool, bool]]:
        """
        Returns [(text, is_title, is_para_break, is_first_line)].
        Uses word-based packing — character slicing is never used.
        First line of each paragraph: indented (shorter available width).
        All other lines: full width.
        """
        dd        = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        indent_px = int(dd.textlength("m" * self.INDENT_SPACES, font=font))

        segs = []
        if title:
            segs.append((title, True, False, False))
            segs.append(("", False, True, False))

        # Split by double-newline for real paragraph breaks.
        # Single newlines (OCR line breaks) are treated as spaces within a paragraph.
        paragraphs = text.split("\n\n")

        for raw_para in paragraphs:
            # Collapse inner single newlines → one continuous paragraph text
            para = " ".join(raw_para.replace("\n", " ").split())
            if not para:
                continue

            words = para.split()
            if not words:
                continue

            # ── First line (indented) ──────────────────────────────────────
            first_words: List[str] = []
            for word in words:
                candidate = " ".join(first_words + [word])
                if dd.textlength(candidate, font=font) <= max_px - indent_px:
                    first_words.append(word)
                else:
                    break

            if not first_words:
                first_words = [words[0]]

            segs.append((" ".join(first_words), False, False, True))
            remaining_words = words[len(first_words):]

            # ── Remaining lines (full width, no indent) ────────────────────
            current: List[str] = []
            for word in remaining_words:
                candidate = " ".join(current + [word])
                if dd.textlength(candidate, font=font) <= max_px:
                    current.append(word)
                else:
                    if current:
                        segs.append((" ".join(current), False, False, False))
                    current = [word]
            if current:
                segs.append((" ".join(current), False, False, False))

            segs.append(("", False, True, False))   # paragraph break

        while segs and segs[-1][2]:
            segs.pop()
        return segs

    def _measure_height(
        self, segments: list, lh: int, para_gap: int
    ) -> int:
        h = 0
        for seg in segments:
            is_break = seg[2]
            h += para_gap if is_break else lh
        return h

    def _draw_line(
        self,
        draw:   ImageDraw.Draw,
        img:    Image.Image,
        text:   str,
        font:   ImageFont.FreeTypeFont,
        x: int, y: int,
        angle:  float,
        colour: Tuple[int, int, int],
    ) -> None:
        if not text:
            return
        if abs(angle) < 0.15:
            draw.text((x, y), text, font=font, fill=colour)
            return
        bbox    = draw.textbbox((0, 0), text, font=font)
        tw, th  = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20
        layer   = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        ld      = ImageDraw.Draw(layer)
        ld.text((8, 6), text, font=font, fill=(*colour, 255))
        rotated = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        img.paste(rotated, (x - 8, y - 6), mask=rotated.split()[3])

    def _make_paper_bg(self) -> Image.Image:
        style = getattr(self, "paper_style", "lined")
        if style == "notebook":
            bg = (255, 252, 220)   # yellow-ish notebook
        elif style == "grid" or style == "blank":
            bg = (254, 254, 254)   # near-white
        else:
            bg = (253, 253, 251)   # warm white (lined/default)
        arr   = np.full((self.A4_H, self.A4_W, 3), bg, dtype=np.int16)
        noise = np.random.RandomState(0).randint(-2, 3, arr.shape, dtype=np.int16)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    def _draw_paper_lines(self, draw: ImageDraw.Draw, lh: int) -> None:
        style = getattr(self, "paper_style", "lined")
        if style == "blank":
            return

        if style in ("lined", "notebook"):
            col   = (210, 215, 235) if style == "lined" else (200, 200, 160)
            y     = self.MARGIN_TOP + lh
            while y < self.A4_H - self.MARGIN_BOT:
                draw.line([(self.MARGIN_X - 5, y),
                           (self.A4_W - self.MARGIN_X + 5, y)],
                          fill=col, width=1)
                y += lh
            if style == "notebook":
                # Red margin line
                draw.line([(self.MARGIN_X - 24, 0),
                           (self.MARGIN_X - 24, self.A4_H)],
                          fill=(200, 80, 80), width=2)

        elif style == "grid":
            col  = (210, 215, 235)
            step = lh // 2
            # Horizontal
            y = self.MARGIN_TOP
            while y < self.A4_H - self.MARGIN_BOT:
                draw.line([(0, y), (self.A4_W, y)], fill=col, width=1)
                y += step
            # Vertical
            x = self.MARGIN_X
            while x < self.A4_W - self.MARGIN_X:
                draw.line([(x, 0), (x, self.A4_H)], fill=col, width=1)
                x += step

    def _vary_ink(self, rng: random.Random) -> Tuple[int, int, int]:
        r, g, b = self.INK
        v = rng.randint(-10, 11)
        return (max(0, min(255, r + v)),
                max(0, min(255, g + v)),
                max(0, min(255, b + v)))
