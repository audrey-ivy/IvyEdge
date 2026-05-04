"""
IvyEdge Image Card Generator

Generates 1080x1080 branded static image cards for Instagram and Threads.
Uses IvyEdge brand colors and typography from brand_voice.md.
Downloads required fonts on first run to assets/fonts/.
"""

from __future__ import annotations

import logging
import textwrap
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("ivyedge.image_card")

# ---------------------------------------------------------------------------
# Brand colors
# ---------------------------------------------------------------------------

FOREST_GREEN  = "#1c6350"
SAGE_GREEN    = "#95c590"
MINT          = "#9ce3d0"
NEAR_BLACK    = "#000501"
SILVER        = "#bebbbb"
CORAL_PINK    = "#ff7b9c"
DEEP_PLUM     = "#62466b"
WHITE         = "#ffffff"


def _hex(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore


# ---------------------------------------------------------------------------
# Fonts — downloaded once to assets/fonts/
# ---------------------------------------------------------------------------

FONT_DIR = Path(__file__).parent / "assets" / "fonts"

FONT_URLS = {
    # Variable fonts from google/fonts — one file covers all weights
    "Fraunces.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/fraunces/"
        "Fraunces%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf"
    ),
    "DMSans.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/"
        "DMSans%5Bopsz%2Cwght%5D.ttf"
    ),
    "DMMono-Regular.ttf": (
        "https://raw.githubusercontent.com/google/fonts/main/ofl/dmmono/"
        "DMMono-Regular.ttf"
    ),
}


def _ensure_fonts() -> dict[str, Path]:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in FONT_URLS.items():
        dest = FONT_DIR / name
        if not dest.exists():
            logger.info("Downloading font: %s", name)
            try:
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(url, context=ctx) as resp:
                    dest.write_bytes(resp.read())
            except Exception as e:
                logger.warning("Could not download %s: %s — will use fallback", name, e)
        paths[name] = dest
    return paths


def _load_font(paths: dict[str, Path], name: str, size: int) -> ImageFont.FreeTypeFont:
    path = paths.get(name)
    if path and path.exists():
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Pillar tag labels
# ---------------------------------------------------------------------------

PILLAR_TAGS = {
    "Pillar 1": "FINANCIAL EDUCATION",
    "Pillar 2": "MONEY DEMYSTIFIED",
    "Pillar 4": "BEHAVIORAL FINANCE",
    "Pillar 5": "INDUSTRY INSIGHT",
    "Brand Story": "OUR STORY",
}


def _pillar_tag(pillar: str) -> str:
    for key, label in PILLAR_TAGS.items():
        if key in pillar:
            return label
    return "FINANCIAL EDUCATION"


# ---------------------------------------------------------------------------
# Text wrap helper
# ---------------------------------------------------------------------------

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
               max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Card generator
# ---------------------------------------------------------------------------

SIZE = 1080
PADDING = 72
CONTENT_W = SIZE - 2 * PADDING


def generate_card(
    title: str,
    pillar: str,
    pull_quote: str = "",
    output_path: Optional[Path] = None,
    dark: bool = True,
) -> Path:
    """
    Generate a 1080x1080 branded image card.

    Args:
        title:       Post headline — shown large in Fraunces Bold.
        pillar:      Content pillar — determines the section tag label.
        pull_quote:  Short stat or hook line shown below the title.
        output_path: Where to save the PNG. Defaults to a temp path.
        dark:        True = Forest Green bg (default). False = white bg.

    Returns:
        Path to the generated PNG.
    """
    fonts = _ensure_fonts()

    bg_color      = _hex(FOREST_GREEN) if dark else (255, 255, 255)
    title_color   = _hex(WHITE)        if dark else _hex(FOREST_GREEN)
    tag_color     = _hex(CORAL_PINK)
    quote_color   = _hex(MINT)         if dark else _hex(NEAR_BLACK)
    footer_color  = _hex(SILVER)
    accent_color  = _hex(SAGE_GREEN)

    img  = Image.new("RGB", (SIZE, SIZE), color=bg_color)
    draw = ImageDraw.Draw(img)

    # ── Coral accent bar (top) ──────────────────────────────────────────
    draw.rectangle([(0, 0), (SIZE, 10)], fill=_hex(CORAL_PINK))

    # ── Section tag ─────────────────────────────────────────────────────
    tag_font = _load_font(fonts, "DMSans.ttf", 26)
    tag_text = _pillar_tag(pillar)
    draw.text((PADDING, 44), tag_text, font=tag_font, fill=tag_color)

    # ── Divider ─────────────────────────────────────────────────────────
    draw.rectangle([(PADDING, 88), (SIZE - PADDING, 91)], fill=accent_color)

    # ── Title (Fraunces Bold, large) ─────────────────────────────────────
    title_font_size = 80
    title_font = _load_font(fonts, "Fraunces.ttf", title_font_size)

    title_lines = _wrap_text(draw, title, title_font, CONTENT_W)
    # Shrink if too many lines
    while len(title_lines) > 4 and title_font_size > 52:
        title_font_size -= 6
        title_font  = _load_font(fonts, "Fraunces.ttf", title_font_size)
        title_lines = _wrap_text(draw, title, title_font, CONTENT_W)

    line_h = title_font_size + 16
    title_block_h = len(title_lines) * line_h
    title_y = max(120, (SIZE // 2) - (title_block_h // 2) - 60)

    for i, line in enumerate(title_lines):
        draw.text((PADDING, title_y + i * line_h), line, font=title_font, fill=title_color)

    # ── Pull quote / stat ────────────────────────────────────────────────
    if pull_quote:
        quote_font = _load_font(fonts, "DMSans.ttf", 34)
        quote_lines = _wrap_text(draw, f'"{pull_quote}"', quote_font, CONTENT_W)
        quote_y = title_y + title_block_h + 48
        for i, line in enumerate(quote_lines[:3]):  # max 3 lines
            draw.text((PADDING, quote_y + i * 46), line, font=quote_font, fill=quote_color)

    # ── Footer divider ───────────────────────────────────────────────────
    draw.rectangle([(PADDING, SIZE - 110), (SIZE - PADDING, SIZE - 107)], fill=accent_color)

    # ── IvyEdge wordmark ─────────────────────────────────────────────────
    wordmark_font  = _load_font(fonts, "DMSans.ttf", 36)
    tagline_font   = _load_font(fonts, "DMSans.ttf", 22)
    draw.text((PADDING, SIZE - 96), "IvyEdge",       font=wordmark_font, fill=title_color)
    draw.text((PADDING, SIZE - 52), "Grow through anything.", font=tagline_font, fill=footer_color)

    # ── ivyedge.co (right-aligned) ───────────────────────────────────────
    url_font = _load_font(fonts, "DMSans.ttf", 22)
    url_text = "ivyedge.co"
    url_bbox = draw.textbbox((0, 0), url_text, font=url_font)
    url_w = url_bbox[2] - url_bbox[0]
    draw.text((SIZE - PADDING - url_w, SIZE - 52), url_text, font=url_font, fill=footer_color)

    # ── Save ─────────────────────────────────────────────────────────────
    if output_path is None:
        output_path = Path("/tmp/ivyedge_card.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG")
    logger.info("Image card saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI — quick preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    title = " ".join(sys.argv[1:]) or "Why Your Career Gap Lowered Your Credit Score"
    out = Path("assets/preview_card.png")
    generate_card(
        title=title,
        pillar="Pillar 1: Financial Education for Non-Traditional Paths",
        pull_quote="Women with identical risk profiles receive lower credit scores — even though they default less.",
        output_path=out,
        dark=True,
    )
    print(f"Preview saved: {out}")
