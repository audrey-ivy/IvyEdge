"""
IvyEdge Video Generator

Produces a branded TikTok/Reels MP4 from the TikTok script in 06_social.md.

Pipeline:
  1. Parse [SPOKEN] lines from the script → full narration text
  2. Call ElevenLabs TTS → narration.mp3
  3. Load ivy background video (assets/ivy_background.mp4), loop to match audio
  4. Generate text overlay cards for each [TEXT:] cue using Pillow
  5. Composite: background + text cards + audio → output MP4

Required in .env:
  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE_ID=...   (optional — defaults to a warm female voice)

Required asset:
  assets/ivy_background.mp4  (royalty-free ivy/leaves loop — see README)
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ivyedge.video")

ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
# Default: "Rachel" — calm, warm, professional female voice
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL    = "eleven_multilingual_v2"

BACKGROUND_VIDEO    = Path(__file__).parent / "assets" / "ivy_background.mp4"
ASSETS_DIR          = Path(__file__).parent / "assets"

# Brand colors
FOREST_GREEN = (28, 99, 80)
CORAL_PINK   = (255, 123, 156)
WHITE        = (255, 255, 255)
NEAR_BLACK   = (0, 5, 1)
MINT         = (156, 227, 208)

VIDEO_W, VIDEO_H = 1080, 1920   # 9:16 vertical for TikTok/Reels


# ---------------------------------------------------------------------------
# Script parser
# ---------------------------------------------------------------------------

def parse_tiktok_script(social_md: str) -> dict:
    """
    Extract spoken lines, on-screen text cues, and section timings
    from the TikTok/Reels script in 06_social.md.

    Returns:
        {
            "spoken": "Full narration text for TTS",
            "text_cues": [{"text": "...", "section": "HOOK"}, ...],
            "sections": ["HOOK", "PROBLEM", "INSIGHT 1", ...],
        }
    """
    # Find the TikTok section
    tiktok_match = re.search(
        r"##\s*TikTok.*?Script\s*\n(.*?)(?=\n##\s*Production notes|\Z)",
        social_md, re.DOTALL | re.IGNORECASE
    )
    if not tiktok_match:
        # Try broader match
        tiktok_match = re.search(
            r"###\s*Script\s*\n(.*?)(?=\n###|\Z)",
            social_md, re.DOTALL
        )

    script_text = tiktok_match.group(1) if tiktok_match else social_md

    spoken_lines: list[str] = []
    text_cues:    list[dict] = []
    current_section = "INTRO"

    for line in script_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Section header e.g. [HOOK - 0:00-0:03]
        sec_match = re.match(r'\[([A-Z][A-Z\s\d]+?)[\s\-].*?\]', line)
        if sec_match:
            current_section = sec_match.group(1).strip()
            continue

        # On-screen text cue e.g. [TEXT: Your income is real.]
        text_match = re.match(r'\[TEXT:\s*(.+?)\]', line, re.IGNORECASE)
        if text_match:
            text_cues.append({"text": text_match.group(1).strip(), "section": current_section})
            continue

        # Visual cue — skip
        if re.match(r'\[VISUAL:', line, re.IGNORECASE):
            continue

        # Spoken dialogue — any remaining non-empty, non-bracket line
        if not line.startswith("["):
            spoken_lines.append(line)

    spoken = " ".join(spoken_lines).strip()
    return {
        "spoken": spoken,
        "text_cues": text_cues,
        "sections": list(dict.fromkeys(c["section"] for c in text_cues)),
    }


# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------

def generate_voiceover(text: str, output_path: Path) -> Path:
    """Call ElevenLabs and save the audio to output_path."""
    if not ELEVENLABS_API_KEY:
        raise ValueError(
            "ELEVENLABS_API_KEY not set in .env. "
            "Sign up at elevenlabs.io and add your key."
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.80,
            "style": 0.20,
            "use_speaker_boost": True,
        },
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)
    logger.info("Voiceover saved: %s (%d bytes)", output_path, len(resp.content))
    return output_path


# ---------------------------------------------------------------------------
# Text overlay frame generator
# ---------------------------------------------------------------------------

FONT_DIR = Path(__file__).parent / "assets" / "fonts"


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if path.exists():
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            pass
    return ImageFont.load_default()


def _make_text_overlay(text: str, section: str, size: tuple[int, int]) -> Image.Image:
    """Return a transparent RGBA image with styled text overlay."""
    img  = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    w, h = size
    pad  = 60

    # Semi-transparent backing panel
    panel_h = 220
    panel_y = h - panel_h - 120
    draw.rectangle(
        [(0, panel_y), (w, panel_y + panel_h)],
        fill=(28, 99, 80, 200),   # Forest green, 78% opacity
    )

    # Coral accent bar at top of panel
    draw.rectangle([(0, panel_y), (w, panel_y + 6)], fill=(*CORAL_PINK, 255))

    # Section label
    label_font = _load_font("DMSans.ttf", 28)
    draw.text((pad, panel_y + 18), section.upper(), font=label_font, fill=(*CORAL_PINK, 255))

    # Main text
    text_font  = _load_font("Fraunces.ttf", 52)
    # Word-wrap
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=text_font)
        if bbox[2] > w - 2 * pad and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    text_y = panel_y + 58
    for i, line in enumerate(lines[:3]):
        draw.text((pad, text_y + i * 62), line, font=text_font, fill=(*WHITE, 255))

    # IvyEdge watermark bottom right
    wm_font = _load_font("DMSans.ttf", 26)
    wm_text = "IvyEdge"
    bbox    = draw.textbbox((0, 0), wm_text, font=wm_font)
    draw.text(
        (w - bbox[2] - pad, h - 56),
        wm_text, font=wm_font, fill=(*WHITE, 180)
    )

    return img


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def generate_video(
    social_md_path: Path,
    output_path: Path,
    title: str = "",
) -> Path:
    """
    Generate a branded TikTok/Reels MP4 from the social copy file.

    Args:
        social_md_path: Path to 06_social.md
        output_path:    Where to save the final MP4
        title:          Post title (for logging)

    Returns:
        Path to the generated MP4
    """
    try:
        from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, ColorClip
    except ImportError:
        raise ImportError("moviepy not installed. Run: pip install moviepy")

    if not BACKGROUND_VIDEO.exists():
        raise FileNotFoundError(
            f"Ivy background video not found: {BACKGROUND_VIDEO}\n"
            "Download a royalty-free ivy/leaves video from pexels.com and save it there.\n"
            "Search: https://www.pexels.com/search/videos/ivy%20leaves/"
        )

    social_text = social_md_path.read_text(encoding="utf-8")
    parsed      = parse_tiktok_script(social_text)

    if not parsed["spoken"]:
        raise ValueError("No spoken dialogue found in TikTok script.")

    logger.info("Generating voiceover (%d chars)...", len(parsed["spoken"]))
    with tempfile.TemporaryDirectory() as tmp:
        audio_path  = Path(tmp) / "narration.mp3"
        generate_voiceover(parsed["spoken"], audio_path)

        audio_clip  = AudioFileClip(str(audio_path))
        duration    = audio_clip.duration

        # ── Background (looped ivy video, cropped to 9:16) ──────────────
        bg = VideoFileClip(str(BACKGROUND_VIDEO), audio=False)
        # Loop to match audio duration
        if bg.duration < duration:
            loops = int(duration / bg.duration) + 1
            from moviepy import concatenate_videoclips
            bg = concatenate_videoclips([bg] * loops)
        bg = bg.subclipped(0, duration)

        # Resize/crop to 1080x1920
        bg_w, bg_h = bg.size
        scale = max(VIDEO_W / bg_w, VIDEO_H / bg_h)
        new_w, new_h = int(bg_w * scale), int(bg_h * scale)
        bg = bg.resized((new_w, new_h))
        x_off = (new_w - VIDEO_W) // 2
        y_off = (new_h - VIDEO_H) // 2
        bg = bg.cropped(x1=x_off, y1=y_off, x2=x_off + VIDEO_W, y2=y_off + VIDEO_H)

        # ── Text overlays ────────────────────────────────────────────────
        overlay_clips = []
        n_cues   = len(parsed["text_cues"])
        seg_dur  = duration / max(n_cues, 1)

        for i, cue in enumerate(parsed["text_cues"]):
            overlay_img  = _make_text_overlay(cue["text"], cue["section"], (VIDEO_W, VIDEO_H))
            arr = __import__("numpy").array(overlay_img)
            clip = (
                ImageClip(arr, duration=seg_dur)
                .with_start(i * seg_dur)
                .with_effects([__import__("moviepy").video.fx.CrossFadeIn(0.3)])
            )
            overlay_clips.append(clip)

        # ── Composite ────────────────────────────────────────────────────
        final = CompositeVideoClip([bg] + overlay_clips).with_audio(audio_clip)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(Path(tmp) / "temp_audio.m4a"),
            remove_temp=True,
            logger=None,
        )

    logger.info("Video saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    social_md = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not social_md or not social_md.exists():
        print("Usage: python video_generator.py path/to/06_social.md")
        sys.exit(1)
    out = social_md.parent / "07_video.mp4"
    generate_video(social_md, out)
    print(f"Video saved: {out}")
