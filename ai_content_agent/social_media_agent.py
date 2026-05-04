"""
IvyEdge Social Media Agent

Scans the output/ directory for posts that have been generated but not yet
posted to social media. For each one:

  1. Generates a branded 1080x1080 image card (Instagram / Threads)
  2. Generates a TikTok/Reels MP4 (ivy background + ElevenLabs voiceover)
  3. Posts the image + caption to Instagram
  4. Posts the image + text to Threads
  5. Saves a social_posted.json receipt so the post is never double-posted

Called automatically from run_monday.sh after the content pipeline runs.
Can also be run manually:

    python social_media_agent.py                    # process all unpublished
    python social_media_agent.py --folder output/2026-05-06_why-your-career-gap...
    python social_media_agent.py --cards-only       # generate cards, skip posting
    python social_media_agent.py --video-only       # generate videos only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ivyedge.social")

OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Instagram caption parser
# ---------------------------------------------------------------------------

def _parse_instagram_caption(social_md: str) -> str:
    """Extract the Instagram caption from 06_social.md."""
    match = re.search(
        r"###\s*Caption\s*\n(.*?)(?=\n###|\n---|\Z)",
        social_md, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    # Fallback: return first 2200 chars of the file
    return social_md[:2200].strip()


def _parse_threads_post(social_md: str) -> str:
    """Extract the best X/Threads option from 06_social.md."""
    # Take Option 1 (the recommended one)
    match = re.search(
        r"###\s*Option 1\s*\n(.*?)(?=\n###\s*Option|\n---|\Z)",
        social_md, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""


def _parse_pull_quote(social_md: str) -> str:
    """Pull a short stat or hook from the Instagram caption for the image card."""
    caption = _parse_instagram_caption(social_md)
    # Grab the first sentence that looks like a stat or bold claim
    sentences = re.split(r'(?<=[.!?])\s+', caption)
    for s in sentences:
        if any(char.isdigit() for char in s) or len(s) < 120:
            return s.strip().lstrip('"').rstrip('"')
    return sentences[0].strip() if sentences else ""


# ---------------------------------------------------------------------------
# Per-folder processor
# ---------------------------------------------------------------------------

def process_folder(
    folder: Path,
    cards_only: bool = False,
    video_only: bool = False,
    skip_post: bool = False,
) -> dict:
    """Process a single output folder. Returns a result dict."""
    receipt_path = folder / "social_posted.json"
    if receipt_path.exists():
        logger.info("Already posted — skipping: %s", folder.name)
        return {"status": "already_posted", "folder": str(folder)}

    meta_path   = folder / "meta.json"
    social_path = folder / "06_social.md"

    if not social_path.exists():
        logger.warning("No 06_social.md in %s — skipping", folder.name)
        return {"status": "no_social_file"}

    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    title  = meta.get("topic", folder.name)
    pillar = meta.get("pillar", "Pillar 1: Financial Education for Non-Traditional Paths")
    social_text = social_path.read_text(encoding="utf-8")

    result: dict = {
        "folder":     str(folder),
        "title":      title,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "image_card": None,
        "video":      None,
        "instagram":  None,
        "threads":    None,
    }

    # ── 1. Image card ───────────────────────────────────────────────────
    if not video_only:
        try:
            from image_card_generator import generate_card
            pull_quote  = _parse_pull_quote(social_text)
            card_path   = folder / "07_image_card.png"
            generate_card(
                title=title,
                pillar=pillar,
                pull_quote=pull_quote,
                output_path=card_path,
                dark=True,
            )
            result["image_card"] = str(card_path)
            logger.info("Image card: %s", card_path.name)
        except Exception as e:
            logger.error("Image card failed for %s: %s", folder.name, e)

    # ── 2. Video ────────────────────────────────────────────────────────
    if not cards_only:
        try:
            from video_generator import generate_video, BACKGROUND_VIDEO, ELEVENLABS_API_KEY
            if not ELEVENLABS_API_KEY:
                logger.warning("ELEVENLABS_API_KEY not set — skipping video for %s", folder.name)
            elif not BACKGROUND_VIDEO.exists():
                logger.warning("Ivy background video missing — skipping video for %s", folder.name)
            else:
                video_path = folder / "08_video.mp4"
                generate_video(social_path, video_path, title=title)
                result["video"] = str(video_path)
                logger.info("Video: %s", video_path.name)
        except Exception as e:
            logger.error("Video generation failed for %s: %s", folder.name, e)

    # ── 3. Post to Instagram + Threads ──────────────────────────────────
    if not cards_only and not video_only and not skip_post:
        card_path_obj = Path(result["image_card"]) if result["image_card"] else None

        # Instagram
        try:
            from meta_poster import post_to_instagram
            ig_caption  = _parse_instagram_caption(social_text)
            if card_path_obj and card_path_obj.exists():
                ig_url = post_to_instagram(ig_caption, card_path_obj)
                result["instagram"] = ig_url
            else:
                logger.warning("No image card — skipping Instagram post")
        except Exception as e:
            logger.error("Instagram post failed for %s: %s", folder.name, e)

        # Threads
        try:
            from meta_poster import post_to_threads
            threads_text = _parse_threads_post(social_text)
            if threads_text:
                threads_url = post_to_threads(threads_text, card_path_obj)
                result["threads"] = threads_url
            else:
                logger.warning("No Threads text found — skipping")
        except Exception as e:
            logger.error("Threads post failed for %s: %s", folder.name, e)

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["status"]      = "done"

    # ── Save receipt (prevents re-posting) ──────────────────────────────
    receipt_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Receipt saved: %s", receipt_path)
    return result


# ---------------------------------------------------------------------------
# Scan all unpublished output folders
# ---------------------------------------------------------------------------

def process_all(
    output_dir: Path = OUTPUT_DIR,
    cards_only: bool = False,
    video_only: bool = False,
    skip_post: bool = False,
) -> list[dict]:
    folders = sorted(
        [f for f in output_dir.iterdir() if f.is_dir()],
        key=lambda f: f.name,
    )
    if not folders:
        logger.info("No output folders found in %s", output_dir)
        return []

    results = []
    for folder in folders:
        if (folder / "social_posted.json").exists():
            continue  # already done
        if not (folder / "06_social.md").exists():
            continue  # no social copy yet
        logger.info("Processing: %s", folder.name)
        r = process_folder(folder, cards_only=cards_only,
                           video_only=video_only, skip_post=skip_post)
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="IvyEdge social media agent")
    parser.add_argument("--folder", help="Process a single output folder")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Output directory to scan (default: output/)")
    parser.add_argument("--cards-only", action="store_true",
                        help="Generate image cards only — skip video and posting")
    parser.add_argument("--video-only", action="store_true",
                        help="Generate videos only — skip cards and posting")
    parser.add_argument("--no-post", action="store_true",
                        help="Generate assets but do not post to social media")
    args = parser.parse_args(argv)

    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Folder not found: {folder}", file=sys.stderr)
            return 1
        result = process_folder(
            folder,
            cards_only=args.cards_only,
            video_only=args.video_only,
            skip_post=args.no_post,
        )
        print(json.dumps(result, indent=2))
        return 0

    results = process_all(
        output_dir=Path(args.output_dir),
        cards_only=args.cards_only,
        video_only=args.video_only,
        skip_post=args.no_post,
    )

    done    = [r for r in results if r.get("status") == "done"]
    skipped = [r for r in results if r.get("status") == "already_posted"]

    print(f"\nDone: {len(done)} posts processed, {len(skipped)} already posted.")
    for r in done:
        ig  = r.get("instagram") or "—"
        thr = r.get("threads")   or "—"
        vid = "✓" if r.get("video") else "—"
        print(f"  {Path(r['folder']).name}")
        print(f"    Instagram: {ig}")
        print(f"    Threads:   {thr}")
        print(f"    Video:     {vid}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
