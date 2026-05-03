"""
IvyEdge Content Agent — CLI runner

Modes:

    # Founding introduction post (run once)
    python run_pipeline.py intro [--publish]

    # Generate a single post from CLI args
    python run_pipeline.py single \\
        --topic "How freelancers can prove income stability" \\
        --persona Maya \\
        --pillar "Pillar 1: Financial Education for Non-Traditional Paths" \\
        --keywords "freelance income proof,1099 loan approval,freelancer credit" \\
        [--publish]

    # Generate every row in editorial_calendar.csv where status == 'queued'
    python run_pipeline.py batch --calendar editorial_calendar.csv [--publish]

Add --publish to any mode to push directly to Substack after generation.
Requires SUBSTACK_SID in .env.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from ivyedge_content_agent import IvyEdgeContentAgent, GenerationResult
from substack_publisher import SubstackPublisher


import ssl
import nltk
import textstat

def _ensure_nltk_data() -> None:
    """Download cmudict (needed by textstat) on first run, bypassing SSL issues on macOS."""
    try:
        nltk.data.find("corpora/cmudict")
    except LookupError:
        _orig = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.download("cmudict", quiet=True)
        ssl._create_default_https_context = _orig

_ensure_nltk_data()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ivyedge.cli")

DALE_CHALL_TARGET = 8.5

SUBSTACK_FOOTER = """
---

IvyEdge is being built for every woman who has been underestimated by a system that never genuinely evaluated her.

If that's you, we want you close when we launch.

[Get on the IvyEdge waitlist →](https://substack.com/@joinivyedge)

*Be first. You've waited long enough.*
"""


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text[:60] or "post"


def _save_result(result: GenerationResult, out_root: Path) -> tuple[Path, float]:
    """Write all artifacts for a single generation to its own folder."""
    date = datetime.utcnow().strftime("%Y-%m-%d")
    slug = _slugify(result.brief.topic)
    folder = out_root / f"{date}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "00_brief.json").write_text(
        json.dumps(result.brief.__dict__, indent=2), encoding="utf-8"
    )
    if result.format_analysis:
        (folder / "00_format_analysis.md").write_text(result.format_analysis, encoding="utf-8")
    (folder / "01_research.md").write_text(result.research, encoding="utf-8")
    (folder / "02_outline.md").write_text(result.outline, encoding="utf-8")
    (folder / "03_first_draft.md").write_text(result.first_draft, encoding="utf-8")
    (folder / "04_edited_draft.md").write_text(result.edited_draft, encoding="utf-8")
    (folder / "05_final_draft.md").write_text(result.final_draft, encoding="utf-8")
    if result.social:
        (folder / "06_social.md").write_text(result.social, encoding="utf-8")

    # Dale-Chall readability score on the final draft
    plain = re.sub(r"[#*_`\[\]()]", "", result.final_draft)
    dale_chall = round(textstat.dale_chall_readability_score(plain), 2)

    meta = {
        "topic": result.brief.topic,
        "persona": result.brief.persona,
        "pillar": result.brief.pillar,
        "primary_keyword": result.brief.primary_keyword,
        "secondary_keywords": result.brief.secondary_keywords,
        "meta_description": result.meta_description,
        "model": result.model,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "token_usage": result.token_usage,
        "dale_chall_score": dale_chall,
        "dale_chall_target": DALE_CHALL_TARGET,
        "dale_chall_delta": round(dale_chall - DALE_CHALL_TARGET, 2),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    delta = dale_chall - DALE_CHALL_TARGET
    flag = "✓" if abs(delta) <= 0.5 else ("↑ too complex" if delta > 0 else "↓ too simple")
    logger.info("Dale-Chall: %.2f (target %.1f) %s", dale_chall, DALE_CHALL_TARGET, flag)
    logger.info("Saved draft to %s", folder)
    return folder, dale_chall


# ---------------------------------------------------------------------------
# Substack publish helper
# ---------------------------------------------------------------------------

def _maybe_publish(result: GenerationResult, folder: Path, publish: bool) -> None:
    if not publish:
        return
    try:
        publisher = SubstackPublisher()
    except ValueError as e:
        logger.error("Cannot publish: %s", e)
        return

    title = result.brief.topic
    subtitle = result.meta_description or ""
    body = result.final_draft.rstrip() + "\n" + SUBSTACK_FOOTER
    post_url = publisher.publish(title=title, body_markdown=body, subtitle=subtitle)
    print(f"  Published to Substack: {post_url}")
    (folder / "substack_url.txt").write_text(post_url, encoding="utf-8")


# ---------------------------------------------------------------------------
# Intro post mode
# ---------------------------------------------------------------------------

def cmd_intro(args: argparse.Namespace) -> int:
    agent = IvyEdgeContentAgent(model=args.model, context_dir=args.context_dir)
    print("\nGenerating IvyEdge introduction post...\n")

    result = agent.generate_intro_post(
        on_phase=lambda name, _: print(f"  [done] {name}"),
    )

    folder, dc = _save_result(result, Path(args.output))
    print(f"\nDone. Intro post: {folder / '05_final_draft.md'}")
    print(f"      Social copy: {folder / '06_social.md'}")
    print(f"Dale-Chall: {dc} (target {DALE_CHALL_TARGET})")
    _maybe_publish(result, folder, args.publish)
    return 0


# ---------------------------------------------------------------------------
# Single-post mode
# ---------------------------------------------------------------------------

def cmd_single(args: argparse.Namespace) -> int:
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    agent = IvyEdgeContentAgent(model=args.model, context_dir=args.context_dir)

    print(f"\nGenerating post on: {args.topic}\n  persona={args.persona}  pillar={args.pillar}")
    print(f"  keywords={keywords}\n")

    result = agent.generate_blog_post(
        topic=args.topic,
        persona=args.persona,
        pillar=args.pillar,
        keywords=keywords,
        content_format=args.format,
        notes=args.notes,
        on_phase=lambda name, _: print(f"  [done] {name}"),
    )

    folder, dc = _save_result(result, Path(args.output))
    print(f"\nDone. Final draft: {folder / '05_final_draft.md'}")
    print(f"      Social copy:  {folder / '06_social.md'}")
    print(f"Tokens: in={result.token_usage.get('input_tokens', 0)} "
          f"out={result.token_usage.get('output_tokens', 0)}")
    print(f"Dale-Chall: {dc} (target {DALE_CHALL_TARGET})")
    _maybe_publish(result, folder, args.publish)
    return 0


# ---------------------------------------------------------------------------
# Batch mode (reads editorial_calendar.csv)
# ---------------------------------------------------------------------------

REQUIRED_CSV_COLUMNS = [
    "publish_date", "title", "persona", "pillar",
    "primary_keyword", "secondary_keywords", "format", "status",
]


def cmd_batch(args: argparse.Namespace) -> int:
    calendar_path = Path(args.calendar)
    if not calendar_path.exists():
        print(f"Calendar not found: {calendar_path}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(calendar_path.open(encoding="utf-8")))
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in (rows[0].keys() if rows else [])]
    if missing:
        print(f"Calendar is missing columns: {missing}", file=sys.stderr)
        return 1

    queued = [r for r in rows if r.get("status", "").strip().lower() == "queued"]
    if not queued:
        print("No rows with status='queued'. Nothing to do.")
        return 0

    print(f"Found {len(queued)} queued post(s). Generating...\n")

    agent = IvyEdgeContentAgent(model=args.model, context_dir=args.context_dir)
    out_root = Path(args.output)

    for row in queued:
        keywords = [row["primary_keyword"]] + [
            k.strip() for k in (row.get("secondary_keywords") or "").split("|") if k.strip()
        ]
        try:
            result = agent.generate_blog_post(
                topic=row["title"],
                persona=row["persona"],
                pillar=row["pillar"],
                keywords=keywords,
                content_format=row.get("format") or "educational",
                notes=row.get("notes", ""),
                on_phase=lambda name, _: print(f"  [{row['title'][:40]}] {name}"),
            )
        except Exception as e:
            logger.exception("Failed for row: %s", row.get("title"))
            row["status"] = "error"
            row["error"] = str(e)[:200]
            continue

        folder, dc = _save_result(result, out_root)
        row["status"] = "drafted"
        row["draft_folder"] = str(folder)
        row["dale_chall"] = dc
        row["drafted_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        _maybe_publish(result, folder, args.publish)

    # Rewrite calendar with updated statuses (preserves all original columns
    # plus draft_folder / drafted_at / error if we added them).
    fieldnames = list({*rows[0].keys(), "draft_folder", "drafted_at", "error"})
    with calendar_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBatch complete. Calendar updated: {calendar_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IvyEdge AI content agent")
    parser.add_argument("--model", default=None, help="Override default model (e.g. claude-sonnet-4-6)")
    parser.add_argument("--context-dir", default="context", help="Context library folder")
    parser.add_argument("--output", default="output", help="Where to write drafts")
    parser.add_argument("--publish", action="store_true",
                        help="Publish to Substack immediately after generation (requires SUBSTACK_SID in .env)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_intro = sub.add_parser("intro", help="Generate the one-time IvyEdge introduction post")
    p_intro.set_defaults(func=cmd_intro)

    p_single = sub.add_parser("single", help="Generate one post from flags")
    p_single.add_argument("--topic", required=True)
    p_single.add_argument("--persona", required=True, help="Priya | Maya | Carmen | Dominique | All")
    p_single.add_argument("--pillar", required=True)
    p_single.add_argument("--keywords", required=True, help="Comma-separated; first is primary")
    p_single.add_argument("--format", default="educational",
                          choices=["educational", "customer_story", "behavioral", "industry"])
    p_single.add_argument("--notes", default="")
    p_single.set_defaults(func=cmd_single)

    p_batch = sub.add_parser("batch", help="Generate all queued rows in editorial_calendar.csv")
    p_batch.add_argument("--calendar", default="editorial_calendar.csv")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
