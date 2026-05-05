"""
IvyEdge Engagement Agent

Discovers conversations about IvyEdge's topics across Instagram, Threads,
Reddit, and TikTok, then either queues suggested interactions for human
review or posts them directly (Reddit only, with --auto).

Each run saves a dated folder under engagement_output/YYYY-MM-DD/ containing
a human-readable report.md and the raw opportunities.json.

Goal: prove market demand by finding and genuinely engaging with people who
are already talking about the problems IvyEdge will solve.

Usage:
    python engagement_agent.py                         # discover all platforms, save report
    python engagement_agent.py --platform instagram    # Instagram hashtags only
    python engagement_agent.py --platform reddit       # Reddit only
    python engagement_agent.py --platform threads      # Threads reply monitor only
    python engagement_agent.py --platform tiktok       # TikTok hashtags only
    python engagement_agent.py --auto                  # also auto-post Reddit comments
    python engagement_agent.py --dry-run               # discover + score but don't save or post
    python engagement_agent.py --show-queue            # print pending review queue
    python engagement_agent.py --stats                 # show engagement log summary
"""

from __future__ import annotations

import argparse
import json
import logging
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
logger = logging.getLogger("ivyedge.engagement")

BASE_DIR        = Path(__file__).parent
QUEUE_FILE      = BASE_DIR / "engagement_log" / "review_queue.json"
ACTION_LOG_FILE = BASE_DIR / "engagement_log" / "actions.json"
OUTPUT_DIR      = BASE_DIR / "engagement_output"

PLATFORM_ICONS = {
    "instagram": "📸",
    "threads":   "🧵",
    "reddit":    "🤖",
    "tiktok":    "🎵",
}


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def _load_queue() -> list[dict]:
    if QUEUE_FILE.exists():
        return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
    return []


def _save_queue(items: list[dict]) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _append_to_action_log(items: list[dict]) -> None:
    existing = []
    if ACTION_LOG_FILE.exists():
        existing = json.loads(ACTION_LOG_FILE.read_text(encoding="utf-8"))
    existing.extend(items)
    ACTION_LOG_FILE.write_text(json.dumps(existing[-2000:], indent=2), encoding="utf-8")


def _enqueue(opportunities, dry_run: bool = False) -> int:
    existing = _load_queue()
    existing_ids = {item["post_id"] for item in existing}
    new_items = [
        o.to_dict() for o in opportunities
        if o.post_id not in existing_ids and o.suggested_comment
    ]
    if not dry_run and new_items:
        _save_queue(existing + new_items)
        logger.info("Added %d item(s) to review queue", len(new_items))
    return len(new_items)


# ---------------------------------------------------------------------------
# Dated folder report
# ---------------------------------------------------------------------------

def _save_report(opportunities: list[dict], summary: dict, dry_run: bool = False) -> Optional[Path]:
    if dry_run or not opportunities:
        return None

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder   = OUTPUT_DIR / date_str
    folder.mkdir(parents=True, exist_ok=True)

    # ── Raw JSON ────────────────────────────────────────────────────────────
    json_path = folder / "opportunities.json"
    json_path.write_text(json.dumps({
        "summary": summary,
        "opportunities": opportunities,
    }, indent=2), encoding="utf-8")

    # ── Markdown report ─────────────────────────────────────────────────────
    lines = [
        f"# IvyEdge Engagement Brief — {date_str}",
        "",
        f"**{len(opportunities)} opportunities** across "
        + ", ".join(
            f"{PLATFORM_ICONS.get(p,'•')} {p.title()} ({n})"
            for p, n in sorted(
                {i.get("platform","?"): 0 for i in opportunities}.items()
            )
        ),
    ]

    # fill in real counts
    by_platform: dict[str, list[dict]] = {}
    for item in opportunities:
        by_platform.setdefault(item.get("platform", "other"), []).append(item)

    lines[2] = (
        f"**{len(opportunities)} opportunities** across "
        + ", ".join(
            f"{PLATFORM_ICONS.get(p,'•')} {p.title()} ({len(items)})"
            for p, items in sorted(by_platform.items())
        )
    )

    if summary.get("posted"):
        lines.append(f"✓ {summary['posted']} Reddit comment(s) auto-posted")
    lines.append("")

    for platform in ["reddit", "instagram", "tiktok", "threads"]:
        items = by_platform.get(platform, [])
        if not items:
            continue
        icon = PLATFORM_ICONS.get(platform, "•")
        lines += [
            f"---",
            f"## {icon} {platform.title()} ({len(items)})",
            "",
        ]
        for i, item in enumerate(items, 1):
            score     = item.get("score", 0)
            url       = item.get("url", "")
            author    = item.get("author", "")
            subreddit = item.get("subreddit", "")
            rationale = item.get("rationale", "")
            content   = item.get("content", "")[:200].strip()
            comment   = item.get("suggested_comment", "").strip()
            action    = item.get("suggested_action", "comment")
            location  = f"r/{subreddit}" if subreddit else f"@{author}"

            lines += [
                f"### {i}. {location} — score {score:.1f}/10",
                f"[Open post]({url})",
                f"",
                f"> {content}{'…' if len(item.get('content','')) > 200 else ''}",
                f"",
                f"**Why engage:** {rationale}",
                f"",
            ]
            if comment:
                lines += [
                    f"**Suggested {action}:**",
                    f"```",
                    comment,
                    f"```",
                    "",
                ]

    md_path = folder / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved: %s", folder)
    return folder


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_queue(pending_only: bool = True) -> None:
    items = _load_queue()
    if pending_only:
        items = [i for i in items if i.get("status") == "pending"]
    if not items:
        print("Review queue is empty.")
        return

    print(f"\n{'='*60}")
    print(f"ENGAGEMENT REVIEW QUEUE  ({len(items)} items)")
    print(f"{'='*60}\n")

    for i, item in enumerate(items, 1):
        platform  = item.get("platform", "?").upper()
        score     = item.get("score", 0)
        url       = item.get("url", "")
        author    = item.get("author", "")
        rationale = item.get("rationale", "")
        comment   = item.get("suggested_comment", "")
        subreddit = item.get("subreddit", "")
        location  = f"r/{subreddit}" if subreddit else platform
        print(f"[{i}] {platform} | {location} | score={score:.1f} | @{author}")
        print(f"    {url}")
        print(f"    Why: {rationale}")
        print(f"    Post: {item.get('content','')[:120].strip()}")
        if comment:
            print(f"    Draft:\n      {comment[:400]}")
        print()


def _print_stats() -> None:
    queue = _load_queue()
    log   = []
    if ACTION_LOG_FILE.exists():
        log = json.loads(ACTION_LOG_FILE.read_text(encoding="utf-8"))

    platforms: dict[str, int] = {}
    for item in log:
        p = item.get("platform", "?")
        platforms[p] = platforms.get(p, 0) + 1

    pending  = sum(1 for i in queue if i.get("status") == "pending")
    actioned = sum(1 for i in queue if i.get("status") == "actioned")

    print(f"\n{'='*40}")
    print("ENGAGEMENT STATS")
    print(f"{'='*40}")
    print(f"Review queue:  {pending} pending, {actioned} actioned")
    print(f"Total logged:  {len(log)}")
    if platforms:
        print("By platform:")
        for p, n in sorted(platforms.items()):
            print(f"  {p}: {n}")

    reports = sorted(OUTPUT_DIR.glob("*/report.md")) if OUTPUT_DIR.exists() else []
    if reports:
        print(f"Reports saved: {len(reports)} (latest: {reports[-1].parent.name})")
    print()


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    platform: Optional[str] = None,
    auto: bool = False,
    dry_run: bool = False,
) -> dict:
    summary: dict = {
        "run_at":     datetime.now(timezone.utc).isoformat(),
        "platform":   platform or "all",
        "dry_run":    dry_run,
        "discovered": 0,
        "queued":     0,
        "posted":     0,
    }

    all_opportunities = []

    # ── Instagram ──────────────────────────────────────────────────────────
    if platform in (None, "instagram"):
        try:
            from platform_agents.instagram_hashtag import discover as ig_discover
            ig_opps = ig_discover(dry_run=dry_run)
            all_opportunities.extend(ig_opps)
            logger.info("Instagram: %d opportunities", len(ig_opps))
        except Exception as e:
            logger.error("Instagram agent failed: %s", e)

    # ── Threads ────────────────────────────────────────────────────────────
    if platform in (None, "threads"):
        try:
            from platform_agents.threads_monitor import discover as threads_discover
            th_opps = threads_discover(dry_run=dry_run)
            all_opportunities.extend(th_opps)
            logger.info("Threads: %d opportunities", len(th_opps))
        except Exception as e:
            logger.error("Threads agent failed: %s", e)

    # ── Reddit ─────────────────────────────────────────────────────────────
    if platform in (None, "reddit"):
        try:
            from platform_agents.reddit_agent import discover as reddit_discover, engage
            reddit_opps = reddit_discover(dry_run=dry_run)
            all_opportunities.extend(reddit_opps)
            logger.info("Reddit: %d opportunities", len(reddit_opps))

            if auto and reddit_opps:
                reddit_opps = engage(reddit_opps, dry_run=dry_run)
                posted = sum(1 for o in reddit_opps if o.status == "actioned")
                summary["posted"] = posted
                if not dry_run:
                    _append_to_action_log(
                        [o.to_dict() for o in reddit_opps if o.status == "actioned"]
                    )
        except Exception as e:
            logger.error("Reddit agent failed: %s", e)

    # ── TikTok ─────────────────────────────────────────────────────────────
    if platform in (None, "tiktok"):
        try:
            from platform_agents.tiktok_agent import discover as tiktok_discover
            tt_opps = tiktok_discover(dry_run=dry_run)
            all_opportunities.extend(tt_opps)
            logger.info("TikTok: %d opportunities", len(tt_opps))
        except Exception as e:
            logger.error("TikTok agent failed: %s", e)

    summary["discovered"] = len(all_opportunities)

    to_queue = [o for o in all_opportunities if o.status == "pending"]
    summary["queued"] = _enqueue(to_queue, dry_run=dry_run)

    folder = _save_report([o.to_dict() for o in to_queue], summary, dry_run=dry_run)
    summary["report_folder"] = str(folder) if folder else None

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="IvyEdge engagement agent")
    parser.add_argument(
        "--platform", choices=["instagram", "reddit", "threads", "tiktok"],
        help="Run only this platform (default: all)"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-post Reddit comments (queues everything else)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover and score posts, but don't save or post"
    )
    parser.add_argument(
        "--show-queue", action="store_true",
        help="Print the pending review queue and exit"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print engagement stats and exit"
    )
    args = parser.parse_args(argv)

    if args.show_queue:
        _print_queue()
        return 0

    if args.stats:
        _print_stats()
        return 0

    summary = run(
        platform=args.platform,
        auto=args.auto,
        dry_run=args.dry_run,
    )

    folder = summary.get("report_folder")
    print(f"\n{'─'*50}")
    print(f"Engagement run complete")
    print(f"  Discovered:  {summary['discovered']} opportunities")
    print(f"  Queued:      {summary['queued']} added to review queue")
    if summary["posted"]:
        print(f"  Posted:      {summary['posted']} Reddit comments")
    if folder:
        print(f"  Report:      {folder}/report.md")
    if summary["dry_run"]:
        print("  (dry-run — nothing saved or posted)")
    print(f"{'─'*50}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
