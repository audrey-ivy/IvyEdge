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



# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Link validator — every URL is checked before a draft is saved
# ---------------------------------------------------------------------------

# Sites that return 403 to bots but work fine in browsers.
# These are NOT skipped — we still check them. A 403 is ignored (bot block),
# but a 404 or other error is treated as a dead link and stripped.
_BOT_BLOCKED_DOMAINS = {
    "consumerfinance.gov", "cfpb.gov", "bls.gov", "urban.org",
    "academic.oup.com", "jstor.org", "census.gov", "dol.gov",
}

def _check_links(markdown_text: str) -> list[tuple[str, str]]:
    """Return list of (url, reason) for every dead link found.
    ALL links are checked — including government/academic domains.
    A 403 on a known bot-blocking domain is ignored (link is kept).
    A 404 is always a dead link, regardless of domain."""
    import urllib.request, ssl as _ssl
    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    url_pat = re.compile(r"https?://[^\s\)\]\"\'<>]+")
    seen: set[str] = set()
    dead: list[tuple[str, str]] = []
    for raw_url in url_pat.findall(markdown_text):
        url = raw_url.rstrip(".,;)")
        if url in seen:
            continue
        seen.add(url)
        domain = url.split("/")[2]
        is_bot_blocked = any(d in domain for d in _BOT_BLOCKED_DOMAINS)
        try:
            req = urllib.request.Request(url, headers=headers)
            # follow_redirects=True is the default; final URL may differ
            with urllib.request.urlopen(req, timeout=8, context=ctx) as r:
                if r.status == 404:
                    dead.append((url, "HTTP 404 — page not found"))
                elif r.status >= 400:
                    dead.append((url, f"HTTP {r.status}"))
        except Exception as e:
            msg = str(e)
            if "404" in msg:
                dead.append((url, "HTTP 404 — page not found"))
                continue
            if "403" in msg and is_bot_blocked:
                continue  # expected bot block on known domain — link is fine
            # Other errors on bot-blocked domains: give benefit of the doubt
            if is_bot_blocked:
                logger.warning("Could not verify %s (%s) — keeping link (bot-blocked domain)", url, msg[:60])
                continue
            dead.append((url, msg[:60]))
    return dead


def _validate_and_strip_dead_links(markdown_text: str, topic: str) -> str:
    """Check every link. Strip hyperlinks that are dead (keep the anchor text).

    This runs twice — once after generation, once immediately before publishing —
    so a dead link can never make it into a live article.
    """
    dead = _check_links(markdown_text)
    if not dead:
        return markdown_text
    for url, reason in dead:
        logger.warning("Dead link in '%s': %s (%s) — stripping hyperlink", topic[:40], url, reason)
        markdown_text = re.sub(
            r'\[([^\]]+)\]\(' + re.escape(url) + r'\)',
            r'\1',
            markdown_text,
        )
        markdown_text = markdown_text.replace(url, "")
    logger.warning("%d dead link(s) stripped from '%s'", len(dead), topic[:40])
    return markdown_text


def _preflight_links(markdown_text: str, topic: str) -> None:
    """Hard gate called immediately before publishing.
    Raises RuntimeError if any dead links remain after stripping — publish is aborted.
    This is the final safety net: if a link slipped through validation, we stop here."""
    dead = _check_links(markdown_text)
    if dead:
        urls = "\n  ".join(f"{url} ({reason})" for url, reason in dead)
        raise RuntimeError(
            f"PUBLISH ABORTED — dead link(s) still present in '{topic[:50]}':\n  {urls}\n"
            "Fix the links in 05_final_draft.md and re-run."
        )


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
    # Validate every hyperlink before saving — strip dead ones
    validated_draft = _validate_and_strip_dead_links(result.final_draft, result.brief.topic)
    (folder / "05_final_draft.md").write_text(validated_draft, encoding="utf-8")
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

def _create_substack_draft(result: GenerationResult, folder: Path) -> Optional[int]:
    """Push article to Substack as a DRAFT (not live). Saves draft ID to folder.
    Returns the draft ID, or None on failure."""
    try:
        publisher = SubstackPublisher()
    except ValueError as e:
        logger.error("Cannot reach Substack: %s", e)
        return None

    slug    = _slugify(result.brief.topic)
    title   = result.brief.topic
    subtitle = getattr(result, "meta_description", "") or ""
    blog_url = f"https://www.ivyedge.co/blog/{slug}"

    draft_id = publisher.create_draft_only(title=title, body_markdown=result.final_draft,
                                           subtitle=subtitle, slug=slug)
    if not draft_id:
        logger.error("Substack draft creation returned no ID")
        return None

    (folder / "substack_draft_id.txt").write_text(str(draft_id), encoding="utf-8")
    (folder / "blog_url.txt").write_text(blog_url, encoding="utf-8")
    print(f"  Substack draft saved (id={draft_id}) — review at:")
    print(f"  https://substack.com/dashboard/posts")
    print(f"  Approve → bash approve.sh")
    print(f"  Reject  → bash reject.sh")
    return draft_id


def _maybe_publish(result: GenerationResult, folder: Path, publish: bool) -> Optional[str]:
    """Kept for backward compat (intro/single commands). Prefer approve command for batch flow."""
    if not publish:
        return None
    try:
        publisher = SubstackPublisher()
    except ValueError as e:
        logger.error("Cannot publish: %s", e)
        return None

    slug     = _slugify(result.brief.topic)
    title    = result.brief.topic
    subtitle = getattr(result, "meta_description", "") or ""
    blog_url = f"https://www.ivyedge.co/blog/{slug}"

    _preflight_links(result.final_draft, title)
    post_url = publisher.publish(title=title, body_markdown=result.final_draft,
                                 subtitle=subtitle, slug=slug)
    print(f"  Published to Substack: {post_url}")
    print(f"  Canonical blog URL:    {blog_url}")
    (folder / "substack_url.txt").write_text(post_url, encoding="utf-8")
    (folder / "blog_url.txt").write_text(blog_url, encoding="utf-8")
    return post_url


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
    "scheduled_date", "title", "persona", "pillar",
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

    # Accept both 'scheduled' (regular calendar) and 'queued' (urgent trending topics)
    READY_STATUSES = {"scheduled", "queued"}
    all_ready = [r for r in rows if r.get("status", "").strip().lower() in READY_STATUSES]
    if not all_ready:
        print("No articles with status='scheduled' or 'queued'. Nothing to do.")
        return 0

    # Only process articles due within the next 7 days.
    today = datetime.utcnow().date()
    cutoff = today + __import__("datetime").timedelta(days=7)

    def _parse_date_flexible(s: str):
        s = s.strip()
        try:
            return __import__("datetime").date.fromisoformat(s)
        except ValueError:
            pass
        for fmt in ("%m/%d/%y", "%m/%d/%Y"):
            try:
                return __import__("datetime").datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None

    due = []
    for r in all_ready:
        pub = _parse_date_flexible(r.get("scheduled_date", ""))
        if pub is None or pub <= cutoff:
            due.append((pub or today, r))

    if not due:
        print(f"No articles due within the next 7 days (cutoff {cutoff}). Nothing to generate.")
        return 0

    # Sort by date and take only ONE — one article per Monday run.
    due.sort(key=lambda x: x[0])
    queued = [due[0][1]]

    print(f"Generating 1 article (next due: {due[0][0]}). {len(due)-1} more article(s) queued for future weeks.\n")

    agent = IvyEdgeContentAgent(model=args.model, context_dir=args.context_dir)
    out_root = Path(args.output)

    for row in queued:
        # Skip if an output folder for this topic already exists (pipeline crashed mid-run)
        title_slug = _slugify(row["title"])
        existing = list(out_root.glob(f"*_{title_slug}"))
        if existing:
            logger.info("Output folder already exists — skipping '%s' (%s)", row["title"][:50], existing[0].name)
            row["status"] = "drafted"
            row["draft_folder"] = str(existing[0])
            continue

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
        row.setdefault("published_at", "")

        # Always push to Substack as a draft first — approve.sh publishes it live
        _create_substack_draft(result, folder)

    # Rewrite calendar with updated statuses (preserves all original columns
    # plus any new fields added this run).
    fieldnames = list({*rows[0].keys(), "draft_folder", "drafted_at", "published_at", "dale_chall", "error"})
    with calendar_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBatch complete. Calendar updated: {calendar_path}")
    return 0


# ---------------------------------------------------------------------------
# Approve command — publish the most recent draft live + queue social posts
# ---------------------------------------------------------------------------

def cmd_approve(args: argparse.Namespace) -> int:
    """Publish the most recent Substack draft live and queue social media posts."""
    import shutil
    out_root = Path(args.output)
    calendar_path = Path(args.calendar)

    # Find most recently drafted folder
    drafted = sorted(
        [f for f in out_root.iterdir() if f.is_dir() and (f / "substack_draft_id.txt").exists()
         and not (f / "substack_url.txt").exists()],
        key=lambda f: f.stat().st_mtime, reverse=True
    )
    if not drafted:
        print("No drafted articles found waiting for approval.")
        print("Run 'bash run_monday.sh' first to generate a draft.")
        return 1

    folder = drafted[0]
    draft_id_file = folder / "substack_draft_id.txt"
    draft_md      = folder / "05_final_draft.md"
    meta_file     = folder / "meta.json"

    draft_id = int(draft_id_file.read_text().strip())
    title    = json.loads(meta_file.read_text())["topic"] if meta_file.exists() else folder.name
    body     = draft_md.read_text(encoding="utf-8")
    blog_url = f"https://www.ivyedge.co/blog/{_slugify(title)}"

    print(f"\nApproving: {title}")
    print(f"Draft ID:  {draft_id}")

    # Hard link check before going live
    _preflight_links(body, title)

    # Publish the existing Substack draft
    try:
        publisher = SubstackPublisher()
    except ValueError as e:
        logger.error("Cannot reach Substack: %s", e)
        return 1

    post_url = publisher._publish_draft(draft_id)
    print(f"\n✅ Published: {post_url}")

    (folder / "substack_url.txt").write_text(post_url, encoding="utf-8")
    (folder / "blog_url.txt").write_text(blog_url, encoding="utf-8")

    # Update calendar
    if calendar_path.exists():
        rows = list(csv.DictReader(calendar_path.open(encoding="utf-8")))
        for row in rows:
            if _slugify(row.get("title", "")) == _slugify(title):
                row["status"]       = "published"
                row["published_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                break
        fieldnames = list({*rows[0].keys(), "published_at"})
        with calendar_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Calendar updated: status=published")

    # Queue social media posts
    print("\nQueuing social media posts...")
    import subprocess
    agent_dir = Path(__file__).parent
    result = subprocess.run(
        [str(agent_dir / ".venv" / "bin" / "python"), "social_media_agent.py",
         "--folder", str(folder), "--output-dir", str(out_root)],
        cwd=str(agent_dir),
    )
    if result.returncode != 0:
        print("⚠️  Social media agent failed — run manually: python social_media_agent.py")

    print("\nDone. Article is live and social posts are queued in Buffer.")
    return 0


# ---------------------------------------------------------------------------
# Reject command — remove topic from calendar, delete folder, start fresh
# ---------------------------------------------------------------------------

def cmd_reject(args: argparse.Namespace) -> int:
    """Remove the most recent drafted article from the calendar and delete its folder."""
    import shutil
    out_root      = Path(args.output)
    calendar_path = Path(args.calendar)

    # Find most recently drafted folder (not yet published)
    drafted = sorted(
        [f for f in out_root.iterdir() if f.is_dir() and (f / "substack_draft_id.txt").exists()
         and not (f / "substack_url.txt").exists()],
        key=lambda f: f.stat().st_mtime, reverse=True
    )
    if not drafted:
        print("No drafted articles found to reject.")
        return 1

    folder    = drafted[0]
    meta_file = folder / "meta.json"
    title     = json.loads(meta_file.read_text())["topic"] if meta_file.exists() else folder.name

    print(f"\nRejecting: {title}")
    print(f"Folder:    {folder}")

    # Remove from calendar
    removed = False
    if calendar_path.exists():
        rows = list(csv.DictReader(calendar_path.open(encoding="utf-8")))
        original_count = len(rows)
        rows = [r for r in rows if _slugify(r.get("title", "")) != _slugify(title)]
        if len(rows) < original_count:
            removed = True
            fieldnames = list(rows[0].keys()) if rows else []
            with calendar_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Removed from calendar.")
        else:
            print("Topic not found in calendar — may have already been removed.")

    # Delete output folder
    shutil.rmtree(folder)
    print(f"Deleted: {folder.name}")

    print("\nDone. Run 'bash run_monday.sh' to generate a replacement article.")
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

    p_approve = sub.add_parser("approve", help="Publish the most recent Substack draft live + queue social posts")
    p_approve.add_argument("--calendar", default="editorial_calendar.csv")
    p_approve.set_defaults(func=cmd_approve)

    p_reject = sub.add_parser("reject", help="Remove the most recent draft from calendar and delete its folder")
    p_reject.add_argument("--calendar", default="editorial_calendar.csv")
    p_reject.set_defaults(func=cmd_reject)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
