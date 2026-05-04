"""
IvyEdge Editorial Calendar Agent

Generates a ready-to-run editorial_calendar.csv from scratch using Claude.
Topics are grounded in women in the economy, consumer lending, and financial
services — aligned to IvyEdge's mission, personas, and content pillars.

Usage:
    # 12-week calendar starting next Monday, 2 posts/week
    python calendar_agent.py

    # Custom run
    python calendar_agent.py \\
        --weeks 8 \\
        --posts-per-week 3 \\
        --start-date 2026-05-11 \\
        --output editorial_calendar.csv

Appends to an existing calendar (skips duplicate titles) or creates a new one.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ivyedge.calendar")

DEFAULT_MODEL = os.getenv("IVYEDGE_MODEL", "claude-sonnet-4-6")
CONTEXT_DIR = Path("context")

CSV_COLUMNS = [
    "publish_date", "title", "persona", "pillar",
    "primary_keyword", "secondary_keywords", "format", "status", "notes",
]

# ---------------------------------------------------------------------------
# Pitch deck brief (extracted from IvyEdge_202604_vNoFinancials.pptx)
# ---------------------------------------------------------------------------

DECK_BRIEF = """
## IvyEdge — Company Brief (from investor deck, April 2026)

### Mission
AI-powered financial platform purpose-built for women — smarter underwriting,
transparent terms, and a community that grows with you.

### The problem
- Most underwriting was built on male financial patterns and penalises career
  breaks, part-time income, and self-employed income.
- Women earn 82¢ per dollar men earn — compounding into lower credit limits
  and higher rate offers.
- Women with identical risk profiles receive lower credit scores even though
  they default less and have higher repayment rates.
- Sole female mortgage applicants are 30% more likely to be denied than sole males.
- Globally, women hold only 18% of C-suite roles in financial services.

### Market size
- $1.9T underserved lending market for women in the U.S.
- $700B in annual revenue left on the table by financial institutions due to
  inability to serve women equitably.
- $34T projected assets owned by women in the U.S. by 2030.

### What IvyEdge is building (products — pre-launch, do NOT describe as live)
- Ivy Smart Loan: $5K–$15K personal loans with ZestAI holistic underwriting.
  Career breaks, part-time income, and flexible work treated as strengths.
  Same-day funding. Rate decreases every 6 months of on-time payments.
- Ivy Checking: fee-free checking, no minimums.
- Ivy Credit Builder Card: secured card reporting to all 3 bureaus.
- Ivy Credit Monitor: free real-time credit score tracking for all members.
- Ivy Circle: community platform with P2P investing, paid mentorship, and
  an Ivy Scholarship for women pursuing university.
- Later phases: Ivy Education Refinance, Ivy Business Launchpad, Ivy Wage
  Advance, Ivy Family.

### Technology
- ZestAI underwriting: scores the whole financial picture, not just a number
  that penalises career breaks.
- LendAPI: onboarding, decisioning, and capital origination.
- Column (BaaS Bridge): bank charter, ACH rails, ECOA/FCRA/TILA compliance.

### Key partners
- Forté Foundation (100,000+ women entering business/finance careers)
- Conferences for Women (3M+ women across leadership events)
- Women's World Banking (global gender-lens financial product research)
- Optimist Economist (finance education platform for ambitious women)

### Why IvyEdge wins vs. competitors
Traditional banks and generic fintechs (Ellevest→Betterment, SoFi,
LendingClub, Chime, Tala) all lack women-first loan underwriting, P2P
community investment, paid mentor models, and founding-by-women credibility.

### Blog's current job (pre-launch)
Prove demand for the thesis. Build the audience. Establish authority on
women's financial lives. NO product CTAs — use waitlist, newsletter, share,
survey, or tell-us-your-story only.
"""

# ---------------------------------------------------------------------------
# Context loader
# ---------------------------------------------------------------------------

def _load_context() -> str:
    files = {
        "brand_voice": "brand_voice.md",
        "personas": "personas.md",
        "content_strategy": "content_strategy.md",
        "inclusive_marketing": "inclusive_marketing.md",
    }
    parts = []
    for key, fname in files.items():
        path = CONTEXT_DIR / fname
        if path.exists():
            parts.append(f"# === {key} ===\n\n{path.read_text(encoding='utf-8')}")
        else:
            logger.warning("Missing context file: %s", path)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Publish date schedule
# This publishes on Tuesdays and Thursdays by default (easy to change).
# ---------------------------------------------------------------------------

PUBLISH_DAYS = {2, 4}  # Tuesday=1, Thursday=3 in Python weekday() — Mon=0


def _publish_dates(start: date, n: int) -> list[date]:
    """Return the next n publish dates on Tue/Thu on or after start."""
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() in PUBLISH_DAYS:
            dates.append(d)
        d += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

def _call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            wait = 2 ** attempt
            logger.warning("API error — retrying in %ss: %s", wait, e)
            time.sleep(wait)
    raise RuntimeError("Claude call failed after 3 retries")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

PILLAR_DISTRIBUTION = """
Distribute posts across pillars following the content strategy:
- Pillar 1 (Financial education for non-traditional paths): ~40% of posts
- Pillar 2 (Demystifying finance): ~25% of posts
- Pillar 4 (Behavioral science insights): ~10% of posts
- Pillar 5 (Industry trends & advocacy): ~5% of posts
- Pillar 1/2 crossover (freelancer finance + explainer hybrid): ~20% of posts
Pillar 3 (reader stories) is PAUSED — do not include it.
"""

PERSONA_GUIDE = """
Rotate across all four personas. Every post targets at least one:
- Maya: freelancer / 1099 worker / gig economy
- Priya: career returner / caregiver / re-entry after gap
- Carmen: small business owner / established entrepreneur
- Dominique: corporate climber / high-earner / wealth builder
Use "All" when the topic applies equally to all four.
"""

FORMAT_OPTIONS = """
Assign one format per post:
- educational: how-to, explainer, "here's how it actually works"
- behavioral: psychology of money, habit design, decision science
- industry: trend, advocacy, regulatory change, market data
(customer_story is reserved for post-launch — do not use it)
"""


def _build_prompt(n_posts: int, context: str) -> str:
    return f"""You are building an editorial calendar for IvyEdge, a pre-launch
AI-powered financial platform built for women.

{DECK_BRIEF}

{context}

---

{PILLAR_DISTRIBUTION}

{PERSONA_GUIDE}

{FORMAT_OPTIONS}

---

TOPIC GUIDANCE
Every post must be rooted in at least one of these three territory areas:
1. Women in the economy — wage gap, wealth gap, workforce participation,
   caregiving economics, female entrepreneurship, return-to-work
2. Consumer lending — credit scoring, underwriting, loan approval, APR,
   debt-to-income, secured vs. unsecured, CFPB rules, fair lending
3. Financial services for women — how the industry was built, what it misses,
   what alternatives exist, what's changing in AI underwriting and open banking

Posts must never mention IvyEdge products as live. CTAs are pre-launch only:
waitlist, newsletter, share, survey, or tell-us-your-story.

Never repeat a topic. Each post should have a distinct angle, even if two posts
share a keyword cluster (e.g., different personas, different stages of the problem).

SEO KEYWORD GUIDANCE
Primary keyword: a real search phrase a woman would type into Google.
Make it specific — not "credit scores" but "how career gaps affect credit score".
Secondary keywords: 2–4 semantically related phrases, pipe-separated.

---

Generate exactly {n_posts} post ideas.

Return ONLY a valid JSON array — no markdown fences, no preamble, no commentary.
Each object must have these exact keys:

{{
  "title": "Post title as it will appear on Substack",
  "persona": "Maya | Priya | Carmen | Dominique | All",
  "pillar": "Pillar 1: Financial Education for Non-Traditional Paths | Pillar 2: Demystifying Finance | Pillar 4: Behavioral Science Insights | Pillar 5: Industry Trends & Advocacy",
  "primary_keyword": "main SEO keyword phrase",
  "secondary_keywords": "keyword two|keyword three|keyword four",
  "format": "educational | behavioral | industry",
  "notes": "1-2 sentence editorial note: the specific angle, the IvyEdge point of view, one data point or hook to anchor it"
}}

The array must contain exactly {n_posts} objects.
"""


# ---------------------------------------------------------------------------
# Parse + write CSV
# ---------------------------------------------------------------------------

def _parse_posts(raw: str) -> list[dict]:
    # Strip accidental markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].strip()
    try:
        posts = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from Claude: %s\nRaw:\n%s", e, raw[:500])
        raise
    if not isinstance(posts, list):
        raise ValueError(f"Expected a JSON array, got {type(posts)}")
    return posts


def _write_calendar(posts: list[dict], dates: list[date], output: Path) -> None:
    existing_titles: set[str] = set()
    existing_rows: list[dict] = []

    if output.exists():
        with output.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                existing_titles.add(row.get("title", "").strip().lower())

    new_rows = []
    date_idx = 0
    for post in posts:
        title = post.get("title", "").strip()
        if title.lower() in existing_titles:
            logger.info("Skipping duplicate: %s", title)
            continue
        if date_idx >= len(dates):
            logger.warning("Ran out of publish dates — increase --weeks or --posts-per-week")
            break
        row = {
            "publish_date": dates[date_idx].isoformat(),
            "title": title,
            "persona": post.get("persona", ""),
            "pillar": post.get("pillar", ""),
            "primary_keyword": post.get("primary_keyword", ""),
            "secondary_keywords": post.get("secondary_keywords", ""),
            "format": post.get("format", "educational"),
            "status": "queued",
            "notes": post.get("notes", ""),
        }
        new_rows.append(row)
        existing_titles.add(title.lower())
        date_idx += 1

    all_rows = existing_rows + new_rows
    fieldnames = list({col for row in all_rows for col in row} | set(CSV_COLUMNS))
    fieldnames = sorted(fieldnames, key=lambda c: CSV_COLUMNS.index(c) if c in CSV_COLUMNS else 99)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("Wrote %d new posts (%d total) → %s", len(new_rows), len(all_rows), output)
    return new_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IvyEdge editorial calendar generator")
    parser.add_argument("--weeks", type=int, default=12,
                        help="Number of weeks to plan (default: 12)")
    parser.add_argument("--posts-per-week", type=int, default=2,
                        help="Posts per week (default: 2)")
    parser.add_argument("--start-date", default=None,
                        help="Start date YYYY-MM-DD (default: next Tuesday)")
    parser.add_argument("--output", default="editorial_calendar.csv",
                        help="Output CSV path (default: editorial_calendar.csv)")
    parser.add_argument("--context-dir", default="context",
                        help="Context library folder")
    args = parser.parse_args(argv)

    global CONTEXT_DIR
    CONTEXT_DIR = Path(args.context_dir)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key)

    # Resolve start date
    if args.start_date:
        start = date.fromisoformat(args.start_date)
    else:
        today = date.today()
        # Find next Tuesday (weekday 1)
        days_ahead = (1 - today.weekday()) % 7 or 7
        start = today + timedelta(days=days_ahead)

    n_posts = args.weeks * args.posts_per_week
    publish_dates = _publish_dates(start, n_posts)

    logger.info(
        "Generating %d posts (%d weeks × %d/week) starting %s",
        n_posts, args.weeks, args.posts_per_week, start.isoformat()
    )

    context = _load_context()
    prompt = _build_prompt(n_posts, context)

    logger.info("Calling Claude (%s)...", DEFAULT_MODEL)
    raw = _call_claude(client, prompt)

    posts = _parse_posts(raw)
    logger.info("Received %d post ideas from Claude", len(posts))

    output = Path(args.output)
    new_rows = _write_calendar(posts, publish_dates, output)

    print(f"\nDone. {len(new_rows)} posts added to {output}")
    print(f"Date range: {publish_dates[0]} → {publish_dates[min(len(new_rows)-1, len(publish_dates)-1)]}")
    print(f"\nRun the batch:\n  python run_pipeline.py batch --calendar {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
