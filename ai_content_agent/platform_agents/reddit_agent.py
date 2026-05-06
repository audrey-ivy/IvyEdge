"""
IvyEdge — Reddit Engagement Agent

Searches relevant subreddits for posts about the problems IvyEdge addresses,
scores them for relevance using Claude, and writes a daily report with direct
links and copy-paste comments for manual engagement.

Discovery uses Reddit's public JSON API — no credentials required.
Auto-posting (--auto) requires PRAW credentials in .env; if not set, the
agent runs in discovery-only mode and outputs links for manual action.

Required for auto-posting (optional — discovery works without these):
  REDDIT_CLIENT_ID=...
  REDDIT_CLIENT_SECRET=...
  REDDIT_USERNAME=JoinIvyEdge
  REDDIT_PASSWORD=...
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import requests
from dotenv import load_dotenv

from platform_agents import EngagementOpportunity

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger("ivyedge.reddit")

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USERNAME      = os.getenv("REDDIT_USERNAME", "JoinIvyEdge")
REDDIT_PASSWORD      = os.getenv("REDDIT_PASSWORD", "")

USER_AGENT   = f"IvyEdgeBot/1.0 by u/{REDDIT_USERNAME}"
REDDIT_BASE  = "https://www.reddit.com"

SUBREDDITS = [
    "freelance",
    "personalfinance",
    "financialindependence",
    "smallbusiness",
    "womeninbusiness",
    "entrepreneur",
    "HENRYfinance",
    "selfemployed",
    "careerguidance",
    "workingmoms",
    "CreditCards",
]

SEARCH_QUERIES = [
    "1099 income loan denied",
    "freelance income mortgage",
    "self employed credit score",
    "career gap credit",
    "gig worker loan",
    "variable income bank",
    "contract work loan",
    "freelancer loan application",
    "self employed denied",
    "career break credit score",
    "non traditional income finance",
    "side hustle income credit",
    "maternity leave credit score",
    "freelance income unstable bank",
]

MIN_RELEVANCE_SCORE  = 6.0
MAX_POSTS_PER_RUN    = 40
MAX_COMMENTS_PER_RUN = 10
POST_DELAY_SECONDS   = 60

SEEN_LOG = Path(__file__).parent.parent / "engagement_log" / "reddit_seen.json"


def _load_seen() -> set[str]:
    if SEEN_LOG.exists():
        data = json.loads(SEEN_LOG.read_text(encoding="utf-8"))
        return set(data.get("seen_ids", []))
    return set()


def _save_seen(seen: set[str]) -> None:
    SEEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    SEEN_LOG.write_text(
        json.dumps({"seen_ids": list(seen)[-1000:]}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public Reddit JSON API (no credentials needed)
# ---------------------------------------------------------------------------

def _reddit_get(path: str, params: dict) -> Optional[dict]:
    """Call Reddit's public JSON API with rate-limit courtesy sleep."""
    try:
        resp = requests.get(
            f"{REDDIT_BASE}{path}.json",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("Reddit rate limited — sleeping 30s")
            time.sleep(30)
            resp = requests.get(
                f"{REDDIT_BASE}{path}.json",
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
        if not resp.ok:
            logger.warning("Reddit request failed (%s): %s", resp.status_code, path)
            return None
        return resp.json()
    except Exception as e:
        logger.warning("Reddit request error: %s", e)
        return None


def _fetch_posts(seen: set[str]) -> list[dict]:
    """Search subreddits using the public API. No credentials required."""
    posts: list[dict] = []
    seen_ids: set[str] = set()

    for query in SEARCH_QUERIES:
        if len(posts) >= MAX_POSTS_PER_RUN:
            break

        data = _reddit_get("/search", {
            "q":           query,
            "sort":        "new",
            "t":           "week",
            "limit":       10,
            "restrict_sr": False,
            "type":        "link",
        })
        if not data:
            continue

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            pid  = post.get("id", "")
            sub  = post.get("subreddit", "")

            if pid in seen or pid in seen_ids:
                continue
            # Skip obviously off-topic subreddits
            blocklist = {"memes", "funny", "gaming", "worldnews", "politics", "askreddit"}
            if sub.lower() in blocklist:
                continue
            if post.get("score", 0) < 1:
                continue
            if not post.get("selftext") and not post.get("title"):
                continue

            posts.append({
                "id":           pid,
                "subreddit":    sub,
                "title":        post.get("title", ""),
                "selftext":     post.get("selftext", "")[:600],
                "url":          f"https://reddit.com{post.get('permalink', '')}",
                "score":        post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "author":       post.get("author", "[deleted]"),
                "created_utc":  post.get("created_utc", 0),
            })
            seen_ids.add(pid)

            if len(posts) >= MAX_POSTS_PER_RUN:
                break

        time.sleep(1)  # be polite to Reddit's servers

    logger.info("Reddit: fetched %d candidate posts", len(posts))
    return posts


# ---------------------------------------------------------------------------
# Claude scoring + comment drafting
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the community engagement strategist for IvyEdge, a pre-launch
consumer finance platform for women with non-traditional financial histories
(freelancers, career returners, entrepreneurs with variable income).

IvyEdge's core thesis:
- Career gaps don't make you a credit risk
- 1099 income is real income
- Five years of business history is an arbitrary threshold
- High earners with non-W-2 income deserve products that match their reality
- Plain-language financial transparency is a baseline, not a feature

IvyEdge is pre-launch with nothing to sell. Comments must be:
  1. Genuinely useful — practical advice the person can act on today
  2. On-brand — reflects IvyEdge's POV
  3. Never promotional — no links, no product mentions
  4. Authentic — smart friend who works in finance, not a press release
  5. Appropriate length — 2-4 sentences for simple questions, up to a short
     paragraph for complex situations"""


BATCH_SIZE = 15  # posts per Claude call to stay within token limits


def _score_batch(posts: list[dict], client: anthropic.Anthropic) -> list[dict]:
    """Score one batch of posts. Returns list of scored dicts."""
    posts_text = "\n\n".join(
        f"POST {i+1} (id={p['id']}, r/{p['subreddit']}, score={p['score']}, "
        f"{p['num_comments']} comments):\n"
        f"Title: {p['title']}\n"
        f"Body: {p['selftext'] or '(link/title-only post)'}"
        for i, p in enumerate(posts)
    )

    prompt = f"""Below are {len(posts)} Reddit posts found via keyword search.

For each, output:
{{
  "post_id": "<id>",
  "score": <0-10 float>,
  "rationale": "<one sentence why this is/isn't worth engaging>",
  "suggested_action": "comment" | "upvote_only" | "skip",
  "suggested_comment": "<if action=comment: full comment text. Empty string otherwise.>"
}}

JSON array only. No prose, no markdown fences.

High scores (≥6): OP is describing a real problem IvyEdge addresses — 1099/gig/freelance
income issues, credit gaps, career breaks, loan denials, variable income frustrations.
Our comment adds concrete, useful advice they can act on today.

Low scores (<6): vague questions, already well-answered, promotional posts,
topics outside IvyEdge's expertise (investing, crypto, etc).

POSTS:
{posts_text}"""

    msg = client.messages.create(
        model=os.getenv("IVYEDGE_MODEL", "claude-sonnet-4-6"),
        max_tokens=5000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude returned unparseable JSON for batch of %d posts", len(posts))
        return []


def _score_and_draft(posts: list[dict], client: anthropic.Anthropic) -> list[EngagementOpportunity]:
    if not posts:
        return []

    # Score in batches to avoid token limits
    scored: list[dict] = []
    for i in range(0, len(posts), BATCH_SIZE):
        batch = posts[i:i + BATCH_SIZE]
        scored.extend(_score_batch(batch, client))
        if i + BATCH_SIZE < len(posts):
            time.sleep(1)

    post_map = {p["id"]: p for p in posts}
    opportunities = []
    for item in scored:
        pid = item.get("post_id", "")
        if item.get("score", 0) < MIN_RELEVANCE_SCORE:
            continue
        if item.get("suggested_action") == "skip":
            continue
        post = post_map.get(pid, {})
        opp = EngagementOpportunity(
            platform="reddit",
            post_id=pid,
            url=post.get("url", ""),
            author=post.get("author", ""),
            content=f"{post.get('title', '')}\n{post.get('selftext', '')}".strip(),
            subreddit=post.get("subreddit"),
            score=float(item.get("score", 0)),
            rationale=item.get("rationale", ""),
            suggested_comment=item.get("suggested_comment", ""),
            suggested_action=item.get("suggested_action", "comment"),
        )
        opportunities.append(opp)

    return sorted(opportunities, key=lambda o: o.score, reverse=True)


# ---------------------------------------------------------------------------
# Auto-posting via PRAW (optional — requires credentials)
# ---------------------------------------------------------------------------

def _praw_client():
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD]):
        return None
    try:
        import praw
        return praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
            user_agent=USER_AGENT,
        )
    except Exception as e:
        logger.warning("PRAW init failed: %s", e)
        return None


def engage(
    opportunities: list[EngagementOpportunity],
    dry_run: bool = False,
) -> list[EngagementOpportunity]:
    """Post comments via PRAW. Falls back to skipping if credentials not set."""
    reddit = _praw_client()
    if not reddit:
        logger.info("Reddit posting credentials not set — skipping auto-post (links saved to report)")
        for opp in opportunities:
            opp.status = "pending"
        return opportunities

    candidates = [
        o for o in opportunities
        if o.suggested_action == "comment" and o.suggested_comment
    ][:MAX_COMMENTS_PER_RUN]

    posted = 0
    for opp in opportunities:
        if opp not in candidates:
            opp.status = "skipped"
            continue
        if dry_run:
            logger.info("[dry-run] Would comment on %s", opp.url)
            opp.status = "actioned"
            continue
        try:
            sub = reddit.submission(id=opp.post_id)
            sub.reply(opp.suggested_comment)
            opp.status = "actioned"
            posted += 1
            logger.info("Posted comment to %s", opp.url)
            if posted < len(candidates):
                time.sleep(POST_DELAY_SECONDS)
        except Exception as e:
            logger.error("Failed to post on %s: %s", opp.url, e)
            opp.status = "skipped"

    logger.info("Reddit: posted %d comment(s)", posted)
    return opportunities


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover(dry_run: bool = False) -> list[EngagementOpportunity]:
    """Find and score Reddit posts. No credentials required."""
    seen = _load_seen()
    posts = _fetch_posts(seen)
    if not posts:
        return []

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    opportunities = _score_and_draft(posts, client)

    if not dry_run:
        for p in posts:
            seen.add(p["id"])
        _save_seen(seen)

    logger.info("Reddit: %d opportunities scored ≥ %.0f", len(opportunities), MIN_RELEVANCE_SCORE)
    return opportunities
