"""
IvyEdge — Reddit Engagement Agent

Searches relevant subreddits for posts about the problems IvyEdge addresses,
scores them for relevance using Claude, drafts helpful on-brand replies, and
either queues them for review or posts them directly (--auto).

Reddit is the highest-signal channel for IvyEdge's pre-launch goal: the target
audience (freelancers, career returners, women entrepreneurs) openly describes
their exact pain points in these communities.

Rules we follow:
  - Every comment identifies the account as AI-assisted (per Reddit norms)
  - No product pitches, no links to IvyEdge (pre-launch — nothing to link to)
  - Genuine helpfulness only — bad comments hurt more than silence
  - Rate limited: max 10 comments per run, 60s between posts (Reddit API)
  - We never comment twice on the same post (tracked in engagement_log/)

Required in .env:
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
import praw
from dotenv import load_dotenv

from platform_agents import EngagementOpportunity

load_dotenv()

logger = logging.getLogger("ivyedge.reddit")

REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USERNAME      = os.getenv("REDDIT_USERNAME", "JoinIvyEdge")
REDDIT_PASSWORD      = os.getenv("REDDIT_PASSWORD", "")
REDDIT_USER_AGENT    = f"IvyEdgeBot/1.0 by u/{REDDIT_USERNAME} — engagement agent for IvyEdge.com"

# Subreddits to monitor, in priority order
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
    "Entrepreneur",
    "loanoriginators",
    "CreditCards",
    "povertyfinance",
]

# Keywords that signal a post is worth engaging with
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
    "career break credit",
    "non traditional income",
    "1099 worker finance",
    "side hustle income credit",
    "maternity leave credit score",
]

MIN_RELEVANCE_SCORE  = 7.0   # Higher bar than Instagram — comments are permanent public record
MAX_POSTS_PER_RUN    = 30    # Posts fetched total; Claude scores them, fewer pass
MAX_COMMENTS_PER_RUN = 10    # Hard cap on comments posted in one run
POST_DELAY_SECONDS   = 60    # Reddit API rate limit buffer between posts

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


def _reddit_client() -> Optional[praw.Reddit]:
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD]):
        logger.error(
            "Reddit credentials missing — set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, "
            "REDDIT_USERNAME, REDDIT_PASSWORD in .env"
        )
        return None
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        user_agent=REDDIT_USER_AGENT,
    )


# ---------------------------------------------------------------------------
# Post fetching
# ---------------------------------------------------------------------------

def _fetch_posts(reddit: praw.Reddit, seen: set[str]) -> list[dict]:
    """Search subreddits for relevant posts. Returns raw dicts."""
    posts: list[dict] = []
    seen_urls: set[str] = set()

    for query in SEARCH_QUERIES:
        if len(posts) >= MAX_POSTS_PER_RUN:
            break
        try:
            for sub in reddit.subreddit("+".join(SUBREDDITS)).search(
                query, sort="new", time_filter="week", limit=5
            ):
                if sub.id in seen or sub.url in seen_urls:
                    continue
                if sub.score < 1:
                    continue
                posts.append({
                    "id":         sub.id,
                    "subreddit":  sub.subreddit.display_name,
                    "title":      sub.title,
                    "selftext":   sub.selftext[:800],
                    "url":        f"https://reddit.com{sub.permalink}",
                    "score":      sub.score,
                    "num_comments": sub.num_comments,
                    "created_utc": sub.created_utc,
                    "author":     str(sub.author) if sub.author else "[deleted]",
                })
                seen_urls.add(sub.url)
                if len(posts) >= MAX_POSTS_PER_RUN:
                    break
        except Exception as e:
            logger.warning("Reddit search failed for '%s': %s", query, e)

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

IvyEdge is pre-launch with nothing to sell yet. Comments must be:
  1. Genuinely useful — practical advice the person can act on today
  2. On-brand — reflects IvyEdge's POV (the system was built wrong, here's how to work within it)
  3. Never promotional — no links, no product mentions, no "we're building X"
  4. Transparent — end every comment with: "(AI-assisted account | IvyEdge)"
  5. Authentic — should sound like a smart friend who works in finance, not a press release"""


def _score_and_draft(posts: list[dict], client: anthropic.Anthropic) -> list[EngagementOpportunity]:
    if not posts:
        return []

    posts_text = "\n\n".join(
        f"POST {i+1} (id={p['id']}, r/{p['subreddit']}, score={p['score']}):\n"
        f"Title: {p['title']}\n"
        f"Body: {p['selftext'] or '(link post)'}"
        for i, p in enumerate(posts)
    )

    prompt = f"""Below are {len(posts)} Reddit posts found via keyword search.

For each, output a JSON object:
{{
  "post_id": "<id>",
  "score": <0-10 float>,
  "rationale": "<one sentence why this is/isn't worth engaging>",
  "suggested_action": "comment" | "upvote_only" | "skip",
  "suggested_comment": "<if action=comment: full comment text ready to post — include the (AI-assisted account | IvyEdge) disclosure at the end. Empty string otherwise.>"
}}

Output ONLY a JSON array. No prose, no markdown fences.

HIGH-score posts (≥7) have ALL of:
- OP is experiencing a real problem IvyEdge's thesis addresses
- Our comment would give concrete, useful advice
- Post is recent (within 7 days) and has real engagement
- No competing authoritative answers already covering what we'd say

LOW-score posts (<7):
- Vague questions we can't add real value to
- Already have great answers from credible sources
- Promotional, brand, or bot posts
- Topics outside IvyEdge's expertise

POSTS:
{posts_text}"""

    msg = client.messages.create(
        model=os.getenv("IVYEDGE_MODEL", "claude-sonnet-4-6"),
        max_tokens=3000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]

    try:
        scored = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude returned unparseable JSON for Reddit scoring")
        return []

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
# Posting
# ---------------------------------------------------------------------------

def _post_comment(reddit: praw.Reddit, submission_id: str, text: str) -> bool:
    try:
        sub = reddit.submission(id=submission_id)
        sub.reply(text)
        logger.info("Posted comment to reddit.com/r/%s (id=%s)", sub.subreddit, submission_id)
        return True
    except Exception as e:
        logger.error("Failed to post Reddit comment on %s: %s", submission_id, e)
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover(dry_run: bool = False) -> list[EngagementOpportunity]:
    """Find and score Reddit posts. Does NOT post comments (use engage() for that)."""
    reddit = _reddit_client()
    if not reddit:
        return []

    seen = _load_seen()
    posts = _fetch_posts(reddit, seen)
    if not posts:
        return []

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    opportunities = _score_and_draft(posts, client)

    # Mark all fetched posts as seen regardless of score (don't re-evaluate same posts)
    if not dry_run:
        for p in posts:
            seen.add(p["id"])
        _save_seen(seen)

    logger.info("Reddit: %d opportunities scored ≥ %.0f", len(opportunities), MIN_RELEVANCE_SCORE)
    return opportunities


def engage(
    opportunities: list[EngagementOpportunity],
    dry_run: bool = False,
) -> list[EngagementOpportunity]:
    """
    Post comments for the given opportunities.
    Respects MAX_COMMENTS_PER_RUN and rate limits.
    Returns the list with status updated to "actioned" or "skipped".
    """
    reddit = _reddit_client()
    if not reddit:
        logger.error("Cannot post Reddit comments — credentials not set")
        return opportunities

    comment_candidates = [
        o for o in opportunities
        if o.suggested_action == "comment" and o.suggested_comment
    ][:MAX_COMMENTS_PER_RUN]

    posted = 0
    for opp in opportunities:
        if opp not in comment_candidates:
            opp.status = "skipped"
            continue

        if dry_run:
            logger.info("[dry-run] Would comment on %s (score=%.1f)", opp.url, opp.score)
            logger.info("  Comment preview: %s", opp.suggested_comment[:120])
            opp.status = "actioned"
            continue

        success = _post_comment(reddit, opp.post_id, opp.suggested_comment)
        opp.status = "actioned" if success else "skipped"
        if success:
            posted += 1
            if posted < len(comment_candidates):
                time.sleep(POST_DELAY_SECONDS)

    logger.info("Reddit: posted %d comment(s)", posted)
    return opportunities
