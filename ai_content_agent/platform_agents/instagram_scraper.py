"""
IvyEdge — Instagram Hashtag Scraper

Discovers public Instagram posts via hashtag search using instaloader.
No Meta developer account or API token required — reads only public data.

Note: This uses Instagram's public web interface, which is against their ToS.
Risk is low since we only read (never write) and run conservatively once daily.
All engagement (liking, commenting) still happens manually in the app.

Setup:
    pip install instaloader
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

from platform_agents import EngagementOpportunity

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger("ivyedge.instagram_scraper")

MIN_RELEVANCE_SCORE = 6.0
MAX_POSTS_PER_RUN   = 25
POSTS_PER_HASHTAG   = 5    # conservative — avoids rate limiting
SLEEP_BETWEEN       = 4    # seconds between hashtag fetches

SEEN_LOG = Path(__file__).parent.parent / "engagement_log" / "instagram_scraper_seen.json"

HASHTAGS = [
    "freelancefinance",
    "selfemployedlife",
    "1099life",
    "careergap",
    "womenentrepreneurs",
    "creditbuilding",
    "solopreneur",
    "freelancerproblems",
    "returntowork",
    "gigeconomy",
    "womeninfinance",
    "independentcontractor",
    "sidehustlemoney",
    "mompreneurs",
    "financialindependence",
]

SIGNAL_KEYWORDS = [
    "1099", "freelance", "self employed", "self-employed", "gig",
    "career gap", "credit", "loan", "income", "contractor",
    "side hustle", "variable income", "career break", "denied",
    "mortgage", "unstable income", "non traditional",
]


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


def _has_signal(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SIGNAL_KEYWORDS)


# ---------------------------------------------------------------------------
# Instaloader fetch
# ---------------------------------------------------------------------------

def _fetch_posts(seen: set[str]) -> list[dict]:
    try:
        import instaloader
    except ImportError:
        logger.error("instaloader not installed — run: pip install instaloader")
        return []

    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        quiet=True,
    )

    posts: list[dict] = []
    seen_ids: set[str] = set()

    for hashtag in HASHTAGS:
        if len(posts) >= MAX_POSTS_PER_RUN:
            break
        try:
            tag = instaloader.Hashtag.from_name(L.context, hashtag)
            count = 0
            for post in tag.get_posts():
                if count >= POSTS_PER_HASHTAG:
                    break
                shortcode = post.shortcode
                if shortcode in seen or shortcode in seen_ids:
                    count += 1
                    continue
                caption = post.caption or ""
                if not _has_signal(caption) and not _has_signal(hashtag):
                    count += 1
                    continue

                posts.append({
                    "id":        shortcode,
                    "url":       f"https://www.instagram.com/p/{shortcode}/",
                    "author":    post.owner_username,
                    "caption":   caption[:600],
                    "hashtag":   hashtag,
                    "likes":     post.likes,
                    "comments":  post.comments,
                })
                seen_ids.add(shortcode)
                count += 1

                if len(posts) >= MAX_POSTS_PER_RUN:
                    break

            logger.info("Instagram #%s: %d posts", hashtag, count)
            time.sleep(SLEEP_BETWEEN)

        except Exception as e:
            logger.warning("Instagram hashtag #%s failed: %s", hashtag, e)
            time.sleep(SLEEP_BETWEEN * 2)

    logger.info("Instagram: fetched %d candidate posts", len(posts))
    return posts


# ---------------------------------------------------------------------------
# Claude scoring + comment drafting
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the community engagement voice for IvyEdge, a pre-launch
consumer finance platform for women with non-traditional financial histories
(freelancers, career returners, entrepreneurs with variable income).

IvyEdge's thesis:
- Career gaps don't make you a credit risk
- 1099 income is real income
- High earners with non-W-2 income deserve products that match their reality
- Plain-language financial transparency is a baseline, not a feature

Instagram comment norms:
- 1-3 sentences, warm and specific to the post
- No links, no product names, nothing promotional
- Can reference working in fintech/finance to signal credibility
- Emoji are fine if they fit the tone
- Should feel like a genuine comment from a smart follower"""


def _score_and_draft(posts: list[dict], client: anthropic.Anthropic) -> list[EngagementOpportunity]:
    if not posts:
        return []

    posts_text = "\n\n".join(
        f"POST {i+1} (id={p['id']}, @{p['author']}, #{p['hashtag']}, "
        f"{p['likes']} likes, {p['comments']} comments):\n{p['caption'] or '(no caption)'}"
        for i, p in enumerate(posts)
    )

    prompt = f"""Below are {len(posts)} Instagram posts found via hashtag search.

For each, output:
{{
  "post_id": "<shortcode id>",
  "score": <0-10 float>,
  "rationale": "<one sentence>",
  "suggested_comment": "<if score >= 6: ready-to-post Instagram comment. Empty string otherwise.>"
}}

JSON array only. No prose, no markdown fences.

High scores (≥6): Creator is sharing a real experience with freelance income,
career gaps, credit struggles, loan denials, or variable-income financial stress.
Our comment adds genuine value or validation.

Low scores (<6): Generic motivational content, brand posts, already has great
engagement that doesn't need us, or topics we can't add anything specific to.

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
        logger.warning("Claude returned unparseable JSON for Instagram scoring")
        return []

    post_map = {p["id"]: p for p in posts}
    opportunities = []
    for item in scored:
        pid = item.get("post_id", "")
        if item.get("score", 0) < MIN_RELEVANCE_SCORE:
            continue
        p = post_map.get(pid, {})
        opp = EngagementOpportunity(
            platform="instagram",
            post_id=pid,
            url=p.get("url", ""),
            author=p.get("author", ""),
            content=p.get("caption", ""),
            hashtags=[p.get("hashtag", "")],
            score=float(item.get("score", 0)),
            rationale=item.get("rationale", ""),
            suggested_comment=item.get("suggested_comment", ""),
            suggested_action="comment",
        )
        opportunities.append(opp)

    return sorted(opportunities, key=lambda o: o.score, reverse=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover(dry_run: bool = False) -> list[EngagementOpportunity]:
    """Scrape Instagram hashtags and return scored opportunities."""
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

    logger.info("Instagram: %d posts worth engaging with", len(opportunities))
    return opportunities
