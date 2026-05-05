"""
IvyEdge — Threads Monitor

The Threads API (via Meta Graph) does not support hashtag search or feed
discovery — it's primarily a publishing API. What we *can* do:

  1. Fetch replies to IvyEdge's own Threads posts
  2. Identify replies that deserve a response (questions, stories, disagreements)
  3. Draft Claude-powered replies for human review

This turns Threads into a two-way conversation rather than a broadcast channel,
which builds the engaged audience IvyEdge needs pre-launch.

Required in .env:
  META_ACCESS_TOKEN=...
  THREADS_USER_ID=...
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import requests
from dotenv import load_dotenv

from platform_agents import EngagementOpportunity

load_dotenv()

logger = logging.getLogger("ivyedge.threads")

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
THREADS_USER_ID   = os.getenv("THREADS_USER_ID", "")
THREADS_BASE      = "https://graph.threads.net/v1.0"

MIN_RELEVANCE_SCORE = 6.0
SEEN_LOG = Path(__file__).parent.parent / "engagement_log" / "threads_seen.json"


def _load_seen() -> set[str]:
    if SEEN_LOG.exists():
        data = json.loads(SEEN_LOG.read_text(encoding="utf-8"))
        return set(data.get("seen_ids", []))
    return set()


def _save_seen(seen: set[str]) -> None:
    SEEN_LOG.parent.mkdir(parents=True, exist_ok=True)
    SEEN_LOG.write_text(
        json.dumps({"seen_ids": list(seen)[-500:]}, indent=2),
        encoding="utf-8",
    )


def _check_credentials() -> bool:
    if not META_ACCESS_TOKEN or not THREADS_USER_ID:
        logger.error("META_ACCESS_TOKEN or THREADS_USER_ID not set — skipping Threads")
        return False
    return True


# ---------------------------------------------------------------------------
# Fetch your own recent Threads posts and their replies
# ---------------------------------------------------------------------------

def _fetch_my_posts(limit: int = 20) -> list[dict]:
    url = f"{THREADS_BASE}/{THREADS_USER_ID}/threads"
    resp = requests.get(url, params={
        "fields":       "id,text,timestamp,permalink",
        "limit":        limit,
        "access_token": META_ACCESS_TOKEN,
    }, timeout=15)
    if not resp.ok:
        logger.warning("Failed to fetch Threads posts: %s", resp.text[:200])
        return []
    return resp.json().get("data", [])


def _fetch_replies(thread_id: str) -> list[dict]:
    """Fetch direct replies to a specific Threads post."""
    url = f"{THREADS_BASE}/{thread_id}/replies"
    resp = requests.get(url, params={
        "fields":       "id,text,timestamp,username",
        "access_token": META_ACCESS_TOKEN,
    }, timeout=15)
    if not resp.ok:
        return []
    return resp.json().get("data", [])


def _fetch_all_replies(seen: set[str]) -> list[dict]:
    """Collect all unseen replies across recent posts."""
    posts = _fetch_my_posts()
    all_replies = []
    for post in posts:
        replies = _fetch_replies(post["id"])
        for r in replies:
            if r.get("id") in seen or not r.get("text"):
                continue
            r["parent_post_text"] = post.get("text", "")[:300]
            r["parent_post_id"]   = post["id"]
            all_replies.append(r)
    return all_replies


# ---------------------------------------------------------------------------
# Claude scoring + reply drafting
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the community voice for IvyEdge, a pre-launch consumer finance
platform for women with non-traditional financial histories.

When responding to replies on IvyEdge's Threads posts:
- Prioritize: questions, personal stories, disagreements that need nuance
- Skip: pure validation ("great post!"), obvious spam, irrelevant comments
- Replies should be warm, specific, and genuinely helpful
- Never mention unreleased products
- Max 3 sentences — Threads is a conversational medium
- You can invite them to join the waitlist or newsletter if the conversation calls for it"""


def _score_and_draft(replies: list[dict], client: anthropic.Anthropic) -> list[EngagementOpportunity]:
    if not replies:
        return []

    replies_text = "\n\n".join(
        f"REPLY {i+1} (id={r['id']}, from=@{r.get('username','?')}):\n"
        f"[On our post: \"{r.get('parent_post_text','')[:150]}...\"]\n"
        f"Their reply: {r.get('text','')}"
        for i, r in enumerate(replies)
    )

    prompt = f"""Below are {len(replies)} replies to IvyEdge's Threads posts.

For each, output:
{{
  "reply_id": "<id>",
  "score": <0-10>,
  "rationale": "<one sentence>",
  "suggested_reply": "<ready-to-post reply text if score >= 6, else empty string>"
}}

JSON array only. No prose, no fences.

{replies_text}"""

    msg = client.messages.create(
        model=os.getenv("IVYEDGE_MODEL", "claude-sonnet-4-6"),
        max_tokens=1500,
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
        logger.warning("Claude returned unparseable JSON for Threads scoring")
        return []

    reply_map = {r["id"]: r for r in replies}
    opportunities = []
    for item in scored:
        rid = item.get("reply_id", "")
        if item.get("score", 0) < MIN_RELEVANCE_SCORE:
            continue
        r = reply_map.get(rid, {})
        opp = EngagementOpportunity(
            platform="threads",
            post_id=rid,
            url=f"https://www.threads.net/t/{rid}",
            author=r.get("username", ""),
            content=r.get("text", ""),
            score=float(item.get("score", 0)),
            rationale=item.get("rationale", ""),
            suggested_comment=item.get("suggested_reply", ""),
            suggested_action="comment",
        )
        opportunities.append(opp)

    return sorted(opportunities, key=lambda o: o.score, reverse=True)


# ---------------------------------------------------------------------------
# Post a reply via Threads API
# ---------------------------------------------------------------------------

def post_reply(thread_id: str, text: str) -> Optional[str]:
    """Reply to a Threads post. Returns the new post URL or None."""
    if not _check_credentials():
        return None

    create_url = f"{THREADS_BASE}/{THREADS_USER_ID}/threads"
    resp = requests.post(create_url, data={
        "media_type":     "TEXT",
        "text":           text,
        "reply_to_id":    thread_id,
        "access_token":   META_ACCESS_TOKEN,
    }, timeout=15)
    if not resp.ok:
        logger.error("Threads reply creation failed: %s", resp.text[:200])
        return None

    container_id = resp.json().get("id")
    publish_url  = f"{THREADS_BASE}/{THREADS_USER_ID}/threads_publish"
    pub_resp = requests.post(publish_url, data={
        "creation_id":  container_id,
        "access_token": META_ACCESS_TOKEN,
    }, timeout=15)
    if not pub_resp.ok:
        logger.error("Threads reply publish failed: %s", pub_resp.text[:200])
        return None

    post_id = pub_resp.json().get("id", "")
    return f"https://www.threads.net/t/{post_id}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def discover(dry_run: bool = False) -> list[EngagementOpportunity]:
    """Fetch replies to IvyEdge's Threads posts and score them."""
    if not _check_credentials():
        return []

    seen = _load_seen()
    replies = _fetch_all_replies(seen)
    if not replies:
        logger.info("Threads: no new replies found")
        return []

    logger.info("Threads: scoring %d new replies with Claude", len(replies))
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    opportunities = _score_and_draft(replies, client)

    if not dry_run:
        for r in replies:
            seen.add(r["id"])
        _save_seen(seen)

    logger.info("Threads: %d replies worth responding to", len(opportunities))
    return opportunities
