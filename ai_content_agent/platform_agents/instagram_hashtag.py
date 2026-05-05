"""
IvyEdge — Instagram Hashtag Monitor

Searches relevant hashtags via the Meta Graph API, scores posts for
relevance using Claude, and writes a review queue for manual engagement.

The Meta Graph API does NOT allow liking or commenting on other users'
posts — that requires human action. This module finds the right posts
and drafts what to say; you do the tapping.

Limits:
  - 30 unique hashtags searchable per IG user per week (Meta enforces this).
  - Up to 10 posts returned per hashtag per call.
  - We rotate through hashtags across runs to stay under the cap.
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

logger = logging.getLogger("ivyedge.ig_hashtag")

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
IG_USER_ID        = os.getenv("IG_USER_ID", "")
GRAPH_BASE        = "https://graph.facebook.com/v19.0"

# Hashtags rotated across weekly runs (30-hashtag cap per week per user)
HASHTAG_POOL = [
    "freelancefinance",
    "freelancermoney",
    "selfemployedlife",
    "1099life",
    "careergap",
    "womenentrepreneurs",
    "womeninfinance",
    "creditbuilding",
    "solopreneur",
    "womenbusiness",
    "mompreneurs",
    "returntowork",
    "freelancerproblems",
    "gigeconomy",
    "independentcontractor",
    "femaleentrepreneur",
    "womenownedbusiness",
    "sidehustlemoney",
    "alternativeincome",
    "creditscorehelp",
]

# State file tracks which hashtags were searched this week to stay under cap
HASHTAG_STATE_FILE = Path(__file__).parent.parent / "engagement_log" / "ig_hashtag_state.json"

MIN_RELEVANCE_SCORE = 6.0   # Only queue posts scoring ≥ this (0–10)
MAX_POSTS_PER_RUN   = 20    # Cap total posts sent to Claude per run


def _load_hashtag_state() -> dict:
    if HASHTAG_STATE_FILE.exists():
        return json.loads(HASHTAG_STATE_FILE.read_text(encoding="utf-8"))
    return {"week": "", "searched": [], "seen_ids": []}


def _save_hashtag_state(state: dict) -> None:
    HASHTAG_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASHTAG_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _current_week() -> str:
    """ISO week string like '2026-W18'."""
    d = datetime.now(timezone.utc)
    return f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"


def _next_hashtags(state: dict, n: int = 10) -> list[str]:
    """Pick the next n hashtags not yet searched this week."""
    searched = set(state.get("searched", []))
    remaining = [h for h in HASHTAG_POOL if h not in searched]
    return remaining[:n]


# ---------------------------------------------------------------------------
# Meta Graph API helpers
# ---------------------------------------------------------------------------

def _get_hashtag_id(hashtag: str) -> Optional[str]:
    url = f"{GRAPH_BASE}/ig_hashtag_search"
    resp = requests.get(url, params={
        "user_id":     IG_USER_ID,
        "q":           hashtag,
        "access_token": META_ACCESS_TOKEN,
    }, timeout=15)
    if not resp.ok:
        logger.warning("Hashtag lookup failed for #%s: %s", hashtag, resp.text[:200])
        return None
    data = resp.json().get("data", [])
    return data[0]["id"] if data else None


def _fetch_hashtag_posts(hashtag_id: str, seen_ids: set) -> list[dict]:
    url = f"{GRAPH_BASE}/{hashtag_id}/recent_media"
    resp = requests.get(url, params={
        "user_id": IG_USER_ID,
        "fields":  "id,caption,media_type,permalink,timestamp",
        "access_token": META_ACCESS_TOKEN,
    }, timeout=15)
    if not resp.ok:
        logger.warning("recent_media failed for hashtag_id %s: %s", hashtag_id, resp.text[:200])
        return []
    posts = []
    for item in resp.json().get("data", []):
        if item.get("id") in seen_ids:
            continue
        if not item.get("caption"):
            continue
        posts.append(item)
    return posts


# ---------------------------------------------------------------------------
# Claude scoring + comment generation
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the community engagement strategist for IvyEdge, a pre-launch
consumer finance platform built for women with non-traditional financial histories
(freelancers, career returners, entrepreneurs with variable income).

IvyEdge's mission: prove that women are underserved by the current financial system
and build an audience before launch. Right now the goal is to find conversations
where IvyEdge's perspective adds genuine value.

IvyEdge's core thesis:
- Career gaps don't make you a credit risk
- 1099 income is real income
- Five years of business history is an arbitrary threshold
- High earners with non-W-2 income deserve products that match their reality
- Plain-language financial transparency is a baseline, not a feature"""


def _score_and_draft(posts: list[dict], client: anthropic.Anthropic) -> list[EngagementOpportunity]:
    if not posts:
        return []

    posts_text = "\n\n".join(
        f"POST {i+1} (id={p['id']}, url={p.get('permalink','?')}):\n{p.get('caption','')[:500]}"
        for i, p in enumerate(posts)
    )

    prompt = f"""Below are {len(posts)} Instagram posts found via hashtag search.

For each post, output a JSON object in this exact format:
{{
  "post_id": "<id from the post header>",
  "score": <0-10 float — how relevant/valuable is engaging here for IvyEdge>,
  "rationale": "<one sentence: why this is or isn't worth engaging>",
  "suggested_comment": "<if score >= 6: a genuine, helpful Instagram comment, 1-3 sentences, no promotional language, no product mentions, sounds like a smart friend who works in finance — leave blank string if score < 6>"
}}

Output ONLY a JSON array of these objects. No prose, no markdown fences.

Scoring criteria (higher = more relevant):
- Person is describing a problem IvyEdge's thesis addresses (1099/gig income, credit gaps, career breaks)
- Post has genuine engagement or reach
- Our comment would add real value, not just validate
- The person seems like Maya (freelancer), Priya (career returner), Carmen (entrepreneur), or Dominique (high-earner)

Low scores (< 6) for:
- Generic finance content not connected to our thesis
- Brand/promotional posts
- Posts from large financial institutions
- Posts we've already commented on

POSTS:
{posts_text}"""

    msg = client.messages.create(
        model=os.getenv("IVYEDGE_MODEL", "claude-sonnet-4-6"),
        max_tokens=2000,
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
        logger.warning("Claude returned unparseable JSON for IG scoring")
        return []

    post_map = {p["id"]: p for p in posts}
    opportunities = []
    for item in scored:
        pid = item.get("post_id", "")
        if item.get("score", 0) < MIN_RELEVANCE_SCORE:
            continue
        post = post_map.get(pid, {})
        opp = EngagementOpportunity(
            platform="instagram",
            post_id=pid,
            url=post.get("permalink", ""),
            author="(instagram user)",
            content=post.get("caption", "")[:500],
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
    """
    Search Instagram hashtags and return scored engagement opportunities.
    Updates the hashtag rotation state unless dry_run=True.
    """
    if not META_ACCESS_TOKEN or not IG_USER_ID:
        logger.error("META_ACCESS_TOKEN or IG_USER_ID not set — skipping Instagram")
        return []

    state = _load_hashtag_state()
    week  = _current_week()

    # Reset rotation at the start of a new week
    if state.get("week") != week:
        state = {"week": week, "searched": [], "seen_ids": []}
        logger.info("New week (%s) — resetting hashtag rotation", week)

    hashtags_to_search = _next_hashtags(state, n=10)
    if not hashtags_to_search:
        logger.info("All %d hashtags already searched this week — done", len(HASHTAG_POOL))
        return []

    logger.info("Searching %d Instagram hashtags: %s", len(hashtags_to_search),
                ", ".join(f"#{h}" for h in hashtags_to_search))

    seen_ids = set(state.get("seen_ids", []))
    all_posts: list[dict] = []

    for hashtag in hashtags_to_search:
        hid = _get_hashtag_id(hashtag)
        if not hid:
            continue
        posts = _fetch_hashtag_posts(hid, seen_ids)
        logger.info("  #%s → %d new posts", hashtag, len(posts))
        all_posts.extend(posts)
        if not dry_run:
            state["searched"].append(hashtag)
            for p in posts:
                seen_ids.add(p["id"])

    state["seen_ids"] = list(seen_ids)[-500:]   # keep last 500 to avoid unbounded growth
    if not dry_run:
        _save_hashtag_state(state)

    if not all_posts:
        logger.info("No new Instagram posts found this run")
        return []

    # Cap what we send to Claude
    all_posts = all_posts[:MAX_POSTS_PER_RUN]
    logger.info("Scoring %d posts with Claude...", len(all_posts))

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    opportunities = _score_and_draft(all_posts, client)
    logger.info("Instagram: %d opportunities scored ≥ %.0f", len(opportunities), MIN_RELEVANCE_SCORE)
    return opportunities
