"""
IvyEdge — TikTok Engagement Agent

Discovers TikTok videos about topics IvyEdge cares about, scores them with
Claude, and queues suggested comments for manual posting.

TikTok API reality:
  - The official TikTok API supports posting videos only (no content discovery).
  - This module uses the unofficial TikTokApi library (pip install TikTokApi)
    which uses a headless Playwright browser. It reads public data only.
  - Commenting / liking still requires manual action in the TikTok app.
  - If you get approved for the TikTok Research API (apply at
    developers.tiktok.com/products/research-api), swap in _research_api_search()
    below to replace the unofficial calls.

Setup:
    pip install TikTokApi playwright
    python -m playwright install chromium

No additional .env keys required for discovery.
Optional: TIKTOK_MS_TOKEN=...  (session token — improves results, see README)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

from platform_agents import EngagementOpportunity

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger("ivyedge.tiktok")

MIN_RELEVANCE_SCORE = 5.5
MAX_VIDEOS_PER_RUN  = 25
SEEN_LOG = Path(__file__).parent.parent / "engagement_log" / "tiktok_seen.json"

# Hashtags to search — TikTok-specific (shorter, trend-aware)
HASHTAGS = [
    "freelancefinance",
    "selfemployed",
    "1099life",
    "girlboss",
    "womenentrepreneurs",
    "creditbuilding",
    "personalfinance",
    "freelancerlife",
    "solopreneur",
    "careergap",
    "sidehustle",
    "womeninbusiness",
    "financetok",
    "moneytok",
    "creditscorehelp",
    "gigseconomy",
    "wfh",
    "returntowork",
]

# Keywords for caption-level filtering before sending to Claude
SIGNAL_KEYWORDS = [
    "1099", "freelance", "self employed", "self-employed", "gig work",
    "career gap", "credit score", "loan denied", "side hustle", "contractor",
    "variable income", "non traditional", "career break",
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
# TikTok data fetching (unofficial TikTokApi)
# ---------------------------------------------------------------------------

async def _fetch_videos_async(seen: set[str]) -> list[dict]:
    try:
        from TikTokApi import TikTokApi
    except ImportError:
        logger.error(
            "TikTokApi not installed. Run: pip install TikTokApi && python -m playwright install chromium"
        )
        return []

    ms_token = os.getenv("TIKTOK_MS_TOKEN")
    videos: list[dict] = []
    seen_urls: set[str] = set()

    async with TikTokApi() as api:
        session_kwargs = dict(
            num_sessions=1,
            sleep_after=5,
            headless=True,
            browser="webkit",   # webkit has better TikTok success rate than chromium
        )
        if ms_token:
            session_kwargs["ms_tokens"] = [ms_token]
        try:
            await api.create_sessions(**session_kwargs)
        except Exception as e:
            logger.warning("TikTok session creation failed: %s", e)
            return []

        for hashtag in HASHTAGS:
            if len(videos) >= MAX_VIDEOS_PER_RUN:
                break
            try:
                tag = api.hashtag(name=hashtag)
                async for video in tag.videos(count=8):
                    vid_id  = str(video.id)
                    vid_url = f"https://www.tiktok.com/@{video.author.username}/video/{vid_id}"
                    caption = getattr(video, "desc", "") or ""

                    if vid_id in seen or vid_url in seen_urls:
                        continue
                    if not _has_signal(caption) and not _has_signal(hashtag):
                        continue

                    videos.append({
                        "id":       vid_id,
                        "author":   video.author.username,
                        "caption":  caption[:600],
                        "url":      vid_url,
                        "hashtag":  hashtag,
                        "plays":    getattr(video.stats, "playCount", 0),
                        "likes":    getattr(video.stats, "diggCount", 0),
                        "comments": getattr(video.stats, "commentCount", 0),
                    })
                    seen_urls.add(vid_url)
                    if len(videos) >= MAX_VIDEOS_PER_RUN:
                        break
            except Exception as e:
                logger.warning("TikTok hashtag #%s failed: %s", hashtag, e)

    logger.info("TikTok: fetched %d candidate videos", len(videos))
    return videos


def _fetch_videos(seen: set[str]) -> list[dict]:
    return asyncio.run(_fetch_videos_async(seen))


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

TikTok comment norms:
- Short: 1-2 sentences max
- Direct, warm, and specific to the video
- No links, no product names, no "we're building something"
- Can say "As someone who works in fintech..." to signal credibility
- Should feel native to TikTok — not corporate, not salesy
- Emojis are fine if they fit the tone"""


def _score_and_draft(videos: list[dict], client: anthropic.Anthropic) -> list[EngagementOpportunity]:
    if not videos:
        return []

    videos_text = "\n\n".join(
        f"VIDEO {i+1} (id={v['id']}, @{v['author']}, #{v['hashtag']}, "
        f"{v['plays']:,} plays, {v['likes']:,} likes):\n{v['caption'] or '(no caption)'}"
        for i, v in enumerate(videos)
    )

    prompt = f"""Below are {len(videos)} TikTok videos found via hashtag search.

For each, output:
{{
  "video_id": "<id>",
  "score": <0-10 float>,
  "rationale": "<one sentence>",
  "suggested_comment": "<if score >= 6.5: ready-to-post TikTok comment, else empty string>"
}}

JSON array only. No prose, no markdown fences.

High scores (≥6.5): Creator is sharing a real experience with 1099/freelance income, credit gaps,
career breaks, loan denials, or variable-income financial struggles. Our comment adds real value.

Low scores (<6.5): Generic finance content, brand accounts, purely motivational with no financial
substance, or topics where we can't add anything specific.

VIDEOS:
{videos_text}"""

    msg = client.messages.create(
        model=os.getenv("IVYEDGE_MODEL", "claude-sonnet-4-6"),
        max_tokens=2500,
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
        logger.warning("Claude returned unparseable JSON for TikTok scoring")
        return []

    video_map = {v["id"]: v for v in videos}
    opportunities = []
    for item in scored:
        vid_id = item.get("video_id", "")
        if item.get("score", 0) < MIN_RELEVANCE_SCORE:
            continue
        v = video_map.get(vid_id, {})
        opp = EngagementOpportunity(
            platform="tiktok",
            post_id=vid_id,
            url=v.get("url", ""),
            author=v.get("author", ""),
            content=v.get("caption", ""),
            hashtags=[v.get("hashtag", "")],
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
    """Discover relevant TikTok videos and return scored opportunities."""
    seen = _load_seen()

    try:
        videos = _fetch_videos(seen)
    except Exception as e:
        logger.error("TikTok fetch failed: %s", e)
        return []

    if not videos:
        logger.info("TikTok: no new videos found")
        return []

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    opportunities = _score_and_draft(videos, client)

    if not dry_run:
        for v in videos:
            seen.add(v["id"])
        _save_seen(seen)

    logger.info("TikTok: %d videos worth engaging with", len(opportunities))
    return opportunities
