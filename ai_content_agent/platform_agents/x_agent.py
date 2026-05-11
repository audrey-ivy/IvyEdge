"""
IvyEdge — X (Twitter) Engagement Agent (Playwright)

Discovers posts on X via keyword search using a headless Playwright browser
with your session cookies. Free, no API credentials needed.

X's internal API requires a JS-generated transaction ID on every request,
so direct HTTP calls are blocked (403). Playwright runs a real browser that
handles X's anti-bot JS automatically.

Setup:
    pip install playwright
    python -m playwright install chromium

First-time cookie setup (do this once):
    1. Log in to x.com in your browser
    2. Open DevTools (F12) → Application → Cookies → https://x.com
    3. Copy the values for: auth_token, ct0
    4. Add to your .env:
         X_AUTH_TOKEN=...
         X_CT0=...
    5. Run: python -m platform_agents.x_agent --setup
"""

from __future__ import annotations

import asyncio
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

logger = logging.getLogger("ivyedge.x")

MIN_RELEVANCE_SCORE = 6.0
MAX_POSTS_PER_RUN   = 30
COOKIES_FILE = Path(__file__).parent.parent / "engagement_log" / "x_cookies.json"
SEEN_LOG     = Path(__file__).parent.parent / "engagement_log" / "x_seen.json"

SEARCH_QUERIES = [
    # Pillar 1 — non-traditional income & credit
    "1099 income credit OR loan",
    "freelance income denied mortgage OR loan",
    "self employed credit score",
    "career gap credit OR loan",
    "gig worker loan denied",
    "variable income bank OR mortgage",
    "non traditional income finance",
    "side hustle income credit",
    "maternity leave credit OR loan",
    "independent contractor income loan",
    # Pillar 6 — workplace flexibility & women in workforce
    "4 day work week women",
    "return to office women quit",
    "RTO mandate women leaving",
    "remote work caregiving women",
    "flexible work women workforce",
    "paid parental leave US",
    "caregiver penalty career women",
    "women leaving workforce childcare",
    "4 day week productivity",
    "student loans women workforce",
]

SIGNAL_KEYWORDS = [
    # Pillar 1
    "1099", "freelance", "self employed", "self-employed", "gig work",
    "career gap", "credit score", "loan denied", "side hustle", "contractor",
    "variable income", "non traditional", "career break", "mortgage denied",
    # Pillar 6
    "4 day", "four day", "remote work", "work from home", "RTO",
    "return to office", "parental leave", "maternity", "caregiver",
    "childcare", "women leaving", "flexible work",
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


def _load_cookies() -> Optional[dict]:
    if COOKIES_FILE.exists():
        return json.loads(COOKIES_FILE.read_text(encoding="utf-8"))
    auth_token = os.getenv("X_AUTH_TOKEN", "")
    ct0        = os.getenv("X_CT0", "")
    if auth_token and ct0:
        cookies = {"auth_token": auth_token, "ct0": ct0}
        if os.getenv("X_GUEST_ID"):
            cookies["guest_id"] = os.getenv("X_GUEST_ID")
        return cookies
    return None


def _save_cookies(cookies: dict) -> None:
    COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOKIES_FILE.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
    logger.info("X cookies saved to %s", COOKIES_FILE)


# ---------------------------------------------------------------------------
# Playwright fetch
# ---------------------------------------------------------------------------

async def _fetch_posts_async(seen: set[str]) -> list[dict]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.error("playwright not installed — run: pip install playwright && python -m playwright install chromium")
        return []

    cookies = _load_cookies()
    if not cookies:
        logger.error("X agent: no cookies. Add X_AUTH_TOKEN + X_CT0 to .env then run --setup")
        return []

    posts: list[dict] = []
    seen_ids: set[str] = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        await context.add_cookies([
            {"name": "auth_token", "value": cookies["auth_token"], "domain": ".x.com", "path": "/"},
            {"name": "ct0",        "value": cookies["ct0"],        "domain": ".x.com", "path": "/"},
        ])
        if cookies.get("guest_id"):
            await context.add_cookies([
                {"name": "guest_id", "value": cookies["guest_id"], "domain": ".x.com", "path": "/"},
            ])

        page = await context.new_page()
        tweet_responses: list[dict] = []

        async def handle_response(response):
            if "search/adaptive" in response.url or "SearchTimeline" in response.url:
                try:
                    body = await response.json()
                    tweet_responses.append(body)
                except Exception:
                    pass

        page.on("response", handle_response)

        for query in SEARCH_QUERIES:
            if len(posts) >= MAX_POSTS_PER_RUN:
                break
            tweet_responses.clear()
            search_url = (
                f"https://x.com/search?q={query}+lang%3Aen+-filter%3Aretweets"
                f"&src=typed_query&f=live"
            )
            try:
                await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(5)

                for resp_data in tweet_responses:
                    _extract_tweets_from_response(resp_data, posts, seen, seen_ids, query)
                    if len(posts) >= MAX_POSTS_PER_RUN:
                        break

                if not tweet_responses:
                    dom_posts = await _extract_tweets_from_dom(page, query, seen, seen_ids)
                    posts.extend(dom_posts)

                logger.info("X query %r: %d posts so far", query, len(posts))
                await asyncio.sleep(2)

            except Exception as e:
                logger.warning("X search failed for %r: %s", query, e)
                await asyncio.sleep(5)

        await browser.close()

    logger.info("X: fetched %d candidate posts", len(posts))
    return posts[:MAX_POSTS_PER_RUN]


def _extract_tweets_from_response(data, posts, seen, seen_ids, query):
    # adaptive.json format
    tweet_map = data.get("globalObjects", {}).get("tweets", {})
    user_map  = data.get("globalObjects", {}).get("users", {})
    for tid, tw in tweet_map.items():
        if tid in seen or tid in seen_ids:
            continue
        text = tw.get("full_text") or tw.get("text") or ""
        if not _has_signal(text) and not _has_signal(query):
            continue
        uid    = str(tw.get("user_id_str", ""))
        user   = user_map.get(uid, {})
        screen = user.get("screen_name", "")
        url    = f"https://x.com/{screen}/status/{tid}" if screen else f"https://x.com/i/web/status/{tid}"
        posts.append({
            "id":      tid,
            "url":     url,
            "author":  screen,
            "text":    text[:600],
            "likes":   tw.get("favorite_count", 0),
            "reposts": tw.get("retweet_count", 0),
            "replies": tw.get("reply_count", 0),
        })
        seen_ids.add(tid)

    # GraphQL SearchTimeline format
    try:
        instructions = (
            data.get("data", {})
                .get("search_by_raw_query", {})
                .get("search_timeline", {})
                .get("timeline", {})
                .get("instructions", [])
        )
        for instr in instructions:
            for entry in instr.get("entries", []):
                result = (entry.get("content", {})
                               .get("itemContent", {})
                               .get("tweet_results", {})
                               .get("result", {}))
                _parse_graphql_tweet(result, posts, seen, seen_ids, query)
    except Exception:
        pass


def _parse_graphql_tweet(result, posts, seen, seen_ids, query):
    try:
        legacy = result.get("legacy", {})
        tid    = legacy.get("id_str", "")
        if not tid or tid in seen or tid in seen_ids:
            return
        text = legacy.get("full_text", "")
        if not _has_signal(text) and not _has_signal(query):
            return
        screen = (result.get("core", {})
                        .get("user_results", {})
                        .get("result", {})
                        .get("legacy", {})
                        .get("screen_name", ""))
        url    = f"https://x.com/{screen}/status/{tid}" if screen else f"https://x.com/i/web/status/{tid}"
        posts.append({
            "id":      tid,
            "url":     url,
            "author":  screen,
            "text":    text[:600],
            "likes":   legacy.get("favorite_count", 0),
            "reposts": legacy.get("retweet_count", 0),
            "replies": legacy.get("reply_count", 0),
        })
        seen_ids.add(tid)
    except Exception:
        pass


async def _extract_tweets_from_dom(page, query, seen, seen_ids):
    posts = []
    try:
        articles = await page.query_selector_all('article[data-testid="tweet"]')
        for article in articles[:10]:
            try:
                text_el = await article.query_selector('[data-testid="tweetText"]')
                text    = await text_el.inner_text() if text_el else ""
                if not _has_signal(text) and not _has_signal(query):
                    continue
                link_el = await article.query_selector('a[href*="/status/"]')
                href    = await link_el.get_attribute("href") if link_el else ""
                # href is either "/username/status/id" or full URL
                path    = href.replace("https://x.com", "").strip("/")
                parts   = path.split("/")
                # parts = ["username", "status", "id"]
                tid     = parts[-1] if len(parts) >= 3 else ""
                screen  = parts[0]  if len(parts) >= 3 and parts[0] not in ("", "i") else ""
                if not tid or not tid.isdigit() or tid in seen or tid in seen_ids:
                    continue
                url = f"https://x.com/{screen}/status/{tid}" if screen else f"https://x.com/i/web/status/{tid}"
                posts.append({
                    "id":      tid,
                    "url":     url,
                    "author":  screen,
                    "text":    text[:600],
                    "likes":   0, "reposts": 0, "replies": 0,
                })
                seen_ids.add(tid)
            except Exception:
                continue
    except Exception as e:
        logger.debug("DOM scrape failed: %s", e)
    return posts


def _fetch_posts(seen: set[str]) -> list[dict]:
    return asyncio.run(_fetch_posts_async(seen))


# ---------------------------------------------------------------------------
# Claude scoring + reply drafting
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the community engagement voice for IvyEdge, a pre-launch
consumer finance platform for women with non-traditional financial histories
(freelancers, career returners, entrepreneurs with variable income).

IvyEdge's thesis:
- Career gaps don't make you a credit risk
- 1099 income is real income
- High earners with non-W-2 income deserve products that match their reality
- Plain-language financial transparency is a baseline, not a feature
- Companies that offer 4-day weeks, remote work, and caregiver support keep women in the workforce

X reply norms:
- 1-2 sentences max
- Warm, specific, directly responsive
- No links, no product names
- Can reference working in fintech to signal credibility
- Emojis fine if they fit — don't force them

Reshare guidance:
- Flag posts worth quote-tweeting (compelling data, real stories, strong takes)"""


def _score_and_draft(posts: list[dict], client: anthropic.Anthropic) -> list[EngagementOpportunity]:
    if not posts:
        return []

    posts_text = "\n\n".join(
        f"POST {i+1} (id={p['id']}, @{p['author']}, "
        f"{p['likes']} likes, {p['reposts']} reposts):\n{p['text'] or '(no text)'}"
        for i, p in enumerate(posts)
    )

    prompt = f"""Below are {len(posts)} X posts found via keyword search.

For each, output:
{{
  "post_id": "<id>",
  "score": <0-10 float>,
  "rationale": "<one sentence>",
  "suggested_action": "reply" | "reshare" | "reply_and_reshare" | "skip",
  "suggested_comment": "<if action includes reply and score >= 6: ready-to-post reply. Empty string otherwise.>"
}}

JSON array only. No prose, no markdown fences.

High scores (≥6): Person sharing real experience with 1099/freelance income, career gaps,
credit issues, OR workplace flexibility/RTO/caregiving.
Reshare (quote-tweet) if it's a strong take or data point worth amplifying.

Low scores (<6): Generic content, brand posts, venting with no opening for engagement.

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
        logger.warning("Claude returned unparseable JSON for X scoring")
        return []

    post_map = {p["id"]: p for p in posts}
    opportunities = []
    for item in scored:
        pid = item.get("post_id", "")
        if item.get("score", 0) < MIN_RELEVANCE_SCORE:
            continue
        action = item.get("suggested_action", "reply")
        if action == "skip":
            continue
        p = post_map.get(pid, {})
        opp = EngagementOpportunity(
            platform="x",
            post_id=pid,
            url=p.get("url", ""),
            author=p.get("author", ""),
            content=p.get("text", ""),
            hashtags=[],
            score=float(item.get("score", 0)),
            rationale=item.get("rationale", ""),
            suggested_comment=item.get("suggested_comment", ""),
            suggested_action=action,
        )
        opportunities.append(opp)

    return sorted(opportunities, key=lambda o: o.score, reverse=True)


def discover(dry_run: bool = False) -> list[EngagementOpportunity]:
    """Search X for relevant posts and return scored opportunities."""
    seen = _load_seen()
    try:
        posts = _fetch_posts(seen)
    except Exception as e:
        logger.error("X fetch failed: %s", e)
        return []

    if not posts:
        logger.info("X: no new posts found")
        return []

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    opportunities = _score_and_draft(posts, client)

    if not dry_run:
        for p in posts:
            seen.add(p["id"])
        _save_seen(seen)

    logger.info("X: %d posts worth engaging with", len(opportunities))
    return opportunities


def setup_cookies() -> None:
    auth_token = os.getenv("X_AUTH_TOKEN", "").strip()
    ct0        = os.getenv("X_CT0", "").strip()
    if not auth_token or not ct0:
        print(
            "\nTo set up X cookies:\n"
            "1. Log in to x.com in your browser\n"
            "2. Open DevTools (F12) → Application → Cookies → https://x.com\n"
            "3. Copy 'auth_token' and 'ct0'\n"
            "4. Add to .env: X_AUTH_TOKEN=... and X_CT0=...\n"
            "5. Re-run: python -m platform_agents.x_agent --setup\n"
        )
        return
    _save_cookies({"auth_token": auth_token, "ct0": ct0})
    print(f"Cookies saved to {COOKIES_FILE}")


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        setup_cookies()
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        opps = discover(dry_run="--dry-run" in sys.argv)
        for o in opps:
            print(f"\n[{o.score:.1f}] @{o.author} — {o.url}")
            print(f"  {o.content[:120]}")
            if o.suggested_comment:
                print(f"  Reply: {o.suggested_comment}")
