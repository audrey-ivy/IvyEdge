"""
Competitive format analysis for IvyEdge.

For a given keyword, searches DuckDuckGo, fetches the top free (non-paywalled)
results, measures their structure (H1/H2/H3 counts, word count, lists, images),
then uses Claude to synthesize format recommendations for the IvyEdge post.

This runs as Phase 0 of the pipeline so the outline phase has concrete benchmarks
to hit rather than guessing at structure.
"""

import logging
import os
import re
import time
from urllib.parse import quote_plus

import anthropic
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ivyedge.competitor")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

PAYWALL_PHRASES = [
    "subscribe to read",
    "subscribe to continue reading",
    "this post is for paid subscribers",
    "become a paid subscriber",
    "upgrade your subscription",
    "paid subscribers only",
]

TARGET_FREE_RESULTS = 5
MAX_CANDIDATES = 20
REQUEST_DELAY = 1.2  # seconds between fetches — polite crawling


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _search_duckduckgo(query: str, n: int = MAX_CANDIDATES) -> list[str]:
    """Return up to n result URLs from DuckDuckGo HTML search."""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            headers=HEADERS,
            data={"q": query, "b": ""},
            timeout=12,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    urls: list[str] = []
    for a in soup.find_all("a", class_="result__a"):
        href = a.get("href", "")
        if href.startswith("http") and href not in urls:
            urls.append(href)
        if len(urls) >= n:
            break
    logger.info("DuckDuckGo returned %d candidates for '%s'", len(urls), query)
    return urls


# ---------------------------------------------------------------------------
# Page analysis
# ---------------------------------------------------------------------------

def _is_paywalled(soup: BeautifulSoup) -> bool:
    text = soup.get_text(" ", strip=True).lower()
    return any(phrase in text for phrase in PAYWALL_PHRASES)


def _main_content(soup: BeautifulSoup) -> BeautifulSoup:
    """Return the most content-rich element on the page."""
    for selector in [
        "article",
        {"class": re.compile(r"post-content|article-body|entry-content|body-text|markup")},
        "main",
    ]:
        el = soup.find(selector) if isinstance(selector, str) else soup.find(**{"attrs": selector})
        if el:
            return el
    return soup


def analyze_url(url: str) -> dict | None:
    """
    Fetch a URL, check for paywalls, and return structural metrics.
    Returns None if the page is paywalled, unreachable, or not useful.
    """
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if not resp.ok:
            logger.debug("Skipping %s — HTTP %s", url, resp.status_code)
            return None
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        logger.debug("Failed to fetch %s: %s", url, e)
        return None

    if _is_paywalled(soup):
        logger.debug("Skipping %s — paywalled", url)
        return None

    content = _main_content(soup)

    h1s = [h.get_text(" ", strip=True) for h in content.find_all("h1")]
    h2s = [h.get_text(" ", strip=True) for h in content.find_all("h2")]
    h3s = [h.get_text(" ", strip=True) for h in content.find_all("h3")]

    words = len(re.findall(r"\b\w+\b", content.get_text(" ", strip=True)))
    if words < 200:
        logger.debug("Skipping %s — too short (%d words)", url, words)
        return None

    bullet_lists = len(content.find_all("ul"))
    numbered_lists = len(content.find_all("ol"))
    images = len(content.find_all("img"))

    # Estimate average H2 section length in words
    section_lengths: list[int] = []
    current = 0
    for tag in content.find_all(["h2", "p"]):
        if tag.name == "h2":
            if current:
                section_lengths.append(current)
            current = 0
        else:
            current += len(re.findall(r"\b\w+\b", tag.get_text()))
    if current:
        section_lengths.append(current)
    avg_section_words = int(sum(section_lengths) / len(section_lengths)) if section_lengths else 0

    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else url

    return {
        "url": url,
        "title": page_title[:120],
        "word_count": words,
        "h1_count": len(h1s),
        "h2_count": len(h2s),
        "h3_count": len(h3s),
        "h2_titles": h2s[:8],
        "h3_titles": h3s[:6],
        "bullet_lists": bullet_lists,
        "numbered_lists": numbered_lists,
        "images": images,
        "avg_section_words": avg_section_words,
    }


# ---------------------------------------------------------------------------
# Claude synthesis
# ---------------------------------------------------------------------------

def synthesize_format_guidance(keyword: str, analyses: list[dict]) -> str:
    """Ask Claude to turn raw structural metrics into actionable format guidance."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    rows = ""
    for i, a in enumerate(analyses, 1):
        rows += (
            f"\n### Result {i}: {a['title']}\n"
            f"- URL: {a['url']}\n"
            f"- Word count: {a['word_count']:,}\n"
            f"- Structure: {a['h1_count']} H1 / {a['h2_count']} H2 / {a['h3_count']} H3\n"
            f"- Avg words per section: {a['avg_section_words']}\n"
            f"- Lists: {a['bullet_lists']} bullet, {a['numbered_lists']} numbered\n"
            f"- Images: {a['images']}\n"
        )
        if a["h2_titles"]:
            rows += f"- H2 titles: {' | '.join(a['h2_titles'][:6])}\n"
        if a["h3_titles"]:
            rows += f"- H3 titles: {' | '.join(a['h3_titles'][:4])}\n"

    prompt = f"""You are advising the IvyEdge editorial team on post structure.

We analyzed the top {len(analyses)} free (non-paywalled) results for the keyword:
**"{keyword}"**

Here is what we found:
{rows}

Based on these benchmarks, give the IvyEdge writer a concrete format brief. Be specific
and prescriptive — the writer will use this brief directly when outlining the post.

Output the following sections (markdown, no preamble):

## Recommended word count
[range, with rationale from the data]

## Heading structure
[exact recommended counts for H1 / H2 / H3, with guidance on what each level should do]

## Section-by-section outline template
[list each H2 section the post should include, with a 1-line description of what it covers
and a suggested word budget. Base this on the H2 patterns you see across competitors.]

## List usage
[should we use bullet lists? numbered lists? how many? what for?]

## Images / visuals
[what the data suggests about image use for this keyword]

## What the competition is missing
[1-2 angles or structural choices IvyEdge can make to differentiate — go deeper,
be more direct, serve our persona better than the generic results do]
"""

    msg = client.messages.create(
        model=os.getenv("IVYEDGE_MODEL", "claude-sonnet-4-6"),
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_competitor_analysis(keyword: str, extra_query: str = "") -> tuple[list[dict], str]:
    """
    Search for the keyword, analyze the top free results, synthesize format guidance.

    Returns (analyses, guidance_markdown).
    """
    query = f"{keyword} {extra_query}".strip()
    logger.info("Competitor analysis for: '%s'", query)

    candidates = _search_duckduckgo(query)
    analyses: list[dict] = []

    for url in candidates:
        if len(analyses) >= TARGET_FREE_RESULTS:
            break
        result = analyze_url(url)
        if result:
            analyses.append(result)
            logger.info(
                "  [%d/%d] %s — %d words, H2×%d",
                len(analyses), TARGET_FREE_RESULTS,
                result["title"][:50], result["word_count"], result["h2_count"],
            )

    if not analyses:
        logger.warning("No usable results found for '%s'", query)
        return [], ""

    guidance = synthesize_format_guidance(keyword, analyses)
    return analyses, guidance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    kw = " ".join(sys.argv[1:]) or "freelance income proof"
    _, guidance = run_competitor_analysis(kw)
    print(guidance)
