"""
IvyEdge Mailer

Sends the daily engagement review queue as a formatted HTML email.

Setup (Gmail — easiest):
  1. Enable 2-Step Verification on your Google account
  2. Go to myaccount.google.com/apppasswords → generate an App Password
  3. Add to .env:
       EMAIL_FROM=you@gmail.com
       EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
       NOTIFY_EMAIL=you@gmail.com   (where to send the digest)

Sends via Gmail SMTP (smtp.gmail.com:587). To use a different provider,
set EMAIL_SMTP_HOST and EMAIL_SMTP_PORT in .env.
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ivyedge.mailer")

EMAIL_FROM        = os.getenv("EMAIL_FROM", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
NOTIFY_EMAIL      = os.getenv("NOTIFY_EMAIL", "")
SMTP_HOST         = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.getenv("EMAIL_SMTP_PORT", "587"))

PLATFORM_COLORS = {
    "instagram": "#E1306C",
    "threads":   "#000000",
    "reddit":    "#FF4500",
    "tiktok":    "#010101",
}

PLATFORM_ICONS = {
    "instagram": "📸",
    "threads":   "🧵",
    "reddit":    "🤖",
    "tiktok":    "🎵",
}


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _score_bar(score: float) -> str:
    filled = round(score)
    empty  = 10 - filled
    return "█" * filled + "░" * empty


def _platform_badge(platform: str) -> str:
    color = PLATFORM_COLORS.get(platform, "#888")
    icon  = PLATFORM_ICONS.get(platform, "•")
    label = platform.upper()
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:11px;font-weight:bold;">'
        f'{icon} {label}</span>'
    )


def _opportunity_card(item: dict, index: int) -> str:
    platform  = item.get("platform", "?")
    score     = item.get("score", 0.0)
    url       = item.get("url", "")
    author    = item.get("author", "")
    subreddit = item.get("subreddit", "")
    rationale = item.get("rationale", "")
    content   = item.get("content", "")[:200].replace("<", "&lt;").replace(">", "&gt;")
    comment   = item.get("suggested_comment", "").replace("<", "&lt;").replace(">", "&gt;")
    action    = item.get("suggested_action", "comment")

    location = f"r/{subreddit}" if subreddit else f"@{author}" if author else ""

    return f"""
<div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;
            padding:16px;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <span style="color:#888;font-size:13px;">#{index}</span>
    {_platform_badge(platform)}
    {"<span style='color:#555;font-size:13px;'>" + location + "</span>" if location else ""}
    <span style="margin-left:auto;font-family:monospace;font-size:13px;
                 color:#333;" title="{score:.1f}/10">{_score_bar(score)}&nbsp;{score:.1f}</span>
  </div>

  <div style="font-size:13px;color:#666;margin-bottom:8px;font-style:italic;">
    {rationale}
  </div>

  <div style="background:#f7f7f7;border-radius:4px;padding:10px;
              font-size:13px;color:#444;margin-bottom:12px;
              border-left:3px solid #ddd;">
    {content}{"…" if len(item.get("content","")) > 200 else ""}
  </div>

  {"" if not comment else f'''
  <div style="margin-bottom:12px;">
    <div style="font-size:11px;font-weight:bold;color:#888;
                text-transform:uppercase;margin-bottom:4px;">
      Suggested {action}
    </div>
    <div style="background:#eef4ff;border-radius:4px;padding:10px;
                font-size:13px;color:#333;border-left:3px solid #4f8ef7;
                white-space:pre-wrap;">{comment}</div>
  </div>
  '''}

  <a href="{url}" style="display:inline-block;background:#4f8ef7;color:white;
     padding:6px 14px;border-radius:4px;font-size:13px;text-decoration:none;
     font-weight:bold;">Open post →</a>
</div>"""


def build_html(opportunities: list[dict], summary: dict) -> str:
    today      = datetime.now(timezone.utc).strftime("%B %-d, %Y")
    total      = len(opportunities)
    discovered = summary.get("discovered", total)
    posted     = summary.get("posted", 0)

    # Group by platform
    by_platform: dict[str, list[dict]] = {}
    for item in opportunities:
        p = item.get("platform", "other")
        by_platform.setdefault(p, []).append(item)

    platform_summary = " &nbsp;|&nbsp; ".join(
        f"{PLATFORM_ICONS.get(p,'•')} {p.title()}: {len(items)}"
        for p, items in sorted(by_platform.items())
    )

    cards_html = ""
    idx = 1
    for platform in ["reddit", "instagram", "tiktok", "threads"]:
        items = by_platform.get(platform, [])
        if not items:
            continue
        color = PLATFORM_COLORS.get(platform, "#888")
        icon  = PLATFORM_ICONS.get(platform, "•")
        cards_html += f"""
<h2 style="font-size:16px;color:{color};margin:24px 0 8px;border-bottom:2px solid {color};
           padding-bottom:6px;">{icon} {platform.title()} ({len(items)})</h2>"""
        for item in items:
            cards_html += _opportunity_card(item, idx)
            idx += 1

    if not cards_html:
        cards_html = """
<div style="text-align:center;padding:40px;color:#888;">
  No high-scoring opportunities found today. Check back tomorrow.
</div>"""

    posted_line = (
        f'<div style="color:#2e7d32;">✓ {posted} Reddit comment{"s" if posted != 1 else ""} auto-posted</div>'
        if posted else ""
    )

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,
             'Segoe UI',sans-serif;">
<div style="max-width:680px;margin:24px auto;background:#fff;border-radius:12px;
            overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
              padding:28px 32px;">
    <div style="color:#a78bfa;font-size:12px;font-weight:bold;
                text-transform:uppercase;letter-spacing:1px;">IvyEdge</div>
    <h1 style="color:white;margin:6px 0 4px;font-size:22px;">
      Daily Engagement Brief
    </h1>
    <div style="color:#94a3b8;font-size:14px;">{today}</div>
  </div>

  <!-- Stats bar -->
  <div style="background:#f8f9fa;padding:16px 32px;border-bottom:1px solid #eee;
              font-size:13px;color:#555;">
    <strong>{total}</strong> opportunities queued for review
    &nbsp;·&nbsp; {discovered} discovered across all platforms
    {("&nbsp;·&nbsp;" + posted_line) if posted else ""}
    <div style="margin-top:4px;color:#888;">{platform_summary}</div>
  </div>

  <!-- Body -->
  <div style="padding:24px 32px;">
    <p style="font-size:14px;color:#555;margin-top:0;">
      Below are today's highest-scoring engagement opportunities.
      Copy the suggested comment, open the post, and paste it in.
      Reddit comments marked ✓ were auto-posted.
    </p>

    {cards_html}
  </div>

  <!-- Footer -->
  <div style="background:#f8f9fa;padding:16px 32px;border-top:1px solid #eee;
              font-size:12px;color:#999;text-align:center;">
    IvyEdge Engagement Agent &nbsp;·&nbsp;
    Run: <code>python engagement_agent.py --show-queue</code> to view full queue
  </div>

</div>
</body>
</html>"""


def build_plain(opportunities: list[dict], summary: dict) -> str:
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total    = len(opportunities)
    posted   = summary.get("posted", 0)

    lines = [
        f"IvyEdge Daily Engagement Brief — {today}",
        f"{'='*50}",
        f"{total} opportunities | {summary.get('discovered', total)} discovered",
    ]
    if posted:
        lines.append(f"{posted} Reddit comment(s) auto-posted")
    lines.append("")

    for i, item in enumerate(opportunities, 1):
        platform  = item.get("platform", "?").upper()
        score     = item.get("score", 0.0)
        url       = item.get("url", "")
        rationale = item.get("rationale", "")
        comment   = item.get("suggested_comment", "")
        lines += [
            f"[{i}] {platform} — score {score:.1f}/10",
            f"    {url}",
            f"    Why: {rationale}",
        ]
        if comment:
            lines += [f"    Draft comment:", f"    {comment[:300]}"]
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def send(
    opportunities: list[dict],
    summary: dict,
    to: Optional[str] = None,
) -> bool:
    recipient = to or NOTIFY_EMAIL
    if not recipient:
        logger.error("No recipient — set NOTIFY_EMAIL in .env or pass --email")
        return False
    if not EMAIL_FROM or not EMAIL_APP_PASSWORD:
        logger.error("EMAIL_FROM or EMAIL_APP_PASSWORD not set in .env")
        return False

    today = datetime.now(timezone.utc).strftime("%b %-d")
    total = len(opportunities)
    subject = f"IvyEdge Engagement Brief — {today} ({total} opportunities)"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"IvyEdge Agent <{EMAIL_FROM}>"
    msg["To"]      = recipient

    plain = build_plain(opportunities, summary)
    html  = build_html(opportunities, summary)
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_FROM, recipient, msg.as_string())
        logger.info("Engagement brief sent to %s", recipient)
        return True
    except Exception as e:
        logger.error("Failed to send email: %s", e)
        return False
