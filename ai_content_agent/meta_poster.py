"""
IvyEdge Meta Poster — Threads + Instagram

Posts branded image cards and captions to Threads and Instagram
via the Meta Graph API.

Required in .env:
  META_ACCESS_TOKEN=...     Long-lived page/user access token
  IG_USER_ID=...            Instagram Business account user ID
  THREADS_USER_ID=...       Threads user ID (same as IG user ID usually)
  CLOUDINARY_CLOUD_NAME=... For hosting images (Meta API needs a public URL)
  CLOUDINARY_API_KEY=...
  CLOUDINARY_API_SECRET=...

Setup guide: see README — one-time Meta developer app setup required.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ivyedge.meta_poster")

META_ACCESS_TOKEN  = os.getenv("META_ACCESS_TOKEN", "")
IG_USER_ID         = os.getenv("IG_USER_ID", "")
THREADS_USER_ID    = os.getenv("THREADS_USER_ID", "")

GRAPH_BASE         = "https://graph.facebook.com/v19.0"
THREADS_BASE       = "https://graph.threads.net/v1.0"


def _configure_cloudinary() -> None:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
        api_key=os.getenv("CLOUDINARY_API_KEY", ""),
        api_secret=os.getenv("CLOUDINARY_API_SECRET", ""),
    )


def _check_credentials(platform: str) -> bool:
    if not META_ACCESS_TOKEN:
        logger.error("META_ACCESS_TOKEN not set — cannot post to %s", platform)
        return False
    if platform == "instagram" and not IG_USER_ID:
        logger.error("IG_USER_ID not set")
        return False
    if platform == "threads" and not THREADS_USER_ID:
        logger.error("THREADS_USER_ID not set")
        return False
    return True


# ---------------------------------------------------------------------------
# Cloudinary image upload
# ---------------------------------------------------------------------------

def _upload_image(image_path: Path) -> str:
    """Upload image to Cloudinary and return the public HTTPS URL."""
    _configure_cloudinary()
    if not os.getenv("CLOUDINARY_CLOUD_NAME"):
        raise ValueError(
            "CLOUDINARY_CLOUD_NAME not set in .env.\n"
            "Sign up free at cloudinary.com and add your credentials."
        )
    result = cloudinary.uploader.upload(
        str(image_path),
        folder="ivyedge/social",
        overwrite=False,
        resource_type="image",
    )
    url = result.get("secure_url", "")
    logger.info("Uploaded to Cloudinary: %s", url)
    return url


# ---------------------------------------------------------------------------
# Instagram poster
# ---------------------------------------------------------------------------

def post_to_instagram(
    caption: str,
    image_path: Path,
) -> Optional[str]:
    """
    Post a static image to Instagram via Meta Graph API.

    Returns the Instagram post URL, or None on failure.
    """
    if not _check_credentials("instagram"):
        return None

    try:
        image_url = _upload_image(image_path)
    except Exception as e:
        logger.error("Cloudinary upload failed: %s", e)
        return None

    # Step 1 — create media container
    create_url = f"{GRAPH_BASE}/{IG_USER_ID}/media"
    resp = requests.post(create_url, data={
        "image_url":    image_url,
        "caption":      caption,
        "access_token": META_ACCESS_TOKEN,
    }, timeout=30)

    if not resp.ok:
        logger.error("Instagram container creation failed: %s", resp.text)
        return None

    container_id = resp.json().get("id")
    logger.info("Instagram container created: %s", container_id)

    # Wait for container to process
    time.sleep(4)

    # Step 2 — publish
    publish_url = f"{GRAPH_BASE}/{IG_USER_ID}/media_publish"
    resp = requests.post(publish_url, data={
        "creation_id":  container_id,
        "access_token": META_ACCESS_TOKEN,
    }, timeout=30)

    if not resp.ok:
        logger.error("Instagram publish failed: %s", resp.text)
        return None

    post_id  = resp.json().get("id", "")
    post_url = f"https://www.instagram.com/p/{post_id}/"
    logger.info("Posted to Instagram: %s", post_url)
    return post_url


# ---------------------------------------------------------------------------
# Threads poster
# ---------------------------------------------------------------------------

def post_to_threads(
    text: str,
    image_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Post to Threads via the Threads Graph API.

    Returns the Threads post URL, or None on failure.
    """
    if not _check_credentials("threads"):
        return None

    image_url = None
    if image_path:
        try:
            image_url = _upload_image(image_path)
        except Exception as e:
            logger.warning("Image upload failed, posting text-only to Threads: %s", e)

    # Step 1 — create container
    create_url = f"{THREADS_BASE}/{THREADS_USER_ID}/threads"
    payload: dict = {
        "text":         text,
        "access_token": META_ACCESS_TOKEN,
    }
    if image_url:
        payload["media_type"] = "IMAGE"
        payload["image_url"]  = image_url
    else:
        payload["media_type"] = "TEXT"

    resp = requests.post(create_url, data=payload, timeout=30)
    if not resp.ok:
        logger.error("Threads container creation failed: %s", resp.text)
        return None

    container_id = resp.json().get("id")
    logger.info("Threads container created: %s", container_id)

    time.sleep(4)

    # Step 2 — publish
    publish_url = f"{THREADS_BASE}/{THREADS_USER_ID}/threads_publish"
    resp = requests.post(publish_url, data={
        "creation_id":  container_id,
        "access_token": META_ACCESS_TOKEN,
    }, timeout=30)

    if not resp.ok:
        logger.error("Threads publish failed: %s", resp.text)
        return None

    post_id  = resp.json().get("id", "")
    post_url = f"https://www.threads.net/t/{post_id}"
    logger.info("Posted to Threads: %s", post_url)
    return post_url
