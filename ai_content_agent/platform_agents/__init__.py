"""
Shared types for IvyEdge platform engagement agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class EngagementOpportunity:
    """A discovered post worth engaging with."""
    platform: str                        # "instagram" | "reddit" | "threads"
    post_id: str
    url: str
    author: str
    content: str                         # post text / caption
    subreddit: Optional[str] = None      # Reddit only
    hashtags: list[str] = field(default_factory=list)
    score: float = 0.0                   # 0–10 relevance score from Claude
    rationale: str = ""                  # why this is worth engaging
    suggested_comment: str = ""          # Claude-drafted reply
    suggested_action: str = "comment"   # "comment" | "like" | "share" | "upvote"
    discovered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "pending"             # "pending" | "actioned" | "skipped"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "EngagementOpportunity":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
