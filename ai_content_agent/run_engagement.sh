#!/usr/bin/env bash
# IvyEdge daily engagement agent — runs every day at 8am via launchd.
# Discovers relevant posts across Instagram, Reddit, TikTok, and Threads,
# auto-posts Reddit comments, and emails a review brief.

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$AGENT_DIR/.venv/bin/python"
LOG_DIR="$AGENT_DIR/logs"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d)_engagement.log"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cd "$AGENT_DIR"

log "========================================"
log "IvyEdge engagement agent starting"
log "========================================"

# Run discovery across all platforms, auto-post Reddit comments, save dated report
"$PYTHON" engagement_agent.py \
  --auto \
  >> "$LOG_FILE" 2>&1

log "========================================"
log "Engagement run complete. Log: $LOG_FILE"
log "========================================"
