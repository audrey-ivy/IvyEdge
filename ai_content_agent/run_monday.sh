#!/usr/bin/env bash
# IvyEdge Monday pipeline — runs automatically every Monday at 9am.
# Order: extend calendar → inject trending topics → generate + publish all queued posts

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$AGENT_DIR/.venv/bin/python"
CALENDAR="$AGENT_DIR/editorial_calendar.csv"
LOG_DIR="$AGENT_DIR/logs"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d)_monday.log"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cd "$AGENT_DIR"

log "========================================"
log "IvyEdge Monday pipeline starting"
log "========================================"

# ── Step 1: Extend the editorial calendar (4 weeks ahead, deduplicates safely)
log "Step 1: Extending editorial calendar..."
"$PYTHON" calendar_agent.py \
  --weeks 4 \
  --posts-per-week 2 \
  --output "$CALENDAR" \
  >> "$LOG_FILE" 2>&1
log "Step 1 done."

# ── Step 2: Inject any trending topics from Google News
log "Step 2: Checking for trending topics..."
"$PYTHON" trend_monitor.py \
  --suggest-posts \
  --add-to-calendar "$CALENDAR" \
  >> "$LOG_FILE" 2>&1
log "Step 2 done."

# ── Step 3: Generate and publish all queued posts
log "Step 3: Running content pipeline..."
"$PYTHON" run_pipeline.py batch \
  --calendar "$CALENDAR" \
  --publish \
  >> "$LOG_FILE" 2>&1
log "Step 3 done."

# ── Step 4: Generate image cards + videos, post to Instagram and Threads
log "Step 4: Running social media agent..."
"$PYTHON" social_media_agent.py \
  --output-dir "$AGENT_DIR/output" \
  >> "$LOG_FILE" 2>&1
log "Step 4 done."

log "========================================"
log "Monday pipeline complete. Log: $LOG_FILE"
log "========================================"
