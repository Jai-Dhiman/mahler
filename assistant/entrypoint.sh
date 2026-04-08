#!/usr/bin/env bash
set -euo pipefail

# Write env vars into Hermes .env file (Hermes reads from ~/.hermes/.env, not process env)
HERMES_ENV="$HOME/.hermes/.env"

{
  echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}"
  echo "DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN:-}"
  echo "DISCORD_HOME_CHANNEL=${DISCORD_HOME_CHANNEL:-}"
  echo "DISCORD_ALLOWED_USERS=${DISCORD_ALLOWED_USERS:-}"
  echo "GMAIL_CLIENT_ID=${GMAIL_CLIENT_ID:-}"
  echo "GMAIL_CLIENT_SECRET=${GMAIL_CLIENT_SECRET:-}"
  echo "GMAIL_REFRESH_TOKEN=${GMAIL_REFRESH_TOKEN:-}"
  echo "OUTLOOK_EMAIL=${OUTLOOK_EMAIL:-}"
  echo "OUTLOOK_APP_PASSWORD=${OUTLOOK_APP_PASSWORD:-}"
  echo "CF_ACCOUNT_ID=${CF_ACCOUNT_ID:-}"
  echo "CF_D1_DATABASE_ID=${CF_D1_DATABASE_ID:-}"
  echo "CF_API_TOKEN=${CF_API_TOKEN:-}"
  echo "DISCORD_TRIAGE_WEBHOOK=${DISCORD_TRIAGE_WEBHOOK:-}"
  echo "OPENROUTER_MODEL=${OPENROUTER_MODEL:-x-ai/grok-4.1-fast}"
} > "$HERMES_ENV"

# Register email triage cron job if not already registered
CRON_DIR="$HOME/.hermes/cron"
JOBS_FILE="$CRON_DIR/jobs.json"
mkdir -p "$CRON_DIR"
if [ ! -f "$JOBS_FILE" ] || ! grep -q "email-triage" "$JOBS_FILE" 2>/dev/null; then
  python3 -c "
import json, uuid
from datetime import datetime, timezone

jobs_file = '$JOBS_FILE'
try:
    with open(jobs_file, 'r') as f:
        jobs = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    jobs = []

# Only add if not already present
if not any(j.get('skill') == 'email-triage' or 'email-triage' in j.get('skills', []) for j in jobs):
    now = datetime.now(timezone.utc).isoformat()
    jobs.append({
        'id': str(uuid.uuid4()),
        'skills': ['email-triage'],
        'skill': 'email-triage',
        'prompt': 'Run email triage: fetch unread emails from Gmail and Outlook, classify them using the priority map, store results in D1, and send Discord alerts for any URGENT emails.',
        'cron': '*/15 * * * *',
        'enabled': True,
        'created_at': now,
        'updated_at': now,
    })
    with open(jobs_file, 'w') as f:
        json.dump(jobs, f, indent=2)
    print('Registered email-triage cron job (every 15 minutes)')
"
fi

exec hermes gateway
