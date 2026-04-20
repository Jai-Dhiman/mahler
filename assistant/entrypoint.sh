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
  echo "OUTLOOK_CLIENT_ID=${OUTLOOK_CLIENT_ID:-}"
  echo "OUTLOOK_CLIENT_SECRET=${OUTLOOK_CLIENT_SECRET:-}"
  echo "OUTLOOK_REFRESH_TOKEN=${OUTLOOK_REFRESH_TOKEN:-}"
  echo "CF_ACCOUNT_ID=${CF_ACCOUNT_ID:-}"
  echo "CF_D1_DATABASE_ID=${CF_D1_DATABASE_ID:-}"
  echo "CF_API_TOKEN=${CF_API_TOKEN:-}"
  echo "DISCORD_TRIAGE_WEBHOOK=${DISCORD_TRIAGE_WEBHOOK:-}"
  echo "OPENROUTER_MODEL=${OPENROUTER_MODEL:-x-ai/grok-4.1-fast}"
  echo "HONCHO_API_KEY=${HONCHO_API_KEY:-}"
} > "$HERMES_ENV"

: "${NOTION_WIKI_READ_TOKEN:=}"
: "${NOTION_WIKI_SOURCES_DB_ID:=}"
: "${NOTION_WIKI_CONCEPTS_DB_ID:=}"
{
  echo "NOTION_WIKI_READ_TOKEN=${NOTION_WIKI_READ_TOKEN}"
  echo "NOTION_WIKI_SOURCES_DB_ID=${NOTION_WIKI_SOURCES_DB_ID}"
  echo "NOTION_WIKI_CONCEPTS_DB_ID=${NOTION_WIKI_CONCEPTS_DB_ID}"
  echo "NOTION_API_TOKEN=${NOTION_API_TOKEN:-}"
  echo "NOTION_DATABASE_ID=${NOTION_DATABASE_ID:-}"
} >> "$HOME/.hermes/.env"

# Register cron jobs in Hermes cron format (schedule object + next_run_at).
# Format was introduced in v0.7.0 and is still used through v0.9.0.
CRON_DIR="$HOME/.hermes/cron"
JOBS_FILE="$CRON_DIR/jobs.json"
mkdir -p "$CRON_DIR"

python3 -c "
import json, uuid
from datetime import datetime, timezone, timedelta

jobs_file = '$JOBS_FILE'
try:
    with open(jobs_file, 'r') as f:
        data = json.load(f)
        # Handle both old list format and new {jobs: [...]} format
        jobs = data.get('jobs', data) if isinstance(data, dict) else data
except (FileNotFoundError, json.JSONDecodeError):
    jobs = []

def next_run_for(cron_expr, now):
    \"\"\"Compute next run for simple cron patterns without croniter.\"\"\"
    parts = cron_expr.strip().split()
    minute_field, hour_field = parts[0], parts[1]
    base = now.replace(second=0, microsecond=0)
    if minute_field.startswith('*/') and hour_field == '*':
        # e.g. */15 * * * *
        interval = int(minute_field[2:])
        delta = interval - (base.minute % interval)
        return base + timedelta(minutes=delta)
    elif minute_field == '0' and hour_field.isdigit():
        # e.g. 0 16 * * *
        h = int(hour_field)
        candidate = base.replace(hour=h, minute=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate
    else:
        # Fallback: 1 minute from now (Hermes will correct it after first run)
        return base + timedelta(minutes=1)

def make_job(skills, prompt, cron_expr):
    now = datetime.now(timezone.utc)
    next_run = next_run_for(cron_expr, now)
    return {
        'id': uuid.uuid4().hex[:12],
        'name': skills[0],
        'prompt': prompt,
        'skills': skills,
        'skill': skills[0],
        'model': None,
        'provider': None,
        'base_url': None,
        'script': None,
        'schedule': {'kind': 'cron', 'expr': cron_expr, 'display': cron_expr},
        'schedule_display': cron_expr,
        'repeat': {'times': None, 'completed': 0},
        'enabled': True,
        'state': 'scheduled',
        'paused_at': None,
        'paused_reason': None,
        'created_at': now.isoformat(),
        'next_run_at': next_run.isoformat(),
        'last_run_at': None,
        'last_status': None,
        'last_error': None,
        'deliver': 'local',
        'origin': None,
    }

existing_skills = set()
for j in jobs:
    for s in j.get('skills', [j.get('skill', '')]):
        existing_skills.add(s)

added = []
if 'email-triage' not in existing_skills:
    jobs.append(make_job(
        ['email-triage'],
        'Run email triage: fetch unread emails from Gmail and Outlook, classify them using the priority map, store results in D1, and send Discord alerts for any URGENT emails.',
        '0 * * * *',
    ))
    added.append('email-triage (every 15 min)')

if 'morning-brief' not in existing_skills:
    jobs.append(make_job(
        ['morning-brief'],
        'Post the morning email brief: query the last 12 hours of triage results and post a structured summary to Discord.',
        '0 16 * * *',
    ))
    jobs.append(make_job(
        ['morning-brief'],
        'Post the evening email brief: query the last 12 hours of triage results and post a structured summary to Discord.',
        '0 4 * * *',
    ))
    added.append('morning-brief (8am + 8pm PST)')

if 'meeting-prep' not in existing_skills:
    jobs.append(make_job(
        ['meeting-prep', 'google-calendar', 'notion-tasks', 'notion-wiki'],
        'Check if there is a meeting starting in 45 to 75 minutes. If so, check deduplication, gather context from recent emails, open tasks, and the wiki, synthesize a prep brief, post it to Discord, and log the event.',
        '*/15 * * * *',
    ))
    added.append('meeting-prep (every 15 min)')

if 'kaizen-reflection' not in existing_skills:
    jobs.append(make_job(
        ['kaizen-reflection'],
        'Run the weekly kaizen reflection: analyze email triage patterns from the past 7 days, generate reclassification proposals, and present each to Discord with approve/deny buttons.',
        '0 18 * * 0',
    ))
    added.append('kaizen-reflection (Sundays 18:00 UTC)')

if 'evening-sweep' not in existing_skills:
    jobs.append(make_job(
        ['evening-sweep'],
        'Run the evening task sweep: query today\'s completed, past-due, and open tasks from Notion, pick the top 3 priorities for tomorrow, post a summary to Discord, and check in on any overdue items.',
        '0 1 * * *',
    ))
    added.append('evening-sweep (01:00 UTC / 6pm Pacific)')

if 'relationship-manager' not in existing_skills:
    jobs.append(make_job(
        ['relationship-manager'],
        'Run the daily calendar sync for the relationship CRM: call python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py sync-calendar --days 1 to fetch yesterday\'s Google Calendar events and update last_contact for any attendees that match known contacts.',
        '0 8 * * *',
    ))
    added.append('relationship-manager (08:00 UTC / midnight Pacific)')

if 'reflection-journal' not in existing_skills:
    jobs.append(make_job(
        ['reflection-journal'],
        'Run the weekly reflection journal: post the three reflection questions to Discord and wait for the user\'s reply. Once the user replies, record the response with --record.',
        '0 2 * * 0',
    ))
    added.append('reflection-journal (Sundays 02:00 UTC)')

with open(jobs_file, 'w') as f:
    json.dump({'jobs': jobs, 'updated_at': datetime.now(timezone.utc).isoformat()}, f, indent=2)

if added:
    print('Registered cron jobs:', ', '.join(added))
else:
    print('Cron jobs already registered, skipping.')
"

exec hermes gateway
