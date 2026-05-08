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
  echo "OPENROUTER_MODEL=${OPENROUTER_MODEL:-openai/gpt-5-nano}"
  echo "HONCHO_API_KEY=${HONCHO_API_KEY:-}"
  echo "TAVILY_API_KEY=${TAVILY_API_KEY:-}"
  echo "MAHLER_OWNER_EMAIL=${MAHLER_OWNER_EMAIL:-}"
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
        jobs = data.get('jobs', data) if isinstance(data, dict) else data
except (FileNotFoundError, json.JSONDecodeError):
    jobs = []

def next_run_for(cron_expr, now):
    parts = cron_expr.strip().split()
    minute_field, hour_field, dow_field = parts[0], parts[1], parts[4]
    base = now.replace(second=0, microsecond=0)
    if minute_field.startswith('*/') and hour_field == '*':
        interval = int(minute_field[2:])
        delta = interval - (base.minute % interval)
        return base + timedelta(minutes=delta)
    elif minute_field == '0' and hour_field.isdigit():
        h = int(hour_field)
        candidate = base.replace(hour=h, minute=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        if dow_field != '*':
            if '-' in dow_field:
                start_day, end_day = [int(x) for x in dow_field.split('-')]
                py_start = (start_day + 6) % 7
                py_end = (end_day + 6) % 7
                while not (py_start <= candidate.weekday() <= py_end):
                    candidate += timedelta(days=1)
            else:
                py_target = (int(dow_field) + 6) % 7
                while candidate.weekday() != py_target:
                    candidate += timedelta(days=1)
        return candidate
    else:
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

# Build index of existing jobs by primary skill name so we can update
# prompts without resetting next_run_at or last_run_at.
existing_by_skill = {}
for j in jobs:
    primary = j.get('skill', j.get('name', ''))
    if primary and primary not in existing_by_skill:
        existing_by_skill[primary] = j

def upsert(skills, prompt, cron_expr):
    primary = skills[0]
    if primary in existing_by_skill:
        existing_by_skill[primary]['prompt'] = prompt
        existing_by_skill[primary]['skills'] = skills
        return 'updated'
    jobs.append(make_job(skills, prompt, cron_expr))
    return 'added'

results = []
results.append(('email-triage', upsert(
    ['email-triage'],
    'Run email triage: fetch unread emails from Gmail and Outlook, classify them using the priority map, store results in D1, and send Discord alerts for any URGENT emails.',
    '0 * * * *',
)))
results.append(('morning-brief', upsert(
    ['morning-brief'],
    'Post the morning email brief: query the last 12 hours of triage results and post a structured summary to Discord.',
    '0 16 * * *',
)))
results.append(('meeting-followthrough', upsert(
    ['meeting-followthrough', 'relationship-manager', 'notion-tasks'],
    'Check the Fathom meeting queue for pending completed meetings. If there are pending meetings, process each one: gather CRM context for attendees, generate action items, create Notion tasks, update CRM last_contact, mark the meeting as done, and post a summary to Discord. If there are no pending meetings, do nothing.',
    '*/5 * * * *',
)))
results.append(('meeting-prep', upsert(
    ['meeting-prep', 'google-calendar', 'notion-tasks', 'notion-wiki'],
    'Check if there is a meeting starting in 45 to 75 minutes. If so, check deduplication, gather context from recent emails, open tasks, and the wiki, synthesize a prep brief, post it to Discord, and log the event.',
    '*/15 * * * *',
)))
results.append(('kaizen-reflection', upsert(
    ['kaizen-reflection'],
    'Run the weekly kaizen reflection: analyze email triage patterns from the past 7 days, generate reclassification proposals, and present each to Discord with approve/deny buttons. Post the results to Discord.',
    '0 18 * * 0',
)))
results.append(('evening-sweep', upsert(
    ['evening-sweep'],
    \"Run the evening task sweep: query today's completed, past-due, and open tasks from Notion, pick the top 3 priorities for tomorrow, post a summary to Discord, and check in on any overdue items.\",
    '0 1 * * *',
)))
results.append(('relationship-manager', upsert(
    ['relationship-manager'],
    \"Run the daily calendar sync for the relationship CRM: call python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py sync-calendar --days 1 to fetch yesterday's Google Calendar events and update last_contact for any attendees that match known contacts.\",
    '0 8 * * *',
)))
results.append(('reflection-journal', upsert(
    ['reflection-journal'],
    \"Run the weekly reflection journal: post the three reflection questions to Discord and wait for the user's reply. Once the user replies, record the response with --record. Always post the questions to Discord.\",
    '0 2 * * 0',
)))
results.append(('project-synthesis', upsert(
    ['project-synthesis'],
    'Run the weekly project synthesis: run python3 ~/.hermes/skills/project-synthesis/scripts/synthesize.py --run, then post the result message to Discord verbatim.',
    '0 18 * * 0',
)))
results.append(('memory-kaizen', upsert(
    ['memory-kaizen'],
    'Run the weekly memory kaizen: run python3 ~/.hermes/skills/memory-kaizen/scripts/kaizen.py --run, then post the result message to Discord verbatim.',
    '0 19 * * 0',
)))
results.append(('synthesis-brief', upsert(
    ['synthesis-brief'],
    'Run the daily synthesis brief: run python3 ~/.hermes/skills/synthesis-brief/scripts/synthesize.py --run, then print the result line to stdout. Do not post to Discord; the 8am morning-brief picks up the result from mahler_kv.',
    '0 13 * * 1-5',
)))

with open(jobs_file, 'w') as f:
    json.dump({'jobs': jobs, 'updated_at': datetime.now(timezone.utc).isoformat()}, f, indent=2)

added = [name for name, action in results if action == 'added']
updated = [name for name, action in results if action == 'updated']
if added:
    print('Registered new cron jobs:', ', '.join(added))
if updated:
    print('Updated existing cron prompts:', ', '.join(updated))
"

exec hermes gateway
