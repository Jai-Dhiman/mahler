---
name: meeting-followthrough
description: >
  Process completed meetings from the Fathom queue in D1. Gather CRM context for
  each attendee, generate smart context-aware action items using current tasks
  and priorities, push to Notion, update last_contact in the CRM, and post a
  summary to Discord.
triggers:
  - process fathom meeting
env:
  - CF_ACCOUNT_ID
  - CF_D1_DATABASE_ID
  - CF_API_TOKEN
  - NOTION_API_TOKEN
  - NOTION_DATABASE_ID
  - DISCORD_TRIAGE_WEBHOOK
---

# Meeting Follow-Through

Closes the loop after any recorded meeting. Triggered by cron every 5 minutes.
The `fathom-webhook` Cloudflare Worker writes completed meetings to D1; this
skill polls for pending entries, processes each one, and marks them done.

## Procedure

### Step 1 — Check for pending meetings

```bash
python3 ~/.hermes/skills/meeting-followthrough/scripts/poll.py fetch
```

If the output is `NO_PENDING_MEETINGS`, stop immediately. Do not post anything.

Otherwise the output contains one or more meeting blocks separated by
`---END_MEETING---`. For each block, extract:
- `RECORDING_ID`: integer ID (needed for mark-done)
- `TITLE`: meeting title
- `ATTENDEES`: comma-separated `name <email>` pairs
- `SUMMARY`: everything after `SUMMARY:` to end of block

Process each meeting through Steps 2–7, then mark it done.

### Step 2 — Fetch CRM context for each external attendee

For each attendee in `ATTENDEES`, attempt:

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py summarize \
  --name "ATTENDEE_NAME"
```

- If the command succeeds: record the output (last contact date, context, open tasks).
- If the command fails (contact not in CRM): note "not in CRM" for that attendee and continue.

### Step 3 — Fetch current open tasks

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py list \
  --status "Not started"
```

Record the full output. Empty output means no existing open tasks — this is
valid state. If this command fails with a non-zero exit code, surface the error
in Discord and stop. Cannot safely generate action items without knowing
existing tasks.

### Step 4 — Generate action items

Using all gathered context — meeting summary, CRM outputs from Step 2, open
tasks from Step 3, and the injected context from project-log, kaizen priorities,
and Honcho memory — reason about what action items arise from this meeting.

Rules:
- Only create tasks for concrete commitments (things said, agreed, or promised).
- Do not create a task if an equivalent open task already exists from Step 3.
- If the action item relates to a specific attendee in the CRM, prefix the task
  title with `[Attendee Name]` (e.g., `[Alice Chen] Send Q2 IC memo`).
- Default priority: Medium. Use High only for explicit deadlines or blockers.

### Step 5 — Create Notion tasks

For each action item:

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py create \
  --title "TASK_TITLE" \
  --priority PRIORITY
```

If `tasks.py create` fails for any task: surface the error in Discord and stop.
Do not silently skip.

### Step 6 — Update CRM last_contact

For each attendee whose `contacts.py summarize` succeeded:

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py talked-to \
  --name "ATTENDEE_NAME"
```

If this fails: surface the error in Discord but continue to Step 7.

### Step 7 — Mark the meeting as processed

```bash
python3 ~/.hermes/skills/meeting-followthrough/scripts/poll.py mark-done \
  --recording-id RECORDING_ID
```

This must run even if no action items were generated. If it fails, surface the
error — the meeting will be reprocessed on the next cron run if not marked done.

### Step 8 — Post summary to Discord

Post a single Discord message via the triage webhook with:
- **Meeting:** `TITLE`
- **Action items created:** bulleted list of task titles, or "None"
- **CRM updated:** comma-separated contact names, or "No CRM matches"

Example:

    Post-meeting: 1:1 with Alice Chen
    Action items created:
      · [Alice Chen] Send Q2 IC memo
      · Follow up on Series A timeline
    CRM updated: Alice Chen

## Failure modes

- `contacts.py summarize` fails → note "not in CRM", continue (non-fatal)
- `tasks.py list` fails → surface error and stop
- `tasks.py create` fails → surface error and stop
- `contacts.py talked-to` fails → surface error, continue to Step 7
- `poll.py mark-done` fails → surface error (meeting will reprocess next run)
