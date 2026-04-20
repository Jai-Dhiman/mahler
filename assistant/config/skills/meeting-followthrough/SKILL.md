---
name: meeting-followthrough
description: >
  Process a completed meeting forwarded from Fathom. Gather CRM context for
  each attendee, generate smart context-aware action items using current tasks
  and priorities, push to Notion, and update last_contact in the CRM.
triggers:
  - "[FATHOM_MEETING]"
  - process fathom meeting
env:
  - CF_ACCOUNT_ID
  - CF_D1_DATABASE_ID
  - CF_API_TOKEN
  - NOTION_API_TOKEN
  - NOTION_DATABASE_ID
---

# Meeting Follow-Through

Closes the loop after any recorded meeting. Triggered automatically when the
`fathom-webhook` Cloudflare Worker posts a structured @Mahler message after a
Fathom recording completes.

## Message format (posted by CF Worker)

    @Mahler [FATHOM_MEETING]
    Meeting: {title}
    Attendees: {name <email>, ...}

    Summary:
    {chronological_summary_markdown}

## Procedure

### Step 1 — Parse the message

Extract from the triggering Discord message:
- `MEETING_TITLE`: the value after `Meeting:`
- `ATTENDEES`: list of `name <email>` pairs from the `Attendees:` line
- `SUMMARY`: everything after `Summary:` to end of message

### Step 2 — Fetch CRM context for each external attendee

For each attendee in `ATTENDEES`, attempt:

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py summarize \
  --name "ATTENDEE_NAME"
```

- If the command succeeds: record the output (last contact date, context, open tasks).
- If the command fails (contact not in CRM): note "not in CRM" for that attendee and continue. Do not stop.

### Step 3 — Fetch current open tasks

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py list \
  --status "Not started"
```

Record the full output. If the output is empty, that means there are no existing open tasks — this is valid state, not an error. Use it to avoid creating duplicate tasks in Step 4.

If this command fails with a non-zero exit code, surface the error in Discord and stop. Cannot safely generate action items without knowing existing tasks.

### Step 4 — Generate action items

Using all gathered context — meeting summary, CRM outputs from Step 2, open tasks from Step 3, and the injected context from project-log, kaizen priorities, and Honcho memory — reason about what action items arise from this meeting.

Rules for generating action items:
- Only create tasks for concrete commitments from the meeting (things said, agreed, or promised).
- Do not create a task if an equivalent open task already exists from Step 3.
- If the action item relates to a specific attendee who is in the CRM, prefix the task title with `[Attendee Name]` (e.g., `[Alice Chen] Send Q2 IC memo`).
- If the action item is general (not tied to a specific attendee), use no prefix.
- Default priority: Medium. Use High only for explicit deadlines or blockers mentioned in the meeting.

### Step 5 — Create Notion tasks

For each action item determined in Step 4:

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py create \
  --title "TASK_TITLE" \
  --priority PRIORITY
```

- If `tasks.py create` fails for any task: surface the error immediately in Discord and stop creating further tasks. Do not silently skip.
- Record each created task's title for the summary in Step 7.

### Step 6 — Update CRM last_contact

For each attendee from Step 2 whose `contacts.py summarize` succeeded:

```bash
python3 ~/.hermes/skills/relationship-manager/scripts/contacts.py talked-to \
  --name "ATTENDEE_NAME"
```

- If `contacts.py talked-to` fails: surface the error in Discord but continue to Step 7 (CRM update failure must not block the summary).

### Step 7 — Post summary to Discord

Post a single Discord message with:
- **Meeting:** `MEETING_TITLE`
- **Action items created:** bulleted list of task titles from Step 5, or "None" if no action items were generated.
- **CRM updated:** comma-separated names of contacts whose `last_contact` was updated, or "No CRM matches" if none.

Example output:

    Post-meeting: 1:1 with Alice Chen
    Action items created:
      · [Alice Chen] Send Q2 IC memo
      · [Alice Chen] Intro to Marcus at Benchmark
      · Follow up on Series A timeline
    CRM updated: Alice Chen

If no action items were generated, say so explicitly rather than posting nothing.

## Failure modes

- `contacts.py summarize` fails → note "not in CRM", continue (non-fatal)
- `tasks.py list` fails → surface error and stop (cannot safely generate without knowing existing tasks)
- `tasks.py create` fails → surface error and stop (partial task list is worse than none)
- `contacts.py talked-to` fails → surface error, continue to Step 7
