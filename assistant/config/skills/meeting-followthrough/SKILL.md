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

Closes the loop after any recorded meeting. Triggered by cron every 15 minutes.
Everything is done by `orchestrate.py` — this skill just invokes it and reports the result.

## Procedure

Run the orchestrator:

```bash
python3 ~/.hermes/skills/meeting-followthrough/scripts/orchestrate.py
```

Report whatever it prints to stdout verbatim. If the exit code is non-zero, the orchestrator will have already posted a per-meeting error message to the Discord triage channel; repeat the error in your reply so the user sees it in the agent log too.

If stdout is `NO_WORK`, stop — no pending meetings this tick.

## Failure modes

- OpenRouter returns 5xx while generating action items → orchestrator catches, posts error to Discord for the affected meeting, leaves the meeting UN-marked (next tick retries).
- `contacts.py summarize` returns non-zero → attendee is treated as "not in CRM" and skipped for CRM fan-out.
- `tasks.py create` returns non-zero for one action item → that item is dropped from the Discord summary; other items proceed.
- `post_discord.py` fails → orchestrator exits non-zero; SessionStop hook surfaces the error.
