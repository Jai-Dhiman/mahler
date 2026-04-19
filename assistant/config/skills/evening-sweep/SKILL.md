---
name: evening-sweep
description: Run the evening task sweep — query today's completed, past-due, and open tasks from Notion, pick tomorrow's top 3 priorities, and check in on overdue items.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [tasks, notion, evening, productivity, cron]
    related_skills: [notion-tasks]
---

## When to use

- Cron-triggered at 01:00 UTC (6pm Pacific) daily
- When the user asks for an evening summary, daily close, or "what's on for tomorrow"

## Prerequisites

| Variable | Purpose |
|---|---|
| `NOTION_API_TOKEN` | Notion internal integration token |
| `NOTION_DATABASE_ID` | ID of the Notion tasks database |

Both must be set as Fly.io secrets. The script raises `RuntimeError` if either is missing.

## Procedure

Run the task sweep:

```bash
python3 ~/.hermes/skills/evening-sweep/scripts/sweep.py
```

The script prints three sections to stdout:
- `=== COMPLETED TODAY ===` — tasks with status Done and last_edited_time today
- `=== PAST DUE (not done) ===` — tasks with a past due date and status not Done, with days overdue
- `=== OPEN TASKS ===` — all tasks, with title, due date, and priority

## After running sweep.py

1. From `=== OPEN TASKS ===`, select the **top 3 tasks for tomorrow** by: High priority first, then soonest due date, then Medium priority. State a one-line reason for each pick.
2. For each task in `=== PAST DUE (not done) ===`, ask the user: "Is [task title] done? If so, say `mark [task title] done` and I will complete it."
3. Post a single Discord message with three parts:
   - Completed today: count and task titles
   - Tomorrow's focus: the top 3 picks with one-line reasons
   - Past-due check-in: the question(s) from step 2, or "No overdue tasks." if the section is empty

If the sweep script fails, surface the error message to the user directly. Do not retry silently.
