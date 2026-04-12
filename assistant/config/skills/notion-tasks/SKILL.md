---
name: notion-tasks
description: Create, list, update, complete, and delete tasks in the user's Notion task database. Full CRUD for personal task management.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [tasks, todo, notion, productivity]
    related_skills: []
---

## When to use

- When the user asks to add, create, or log a new task or todo
- When the user asks what tasks or todos they have, wants to see their list, or asks "what do I need to do?"
- When the user asks to update, modify, or change a task's title, status, priority, or due date
- When the user says a task is done, finished, or complete
- When the user asks to remove or delete a task

## Prerequisites

| Variable | Purpose |
|---|---|
| `NOTION_API_TOKEN` | Notion internal integration token |
| `NOTION_DATABASE_ID` | ID of the Notion tasks database |

Both must be set as Fly.io secrets. The script raises `RuntimeError` if either is missing.

## Date handling

All dates must be in ISO 8601 format: `YYYY-MM-DD`. Before invoking any command, convert relative dates from the user's message to absolute dates using today's date. Example: "Friday" → the date of the upcoming Friday, "next week" → the Monday of next week.

## Operations

### Create a task

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py create \
  --title "TITLE" \
  [--due YYYY-MM-DD] \
  [--priority High|Medium|Low]
```

If the user does not state a priority, infer it:
- Deadlines within 2 days, blocking other work, or urgent language → `High`
- Clear action items without urgency → `Medium`
- Nice-to-have, someday, or low-stakes tasks → `Low`

### List tasks

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py list \
  [--status "Not started"|"In progress"|"Done"] \
  [--priority "High"|"Medium"|"Low"] \
  [--due-before YYYY-MM-DD]
```

Output format per task:
```
[page-id] Task title
  (status=Todo, priority=High, due=2026-04-17)
```

The page ID on each task's first line is required for follow-up update, complete, or delete operations. If the user asks to act on a task and you do not have its page ID, run `list` first.

### Update a task

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py update \
  --id PAGE_ID \
  [--title "NEW TITLE"] \
  [--status "Not started"|"In progress"|"Done"] \
  [--due YYYY-MM-DD] \
  [--priority "High"|"Medium"|"Low"]
```

Include only the flags for fields the user wants to change.

### Complete a task

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py complete --id PAGE_ID
```

Use this when the user says a task is done, finished, or complete. Sets status to Done.

### Delete a task

```bash
python3 ~/.hermes/skills/notion-tasks/scripts/tasks.py delete --id PAGE_ID
```

**Always confirm with the user before running delete.** Ask: "Are you sure you want to delete [task title]?" and only proceed if they confirm. Deleted tasks are archived in Notion (recoverable from the Notion UI) but treated as permanently removed in this interface.

## Output

Each command prints a single confirmation line to stdout:
- `create` → `Created: {page_id} — {title}`
- `list` → one formatted entry per task, or `No tasks found.`
- `update` → `Updated: {page_id} — {title}`
- `complete` → `Completed: {page_id} — {title}`
- `delete` → `Deleted: {page_id}`

Any failure raises `RuntimeError` and exits non-zero. Surface the error message to the user directly.
