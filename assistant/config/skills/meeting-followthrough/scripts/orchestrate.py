"""Meeting follow-through orchestrator. Invoked by cron every 15 min."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path


_DEFAULT_MODEL = "openai/gpt-5-nano"


def _load_hermes_env() -> None:
    hermes_env = Path.home() / ".hermes" / ".env"
    if not hermes_env.exists():
        print(f"WARNING: hermes env file not found: {hermes_env}", file=sys.stderr)
        return
    with open(hermes_env, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def _parse_attendees(raw: str) -> list[dict]:
    try:
        return json.loads(raw) if isinstance(raw, str) else list(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _fetch_crm_context(attendees: list[dict], runner) -> dict[str, str]:
    owner_email = os.environ.get("MAHLER_OWNER_EMAIL", "").lower()
    ctx: dict[str, str] = {}
    for a in attendees:
        email = (a.get("email") or "").lower()
        name = a.get("name")
        if not name or not email or email == owner_email:
            continue
        result = runner(
            ["python3", str(Path.home() / ".hermes" / "skills" / "relationship-manager" / "scripts" / "contacts.py"), "summarize", "--name", name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            ctx[email] = result.stdout.strip()
    return ctx


def _fetch_open_tasks(runner) -> list[str]:
    result = runner(
        ["python3", str(Path.home() / ".hermes" / "skills" / "notion-tasks" / "scripts" / "tasks.py"), "list", "--status", "Not started"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"WARNING: tasks.py list failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return []
    # tasks.py list format: "[uuid] Title\n  (status=..., priority=...)"
    # Extract only title lines (start with "[") and strip the UUID prefix.
    titles = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "] " in stripped:
            titles.append(stripped[stripped.index("] ") + 2:])
    return titles


def process_meeting(row, *, runner, llm_caller, discord_poster, d1_client) -> str:
    title = row["title"]
    attendees = _parse_attendees(row["attendees"])
    crm_context = _fetch_crm_context(attendees, runner)
    action_items = generate_action_items(
        summary=row["summary"],
        attendees=attendees,
        crm_context=crm_context,
        open_tasks=_fetch_open_tasks(runner),
        llm_caller=llm_caller,
    )
    if action_items:
        action_lines = "\n".join(f"  · {i['title']}" for i in action_items)
    else:
        action_lines = "  None"
    crm_line = "CRM updated: No CRM matches"
    summary = (
        f"Post-meeting: {title}\n"
        f"Action items created:\n"
        f"{action_lines}\n"
        f"{crm_line}"
    )
    discord_poster(summary)
    d1_client.mark_done(row["recording_id"])
    return summary


def generate_action_items(summary, attendees, crm_context, open_tasks, llm_caller) -> list[dict]:
    prompt = _build_prompt(summary, attendees, crm_context, open_tasks)
    raw = llm_caller(prompt)
    return _parse_action_items(raw)


def _build_prompt(summary, attendees, crm_context, open_tasks) -> str:
    attendees_block = "\n".join(
        f"- {a.get('name') or a.get('email') or 'unknown'}: {crm_context.get(a.get('email', ''), 'not in CRM')}"
        for a in attendees
    ) or "- none"
    open_tasks_block = "\n".join(f"- {t}" for t in open_tasks) or "- none"
    return (
        "You are generating post-meeting action items.\n\n"
        f"Meeting summary:\n{summary}\n\n"
        f"Attendees:\n{attendees_block}\n\n"
        f"Existing open tasks (do NOT duplicate these):\n{open_tasks_block}\n\n"
        "Output format: one action item per line, prefixed with 'TASK: '. "
        "Include priority as '| PRIORITY: High|Medium|Low'. "
        "Prefix the title with '[Attendee Name]' when the item relates to a specific attendee. "
        "If there are no action items, respond with exactly 'no action items'."
    )


def _parse_action_items(raw: str) -> list[dict]:
    items: list[dict] = []
    for line in raw.splitlines():
        if line.startswith("TASK:"):
            items.append(_parse_task_line(line))
    return items


def _parse_task_line(line: str) -> dict:
    body = line[len("TASK:"):].strip()
    title_part, _, priority_part = body.partition("| PRIORITY:")
    title = title_part.strip()
    priority = priority_part.strip() or "Medium"
    attendee = None
    if title.startswith("["):
        end = title.find("]")
        if end > 0:
            attendee = title[1:end]
    return {"title": title, "priority": priority, "attendee": attendee}


def main(argv, *, d1_client, runner, llm_caller, discord_poster) -> int:
    rows = d1_client.fetch_pending()
    if not rows:
        print("NO_WORK")
        return 0
    return 0
