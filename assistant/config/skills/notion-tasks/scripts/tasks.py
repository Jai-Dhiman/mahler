"""
Notion task management CLI for Mahler.

Usage:
    python3 tasks.py create --title TITLE [--due YYYY-MM-DD] [--priority High|Medium|Low]
    python3 tasks.py list [--status STATUS] [--priority PRIORITY] [--due-before YYYY-MM-DD]
    python3 tasks.py update --id PAGE_ID [--title TITLE] [--status STATUS] [--due DATE] [--priority PRIORITY]
    python3 tasks.py complete --id PAGE_ID
    python3 tasks.py delete --id PAGE_ID
"""

import argparse
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))


def _supplement_env_from_hermes() -> None:
    hermes_env = Path.home() / ".hermes" / ".env"
    if not hermes_env.exists():
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


_supplement_env_from_hermes()

from notion_client import NotionClient  # noqa: E402


def _get_client() -> NotionClient:
    api_token = os.environ.get("NOTION_API_TOKEN")
    database_id = os.environ.get("NOTION_DATABASE_ID")
    if not api_token:
        raise RuntimeError("NOTION_API_TOKEN environment variable is not set")
    if not database_id:
        raise RuntimeError("NOTION_DATABASE_ID environment variable is not set")
    return NotionClient(api_token, database_id)


def _format_task(task: dict) -> str:
    meta = []
    if task.get("status"):
        meta.append(f"status={task['status']}")
    if task.get("priority"):
        meta.append(f"priority={task['priority']}")
    if task.get("due"):
        meta.append(f"due={task['due']}")
    line = f"[{task['id']}] {task['title']}"
    if meta:
        line += f"\n  ({', '.join(meta)})"
    return line


def cmd_create(args: argparse.Namespace) -> None:
    client = _get_client()
    task = client.create_task(title=args.title, due=args.due, priority=args.priority)
    print(f"Created: {task['id']} — {task['title']}")


def cmd_list(args: argparse.Namespace) -> None:
    client = _get_client()
    task_list = client.list_tasks(
        status=args.status,
        priority=args.priority,
        due_before=args.due_before,
    )
    if not task_list:
        print("No tasks found.")
        return
    for task in task_list:
        print(_format_task(task))


def cmd_update(args: argparse.Namespace) -> None:
    client = _get_client()
    fields = {}
    if args.title is not None:
        fields["title"] = args.title
    if args.status is not None:
        fields["status"] = args.status
    if args.due is not None:
        fields["due"] = args.due
    if args.priority is not None:
        fields["priority"] = args.priority
    if not fields:
        raise RuntimeError("No fields specified for update — provide at least one of: --title, --status, --due, --priority")
    task = client.update_task(args.id, **fields)
    print(f"Updated: {task['id']} — {task['title']}")


def cmd_complete(args: argparse.Namespace) -> None:
    client = _get_client()
    task = client.complete_task(args.id)
    print(f"Completed: {task['id']} — {task['title']}")


def cmd_delete(args: argparse.Namespace) -> None:
    client = _get_client()
    client.delete_task(args.id)
    print(f"Deleted: {args.id}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler Notion task manager")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create")
    p_create.add_argument("--title", required=True)
    p_create.add_argument("--due", default=None)
    p_create.add_argument("--priority", choices=["High", "Medium", "Low"], default=None)

    p_list = sub.add_parser("list")
    p_list.add_argument("--status", choices=["Todo", "In Progress", "Done"], default=None)
    p_list.add_argument("--priority", choices=["High", "Medium", "Low"], default=None)
    p_list.add_argument("--due-before", dest="due_before", default=None)

    p_update = sub.add_parser("update")
    p_update.add_argument("--id", required=True)
    p_update.add_argument("--title", default=None)
    p_update.add_argument("--status", choices=["Todo", "In Progress", "Done"], default=None)
    p_update.add_argument("--due", default=None)
    p_update.add_argument("--priority", choices=["High", "Medium", "Low"], default=None)

    p_complete = sub.add_parser("complete")
    p_complete.add_argument("--id", required=True)

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("--id", required=True)

    args = parser.parse_args(argv)
    dispatch = {
        "create": cmd_create,
        "list": cmd_list,
        "update": cmd_update,
        "complete": cmd_complete,
        "delete": cmd_delete,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
