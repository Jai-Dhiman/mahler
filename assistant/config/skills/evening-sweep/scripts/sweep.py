import os
import sys
from datetime import date, timedelta
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
    return NotionClient(api_token, database_id)


def main(argv=None, _today=None):
    today = _today if _today is not None else date.today()
    yesterday = (today - timedelta(days=1)).isoformat()
    today_str = today.isoformat()

    client = _get_client()

    completed = client.list_tasks(status="Done", last_edited_after=today_str)
    overdue_raw = client.list_tasks(due_before=yesterday)

    print("=== COMPLETED TODAY ===")
    for t in completed:
        priority_part = f", priority={t['priority']}" if t["priority"] else ""
        print(f"- {t['title']}{priority_part}")

    print("\n=== PAST DUE (not done) ===")
    past_due = [t for t in overdue_raw if t["status"] != "Done"]
    for t in past_due:
        days_over = (today - date.fromisoformat(t["due"])).days
        day_word = "day" if days_over == 1 else "days"
        print(f"- {t['title']} (due={t['due']}, {days_over} {day_word} overdue)")


if __name__ == "__main__":
    main()
