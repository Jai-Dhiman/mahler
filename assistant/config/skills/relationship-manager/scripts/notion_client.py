import sys
from pathlib import Path

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from notion_base import _NotionBase


def _extract_task(page: dict) -> dict:
    props = page.get("properties", {})
    title_list = props.get("Task name", {}).get("title", [])
    title = title_list[0].get("plain_text", "") if title_list else ""
    status_sel = props.get("Status", {}).get("status")
    status = status_sel["name"] if status_sel else "Todo"
    due_obj = props.get("Due date", {}).get("date")
    due = due_obj["start"] if due_obj else None
    priority_sel = props.get("Priority", {}).get("select")
    priority = priority_sel["name"] if priority_sel else None
    return {"id": page["id"], "title": title, "status": status, "due": due, "priority": priority}


class NotionClient(_NotionBase):
    def __init__(self, api_token: str, database_id: str):
        self._token = api_token
        self._database_id = database_id

    def list_tasks_for_contact(self, name: str) -> list[dict]:
        results = []
        cursor = None
        while True:
            body: dict = {
                "filter": {
                    "and": [
                        {"property": "Task name", "title": {"contains": f"[{name}]"}},
                        {"property": "Status", "status": {"does_not_equal": "Done"}},
                    ]
                },
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{self._database_id}/query", body)
            for page in data.get("results", []):
                results.append(_extract_task(page))
            if not data.get("has_more") or not data.get("next_cursor"):
                break
            cursor = data["next_cursor"]
        return results
