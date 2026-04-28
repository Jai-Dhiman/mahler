import sys
from pathlib import Path
from typing import Optional

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from notion_base import _NotionBase, _OPENER  # noqa: F401


def _extract_task(page: dict) -> dict:
    props = page.get("properties", {})
    title_list = props.get("Task name", {}).get("title", [])
    title = title_list[0]["plain_text"] if title_list else ""
    status_sel = props.get("Status", {}).get("status")
    status = status_sel["name"] if status_sel else "Todo"
    due_obj = props.get("Due date", {}).get("date")
    due = due_obj["start"] if due_obj else None
    priority_sel = props.get("Priority", {}).get("select")
    priority = priority_sel["name"] if priority_sel else None
    return {"id": page["id"], "title": title, "status": status, "due": due, "priority": priority}


class NotionClient(_NotionBase):
    def __init__(self, api_token: str, database_id: str):
        if not api_token:
            raise RuntimeError("NOTION_API_TOKEN is required")
        if not database_id:
            raise RuntimeError("NOTION_DATABASE_ID is required")
        self._token = api_token
        self._database_id = database_id

    def create_task(self, title: str, due: Optional[str] = None, priority: Optional[str] = None) -> dict:
        properties: dict = {
            "Task name": {"title": [{"text": {"content": title}}]},
            "Status": {"status": {"name": "Not started"}},
        }
        if due is not None:
            properties["Due date"] = {"date": {"start": due}}
        if priority is not None:
            properties["Priority"] = {"select": {"name": priority}}
        data = self._request("POST", "/pages", {"parent": {"database_id": self._database_id}, "properties": properties})
        return _extract_task(data)

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        due_before: Optional[str] = None,
    ) -> list[dict]:
        filters = []
        if status is not None:
            filters.append({"property": "Status", "status": {"equals": status}})
        if priority is not None:
            filters.append({"property": "Priority", "select": {"equals": priority}})
        if due_before is not None:
            filters.append({"property": "Due date", "date": {"on_or_before": due_before}})

        body: dict = {}
        if len(filters) == 1:
            body["filter"] = filters[0]
        elif len(filters) > 1:
            body["filter"] = {"and": filters}

        results = []
        cursor = None
        while True:
            if cursor is not None:
                body["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{self._database_id}/query", body)
            for page in data.get("results", []):
                results.append(_extract_task(page))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                raise RuntimeError("Notion API returned has_more=True but no next_cursor")
        return results

    def update_task(self, page_id: str, **fields) -> dict:
        properties: dict = {}
        if "title" in fields:
            properties["Task name"] = {"title": [{"text": {"content": fields["title"]}}]}
        if "status" in fields:
            properties["Status"] = {"status": {"name": fields["status"]}}
        if "due" in fields:
            properties["Due date"] = {"date": {"start": fields["due"]}} if fields["due"] else {"date": None}
        if "priority" in fields:
            properties["Priority"] = {"select": {"name": fields["priority"]}} if fields["priority"] else {"select": None}
        try:
            data = self._request("PATCH", f"/pages/{page_id}", {"properties": properties})
        except RuntimeError as e:
            if "404" in str(e):
                raise RuntimeError(f"Task not found: {page_id}")
            raise
        return _extract_task(data)

    def delete_task(self, page_id: str) -> None:
        try:
            self._request("PATCH", f"/pages/{page_id}", {"archived": True})
        except RuntimeError as e:
            if "404" in str(e):
                raise RuntimeError(f"Task not found: {page_id}")
            raise

    def complete_task(self, page_id: str) -> dict:
        return self.update_task(page_id, status="Done")
