import json
import ssl
import urllib.error
import urllib.request
from typing import Optional

_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


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


class NotionClient:
    def __init__(self, api_token: str, database_id: str):
        self.api_token = api_token
        self.database_id = database_id

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{_NOTION_API_BASE}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Notion-Version": _NOTION_VERSION,
                "Content-Type": "application/json",
            },
        )
        try:
            with _OPENER.open(req) as resp:
                status = resp.status
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Notion API error (connection failed): {exc.reason}"
            ) from exc
        if status not in (200, 201):
            raise RuntimeError(f"Notion API error {status}: {raw.decode('utf-8', errors='replace')}")
        return json.loads(raw)

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
            data = self._request("POST", f"/databases/{self.database_id}/query", body)
            for page in data.get("results", []):
                results.append(_extract_task(page))
            if not data.get("has_more") or not data.get("next_cursor"):
                break
            cursor = data["next_cursor"]
        return results
