import json
import ssl
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
    title_list = props.get("Name", {}).get("title", [])
    title = title_list[0]["plain_text"] if title_list else ""
    status_sel = props.get("Status", {}).get("select")
    status = status_sel["name"] if status_sel else "Todo"
    due_obj = props.get("Due", {}).get("date")
    due = due_obj["start"] if due_obj else None
    priority_sel = props.get("Priority", {}).get("select")
    priority = priority_sel["name"] if priority_sel else None
    return {
        "id": page["id"],
        "title": title,
        "status": status,
        "due": due,
        "priority": priority,
    }


class NotionClient:
    def __init__(self, api_token: str, database_id: str):
        if not api_token:
            raise RuntimeError("NOTION_API_TOKEN is required")
        if not database_id:
            raise RuntimeError("NOTION_DATABASE_ID is required")
        self._token = api_token
        self._database_id = database_id

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def create_task(
        self,
        title: str,
        due: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> dict:
        properties: dict = {
            "Name": {"title": [{"text": {"content": title}}]},
            "Status": {"select": {"name": "Todo"}},
        }
        if due is not None:
            properties["Due"] = {"date": {"start": due}}
        if priority is not None:
            properties["Priority"] = {"select": {"name": priority}}

        body = {
            "parent": {"database_id": self._database_id},
            "properties": properties,
        }
        data = self._request("POST", "/pages", body)
        return _extract_task(data)

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        due_before: Optional[str] = None,
    ) -> list[dict]:
        filters = []
        if status is not None:
            filters.append({"property": "Status", "select": {"equals": status}})
        if priority is not None:
            filters.append({"property": "Priority", "select": {"equals": priority}})
        if due_before is not None:
            filters.append({"property": "Due", "date": {"on_or_before": due_before}})

        body: dict = {}
        if len(filters) == 1:
            body["filter"] = filters[0]
        elif len(filters) > 1:
            body["filter"] = {"and": filters}

        results = []
        cursor = None
        while True:
            if cursor:
                body["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{self._database_id}/query", body)
            for page in data.get("results", []):
                results.append(_extract_task(page))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        return results

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{_NOTION_API_BASE}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
        if status not in (200,):
            raise RuntimeError(f"Notion API error {status}: {raw.decode('utf-8', errors='replace')}")
        return json.loads(raw)
