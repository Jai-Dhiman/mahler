import json
import ssl
import time
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


def _summary_to_paragraph_blocks(summary: str) -> list:
    paragraphs = [p.strip() for p in summary.split("\n\n") if p.strip()]
    return [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": p}}],
            },
        }
        for p in paragraphs
    ]


class NotionWikiWriter:
    def __init__(
        self,
        token: str,
        sources_db_id: str,
        concepts_db_id: str,
        log_db_id: str,
    ):
        if not token:
            raise RuntimeError("NOTION_WIKI_WRITE_TOKEN is required")
        if not sources_db_id:
            raise RuntimeError("NOTION_WIKI_SOURCES_DB_ID is required")
        if not concepts_db_id:
            raise RuntimeError("NOTION_WIKI_CONCEPTS_DB_ID is required")
        if not log_db_id:
            raise RuntimeError("NOTION_WIKI_LOG_DB_ID is required")
        self._token = token
        self._sources_db_id = sources_db_id
        self._concepts_db_id = concepts_db_id
        self._log_db_id = log_db_id

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def find_source_by_url(self, url: str) -> Optional[dict]:
        body = {"filter": {"property": "URL", "url": {"equals": url}}}
        data = self._request("POST", f"/databases/{self._sources_db_id}/query", body)
        results = data.get("results", [])
        if not results:
            return None
        return results[0]

    def find_concept_by_title(self, title: str) -> Optional[dict]:
        body = {"filter": {"property": "Title", "title": {"contains": title}}}
        data = self._request("POST", f"/databases/{self._concepts_db_id}/query", body)
        needle = title.strip().lower()
        for page in data.get("results", []):
            props = page.get("properties", {})
            title_parts = props.get("Title", {}).get("title", [])
            stored = "".join(p.get("plain_text", "") for p in title_parts).strip().lower()
            if stored == needle:
                return page
        return None

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{_NOTION_API_BASE}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()
        if status != 200:
            raise RuntimeError(f"Notion API error {status}: {raw.decode('utf-8', errors='replace')}")
        return json.loads(raw)
