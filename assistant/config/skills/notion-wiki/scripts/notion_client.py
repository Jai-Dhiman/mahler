import json
import ssl
import urllib.request
import urllib.parse
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


def _blocks_to_markdown(blocks: list) -> str:
    return ""


class NotionWikiReader:
    def __init__(self, token: str, sources_db_id: str, concepts_db_id: str):
        if not token:
            raise RuntimeError("NOTION_WIKI_READ_TOKEN is required")
        if not sources_db_id:
            raise RuntimeError("NOTION_WIKI_SOURCES_DB_ID is required")
        if not concepts_db_id:
            raise RuntimeError("NOTION_WIKI_CONCEPTS_DB_ID is required")
        self._token = token
        self._sources_db_id = sources_db_id
        self._concepts_db_id = concepts_db_id

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

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

    def search(self, query: str, limit: int = 10) -> list:
        body = {
            "query": query,
            "page_size": limit,
            "filter": {"property": "object", "value": "page"},
        }
        data = self._request("POST", "/search", body)
        results = []
        for page in data.get("results", []):
            parent = page.get("parent", {})
            db_id = parent.get("database_id", "")
            if db_id == self._sources_db_id:
                db_name = "sources"
            elif db_id == self._concepts_db_id:
                db_name = "concepts"
            else:
                continue
            title_parts = page.get("properties", {}).get("Title", {}).get("title", [])
            title = "".join(p.get("plain_text", "") for p in title_parts).strip()
            results.append({
                "id": page["id"],
                "title": title,
                "db": db_name,
            })
        return results
