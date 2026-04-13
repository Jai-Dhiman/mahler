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

    def create_source(
        self,
        url: str,
        title: str,
        type_: str,
        summary: str,
        tags: Optional[list] = None,
        concept_ids: Optional[list] = None,
        ingested: Optional[str] = None,
    ) -> dict:
        properties: dict = {
            "Title": {"title": [{"text": {"content": title}}]},
            "URL": {"url": url},
            "Type": {"select": {"name": type_}},
        }
        if ingested is not None:
            properties["Ingested"] = {"date": {"start": ingested}}
        if tags is not None:
            for tag in tags:
                if "," in tag:
                    raise RuntimeError(
                        f"Tag value contains a comma, which Notion multi_select cannot store: {tag!r}"
                    )
            properties["Tags"] = {
                "multi_select": [{"name": t} for t in tags]
            }
        if concept_ids is not None and len(concept_ids) > 0:
            properties["Concepts"] = {
                "relation": [{"id": cid} for cid in concept_ids]
            }
        children = _summary_to_paragraph_blocks(summary)
        body = {
            "parent": {"database_id": self._sources_db_id},
            "properties": properties,
            "children": children,
        }
        return self._request("POST", "/pages", body)

    def append_log(self, kind: str, detail: str, when: str) -> dict:
        body = {
            "parent": {"database_id": self._log_db_id},
            "properties": {
                "Kind": {"select": {"name": kind}},
                "Detail": {"rich_text": [{"text": {"content": detail}}]},
                "When": {"date": {"start": when}},
            },
        }
        return self._request("POST", "/pages", body)

    def list_all_concepts(self) -> list:
        results = []
        cursor = None
        while True:
            body: dict = {}
            if cursor is not None:
                body["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{self._concepts_db_id}/query", body)
            for page in data.get("results", []):
                title_parts = page.get("properties", {}).get("Title", {}).get("title", [])
                title = "".join(p.get("plain_text", "") for p in title_parts).strip()
                sources_rel = page.get("properties", {}).get("Sources", {}).get("relation", [])
                source_ids = [r["id"] for r in sources_rel]
                body_md = self._fetch_concept_body_markdown(page["id"])
                results.append({
                    "id": page["id"],
                    "title": title,
                    "body_markdown": body_md,
                    "source_ids": source_ids,
                })
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                raise RuntimeError("Notion API returned has_more=True but no next_cursor")
        return results

    def _fetch_concept_body_markdown(self, page_id: str) -> str:
        data = self._request("GET", f"/blocks/{page_id}/children", None)
        parts = []
        for block in data.get("results", []):
            if block.get("type") == "paragraph":
                rich = block["paragraph"].get("rich_text", [])
                parts.append("".join(r.get("plain_text", "") for r in rich))
        return "\n\n".join(parts)

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{_NOTION_API_BASE}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        attempts = 3
        delay = 0.5
        last_status = 0
        last_raw = b""
        for attempt in range(attempts):
            req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
            with _OPENER.open(req) as resp:
                last_status = resp.status
                last_raw = resp.read()
            if last_status == 200:
                return json.loads(last_raw)
            if last_status != 429:
                break
            if attempt < attempts - 1:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(
            f"Notion API error {last_status}: {last_raw.decode('utf-8', errors='replace')}"
        )
