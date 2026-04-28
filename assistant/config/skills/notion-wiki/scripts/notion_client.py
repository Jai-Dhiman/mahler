import sys
import urllib.parse
from pathlib import Path
from typing import Optional

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from notion_base import _NotionBase, _OPENER  # noqa: F401


def _normalize_db_id(db_id: str) -> str:
    return db_id.replace("-", "").lower()


def _rich_text_plain(rich_text: list) -> str:
    return "".join(r.get("plain_text", "") for r in rich_text)


def _render_block(block: dict) -> Optional[str]:
    btype = block.get("type", "")
    if btype == "paragraph":
        return _rich_text_plain(block["paragraph"].get("rich_text", []))
    if btype == "heading_1":
        return "# " + _rich_text_plain(block["heading_1"].get("rich_text", []))
    if btype == "heading_2":
        return "## " + _rich_text_plain(block["heading_2"].get("rich_text", []))
    if btype == "heading_3":
        return "### " + _rich_text_plain(block["heading_3"].get("rich_text", []))
    if btype == "bulleted_list_item":
        return "- " + _rich_text_plain(block["bulleted_list_item"].get("rich_text", []))
    return None


def _blocks_to_markdown(blocks: list) -> str:
    parts = []
    for block in blocks:
        rendered = _render_block(block)
        if rendered is not None:
            parts.append(rendered)
    return "\n\n".join(parts)


class NotionWikiReader(_NotionBase):
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

    def search(self, query: str, limit: int = 10) -> list:
        body = {
            "query": query,
            "page_size": limit,
            "filter": {"property": "object", "value": "page"},
        }
        data = self._request("POST", "/search", body)
        sources_canonical = _normalize_db_id(self._sources_db_id)
        concepts_canonical = _normalize_db_id(self._concepts_db_id)
        results = []
        for page in data.get("results", []):
            parent = page.get("parent", {})
            db_id = _normalize_db_id(parent.get("database_id", ""))
            if db_id == sources_canonical:
                db_name = "sources"
            elif db_id == concepts_canonical:
                db_name = "concepts"
            else:
                continue
            title_parts = page.get("properties", {}).get("Title", {}).get("title", [])
            title = "".join(p.get("plain_text", "") for p in title_parts).strip()
            results.append({"id": page["id"], "title": title, "db": db_name})
        return results

    def read_page(self, page_id: str) -> dict:
        page = self._request("GET", f"/pages/{page_id}", None)
        props = page.get("properties", {})

        title_parts = props.get("Title", {}).get("title", [])
        title = "".join(p.get("plain_text", "") for p in title_parts).strip()

        type_ = ""
        type_prop = props.get("Type", {})
        if type_prop.get("type") == "select" and type_prop.get("select"):
            type_ = type_prop["select"].get("name", "")

        url = ""
        url_prop = props.get("URL", {})
        if url_prop.get("type") == "url":
            url = url_prop.get("url") or ""

        blocks = self._fetch_children(page_id)
        body_markdown = _blocks_to_markdown(blocks)

        related = []
        for prop_name in ("Concepts", "Sources"):
            prop = props.get(prop_name, {})
            if prop.get("type") != "relation":
                continue
            for rel in prop.get("relation", []):
                related.append(self._lookup_title(rel["id"]))

        return {
            "id": page["id"],
            "title": title,
            "type": type_,
            "url": url,
            "body_markdown": body_markdown,
            "related_sources": related,
        }

    def list_index(self, db: str, limit: int = 100) -> list:
        if db == "sources":
            db_id = self._sources_db_id
        elif db == "concepts":
            db_id = self._concepts_db_id
        else:
            raise RuntimeError(f"Unknown db: {db!r} (expected 'sources' or 'concepts')")

        results = []
        cursor = None
        while True:
            body: dict = {"page_size": limit}
            if cursor is not None:
                body["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{db_id}/query", body)
            for page in data.get("results", []):
                title_parts = page.get("properties", {}).get("Title", {}).get("title", [])
                title = "".join(p.get("plain_text", "") for p in title_parts).strip()
                results.append({"id": page["id"], "title": title})
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                raise RuntimeError("Notion API returned has_more=True but no next_cursor")
        return results

    def _lookup_title(self, page_id: str) -> dict:
        page = self._request("GET", f"/pages/{page_id}", None)
        title_parts = page.get("properties", {}).get("Title", {}).get("title", [])
        title = "".join(p.get("plain_text", "") for p in title_parts).strip()
        return {"id": page["id"], "title": title}

    def _fetch_children(self, page_id: str) -> list:
        results = []
        cursor = None
        while True:
            path = f"/blocks/{page_id}/children"
            if cursor is not None:
                path += f"?start_cursor={urllib.parse.quote(cursor)}"
            data = self._request("GET", path, None)
            results.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                raise RuntimeError("Notion API returned has_more=True but no next_cursor")
        return results
