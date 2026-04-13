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
