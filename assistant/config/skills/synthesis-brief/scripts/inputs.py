import json as _json
from dataclasses import dataclass, field


@dataclass
class Item:
    source: str
    id: str
    content: str
    captured_at: str


@dataclass
class InputBundle:
    recent_items: list = field(default_factory=list)
    context_items: list = field(default_factory=list)
    past_briefs: list = field(default_factory=list)
    identifiers: set = field(default_factory=set)


_CREATE_LOCAL_CAPTURE = """
CREATE TABLE IF NOT EXISTS local_capture (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL CHECK(source IN ('memory','git')),
  project TEXT,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  captured_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_SYNTHESIS_BRIEF = """
CREATE TABLE IF NOT EXISTS synthesis_brief (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  posted_at TEXT NOT NULL DEFAULT (datetime('now')),
  connections_json TEXT NOT NULL,
  pattern TEXT NOT NULL,
  question TEXT NOT NULL
)
"""


def _ensure_tables(d1) -> None:
    d1.query(_CREATE_LOCAL_CAPTURE, [])
    d1.query(_CREATE_SYNTHESIS_BRIEF, [])
    d1.query(
        "CREATE INDEX IF NOT EXISTS idx_local_capture_recent ON local_capture(captured_at)",
        [],
    )


def _load_project_wins(d1, context_days: int) -> list:
    rows = d1.query(
        "SELECT id, project, summary, created_at FROM project_log "
        "WHERE entry_type = 'win' AND created_at >= datetime('now', ? || ' days') "
        "ORDER BY created_at DESC",
        [f"-{context_days}"],
    )
    items = []
    for r in rows:
        items.append(Item(
            source="project_log",
            id=f"project_log:{r['id']}",
            content=f"[{r['project']}] {r['summary']}",
            captured_at=r["created_at"],
        ))
    return items


def _load_honcho(honcho, context_days: int) -> list:
    conclusions = honcho.list_conclusions(since_days=context_days)
    items = []
    for i, c in enumerate(conclusions):
        items.append(Item(
            source="honcho",
            id=f"honcho:{i}",
            content=getattr(c, "content", str(c)),
            captured_at=str(getattr(c, "created_at", "")),
        ))
    return items


def _row_to_item(r: dict) -> Item:
    return Item(
        source=r["source"],
        id=f"{r['source']}:{r['id']}",
        content=r["content"],
        captured_at=r["captured_at"],
    )


def _load_local_recent(d1, recent_days: int) -> list:
    rows = d1.query(
        "SELECT id, source, project, content, captured_at FROM local_capture "
        "WHERE captured_at >= datetime('now', ? || ' days') "
        "ORDER BY captured_at DESC",
        [f"-{recent_days}"],
    )
    return [_row_to_item(r) for r in rows]


def _load_local_context(d1, recent_days: int, context_days: int) -> list:
    rows = d1.query(
        "SELECT id, source, project, content, captured_at FROM local_capture "
        "WHERE captured_at >= datetime('now', ? || ' days') "
        "AND captured_at < datetime('now', ? || ' days') "
        "ORDER BY captured_at DESC",
        [f"-{context_days}", f"-{recent_days}"],
    )
    return [_row_to_item(r) for r in rows]


def _load_past_briefs(d1, context_days: int) -> list:
    rows = d1.query(
        "SELECT posted_at, connections_json, pattern, question FROM synthesis_brief "
        "WHERE posted_at >= datetime('now', ? || ' days') ORDER BY posted_at DESC",
        [f"-{context_days}"],
    )
    out = []
    for r in rows:
        try:
            connections = _json.loads(r.get("connections_json") or "[]")
        except ValueError:
            connections = []
        out.append({
            "posted_at": r.get("posted_at"),
            "connections": connections,
            "pattern": r.get("pattern", ""),
            "question": r.get("question", ""),
        })
    return out


def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    bundle = InputBundle()
    bundle.recent_items.extend(_load_local_recent(d1, recent_days))
    bundle.context_items.extend(_load_local_context(d1, recent_days, context_days))
    bundle.context_items.extend(_load_project_wins(d1, context_days))
    bundle.context_items.extend(_load_honcho(honcho, context_days))
    bundle.past_briefs = _load_past_briefs(d1, context_days)
    return bundle
