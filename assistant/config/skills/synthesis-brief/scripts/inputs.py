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


def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    bundle = InputBundle()
    bundle.context_items.extend(_load_project_wins(d1, context_days))
    bundle.context_items.extend(_load_honcho(honcho, context_days))
    return bundle
