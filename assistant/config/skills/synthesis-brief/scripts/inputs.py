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


def load_all(d1, honcho, recent_days: int = 1, context_days: int = 14) -> InputBundle:
    _ensure_tables(d1)
    return InputBundle()
