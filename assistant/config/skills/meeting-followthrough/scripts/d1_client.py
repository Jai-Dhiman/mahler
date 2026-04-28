import sys
from pathlib import Path

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_base import D1Client as _D1Base


class D1Client(_D1Base):
    def ensure_queue_table(self) -> None:
        self.query("""CREATE TABLE IF NOT EXISTS fathom_meeting_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL UNIQUE,
    title TEXT NOT NULL,
    attendees TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    processed_at TEXT
)""")

    def fetch_pending(self) -> list[dict]:
        return self.query(
            "SELECT id, recording_id, title, attendees, summary FROM fathom_meeting_queue "
            "WHERE processed_at IS NULL ORDER BY created_at ASC",
        )

    def mark_done(self, recording_id: int) -> None:
        self.query(
            "UPDATE fathom_meeting_queue SET processed_at = datetime('now') WHERE recording_id = ?",
            [recording_id],
        )
