import sys
from pathlib import Path
from typing import Optional

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_base import D1Client as _D1Base


class D1Client(_D1Base):
    def is_already_notified(self, event_id: str) -> bool:
        rows = self.query(
            "SELECT event_id FROM meeting_prep_log WHERE event_id = ? LIMIT 1",
            [event_id],
        )
        return len(rows) > 0

    def insert_meeting_prep(self, event_id: str, summary: str, start_time: str) -> None:
        self.query(
            "INSERT OR IGNORE INTO meeting_prep_log (event_id, summary, start_time, notified_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            [event_id, summary, start_time],
        )

    def get_upcoming_meeting(self) -> Optional[dict]:
        rows = self.query(
            "SELECT event_id, summary, start_time FROM meeting_prep_log "
            "WHERE start_time > datetime('now') AND start_time < datetime('now', '+2 hours') "
            "ORDER BY start_time ASC LIMIT 1",
            [],
        )
        return rows[0] if rows else None

    def ensure_meeting_prep_table(self) -> None:
        self.query(
            """CREATE TABLE IF NOT EXISTS meeting_prep_log (
    event_id    TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    start_time  TEXT NOT NULL,
    notified_at TEXT NOT NULL
)""",
            [],
        )
