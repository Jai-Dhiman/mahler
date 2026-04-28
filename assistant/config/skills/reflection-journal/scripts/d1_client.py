import sys
from pathlib import Path

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_base import D1Client as _D1Base, _OPENER  # noqa: F401


class D1Client(_D1Base):
    def ensure_table(self) -> None:
        self.query(
            """CREATE TABLE IF NOT EXISTS reflection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_of TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)""",
            [],
        )

    def insert_reflection(self, week_of: str, raw_text: str) -> None:
        self.query(
            "INSERT INTO reflection_log (week_of, raw_text) VALUES (?, ?)",
            [week_of, raw_text],
        )

    def get_recent_reflections(self, since_weeks: int = 4) -> list[dict]:
        if not isinstance(since_weeks, int) or since_weeks <= 0:
            raise ValueError(
                f"since_weeks must be a positive integer, got {since_weeks!r}"
            )
        since_days = since_weeks * 7
        return self.query(
            "SELECT week_of, raw_text, created_at FROM reflection_log "
            "WHERE created_at >= datetime('now', ? || ' days') "
            "ORDER BY created_at DESC",
            [f"-{since_days}"],
        )
