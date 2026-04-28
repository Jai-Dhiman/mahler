import sys
from pathlib import Path

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_base import D1Client as _D1Base


class D1Client(_D1Base):
    def get_triage_patterns(self, since_days: int = 7, min_count: int = 3) -> list[dict]:
        if not isinstance(since_days, int) or since_days <= 0:
            raise ValueError(f"since_days must be a positive integer, got {since_days!r}")
        return self.query(
            """SELECT from_addr, classification, COUNT(*) AS occurrence_count
               FROM email_triage_log
               WHERE processed_at >= datetime('now', ? || ' days')
               GROUP BY from_addr, classification
               HAVING COUNT(*) >= ?
               ORDER BY occurrence_count DESC""",
            [f"-{since_days}", min_count],
        )

    def get_triage_patterns_with_reply_rate(
        self, since_days: int = 7, min_count: int = 3
    ) -> list[dict]:
        if not isinstance(since_days, int) or since_days <= 0:
            raise ValueError(
                f"since_days must be a positive integer, got {since_days!r}"
            )
        return self.query(
            """SELECT from_addr, classification,
                      COUNT(*) AS occurrence_count,
                      COUNT(replied_at) AS reply_count
               FROM email_triage_log
               WHERE processed_at >= datetime('now', ? || ' days')
               GROUP BY from_addr, classification
               HAVING COUNT(*) >= ?
               ORDER BY occurrence_count DESC""",
            [f"-{since_days}", min_count],
        )

    def get_priority_map(self) -> str:
        rows = self.query(
            "SELECT content FROM priority_map ORDER BY version DESC LIMIT 1",
            [],
        )
        if not rows:
            raise RuntimeError(
                "priority_map table is empty — run migrate.py to seed initial content"
            )
        return rows[0]["content"]

    def set_priority_map(self, content: str) -> None:
        self.query(
            "INSERT INTO priority_map (content, version, updated_at) "
            "VALUES (?, COALESCE((SELECT MAX(version) FROM priority_map), 0) + 1, datetime('now'))",
            [content],
        )

    def ensure_priority_map_table(self) -> None:
        self.query(
            """CREATE TABLE IF NOT EXISTS priority_map (
    version INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
            [],
        )

    def get_recent_project_log(self, since_days: int = 7) -> list[dict]:
        if not isinstance(since_days, int) or since_days <= 0:
            raise ValueError(
                f"since_days must be a positive integer, got {since_days!r}"
            )
        return self.query(
            "SELECT project, entry_type, summary, git_ref, created_at "
            "FROM project_log "
            "WHERE created_at >= datetime('now', ? || ' days') "
            "ORDER BY created_at DESC",
            [f"-{since_days}"],
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
