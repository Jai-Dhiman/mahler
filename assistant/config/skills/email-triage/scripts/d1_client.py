import sys
from pathlib import Path
from typing import Optional

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_base import D1Client as _D1Base, _OPENER  # noqa: F401


class D1Client(_D1Base):
    def is_already_processed(self, message_id: str) -> bool:
        rows = self.query(
            "SELECT message_id FROM email_triage_log WHERE message_id = ? LIMIT 1",
            [message_id],
        )
        return len(rows) > 0

    def insert_triage_result(self, result: dict) -> None:
        sql = (
            "INSERT OR IGNORE INTO email_triage_log "
            "(message_id, source, from_addr, subject, received_at, classification, "
            "summary, alerted, classification_error, processed_at, conversation_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = [
            result["message_id"],
            result["source"],
            result.get("from_addr"),
            result.get("subject"),
            result.get("received_at"),
            result["classification"],
            result.get("summary"),
            int(result.get("alerted", 0)),
            int(result.get("classification_error", 0)),
            result["processed_at"],
            result.get("conversation_id"),
        ]
        self.query(sql, params)

    def get_unattributed_recent(self, since_days: int = 3) -> list[dict]:
        return self.query(
            """SELECT message_id, conversation_id, from_addr, subject, classification
               FROM email_triage_log
               WHERE source = 'outlook'
                 AND classification IN ('URGENT', 'NEEDS_ACTION')
                 AND conversation_id IS NOT NULL
                 AND conversation_id != ''
                 AND replied_at IS NULL
                 AND processed_at >= datetime('now', ? || ' days')""",
            [f"-{since_days}"],
        )

    def mark_replied(self, message_id: str, replied_at: str) -> None:
        self.query(
            "UPDATE email_triage_log SET replied_at = ? WHERE message_id = ?",
            [replied_at, message_id],
        )

    def get_kv(self, key: str) -> Optional[str]:
        rows = self.query(
            "SELECT value FROM mahler_kv WHERE key = ? LIMIT 1",
            [key],
        )
        if rows:
            return rows[0].get("value")
        return None

    def set_kv(self, key: str, value: str) -> None:
        self.query(
            "INSERT INTO mahler_kv (key, value, updated_at) VALUES (?, ?, datetime('now'))"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            [key, value],
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

    def get_recent_project_log(self, days: int = 7) -> list[dict]:
        return self.query(
            "SELECT project, entry_type, summary, git_ref, created_at FROM project_log "
            "WHERE created_at >= datetime('now', ? || ' days') ORDER BY created_at DESC",
            [f"-{days}"],
        )

    def insert_project_log(self, project: str, entry_type: str, summary: str, git_ref: str) -> None:
        self.query(
            "INSERT INTO project_log (project, entry_type, summary, git_ref) VALUES (?, ?, ?, ?)",
            [project, entry_type, summary, git_ref],
        )

    def insert_session_heartbeat(self, project: str, git_ref: str, branch: str) -> None:
        self.query(
            "INSERT INTO project_log (project, entry_type, git_ref, branch) VALUES (?, 'session', ?, ?)",
            [project, git_ref, branch],
        )

    def _migrate_project_log_v2(self) -> None:
        cols = self.query("PRAGMA table_info(project_log)", [])
        col_names = {c["name"] for c in cols}
        if "branch" in col_names:
            return
        self.query(
            """CREATE TABLE project_log_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                entry_type TEXT NOT NULL CHECK(entry_type IN ('win', 'blocker', 'session')),
                summary TEXT,
                git_ref TEXT,
                branch TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )""",
            [],
        )
        self.query(
            "INSERT INTO project_log_v2 (id, project, entry_type, summary, git_ref, created_at) "
            "SELECT id, project, entry_type, summary, git_ref, created_at FROM project_log",
            [],
        )
        self.query("DROP TABLE project_log", [])
        self.query("ALTER TABLE project_log_v2 RENAME TO project_log", [])

    def _add_column_if_missing(self, table: str, column: str, col_type: str) -> None:
        """Add a column to a table if not already present.

        Silently ignores the SQLite duplicate-column error. Raises on any other error.
        All arguments must be string literals from call sites — never user-controlled input.
        """
        try:
            self.query(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}", [])
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "duplicate column" not in msg and "already has a column" not in msg:
                raise

    def ensure_tables(self) -> None:
        self.query(
            """CREATE TABLE IF NOT EXISTS email_triage_log (
    message_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    from_addr TEXT,
    subject TEXT,
    received_at TEXT,
    classification TEXT NOT NULL,
    summary TEXT,
    alerted INTEGER DEFAULT 0,
    classification_error INTEGER DEFAULT 0,
    processed_at TEXT NOT NULL
)""",
            [],
        )
        self.query(
            """CREATE TABLE IF NOT EXISTS triage_state (
    source TEXT PRIMARY KEY,
    last_run TEXT,
    last_error TEXT
)""",
            [],
        )
        self.query(
            """CREATE TABLE IF NOT EXISTS mahler_kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
            [],
        )
        self.query(
            """CREATE TABLE IF NOT EXISTS priority_map (
    version INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
            [],
        )
        self.query(
            """CREATE TABLE IF NOT EXISTS project_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT NOT NULL,
    entry_type TEXT NOT NULL CHECK(entry_type IN ('win', 'blocker', 'session')),
    summary TEXT,
    git_ref TEXT,
    branch TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)""",
            [],
        )
        self._migrate_project_log_v2()
        self._add_column_if_missing("email_triage_log", "conversation_id", "TEXT")
        self._add_column_if_missing("email_triage_log", "replied_at", "TEXT")
