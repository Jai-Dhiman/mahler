import json
import re
import ssl
import urllib.request
from typing import Optional


_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')

_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"


def _build_https_opener() -> urllib.request.OpenerDirector:
    """
    Build a urllib opener restricted to HTTPS with certificate verification
    enforced. FileHandler and FTPHandler are intentionally excluded to prevent
    file:// and ftp:// scheme access.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    https_handler = urllib.request.HTTPSHandler(context=ctx)
    opener = urllib.request.OpenerDirector()
    opener.add_handler(https_handler)
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


class D1Client:
    def __init__(self, account_id: str, database_id: str, api_token: str):
        if not _ID_RE.match(account_id):
            raise ValueError(f"Invalid account_id: {account_id!r}")
        if not _ID_RE.match(database_id):
            raise ValueError(f"Invalid database_id: {database_id!r}")

        self.account_id = account_id
        self.database_id = database_id
        self.api_token = api_token
        self._url = _URL_TEMPLATE.format(
            account_id=account_id,
            database_id=database_id,
        )

    def query(self, sql: str, params: Optional[list] = None) -> list[dict]:
        """Execute SQL, return list of row dicts. Raises on error."""
        body = json.dumps({"sql": sql, "params": params or []}).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
        )

        with _OPENER.open(req) as resp:
            status = resp.status
            raw = resp.read()

        if status != 200:
            raise RuntimeError(f"D1 API error {status}: {raw.decode('utf-8', errors='replace')}")

        data = json.loads(raw)

        if not data.get("success") or data.get("errors"):
            errors = data.get("errors", [])
            raise RuntimeError(f"D1 query failed: {errors}")

        results = data.get("result", [])
        if not results:
            return []

        return results[0].get("results") or []

    def is_already_processed(self, message_id: str) -> bool:
        """Return True if message_id exists in email_triage_log."""
        rows = self.query(
            "SELECT message_id FROM email_triage_log WHERE message_id = ? LIMIT 1",
            [message_id],
        )
        return len(rows) > 0

    def insert_triage_result(self, result: dict) -> None:
        """Insert one triage result. result dict has keys matching table columns."""
        sql = (
            "INSERT OR IGNORE INTO email_triage_log "
            "(message_id, source, from_addr, subject, received_at, classification, "
            "summary, alerted, classification_error, processed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
        ]
        self.query(sql, params)

    def get_kv(self, key: str) -> Optional[str]:
        """Read a value from mahler_kv. Returns None if the key doesn't exist."""
        rows = self.query(
            "SELECT value FROM mahler_kv WHERE key = ? LIMIT 1",
            [key],
        )
        if rows:
            return rows[0].get("value")
        return None

    def set_kv(self, key: str, value: str) -> None:
        """Write a value to mahler_kv. Upserts on conflict."""
        self.query(
            "INSERT INTO mahler_kv (key, value, updated_at) VALUES (?, ?, datetime('now'))"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            [key, value],
        )

    def get_priority_map(self) -> str:
        """Read current priority map content from D1. Raises RuntimeError if no row exists."""
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
        """Write updated priority map content to D1, incrementing version."""
        self.query(
            "INSERT INTO priority_map (content, version, updated_at) "
            "VALUES (?, COALESCE((SELECT MAX(version) FROM priority_map), 0) + 1, datetime('now'))",
            [content],
        )

    def insert_project_log(self, project: str, entry_type: str, summary: str, git_ref: str) -> None:
        """Insert one project log entry. Raises RuntimeError on D1 failure."""
        self.query(
            "INSERT INTO project_log (project, entry_type, summary, git_ref) VALUES (?, ?, ?, ?)",
            [project, entry_type, summary, git_ref],
        )

    def ensure_tables(self) -> None:
        """Create tables if they don't exist. Safe to call on every run."""
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
    entry_type TEXT NOT NULL CHECK(entry_type IN ('win', 'blocker')),
    summary TEXT NOT NULL,
    git_ref TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)""",
            [],
        )
