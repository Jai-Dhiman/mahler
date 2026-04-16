import json
import re
import ssl
import urllib.request
from typing import Optional

_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')
_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
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
        self._url = _URL_TEMPLATE.format(account_id=account_id, database_id=database_id)

    def query(self, sql: str, params: Optional[list] = None) -> list[dict]:
        """Execute SQL against D1. Returns list of row dicts. Raises on error."""
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
            raise RuntimeError(f"D1 query failed: {data.get('errors', [])}")
        results = data.get("result", [])
        if not results:
            return []
        return results[0].get("results") or []

    def is_already_notified(self, event_id: str) -> bool:
        """Return True if event_id exists in meeting_prep_log."""
        rows = self.query(
            "SELECT event_id FROM meeting_prep_log WHERE event_id = ? LIMIT 1",
            [event_id],
        )
        return len(rows) > 0

    def insert_meeting_prep(self, event_id: str, summary: str, start_time: str) -> None:
        """Insert a meeting_prep_log row. Idempotent via INSERT OR IGNORE."""
        self.query(
            "INSERT OR IGNORE INTO meeting_prep_log (event_id, summary, start_time, notified_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            [event_id, summary, start_time],
        )

    def ensure_meeting_prep_table(self) -> None:
        """Create meeting_prep_log table if it does not exist. Safe to call on every run."""
        self.query(
            """CREATE TABLE IF NOT EXISTS meeting_prep_log (
    event_id    TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,
    start_time  TEXT NOT NULL,
    notified_at TEXT NOT NULL
)""",
            [],
        )
