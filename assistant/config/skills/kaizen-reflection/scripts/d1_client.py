import json
import re
import ssl
import urllib.error
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
        try:
            with _OPENER.open(req) as resp:
                status = resp.status
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"D1 API error (connection failed): {exc.reason}") from exc
        if status != 200:
            raise RuntimeError(f"D1 API error {status}: {raw.decode('utf-8', errors='replace')}")
        data = json.loads(raw)
        if not data.get("success") or data.get("errors"):
            raise RuntimeError(f"D1 query failed: {data.get('errors', [])}")
        results = data.get("result", [])
        if not results:
            return []
        return results[0].get("results") or []

    def get_triage_patterns(self, since_days: int = 7, min_count: int = 3) -> list[dict]:
        """Return senders appearing >= min_count times at the same tier in the last since_days days."""
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
        """Return triage patterns with occurrence and reply counts.

        Each row: from_addr, classification, occurrence_count, reply_count.
        reply_count uses COUNT(replied_at) which counts non-NULL values only.
        """
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
        """Read the current priority map content from D1. Raises RuntimeError if no row exists."""
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

    def ensure_priority_map_table(self) -> None:
        """Create priority_map table if it does not exist. Does not seed initial content."""
        self.query(
            """CREATE TABLE IF NOT EXISTS priority_map (
    version INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
            [],
        )
