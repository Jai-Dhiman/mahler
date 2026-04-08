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
