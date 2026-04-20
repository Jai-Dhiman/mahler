import json
import re
import ssl
import urllib.error
import urllib.request
from typing import Optional

_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')
_URL_TEMPLATE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}"
    "/d1/database/{database_id}/query"
)


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
        self._url = _URL_TEMPLATE.format(
            account_id=account_id, database_id=database_id
        )

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
            raise RuntimeError(
                f"D1 API error (connection failed): {exc.reason}"
            ) from exc
        if status != 200:
            raise RuntimeError(
                f"D1 API error {status}: {raw.decode('utf-8', errors='replace')}"
            )
        data = json.loads(raw)
        if not data.get("success") or data.get("errors"):
            raise RuntimeError(f"D1 query failed: {data.get('errors', [])}")
        results = data.get("result", [])
        if not results:
            return []
        return results[0].get("results") or []

    def ensure_table(self) -> None:
        """Create reflection_log table if it does not exist."""
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
        """Insert one reflection entry. Raises RuntimeError on D1 failure."""
        self.query(
            "INSERT INTO reflection_log (week_of, raw_text) VALUES (?, ?)",
            [week_of, raw_text],
        )
