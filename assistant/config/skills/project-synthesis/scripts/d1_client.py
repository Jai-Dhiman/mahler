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

    def get_recent_project_log(self, days: int = 7) -> list[dict]:
        """Return project_log rows from the last N days, newest first."""
        return self.query(
            "SELECT project, entry_type, summary, git_ref, created_at FROM project_log "
            "WHERE created_at >= datetime('now', ? || ' days') ORDER BY created_at DESC",
            [f"-{days}"],
        )

    def insert_project_log(self, project: str, entry_type: str, summary: str, git_ref: str) -> None:
        """Insert one project log entry. Raises RuntimeError on D1 failure."""
        self.query(
            "INSERT INTO project_log (project, entry_type, summary, git_ref) VALUES (?, ?, ?, ?)",
            [project, entry_type, summary, git_ref],
        )

    def insert_session_heartbeat(self, project: str, git_ref: str, branch: str) -> None:
        """Insert a lightweight session heartbeat with no LLM summary."""
        self.query(
            "INSERT INTO project_log (project, entry_type, git_ref, branch) VALUES (?, 'session', ?, ?)",
            [project, git_ref, branch],
        )

