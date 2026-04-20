import json
import re
import ssl
import urllib.request

_ID_RE = re.compile(r'^[a-zA-Z0-9_-]+$')
_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query"

_ALLOWED_UPDATE_FIELDS = frozenset({"name", "email", "type", "context"})


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

    def query(self, sql: str, params: list | None = None) -> list[dict]:
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

    def ensure_table(self) -> None:
        self.query("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL CHECK(type IN ('professional', 'personal')),
                last_contact TEXT,
                context TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

    def get_contact(self, name: str) -> dict:
        rows = self.query(
            "SELECT id, name, email, type, last_contact, context, created_at "
            "FROM contacts WHERE lower(name) = lower(?) LIMIT 1",
            [name],
        )
        if not rows:
            raise RuntimeError(f"Contact not found: {name!r}")
        return rows[0]

    def list_contacts(self, type: str | None = None) -> list[dict]:
        if type is not None:
            return self.query(
                "SELECT id, name, email, type, last_contact, context, created_at "
                "FROM contacts WHERE type = ? ORDER BY name",
                [type],
            )
        return self.query(
            "SELECT id, name, email, type, last_contact, context, created_at "
            "FROM contacts ORDER BY name"
        )

    def touch_last_contact(self, name: str, date: str) -> None:
        self.query(
            "UPDATE contacts SET last_contact = ? WHERE lower(name) = lower(?)",
            [date, name],
        )

    def update_contact(self, name: str, field: str, value: str) -> None:
        if field not in _ALLOWED_UPDATE_FIELDS:
            raise ValueError(f"Cannot update field {field!r}. Allowed: {sorted(_ALLOWED_UPDATE_FIELDS)}")
        self.query(
            f"UPDATE contacts SET {field} = ? WHERE lower(name) = lower(?)",
            [value, name],
        )

    def delete_contact(self, name: str) -> None:
        self.query(
            "DELETE FROM contacts WHERE lower(name) = lower(?)",
            [name],
        )

    def upsert_contact(self, name: str, email: str, contact_type: str, context: str) -> None:
        self.query(
            """
            INSERT INTO contacts (name, email, type, context, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                context = excluded.context
            """,
            [name, email, contact_type, context],
        )
