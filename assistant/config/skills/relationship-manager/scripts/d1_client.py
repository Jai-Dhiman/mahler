import sys
from pathlib import Path

_UPDATE_QUERIES = {
    "name":    "UPDATE contacts SET name    = ? WHERE lower(name) = lower(?)",
    "email":   "UPDATE contacts SET email   = ? WHERE lower(name) = lower(?)",
    "type":    "UPDATE contacts SET type    = ? WHERE lower(name) = lower(?)",
    "context": "UPDATE contacts SET context = ? WHERE lower(name) = lower(?)",
}

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_base import D1Client as _D1Base


class D1Client(_D1Base):
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
        sql = _UPDATE_QUERIES.get(field)
        if sql is None:
            raise ValueError(f"Cannot update field {field!r}. Allowed: {sorted(_UPDATE_QUERIES)}")
        self.query(sql, [value, name])

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
