import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_hermes_env() -> None:
    hermes_env = Path.home() / ".hermes" / ".env"
    if not hermes_env.exists():
        return
    with open(hermes_env, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def _query_project_log() -> list[dict]:
    _load_hermes_env()
    triage_scripts = str(Path.home() / ".hermes" / "skills" / "email-triage" / "scripts")
    if triage_scripts not in sys.path:
        sys.path.insert(0, triage_scripts)
    from d1_client import D1Client
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    database_id = os.environ.get("CF_D1_DATABASE_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    if not account_id or not database_id or not api_token:
        return []
    client = D1Client(account_id, database_id, api_token)
    return client.get_recent_project_log(days=3)


def _format_entries(rows: list[dict]) -> str:
    sessions = [r for r in rows if r.get("entry_type") == "session"]
    events = [r for r in rows if r.get("entry_type") in ("win", "blocker")]

    lines = ["Recent project activity (last 3 days):\n"]

    if sessions:
        counts: dict[str, int] = {}
        for r in sessions:
            p = r.get("project", "unknown")
            counts[p] = counts.get(p, 0) + 1
        parts = [f"{p} ({n} session{'s' if n > 1 else ''})" for p, n in counts.items()]
        lines.append(f"Active: {', '.join(parts)}")

    for row in events:
        project = row.get("project", "unknown")
        entry_type = row.get("entry_type", "").upper()
        summary = row.get("summary", "")
        created_at = (row.get("created_at") or "")[:10]
        lines.append(f"[{project}] {created_at} — {entry_type}: {summary}")

    return "\n".join(lines)


def project_context(
    session_id: str,
    user_message: str,
    is_first_turn: bool,
    **kwargs,
) -> dict | None:
    if not is_first_turn:
        return None
    try:
        rows = _query_project_log()
        if not rows:
            return None
        return {"context": _format_entries(rows)}
    except Exception as exc:
        logger.debug("project-context plugin error: %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", project_context)
