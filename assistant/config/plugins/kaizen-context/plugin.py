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


def _query_priority_map() -> str | None:
    """Read current priority map content from D1. Returns None on any error."""
    _load_hermes_env()
    triage_scripts = str(Path.home() / ".hermes" / "skills" / "email-triage" / "scripts")
    if triage_scripts not in sys.path:
        sys.path.insert(0, triage_scripts)
    from d1_client import D1Client
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    database_id = os.environ.get("CF_D1_DATABASE_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    if not account_id or not database_id or not api_token:
        return None
    client = D1Client(account_id, database_id, api_token)
    return client.get_priority_map()


def priority_map_context(
    session_id: str,
    user_message: str,
    is_first_turn: bool,
    **kwargs,
) -> dict | None:
    """Called before each LLM turn. Injects email priority map or returns None."""
    try:
        content = _query_priority_map()
        if not content:
            return None
        return {"context": f"Email priority map (active classification rules):\n\n{content}"}
    except Exception as exc:
        logger.debug("kaizen-context plugin error: %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", priority_map_context)
