"""
Calendar-aware pre_llm_call plugin for Mahler.
Injects a one-line upcoming meeting reminder into every chat turn.
Returns None silently on any failure -- must never break a chat turn.
"""
import logging
import os
import sys
from datetime import datetime, timezone
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


def _query_upcoming_meeting() -> dict | None:
    """Query meeting_prep_log for the next meeting starting within 2 hours."""
    _load_hermes_env()
    meeting_prep_scripts = str(Path.home() / ".hermes" / "skills" / "meeting-prep" / "scripts")
    if meeting_prep_scripts not in sys.path:
        sys.path.insert(0, meeting_prep_scripts)
    from d1_client import D1Client
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    database_id = os.environ.get("CF_D1_DATABASE_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    if not account_id or not database_id or not api_token:
        return None
    client = D1Client(account_id, database_id, api_token)
    return client.get_upcoming_meeting()


def upcoming_meeting_context(
    session_id: str,
    user_message: str,
    is_first_turn: bool,
    _now: datetime | None = None,
    **kwargs,
) -> dict | None:
    """Called before each LLM turn. Injects upcoming meeting context or returns None."""
    try:
        meeting = _query_upcoming_meeting()
        if not meeting:
            return None
        now = _now or datetime.now(timezone.utc)
        start = datetime.fromisoformat(meeting["start_time"].replace("Z", "+00:00"))
        minutes_until = int((start - now).total_seconds() / 60)
        return {"context": f"Upcoming meeting in {minutes_until}min: {meeting['summary']}"}
    except Exception as exc:
        logger.debug("calendar-aware plugin error: %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", upcoming_meeting_context)
