"""
Conversation history pre_llm_call plugin for Mahler.
On the first turn of a new Hermes session, fetches the last 45 minutes of
Discord channel history and injects it as context so the LLM has continuity
with whatever was discussed before this @mention.
Returns None silently on any failure -- must never break a chat turn.
"""
import json
import logging
import os
import re
import ssl
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_WINDOW_MINUTES = 45
_MAX_MESSAGES = 20
_DISCORD_HOST = "discord.com"

# Snowflake IDs are numeric only
_SNOWFLAKE_RE = re.compile(r"^\d+$")

_bot_id_cache: str | None = None


def _build_https_opener() -> urllib.request.OpenerDirector:
    """
    Build an opener restricted to HTTPS with certificate verification enforced.
    FileHandler and FTPHandler are excluded to prevent file:// and ftp:// access.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


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


def _discord_get(path: str, token: str) -> object:
    """GET a Discord API path under https://discord.com/api/v10."""
    url = f"https://{_DISCORD_HOST}/api/v10{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bot {token}"})
    with _OPENER.open(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def _get_bot_id(token: str) -> str:
    global _bot_id_cache
    if _bot_id_cache is None:
        me = _discord_get("/users/@me", token)
        if not isinstance(me, dict):
            raise RuntimeError("Unexpected response from Discord /users/@me")
        _bot_id_cache = str(me["id"])
    return _bot_id_cache


def _fetch_session_messages(
    channel_id: str,
    token: str,
    bot_id: str,
    cutoff: datetime,
) -> list[dict]:
    """Fetch channel messages within the session window, oldest first."""
    if not _SNOWFLAKE_RE.match(channel_id):
        raise ValueError(f"Invalid channel_id: {channel_id!r}")
    path = f"/channels/{channel_id}/messages?limit={_MAX_MESSAGES}"
    raw = _discord_get(path, token)
    if not isinstance(raw, list):
        return []

    result = []
    for msg in reversed(raw):  # Discord returns newest-first; reverse to chronological
        ts_str = msg.get("timestamp", "")
        if not ts_str:
            continue
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts < cutoff:
            continue
        content = msg.get("content", "").strip()
        if not content:
            continue
        author_id = msg["author"]["id"]
        label = "Mahler" if author_id == bot_id else "You"
        result.append({"label": label, "content": content, "ts": ts})
    return result


def conversation_history_context(
    session_id: str,
    user_message: str,
    is_first_turn: bool,
    _now: datetime | None = None,
    **kwargs,
) -> dict | None:
    """Called before each LLM turn. Injects prior Discord history on session start."""
    if not is_first_turn:
        return None
    try:
        _load_hermes_env()
        token = os.environ.get("DISCORD_BOT_TOKEN", "")
        channel_id = os.environ.get("DISCORD_HOME_CHANNEL", "")
        if not token or not channel_id:
            return None
        now = _now or datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=_WINDOW_MINUTES)
        bot_id = _get_bot_id(token)
        messages = _fetch_session_messages(channel_id, token, bot_id, cutoff)
        # Drop the current user message at the tail (Hermes already has it)
        if messages and messages[-1]["label"] == "You" and user_message in messages[-1]["content"]:
            messages = messages[:-1]
        if not messages:
            return None
        lines = [
            f"[{m['ts'].strftime('%H:%M')}] {m['label']}: {m['content']}"
            for m in messages
        ]
        context = "Recent conversation (last 45 min):\n" + "\n".join(lines)
        return {"context": context}
    except Exception as exc:
        logger.debug("conversation-history plugin error: %s", exc)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", conversation_history_context)
