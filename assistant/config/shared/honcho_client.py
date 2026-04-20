# config/shared/honcho_client.py
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


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


def _load_config() -> dict:
    honcho_json = Path.home() / ".hermes" / "honcho.json"
    if not honcho_json.exists():
        raise RuntimeError(f"honcho.json not found at {honcho_json}")
    with open(honcho_json) as f:
        cfg = json.load(f)
    api_key = os.environ.get("HONCHO_API_KEY", "")
    if not api_key:
        raise RuntimeError("HONCHO_API_KEY environment variable not set")
    return {
        "workspace_id": cfg["workspace"],
        "ai_peer_id": cfg["aiPeer"],
        "user_peer_id": cfg["peerName"],
        "api_key": api_key,
    }


def _build_conclusions_client(cfg: dict):
    from honcho import Honcho
    honcho = Honcho(workspace_id=cfg["workspace_id"], api_key=cfg["api_key"])
    return honcho.peer(cfg["ai_peer_id"]).conclusions_of(cfg["user_peer_id"])


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def conclude(text: str, session_id: str = "mahler-memory") -> None:
    """Write one durable conclusion to Honcho. Raises RuntimeError on failure."""
    _load_hermes_env()
    cfg = _load_config()
    conclusions = _build_conclusions_client(cfg)
    try:
        conclusions.create([{"content": text, "session_id": session_id}])
    except Exception as exc:
        raise RuntimeError(f"Honcho conclude failed: {exc}") from exc


def list_conclusions(since_days: int = 30) -> list:
    """Return conclusions written within the last since_days days. Raises RuntimeError on failure."""
    _load_hermes_env()
    cfg = _load_config()
    conclusions = _build_conclusions_client(cfg)
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    try:
        all_items = list(conclusions.list())
    except Exception as exc:
        raise RuntimeError(f"Honcho list_conclusions failed: {exc}") from exc
    return [
        c for c in all_items
        if getattr(c, "created_at", None) is None
        or _parse_dt(c.created_at) >= cutoff
    ]


def query_conclusions(query: str, top_k: int = 10) -> list:
    """Semantic search over conclusions. Raises RuntimeError on failure."""
    _load_hermes_env()
    cfg = _load_config()
    conclusions = _build_conclusions_client(cfg)
    try:
        return list(conclusions.query(query))
    except Exception as exc:
        raise RuntimeError(f"Honcho query_conclusions failed: {exc}") from exc
