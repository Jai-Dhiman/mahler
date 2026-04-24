"""
Post stdin as a Discord message via DISCORD_TRIAGE_WEBHOOK.

Usage:
    echo "message" | python3 post_discord.py
"""
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def _load_env() -> None:
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


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def main() -> None:
    _load_env()
    webhook = os.environ.get("DISCORD_TRIAGE_WEBHOOK")
    if not webhook:
        raise RuntimeError("DISCORD_TRIAGE_WEBHOOK must be set")
    parsed = urlparse(webhook)
    if parsed.scheme != "https" or parsed.hostname != "discord.com":
        raise RuntimeError("DISCORD_TRIAGE_WEBHOOK must be an https://discord.com URL")
    content = sys.stdin.read().strip()
    if not content:
        raise RuntimeError("No message content on stdin")
    if len(content) > 2000:
        content = content[:1997] + "..."
    body = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "mahler-meeting-followthrough/1.0 (+https://github.com/Jai-Dhiman)",
        },
    )
    try:
        with _OPENER.open(req) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"Discord webhook returned HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Discord webhook returned HTTP {exc.code}") from exc
    print("OK")


if __name__ == "__main__":
    main()
