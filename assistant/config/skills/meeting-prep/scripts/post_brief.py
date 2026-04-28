"""
Post a meeting prep brief to the Discord triage webhook.

Usage:
    python3 post_brief.py --title TITLE --start ISO8601_UTC --synthesis TEXT
                          [--attendees TEXT] [--emails TEXT] [--tasks TEXT] [--wiki TEXT]
"""
import argparse
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path


def _supplement_env_from_hermes() -> None:
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


_supplement_env_from_hermes()


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
    opener.add_handler(urllib.request.UnknownHandler())
    return opener


_OPENER = _build_https_opener()


def _format_start(iso_utc: str) -> str:
    try:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        la = dt.astimezone(ZoneInfo("America/Los_Angeles"))
        return la.strftime("%a, %b %-d at %-I:%M %p %Z")
    except Exception:
        return iso_utc


def build_payload(
    title: str,
    start: str,
    synthesis: str,
    attendees: str | None,
    emails: str | None,
    tasks: str | None,
    wiki: str | None,
) -> dict:
    fields = [
        {"name": "Meeting", "value": f"{title} — {_format_start(start)}", "inline": False},
    ]
    if attendees:
        fields.append({"name": "Attendees", "value": attendees, "inline": False})
    if emails:
        fields.append({"name": "Recent emails", "value": emails[:1000], "inline": False})
    if tasks:
        fields.append({"name": "Open tasks", "value": tasks[:1000], "inline": False})
    if wiki:
        fields.append({"name": "Wiki context", "value": wiki[:500], "inline": False})
    fields.append({"name": "What to know", "value": synthesis[:1500], "inline": False})
    return {
        "embeds": [
            {
                "title": "Meeting Prep Brief",
                "color": 3447003,
                "fields": fields,
                "footer": {"text": "Mahler Meeting Prep"},
            }
        ]
    }


def post_brief(webhook_url: str, payload: dict) -> None:
    parsed_scheme = webhook_url.split("://")[0] if "://" in webhook_url else ""
    if parsed_scheme != "https":
        raise RuntimeError(
            f"DISCORD_TRIAGE_WEBHOOK must be an HTTPS URL, got scheme: {parsed_scheme!r}"
        )
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(webhook_url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "DiscordBot (mahler, 1.0)")
    try:
        with _OPENER.open(req) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Discord webhook failed: {exc.code} {body}")
    if status not in (200, 204):
        raise RuntimeError(f"Discord webhook failed: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post a meeting prep brief to Discord.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--start", required=True, help="Meeting start time in UTC (ISO 8601)")
    parser.add_argument("--synthesis", required=True, help="3-5 bullet points of what to know")
    parser.add_argument("--attendees", default=None)
    parser.add_argument("--emails", default=None, help="Recent email context")
    parser.add_argument("--tasks", default=None, help="Open tasks")
    parser.add_argument("--wiki", default=None, help="Wiki context")
    args = parser.parse_args()

    webhook_url = os.environ.get("DISCORD_TRIAGE_WEBHOOK")
    if not webhook_url:
        raise RuntimeError("DISCORD_TRIAGE_WEBHOOK environment variable is not set")

    payload = build_payload(
        title=args.title,
        start=args.start,
        synthesis=args.synthesis,
        attendees=args.attendees,
        emails=args.emails,
        tasks=args.tasks,
        wiki=args.wiki,
    )
    post_brief(webhook_url, payload)
    print("Brief sent.")


if __name__ == "__main__":
    main()
