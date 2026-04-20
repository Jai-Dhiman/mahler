"""
Google Calendar CLI for Mahler.

Usage:
    python3 gcal.py list [--days N] [--hours-ahead N]
    python3 gcal.py upcoming --min-minutes N --max-minutes N
    python3 gcal.py create --title TITLE --start ISO8601 --end ISO8601 [--attendees email1,email2] [--description TEXT]
"""
import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))


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

import gcal_client  # noqa: E402


def _get_credentials() -> tuple[str, str, str]:
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN")
    if not client_id:
        raise RuntimeError("GMAIL_CLIENT_ID environment variable is not set")
    if not client_secret:
        raise RuntimeError("GMAIL_CLIENT_SECRET environment variable is not set")
    if not refresh_token:
        raise RuntimeError("GMAIL_REFRESH_TOKEN environment variable is not set")
    return client_id, client_secret, refresh_token


def cmd_list(args: argparse.Namespace) -> None:
    client_id, client_secret, refresh_token = _get_credentials()
    access_token = gcal_client.refresh_access_token(client_id, client_secret, refresh_token)
    now = datetime.now(timezone.utc)
    if args.hours_ahead is not None:
        time_max = (now + timedelta(hours=args.hours_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        time_max = (now + timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    time_min = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    events = gcal_client.list_events(access_token, time_min, time_max)
    if not events:
        print("No upcoming events.")
        return
    for evt in events:
        print(f"{evt['id']}  {evt['start']}  {evt['summary']}")
        if evt.get("attendees"):
            print(f"  Attendees: {', '.join(evt['attendees'])}")
        if evt.get("description"):
            print(f"  {evt['description'][:80]}")


def _parse_iso8601(dt_str: str) -> datetime:
    """Parse an ISO 8601 datetime string (with or without offset) into a UTC-aware datetime."""
    dt_str = dt_str.strip()
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str).astimezone(timezone.utc)


def _is_blocked(evt: dict, skip_keywords: list[str]) -> bool:
    """Return True if the event title or description matches any blocked keyword (case-insensitive)."""
    haystack = (evt.get("summary", "") + " " + evt.get("description", "")).lower()
    return any(kw in haystack for kw in skip_keywords)


def cmd_upcoming(args: argparse.Namespace) -> None:
    """Print events whose start time falls in [min_minutes, max_minutes] from now (UTC). Skips all-day events."""
    client_id, client_secret, refresh_token = _get_credentials()
    access_token = gcal_client.refresh_access_token(client_id, client_secret, refresh_token)
    skip_keywords = [kw.strip().lower() for kw in args.skip_keywords.split(",") if kw.strip()] if args.skip_keywords else []
    now = datetime.now(timezone.utc)
    time_min = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_max = (now + timedelta(minutes=args.max_minutes + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    events = gcal_client.list_events(access_token, time_min, time_max)
    found = False
    for evt in events:
        start_str = evt["start"]
        if "T" not in start_str:
            continue  # all-day event
        start_utc = _parse_iso8601(start_str)
        minutes_away = (start_utc - now).total_seconds() / 60
        if args.min_minutes <= minutes_away <= args.max_minutes:
            if skip_keywords and _is_blocked(evt, skip_keywords):
                continue
            found = True
            print(f"{evt['id']}  {start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}  {evt['summary']}")
            if evt.get("attendees"):
                print(f"  Attendees: {', '.join(evt['attendees'])}")
            if evt.get("description"):
                print(f"  {evt['description'][:80]}")
    if not found:
        print("No meetings in window.")


def cmd_create(args: argparse.Namespace) -> None:
    client_id, client_secret, refresh_token = _get_credentials()
    access_token = gcal_client.refresh_access_token(client_id, client_secret, refresh_token)
    attendees = [e.strip() for e in args.attendees.split(",")] if args.attendees else None
    result = gcal_client.create_event(
        access_token=access_token,
        summary=args.title,
        start=args.start,
        end=args.end,
        attendees=attendees,
        description=args.description,
    )
    print(f"Created: {result['id']} \u2014 {result['summary']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler Google Calendar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--days", type=int, default=7)
    p_list.add_argument("--hours-ahead", dest="hours_ahead", type=int, default=None)

    p_upcoming = sub.add_parser("upcoming")
    p_upcoming.add_argument("--min-minutes", dest="min_minutes", type=int, default=45)
    p_upcoming.add_argument("--max-minutes", dest="max_minutes", type=int, default=75)
    p_upcoming.add_argument("--skip-keywords", dest="skip_keywords", default=None,
                            help="Comma-separated keywords; events matching any are skipped")

    p_create = sub.add_parser("create")
    p_create.add_argument("--title", required=True)
    p_create.add_argument("--start", required=True)
    p_create.add_argument("--end", required=True)
    p_create.add_argument("--attendees", default=None)
    p_create.add_argument("--description", default=None)

    args = parser.parse_args(argv)
    dispatch = {"list": cmd_list, "upcoming": cmd_upcoming, "create": cmd_create}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
