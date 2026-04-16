"""
Google Calendar CLI for Mahler.

Usage:
    python3 gcal.py list [--days N] [--hours-ahead N]
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
        print(f"{evt['start']}  {evt['summary']}")
        if evt.get("attendees"):
            print(f"  Attendees: {', '.join(evt['attendees'])}")
        if evt.get("description"):
            print(f"  {evt['description'][:80]}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler Google Calendar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--days", type=int, default=7)
    p_list.add_argument("--hours-ahead", dest="hours_ahead", type=int, default=None)

    # create subcommand added in Task 5
    sub.add_parser("create")

    args = parser.parse_args(argv)
    if args.command == "list":
        cmd_list(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
