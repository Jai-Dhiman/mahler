"""
Poll D1 for pending Fathom meetings or mark one as processed.

Usage:
    python3 poll.py fetch
    python3 poll.py mark-done --recording-id 12345
"""
import argparse
import json
import os
import sys
from pathlib import Path


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


_load_env()

sys.path.insert(0, str(Path(__file__).parent))
from d1_client import D1Client


def _client() -> D1Client:
    account_id = os.environ.get("CF_ACCOUNT_ID")
    database_id = os.environ.get("CF_D1_DATABASE_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if not account_id or not database_id or not api_token:
        raise RuntimeError("CF_ACCOUNT_ID, CF_D1_DATABASE_ID, and CF_API_TOKEN must be set")
    return D1Client(account_id, database_id, api_token)


def cmd_fetch(_args: argparse.Namespace) -> None:
    client = _client()
    client.ensure_queue_table()
    rows = client.fetch_pending()
    if not rows:
        print("NO_PENDING_MEETINGS")
        return
    for row in rows:
        attendees = json.loads(row["attendees"]) if isinstance(row["attendees"], str) else row["attendees"]
        attendee_str = ", ".join(
            f"{a['name']} <{a['email']}>" if a.get("name") and a.get("email")
            else a.get("email") or a.get("name") or "unknown"
            for a in attendees
            if a.get("email") or a.get("name")
        ) or "none"
        print(f"RECORDING_ID: {row['recording_id']}")
        print(f"TITLE: {row['title']}")
        print(f"ATTENDEES: {attendee_str}")
        print("SUMMARY:")
        print(row["summary"])
        print("---END_MEETING---")


def cmd_mark_done(args: argparse.Namespace) -> None:
    client = _client()
    client.mark_done(args.recording_id)
    print(f"Marked recording {args.recording_id} as processed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="Print pending meetings to stdout")

    done_p = sub.add_parser("mark-done", help="Mark a meeting as processed")
    done_p.add_argument("--recording-id", type=int, required=True)

    args = parser.parse_args()
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "mark-done":
        cmd_mark_done(args)


if __name__ == "__main__":
    main()
