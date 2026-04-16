"""
Meeting prep deduplication CLI for Mahler.

Usage:
    python3 dedup.py check --event-id EVENT_ID
        Exit 0: not yet notified -- safe to proceed.
        Exit 1: already notified -- stop.
        Non-zero + RuntimeError: D1 failure -- do not proceed.

    python3 dedup.py log --event-id EVENT_ID --summary SUMMARY --start-time ISO8601
"""
import argparse
import os
import sys
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

from d1_client import D1Client  # noqa: E402


def _get_client() -> D1Client:
    account_id = os.environ.get("CF_ACCOUNT_ID")
    database_id = os.environ.get("CF_D1_DATABASE_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if not account_id:
        raise RuntimeError("CF_ACCOUNT_ID environment variable is not set")
    if not database_id:
        raise RuntimeError("CF_D1_DATABASE_ID environment variable is not set")
    if not api_token:
        raise RuntimeError("CF_API_TOKEN environment variable is not set")
    return D1Client(account_id, database_id, api_token)


def cmd_check(args: argparse.Namespace) -> None:
    client = _get_client()
    client.ensure_meeting_prep_table()
    if client.is_already_notified(args.event_id):
        sys.exit(1)


def cmd_log(args: argparse.Namespace) -> None:
    client = _get_client()
    client.insert_meeting_prep(args.event_id, args.summary, args.start_time)
    print(f"Logged: {args.event_id}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler meeting prep dedup")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--event-id", dest="event_id", required=True)

    p_log = sub.add_parser("log")
    p_log.add_argument("--event-id", dest="event_id", required=True)
    p_log.add_argument("--summary", required=True)
    p_log.add_argument("--start-time", dest="start_time", required=True)

    args = parser.parse_args(argv)
    dispatch = {"check": cmd_check, "log": cmd_log}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
