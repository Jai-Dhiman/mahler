"""
Fetch recent URGENT/NEEDS_ACTION emails from specific attendees.

Usage:
    python3 email_context.py email-context --attendees "email1@x.com,email2@y.com"
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


def cmd_email_context(args: argparse.Namespace) -> None:
    attendees = [e.strip() for e in args.attendees.split(",") if e.strip()]
    if not attendees:
        raise RuntimeError("--attendees must contain at least one email address")
    client = _get_client()
    placeholders = ",".join("?" for _ in attendees)
    rows = client.query(
        f"SELECT classification, from_addr, subject, summary FROM email_triage_log "
        f"WHERE from_addr IN ({placeholders}) "
        f"AND classification IN ('URGENT', 'NEEDS_ACTION') "
        f"AND processed_at > datetime('now', '-7 days') "
        f"ORDER BY processed_at DESC LIMIT 10",
        attendees,
    )
    if not rows:
        print("No recent flagged emails from these contacts.")
        return
    print("Recent emails (last 7 days, URGENT/NEEDS_ACTION):")
    for row in rows:
        classification = row.get("classification", "")
        from_addr = row.get("from_addr", "")
        subject = row.get("subject", "(no subject)")
        summary = row.get("summary", "")
        print(f"  [{classification}] {from_addr}: {subject} — {summary}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler meeting email context")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("email-context")
    p.add_argument("--attendees", required=True)
    args = parser.parse_args(argv)
    if args.command == "email-context":
        cmd_email_context(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
