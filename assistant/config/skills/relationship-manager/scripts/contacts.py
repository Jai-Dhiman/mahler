import argparse
import os
import sys
from datetime import date

from d1_client import D1Client
from notion_client import NotionClient


def _supplement_env_from_hermes() -> None:
    env_path = os.path.expanduser("~/.hermes/.env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key not in os.environ:
                os.environ[key] = val


def _d1_client() -> D1Client:
    return D1Client(
        account_id=os.environ["CF_ACCOUNT_ID"],
        database_id=os.environ["CF_D1_DATABASE_ID"],
        api_token=os.environ["CF_API_TOKEN"],
    )


def _notion_client() -> NotionClient:
    return NotionClient(
        api_token=os.environ["NOTION_API_TOKEN"],
        database_id=os.environ["NOTION_DATABASE_ID"],
    )


def _cmd_add(args: argparse.Namespace) -> None:
    db = _d1_client()
    db.ensure_table()
    db.upsert_contact(args.name, args.email, args.type, args.context or "")
    print(f"Added: {args.name} ({args.type})")


def main(argv=None) -> None:
    _supplement_env_from_hermes()
    parser = argparse.ArgumentParser(prog="contacts")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--email", required=True)
    p_add.add_argument("--type", required=True, choices=["professional", "personal"])
    p_add.add_argument("--context", default="")

    # placeholders for later subcommands
    sub.add_parser("summarize").add_argument("--name", required=True)
    sub.add_parser("list").add_argument("--type", choices=["professional", "personal"])
    p_tt = sub.add_parser("talked-to")
    p_tt.add_argument("--name", required=True)
    p_up = sub.add_parser("update")
    p_up.add_argument("--name", required=True)
    p_up.add_argument("--field", required=True)
    p_up.add_argument("--value", required=True)
    sub.add_parser("delete").add_argument("--name", required=True)
    sub.add_parser("sync-calendar").add_argument("--days", type=int, default=1)

    args = parser.parse_args(argv)
    dispatch = {
        "add": _cmd_add,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        print(f"Command '{args.command}' not yet implemented", file=sys.stderr)
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
