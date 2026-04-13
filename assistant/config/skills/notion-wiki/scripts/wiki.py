"""
Notion wiki read CLI (Hermes side).

Usage:
    python3 wiki.py index --db sources|concepts [--limit N]
    python3 wiki.py read --id PAGE_ID
    python3 wiki.py search --query "text" [--limit N]
"""

import argparse
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))


def _load_env_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def _supplement_env_from_hermes() -> None:
    hermes_env = Path.home() / ".hermes" / ".env"
    _load_env_file(str(hermes_env))


_supplement_env_from_hermes()

from notion_client import NotionWikiReader  # noqa: E402


def _get_reader() -> NotionWikiReader:
    token = os.environ.get("NOTION_WIKI_READ_TOKEN", "")
    src = os.environ.get("NOTION_WIKI_SOURCES_DB_ID", "")
    con = os.environ.get("NOTION_WIKI_CONCEPTS_DB_ID", "")
    if not token:
        raise RuntimeError("NOTION_WIKI_READ_TOKEN is not set")
    if not src:
        raise RuntimeError("NOTION_WIKI_SOURCES_DB_ID is not set")
    if not con:
        raise RuntimeError("NOTION_WIKI_CONCEPTS_DB_ID is not set")
    return NotionWikiReader(token=token, sources_db_id=src, concepts_db_id=con)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler notion-wiki read CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index")
    p_index.add_argument("--db", choices=["sources", "concepts"], required=True)
    p_index.add_argument("--limit", type=int, default=100)

    p_read = sub.add_parser("read")
    p_read.add_argument("--id", dest="page_id", required=True)

    p_search = sub.add_parser("search")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--limit", type=int, default=10)

    args = parser.parse_args(argv)
    dispatch = {
        "index": cmd_index,
        "read": cmd_read,
        "search": cmd_search,
    }
    dispatch[args.command](args)


def cmd_index(args: argparse.Namespace) -> None:
    raise NotImplementedError  # Task F2


def cmd_read(args: argparse.Namespace) -> None:
    raise NotImplementedError  # Task F3


def cmd_search(args: argparse.Namespace) -> None:
    reader = _get_reader()
    hits = reader.search(args.query, limit=args.limit)
    if not hits:
        print("No results.")
        return
    for hit in hits:
        print(f"[{hit['id']}] ({hit['db']}) {hit['title']}")


if __name__ == "__main__":
    main()
