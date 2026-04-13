"""
Notion wiki lint CLI.

Usage:
    python3 lint.py lint
"""

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from notion_client import NotionWikiWriter  # noqa: E402


def _load_env_from_dotenv(path: str) -> None:
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


_DEFAULT_DOTENV = str(Path(__file__).parent.parent / ".env")
_load_env_from_dotenv(_DEFAULT_DOTENV)


def _get_writer() -> NotionWikiWriter:
    token = os.environ.get("NOTION_WIKI_WRITE_TOKEN", "")
    src = os.environ.get("NOTION_WIKI_SOURCES_DB_ID", "")
    con = os.environ.get("NOTION_WIKI_CONCEPTS_DB_ID", "")
    log = os.environ.get("NOTION_WIKI_LOG_DB_ID", "")
    if not token:
        raise RuntimeError("NOTION_WIKI_WRITE_TOKEN is not set")
    if not src or not con or not log:
        raise RuntimeError("NOTION_WIKI_*_DB_ID env vars are not fully set")
    return NotionWikiWriter(token=token, sources_db_id=src, concepts_db_id=con, log_db_id=log)


_WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def cmd_lint(args: argparse.Namespace) -> None:
    writer = _get_writer()
    concepts = writer.list_all_concepts()
    titles_by_lower = {c["title"].strip().lower(): c for c in concepts}

    broken = 0
    for concept in concepts:
        for match in _WIKILINK_PATTERN.finditer(concept["body_markdown"]):
            target = match.group(1).strip().lower()
            if target not in titles_by_lower:
                print(f"Broken wikilink: [[{match.group(1)}]] in concept {concept['title']!r}")
                broken += 1

    print(f"Summary: {broken} broken wikilinks")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler notion-wiki linter")
    sub = parser.add_subparsers(dest="command", required=True)
    p_lint = sub.add_parser("lint")
    args = parser.parse_args(argv)
    if args.command == "lint":
        cmd_lint(args)


if __name__ == "__main__":
    main()
