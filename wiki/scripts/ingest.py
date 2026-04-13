# mahler/wiki/scripts/ingest.py
"""
Notion wiki ingest CLI.

Usage:
    python3 ingest.py ingest --url URL --title TITLE --type TYPE --summary-file PATH [--tags T1,T2] [--concepts C1,C2] [--ingested YYYY-MM-DD]
"""

import argparse
import os
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
        raise RuntimeError("NOTION_WIKI_WRITE_TOKEN is not set (check mahler/wiki/.env)")
    if not src:
        raise RuntimeError("NOTION_WIKI_SOURCES_DB_ID is not set")
    if not con:
        raise RuntimeError("NOTION_WIKI_CONCEPTS_DB_ID is not set")
    if not log:
        raise RuntimeError("NOTION_WIKI_LOG_DB_ID is not set")
    return NotionWikiWriter(token=token, sources_db_id=src, concepts_db_id=con, log_db_id=log)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler notion-wiki ingester")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--url", required=True)
    p_ingest.add_argument("--title", required=True)
    p_ingest.add_argument("--type", dest="type_", required=True,
                          choices=["paper", "article", "post", "video", "other"])
    p_ingest.add_argument("--summary-file", dest="summary_file", required=True)
    p_ingest.add_argument("--tags", default="")
    p_ingest.add_argument("--concepts", default="")
    p_ingest.add_argument("--ingested", default=None)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        cmd_ingest(args)


def cmd_ingest(args: argparse.Namespace) -> None:
    writer = _get_writer()
    existing = writer.find_source_by_url(args.url)
    if existing is not None:
        print(f"Already ingested: {existing['id']} ({args.url})")
        return
    summary = Path(args.summary_file).read_text(encoding="utf-8")
    ingested = args.ingested or date.today().isoformat()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    concept_titles = [c.strip() for c in args.concepts.split(",") if c.strip()] if args.concepts else []

    concept_ids: list = []
    # concept resolution added in Task C4

    created = writer.create_source(
        url=args.url,
        title=args.title,
        type_=args.type_,
        summary=summary,
        tags=tags,
        concept_ids=concept_ids or None,
        ingested=ingested,
    )
    print(f"Created: {created['id']} ({args.url})")


if __name__ == "__main__":
    main()
