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

    p_fix = sub.add_parser("fix-url")
    p_fix.add_argument("--page-id", dest="page_id", required=True)
    p_fix.add_argument("--url", required=True)

    p_fixbody = sub.add_parser("fix-body")
    p_fixbody.add_argument("--page-id", dest="page_id", required=True)
    p_fixbody.add_argument("--body-file", dest="body_file", required=True)

    p_link = sub.add_parser("link-concepts")
    p_link.add_argument("--page-id", dest="page_id", required=True)
    p_link.add_argument("--concepts", required=True, help="Comma-separated exact concept titles")

    p_concept = sub.add_parser("create-concept")
    p_concept.add_argument("--title", required=True)
    p_concept.add_argument("--body-file", dest="body_file", required=True)
    p_concept.add_argument("--tags", default="")

    args = parser.parse_args(argv)
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "fix-url":
        cmd_fix_url(args)
    elif args.command == "fix-body":
        cmd_fix_body(args)
    elif args.command == "link-concepts":
        cmd_link_concepts(args)
    elif args.command == "create-concept":
        cmd_create_concept(args)


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
    for title in concept_titles:
        page = writer.find_concept_by_title(title)
        if page is None:
            raise RuntimeError(
                f"Concept not found: {title!r}. Create it in Notion first, then re-run ingest."
            )
        concept_ids.append(page["id"])

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
    writer.append_log(
        kind="INGEST",
        detail=f"ingested {args.url} as {created['id']}",
        when=ingested,
    )


def cmd_fix_url(args: argparse.Namespace) -> None:
    writer = _get_writer()
    writer.update_source_url(args.page_id, args.url)
    print(f"Updated: {args.page_id} -> {args.url}")


def cmd_fix_body(args: argparse.Namespace) -> None:
    writer = _get_writer()
    body = Path(args.body_file).read_text(encoding="utf-8")
    writer.update_source_body(args.page_id, body)
    print(f"Body updated: {args.page_id}")


def cmd_link_concepts(args: argparse.Namespace) -> None:
    writer = _get_writer()
    titles = [t.strip() for t in args.concepts.split(",") if t.strip()]
    concept_ids = []
    for title in titles:
        page = writer.find_concept_by_title(title)
        if page is None:
            raise RuntimeError(f"Concept not found: {title!r}")
        concept_ids.append(page["id"])
    writer.update_source_concepts(args.page_id, concept_ids)
    print(f"Linked {args.page_id} -> {titles}")


def cmd_create_concept(args: argparse.Namespace) -> None:
    writer = _get_writer()
    existing = writer.find_concept_by_exact_title(args.title)
    if existing is not None:
        print(f"Already exists: {existing['id']} ({args.title!r})")
        return
    body = Path(args.body_file).read_text(encoding="utf-8")
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    created = writer.create_concept(title=args.title, body=body, tags=tags)
    print(f"Created concept: {created['id']} ({args.title!r})")
    writer.append_log(
        kind="INGEST",
        detail=f"created concept {args.title!r} as {created['id']}",
        when=date.today().isoformat(),
    )


if __name__ == "__main__":
    main()
