"""
Notion wiki lint CLI.

Usage:
    python3 lint.py lint
    python3 lint.py dump
"""

import argparse
import json
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

    incoming = {c["title"].strip().lower(): set() for c in concepts}
    for concept in concepts:
        for match in _WIKILINK_PATTERN.finditer(concept["body_markdown"]):
            target = match.group(1).strip().lower()
            if target in incoming:
                incoming[target].add(concept["id"])

    orphans = 0
    for concept in concepts:
        key = concept["title"].strip().lower()
        has_incoming = len(incoming.get(key, set())) > 0
        has_sources = len(concept["source_ids"]) > 0
        if not has_incoming and not has_sources:
            print(f"Orphan concept: {concept['title']!r} (no incoming links and no sources)")
            orphans += 1

    sourceless = 0
    for concept in concepts:
        key = concept["title"].strip().lower()
        has_incoming = len(incoming.get(key, set())) > 0
        has_sources = len(concept["source_ids"]) > 0
        if has_incoming and not has_sources:
            print(f"Sourceless concept: {concept['title']!r} (referenced but has no sources)")
            sourceless += 1

    by_norm: dict = {}
    for concept in concepts:
        key = concept["title"].strip().lower()
        by_norm.setdefault(key, []).append(concept)
    duplicates = 0
    for key, group in by_norm.items():
        if len(group) > 1:
            ids = ", ".join(c["id"] for c in group)
            print(f"Duplicate title: {group[0]['title']!r} across pages {ids}")
            duplicates += 1

    print(f"Summary: {broken} broken wikilinks, {orphans} orphans, {sourceless} sourceless, {duplicates} duplicate titles")
    writer.append_log(
        kind="LINT",
        detail=f"{broken} broken wikilinks, {orphans} orphans, {sourceless} sourceless, {duplicates} duplicate titles",
        when=date.today().isoformat(),
    )


def cmd_dump(args: argparse.Namespace) -> None:
    writer = _get_writer()
    concepts = writer.list_all_concepts()
    all_concept_titles = [c["title"] for c in concepts]

    unique_source_ids = list(dict.fromkeys(
        sid for c in concepts for sid in c["source_ids"]
    ))
    sources_by_id = {sid: writer.get_source(sid) for sid in unique_source_ids}

    output = []
    for concept in concepts:
        output.append({
            "title": concept["title"],
            "body": concept["body_markdown"],
            "all_concept_titles": all_concept_titles,
            "sources": [sources_by_id[sid] for sid in concept["source_ids"]],
        })

    print(json.dumps(output, ensure_ascii=False, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Mahler notion-wiki linter")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lint")
    sub.add_parser("dump")
    args = parser.parse_args(argv)
    if args.command == "lint":
        cmd_lint(args)
    elif args.command == "dump":
        cmd_dump(args)


if __name__ == "__main__":
    main()
