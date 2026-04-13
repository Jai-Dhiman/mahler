# Mahler Wiki

A Karpathy-style LLM Wiki backed by Notion. The runtime source of truth is three Notion databases; this directory holds the local write-side tooling and the raw sources that never leave the laptop.

## What's where

```
mahler/wiki/
├── raw/              (gitignored) — original markdown sources
├── assets/           (gitignored) — images, PDFs
├── outputs/          (gitignored) — working drafts
├── schema.md         — Notion DB schema (source of truth)
├── .env.example      — credential template
├── .env              (gitignored) — real credentials
├── scripts/
│   ├── notion_client.py   — write-side Notion client
│   ├── ingest.py          — ingest CLI
│   └── lint.py            — lint CLI
├── tests/
│   ├── test_notion_client.py
│   ├── test_ingest.py
│   └── test_lint.py
└── CLAUDE.md         (this file)
```

## Operations

### Ingest a source

In a local Claude Code session, ask Claude to ingest a URL. Claude will:
1. Download the source (via its own tools — `curl`, `WebFetch`, etc.)
2. Save the raw markdown to `wiki/raw/<slug>.md`
3. Write a short summary (~400 words) as markdown to `wiki/outputs/<slug>-summary.md`
4. Decide which existing concept pages the source should link to (do NOT create new concepts during ingest)
5. Run `python3 wiki/scripts/ingest.py ingest --url URL --title "..." --type paper --summary-file wiki/outputs/<slug>-summary.md [--tags llm,inference] [--concepts "Speculative Decoding,LLM Efficiency"]`
6. The CLI creates the Source page in Notion and links the concepts. Notion auto-populates the reverse relation on each Concept page.
7. The CLI appends a row to the Log DB.

If a concept the user mentions doesn't exist yet, the ingest CLI fails loudly. Create the concept in Notion first, then re-run ingest.

### Create a concept

Concepts are curated by hand. In Notion, create a new row in the Concepts DB with a unique title (case-insensitive). Write the concept body using Markdown with optional `[[wikilinks]]` to other concept titles.

### Lint

```bash
python3 wiki/scripts/lint.py lint
```

Reports broken wikilinks, orphan concepts, sourceless concepts, and duplicate titles. Lint does not modify anything in Notion — fixes are manual.

## Environment

Copy `wiki/.env.example` to `wiki/.env` and fill in the token and DB IDs. The write-side CLI loads `.env` on startup via a tiny stdlib parser (no `python-dotenv` dependency).

## Testing

```bash
cd wiki && python3 -m unittest discover tests
```

Tests mock only the HTTP transport at the `_OPENER` boundary. No tests hit the real Notion API.

## Relationship to the Hermes read skill

The Hermes read-only counterpart to this tooling lives at `assistant/config/skills/notion-wiki/`. That skill has its own `notion_client.py` (read-only), its own tests, and its own credentials (`NOTION_WIKI_READ_TOKEN` and the same DB IDs). The two halves deliberately do not share code — the write-side evolves with retries and idempotency; the read-side stays minimal and fails fast.
