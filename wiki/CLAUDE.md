# Mahler Wiki

A Karpathy-style LLM Wiki backed by Notion. The runtime source of truth is three Notion databases; this directory holds the local write-side tooling and the raw sources that never leave the laptop.

## What's where

```
mahler/wiki/
├── raw/              (gitignored) — original markdown sources
├── assets/           (gitignored) — images, PDFs
├── outputs/          (gitignored) — summary and concept body drafts
├── schema.md         — Notion DB schema (source of truth)
├── .env.example      — credential template
├── .env              (gitignored) — real credentials (NOTION_WIKI_WRITE_TOKEN, DB IDs)
├── scripts/
│   ├── notion_client.py   — write-side Notion client (NotionWikiWriter)
│   ├── ingest.py          — ingest + create-concept CLI
│   └── lint.py            — lint CLI
├── tests/
└── CLAUDE.md         (this file)
```

## Environment

`wiki/.env` is loaded automatically by `ingest.py` at startup — never pass tokens as arguments or set them in the shell. The file contains:

```
NOTION_WIKI_WRITE_TOKEN=secret_...
NOTION_WIKI_SOURCES_DB_ID=...
NOTION_WIKI_CONCEPTS_DB_ID=...
NOTION_WIKI_LOG_DB_ID=...
```

All commands below must be run from the repo root (`mahler/`), not from inside `wiki/`.

## Ingesting a new source

When the user adds a file to `wiki/raw/`, or asks to ingest a URL:

1. If the raw file doesn't exist yet, save the article content to `wiki/raw/<slug>.md`
2. Read the raw file
3. Extract the canonical URL from the file content. For arxiv papers look for the arxiv.org URL. For blog posts look near the top. The URL is the idempotency key — it must be real and unique.
4. Determine the type: `paper` (academic/arxiv), `article` (blog post), `post` (short post), `video`, or `other`
5. Write a ~400-word summary to `wiki/outputs/<slug>-summary.md` as **plain prose paragraphs only** — no headings, no bullet lists, no markdown formatting. One blank line between paragraphs.
6. Decide which existing concepts to link (see concept list below). Do NOT create new concepts during ingest — the CLI will fail loudly if a concept title doesn't exist.
7. Run:

```bash
python3 wiki/scripts/ingest.py ingest \
  --url "https://..." \
  --title "Title of the source" \
  --type article \
  --summary-file wiki/outputs/<slug>-summary.md \
  --tags "tag1,tag2,tag3" \
  --concepts "Agent Harnesses,Skill Design"
```

The CLI prints `Created: <id>` on success or `Already ingested: <id>` if the URL is already in Notion. Both are correct outcomes. Any other output is an error.

Tags must be lowercase, kebab-case, no commas within a single tag value.

`--concepts` is optional. Omit it if none of the existing concepts apply.

## Creating a new concept

Concepts are Karpathy-style wiki pages — the user's own synthesized understanding of a topic across multiple sources. Create one when a recurring theme across sources doesn't have a home yet.

1. Write a ~300-400 word concept body to `wiki/outputs/concept-<slug>.md` as **plain prose paragraphs only** (same format rules as summaries). Use `[[ConceptTitle]]` wikilink syntax to cross-reference other concepts.
2. Run:

```bash
python3 wiki/scripts/ingest.py create-concept \
  --title "Concept Title" \
  --body-file wiki/outputs/concept-<slug>.md \
  --tags "tag1,tag2"
```

The CLI prints `Created concept: <id>` or `Already exists: <id>`.

## Existing concepts

These are the 23 concepts currently in the wiki. Use exact titles (case-insensitive) in `--concepts` flags:

**Music / piano AI**
- `Music Representation Learning`
- `Music AI Systems`
- `Score Following and Music Education`

**ML / AI core**
- `Transformer Architecture`
- `LoRA and Parameter-Efficient Fine-Tuning`
- `Generative Models`
- `Reinforcement Learning for LLMs`
- `Multi-Agent Memory Systems`
- `Speculative Decoding`

**Agent engineering**
- `Agent Harnesses`
- `Natural Language Harnesses`
- `Skill Design`
- `Context Graphs`
- `Self-Improving AI Systems`
- `Evals and Benchmarking`
- `Asynchronous Agent Systems`

**Infrastructure**
- `Prompt Caching`
- `Cloudflare Workers and Durable Objects`

**Product / growth / leadership**
- `AI-Native Product Engineering`
- `Growth Marketing`
- `Startup Fundraising and VC`
- `Technical Leadership`
- `Founder Mindset`

## Lint

```bash
python3 wiki/scripts/lint.py lint
```

Reports broken wikilinks, orphan concepts, sourceless concepts, and duplicate titles. Lint does not modify Notion — fixes are manual.

## Testing

```bash
cd wiki && python3 -m unittest discover tests
```

Tests mock only the HTTP transport at the `_OPENER` boundary. No tests hit the real Notion API.

## Relationship to the Hermes read skill

The Hermes read-only counterpart lives at `assistant/config/skills/notion-wiki/`. It has its own `notion_client.py` (read-only), its own tests, and uses `NOTION_WIKI_READ_TOKEN` (a separate read-only integration). The two halves deliberately do not share code.

## Semantic Analysis

When asked to "semantically lint", "analyze", or "find concept gaps" in the wiki:

1. Run from the repo root:

   ```bash
   python3 wiki/scripts/lint.py dump
   ```

   This prints a JSON array of all concepts with their linked source bodies. It may
   take a minute — it fetches each linked source from Notion.

2. Parse the JSON. Each element has:
   - `title`: concept name
   - `body`: the concept's current synthesized text
   - `sources`: array of `{title, body}` for every source linked to this concept
   - `all_concept_titles`: list of every concept title currently in the wiki

3. Fan out one subagent per concept in parallel. For each concept, pass its element
   from the JSON and ask the subagent two questions:

   a) Given these sources and the existing concept list, is there a sub-concept hiding
      here that warrants its own wiki page? Suggest a title and a one-sentence rationale.
      If no clear sub-concept exists, say "No sub-concept found."

   b) Do any of these sources make contradictory claims? Name the two sources and
      describe the specific tension in one sentence. If no contradiction exists, say
      "No contradictions."

   Concepts with empty `sources` arrays should return "No sources to analyze."

4. Aggregate all subagent findings and print grouped by concept title to session stdout.
   Do not write findings to Notion — the user acts on them manually.
