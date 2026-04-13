# notion-wiki Design

**Goal:** Give Hermes (on Fly.io) and local Claude Code sessions a shared, persistent, Notion-backed knowledge base — local sessions ingest sources into Notion; Hermes reads them remotely to answer questions grounded in the user's own reading and notes.

**Not in scope:**
- Mobile or cloud-side ingestion (ingest is laptop-only, by deliberate design)
- Auto-journaling of coding activity, meeting notes, or project progress (separate future feature)
- Syncing Notion pages back into local markdown (Notion is the runtime truth)
- Vector search, embeddings, BM25 index, or any retrieval infra beyond Notion's native search (wiki is ~62 pages growing to ~1000; read-the-index is sufficient)
- Rewriting or re-running LLM pipelines on already-ingested sources (idempotency is "skip if URL exists", not "re-summarize")
- Automatic concept creation from source ingestion (concepts are curated, not auto-extracted)
- A new Hermes skill that *writes* to Notion wiki (read-only from Fly.io — intentional blast-radius boundary)
- Bootstrapping a UI in Notion beyond the three databases and their default views

## Problem

The user already maintains a local wiki at `~/Documents/wiki/` following the Karpathy LLM Wiki pattern: 62 raw sources (papers, articles, blog posts) and 88 LLM-generated concept/summary pages, cross-linked with Obsidian-style `[[wikilinks]]`. Two things are broken:

1. **Hermes on Fly.io cannot read any of it.** The wiki lives on the laptop filesystem; the Hermes container has no access. Phase 7 of the Mahler roadmap ("Wiki Bridge") was going to KV-sync markdown files, but that's a one-way dump with no query surface and no UI.
2. **There is no disciplined ingest pipeline.** The existing 88 generated pages were produced ad-hoc across many Claude sessions with inconsistent schema, inconsistent wikilink targets, and no lint. Adding a new source today means a fresh LLM conversation deciding what to do from scratch.

The second problem compounds the first: even if Hermes could read the wiki, the data shape is inconsistent enough that a remote reader would struggle to get reliable answers.

## Solution (from the user's perspective)

**On the laptop, in a Claude Code session:**

> "Read this paper and add it to the wiki: https://arxiv.org/abs/2404.xxxxx"

Claude invokes the `notion-wiki` ingest skill, which downloads the source, produces a structured summary, creates a Source page in Notion with properties (URL, title, type, tags, date added) and a Concept page (or updates an existing one), sets the `Source → Concept` relation, appends an entry to the Log database, and reports back with the Notion page URLs. The raw file stays on the laptop in `mahler/wiki/raw/` (gitignored). The user sees the new pages in their normal Notion sidebar, in the normal Notion UI, with working filters and views — for free.

**On Fly.io, in Discord:**

> "@Mahler what do I know about speculative decoding?"

Hermes invokes the `notion-wiki` read skill: it queries the Concepts database for pages matching "speculative decoding", fetches the top hit's page content, follows the `Concept → Sources` relation to pull the underlying source summaries, and composes an answer that cites both the concept page and its sources by Notion URL. If nothing matches, Hermes says so plainly rather than hallucinating — the search failed, the user just hasn't ingested anything on that topic yet.

**Weekly, in a local Claude session:**

> "/lint the wiki"

Claude runs the lint skill, which walks all Concept pages, collects every `[[wikilink]]` reference, and reports broken links, orphans, and concepts with no sources. The user fixes them by hand in Notion or by re-ingesting.

## Design

### Architecture

Two deployment targets, one Notion workspace, two separate integrations (for blast-radius isolation):

```
Laptop (local Claude Code)                 Fly.io (Hermes)
  └─ mahler/wiki/                            └─ ~/.hermes/skills/notion-wiki/
     ├─ raw/          (gitignored)              ├─ SKILL.md
     ├─ assets/       (gitignored)              └─ scripts/
     ├─ outputs/      (gitignored)                 ├─ notion_client.py  (read-only)
     ├─ schema.md                                  └─ wiki.py           (index/read/search CLI)
     └─ scripts/
        ├─ notion_client.py  (read-write)
        ├─ ingest.py         (ingest CLI)
        └─ lint.py           (lint CLI)
         │                                          │
         │    NOTION_WIKI_WRITE_TOKEN (rw)          │    NOTION_WIKI_READ_TOKEN (ro)
         ▼                                          ▼
              Notion workspace
              ├─ Wiki: Sources   (DB — raw source metadata + summary)
              ├─ Wiki: Concepts  (DB — curated concept pages, Karpathy-style)
              └─ Wiki: Log       (DB — ingest/lint activity log)
              with relations: Source.concepts <-> Concept.sources
```

**Key architectural decisions (locked during /brainstorm):**

1. **Notion over Obsidian.** Notion is the user's daily driver, has a stable HTTP API reachable from Fly.io, and its DB views give us filtering and UI for free. Obsidian would require a separate filesystem sync layer.
2. **Local ingest, remote read.** Ingestion is laptop-only by deliberate choice — it involves web fetches, LLM summarization, and human-in-the-loop review, none of which benefit from running on Fly. This keeps Hermes's container boring and read-only, with a smaller attack surface.
3. **Raw sources never leave the laptop.** `mahler/wiki/raw/` is gitignored. Only the summary (short, curated) goes into Notion. The full article stays local for re-reading.
4. **Three databases with relations, not one unified DB.** Sources, Concepts, Log. Two-way relation between Sources and Concepts. This maps 1:1 onto the existing Karpathy schema and lets us get Notion's native bidirectional relation UI, DB filters, and per-DB views for free. One unified DB would collapse the type distinction and force schema contortions.
5. **Read-the-index retrieval, not vector search.** At ~62 pages today scaling to ~1000, Karpathy's three-primitive pattern (list index → read page → search by keyword) is sufficient and avoids a whole subsystem (embeddings store, reindex jobs, drift). Reassess at ~1000 pages.
6. **Two separate Notion integrations.** `NOTION_WIKI_WRITE_TOKEN` lives on the laptop and has read-write access to all three DBs; `NOTION_WIKI_READ_TOKEN` lives on Fly.io and has read-only access. A compromised Hermes cannot corrupt the wiki.
7. **Two separate `notion_client.py` copies, not one shared module.** The write-side client needs retries, rate-limit handling, and idempotency checks; the read-side client should fail fast and stay minimal. Sharing would force both halves to evolve together and couple two separate trust zones.
8. **Clean break on seeding.** The 88 existing LLM-generated pages are inconsistent and will be deleted (they're recoverable from git if needed — they're in `~/Documents/wiki/wiki/`). The 62 raw sources will be re-processed through the new ingester in a one-time bootstrap run (~20 min of LLM time, runs locally).

### Data flow: ingest

```
user: "ingest https://arxiv.org/abs/xxxx"
  │
  ▼
Claude Code (local) invokes skill → ingest.py ingest --url ...
  │
  ▼
ingest.py:
  1. Check if URL already in Sources DB → if yes, print "already ingested" + URL, exit
  2. Download source (arxiv API / raw HTML) → save to mahler/wiki/raw/<slug>.md
  3. Parse source metadata (title, authors, type)
  4. Ask Claude (via the active session, not a subprocess) to produce:
     - Source summary (~400 words, Markdown)
     - List of concepts the source should relate to (existing or new)
  5. For each concept title the user passed:
     - Look up the concept by title in the Concepts DB
     - If it does not exist → FAIL LOUDLY: "Concept 'X' does not exist. Create it first,
       then re-run ingest." (No auto-stubbing — concepts are curated.)
     - If it exists → collect its page id
  6. Create Source page in Sources DB with properties (including the Concepts relation = collected ids) + summary as page body. Notion's bidirectional relation auto-populates `Concept.sources` on the other side.
  7. Append Log entry: INGEST, source_url, page_id, timestamp
  8. Print Source page URL + updated Concept page URLs
```

Note: step 4 is Claude-the-LLM doing the summarization *in the active Claude Code session*, not a subprocess calling an LLM API. `ingest.py` is a shallow argument-marshaling CLI that the skill's SKILL.md directs Claude to invoke after Claude has done the reading and summarization itself. This matches the notion-tasks skill pattern where the CLI is the tool and Claude is the brain.

### Data flow: read (Hermes side)

```
Hermes: "what do I know about speculative decoding?"
  │
  ▼
Hermes invokes wiki.py search --query "speculative decoding" --limit 5
  │
  ▼
wiki.py:
  1. Call Notion search API scoped to the Wiki databases
  2. Return ranked list of {page_id, title, type, snippet}
  │
  ▼
Hermes picks top 1-2 hits, invokes wiki.py read --id PAGE_ID
  │
  ▼
wiki.py:
  1. Fetch page properties + block children from Notion
  2. Flatten blocks to Markdown
  3. For Concept pages, also follow Concept.sources relation and list linked Source titles + URLs
  4. Print Markdown to stdout
  │
  ▼
Hermes composes Discord reply grounded in the fetched content, cites Notion URLs
```

### Data flow: lint

```
user: "lint the wiki"
  │
  ▼
lint.py walks every Concept page's body → extracts all [[wikilinks]]
  │
  ▼
Cross-references against Concepts DB page-title index → flags:
  - Broken wikilinks (target does not exist)
  - Orphans (concept with no incoming links from any source or other concept)
  - Sourceless concepts (concept with empty Concept.sources relation)
  - Duplicate titles (two concepts with the same name — case-insensitive)
  │
  ▼
Appends LINT entry to Log DB with counts, prints report to stdout
```

### Module boundaries

Four modules ship, plus one reference doc (`schema.md`). The deepest module is the LLM itself — the user in a Claude session. Code is intentionally shallow primitives around a medium-depth Notion client. See Modules section below for the depth analysis.

### Error handling

- **Notion API error (non-2xx):** `notion_client.py` raises `RuntimeError` with status code and body. Caller surfaces verbatim. No retries on the read-side client (fail fast, Hermes re-asks). Write-side client retries only on 429 (rate limit) with exponential backoff up to 3 attempts.
- **Missing env var:** CLI scripts raise `RuntimeError` on startup, exit non-zero. Matches notion-tasks pattern.
- **Concept relation target missing:** `ingest.py` raises `RuntimeError` naming the missing concept and tells the user to create it first. No auto-stub.
- **URL already ingested:** `ingest.py` prints the existing Source page URL and exits 0 (not an error, just idempotent).
- **Broken wikilink at read time:** read-side renders the raw `[[text]]` unchanged. Hermes sees it and can mention it in the reply.
- **Lint violations:** always non-fatal. Lint exits 0 and prints a report; fixing is manual.

### Testing strategy

All tests verify behavior through public CLI entrypoints, mocking only the HTTP transport at the `_OPENER` boundary (same pattern as notion-tasks `test_notion_client.py`). No mocking of internal collaborators. No testing of private methods. Each test exercises one end-to-end slice through the CLI.

Tests cover, per vertical slice:
- Write-side: create source page, skip-if-duplicate URL, create with concept relation, fail-on-missing-concept, retry on 429
- Read-side: list index, read page by id, search by query, follow relations on concept read, fail-fast on API error
- Lint: detect broken wikilinks, detect orphans, detect sourceless concepts, detect duplicate titles
- CLI plumbing: env-var bridging from `~/.hermes/.env`, argparse dispatch, exit codes

Bootstrapping (one-time re-ingest of 62 sources) is **not** tested — it's a manual run against the real Notion workspace, verified by the user eyeballing the populated DBs.

## Modules

### `mahler/wiki/scripts/notion_client.py` (write-side) — DEEP

- **Interface:** `NotionWikiWriter(token, sources_db_id, concepts_db_id, log_db_id)` with methods `create_source(url, title, summary, tags, concept_ids) -> dict`, `find_source_by_url(url) -> dict | None`, `find_concept_by_title(title) -> dict | None`, `append_log(kind, detail) -> dict`.
- **Hides:** Notion property JSON shapes, pagination for lookup, 429 retry with backoff, SSRF-safe HTTPS opener, idempotency checks. The `Source → Concepts` relation is set once at create time; Notion's bidirectional relation auto-populates `Concept.sources` on the other side, so no read-modify-write cycle is needed.
- **Tested through:** direct unit tests at the class level using the `_OPENER` mock boundary.

### `mahler/wiki/scripts/ingest.py` — SHALLOW (justified)

- **Interface:** `python3 ingest.py ingest --url URL --title TITLE --summary-file PATH [--tags T1,T2] [--concepts C1,C2]`
- **Hides:** argparse setup, env loading, Notion client instantiation, dispatch to write-side methods. That's it.
- **Depth verdict:** SHALLOW — but justified. The skill pattern (matching notion-tasks) intentionally puts the intelligence in SKILL.md and in Claude's reasoning during the session. The CLI is a tool for Claude to invoke, not an app. Any "deepening" would duplicate what Claude is already doing (choosing concepts, writing summaries) into code, which would be worse.
- **Tested through:** CLI-level tests that mock `NotionWikiWriter` and assert on argparse/dispatch behavior.

### `assistant/config/skills/notion-wiki/scripts/notion_client.py` (read-side) — DEEP

- **Interface:** `NotionWikiReader(token, sources_db_id, concepts_db_id)` with methods `search(query, limit) -> list[dict]`, `read_page(page_id) -> dict` (returns `{title, type, body_markdown, relations}`), `list_index(db, limit, cursor) -> dict`.
- **Hides:** Notion search API quirks, block-tree to Markdown flattening, relation resolution, pagination, SSRF-safe HTTPS opener.
- **Depth verdict:** DEEP — the block-to-Markdown flattening alone is substantial, and callers never have to think about Notion's block model.
- **Tested through:** unit tests at the class level using `_OPENER` mock.

### `assistant/config/skills/notion-wiki/scripts/wiki.py` — SHALLOW (justified)

- **Interface:** `python3 wiki.py {index|read|search} [flags]`
- **Hides:** argparse, env loading from `~/.hermes/.env`, NotionWikiReader instantiation, dispatch.
- **Depth verdict:** SHALLOW — justified identically to `ingest.py`. The skill is the user-facing surface; the script is a thin tool.
- **Tested through:** CLI-level tests with mocked `NotionWikiReader`.

### `mahler/wiki/scripts/lint.py` — medium-depth

- **Interface:** `python3 lint.py lint [--fix-suggestions]`
- **Hides:** iteration over all Concept pages, `[[wikilink]]` regex extraction, cross-reference against title index, orphan detection traversal, sourceless-concept detection.
- **Depth verdict:** medium. Not deep (no abstraction hides from callers — there's one caller, the user via CLI), not shallow (the lint rules themselves are real logic, not plumbing). One-caller utilities don't need to be deep — they need to be correct.
- **Tested through:** CLI-level tests using a mocked `NotionWikiWriter` (lint reads Concepts through the same client) with staged fixture data.

### `mahler/wiki/schema.md` — reference doc

- **Interface:** read-only documentation.
- **Contents:** exact DB property names and types for all three databases, the tag grammar (no commas allowed), the concept-title uniqueness rule, the relation shape, and a worked example.
- **Tested through:** not tested — it's a spec, its correctness is that the code matches it.

## File Changes

| File | Change | Type |
|------|--------|------|
| `~/Documents/wiki/` → `mahler/wiki/` | Manual move (prerequisite, not a task) | Modify |
| `mahler/wiki/wiki/` (88 LLM-generated pages) | Delete (clean-break decision) | Delete |
| `mahler/.gitignore` | Add `wiki/raw/`, `wiki/assets/`, `wiki/outputs/`, `wiki/.env` | Modify |
| `mahler/wiki/schema.md` | Database schema spec | New |
| `mahler/wiki/.env.example` | Template with `NOTION_WIKI_WRITE_TOKEN`, DB IDs | New |
| `mahler/wiki/scripts/notion_client.py` | Write-side Notion client | New |
| `mahler/wiki/scripts/ingest.py` | Ingest CLI | New |
| `mahler/wiki/scripts/lint.py` | Lint CLI | New |
| `mahler/wiki/tests/test_notion_client.py` | Write-client unit tests | New |
| `mahler/wiki/tests/test_ingest.py` | Ingest CLI tests | New |
| `mahler/wiki/tests/test_lint.py` | Lint CLI tests | New |
| `mahler/wiki/CLAUDE.md` | Rewrite (was the old Obsidian-flavored schema) — now documents the new ingest + lint flow | Modify |
| `assistant/config/skills/notion-wiki/SKILL.md` | Read skill for Hermes | New |
| `assistant/config/skills/notion-wiki/scripts/notion_client.py` | Read-side Notion client | New |
| `assistant/config/skills/notion-wiki/scripts/wiki.py` | Read CLI (index/read/search) | New |
| `assistant/config/skills/notion-wiki/tests/test_notion_client.py` | Read-client unit tests | New |
| `assistant/config/skills/notion-wiki/tests/test_wiki.py` | Read CLI tests | New |
| `assistant/Dockerfile` | Add COPY line for `notion-wiki` skill | Modify |
| `assistant/entrypoint.sh` | Add `NOTION_WIKI_READ_TOKEN`, `NOTION_WIKI_SOURCES_DB_ID`, `NOTION_WIKI_CONCEPTS_DB_ID` to the env-bridge block | Modify |
| `assistant/.env.example` | Add the three Hermes-side wiki env vars | Modify |
| `assistant/CLAUDE.md` | Update Phase 7 ("Wiki Bridge") — mark as replaced by `notion-wiki` skill | Modify |

## Open Questions

- **Q: What are the exact Notion property types for tags?** Default: `multi_select` — Notion's built-in and gives per-value coloring for free. The tag grammar (no commas) is enforced at write time in `notion_client.py`.
- **Q: Should `lint.py` be able to auto-fix simple broken wikilinks (e.g., case mismatch)?** Default: no — lint reports only, fixing is manual. A `--fix-suggestions` flag emits suggested edits but does not apply them.
- **Q: Where does `NOTION_WIKI_WRITE_TOKEN` live on the laptop?** Default: `mahler/wiki/.env` (gitignored). Loaded by `ingest.py` and `lint.py` on startup with the same pattern `tasks.py` uses for `~/.hermes/.env`.
- **Q: What happens if the user ingests a source whose auto-detected type is unknown (not paper/article/post)?** Default: default to `article` and proceed. The user can edit the Type property in Notion after the fact.
- **Q: Should the Log DB prune old entries?** Default: no — it's small, append-only, and valuable as a history. Revisit if it crosses 10k rows.
- **Q: How are the three Notion databases initially created?** Default: manual one-time setup by the user in Notion's UI, following `schema.md` exactly. The DB IDs are then copied into `.env`. Scripted creation is out of scope.
