# Wiki Semantic Lint Design

**Goal:** Claude Code can semantically analyze the wiki's concepts — suggesting missing sub-concepts and flagging cross-source contradictions — by running a single command that dumps all needed data, then fanning out per-concept subagents.

**Not in scope:**
- LLM API calls inside any Python script (all LLM work stays in the Claude Code session)
- Analysis of sources not linked to any concept
- Writing findings back to Notion
- Structural lint (broken wikilinks, orphans, duplicates — already in `lint.py lint`)

## Problem

The current `lint.py lint` does structural validation only: broken wikilinks, orphan
concepts, duplicate titles. It cannot tell you that four agent-harness sources are
implicitly about a "thin-harness vs. fat-harness" sub-concept not yet in the wiki, nor
that two sources linked to "Self-Improving AI Systems" make contradictory claims about
feedback loop frequency. Those gaps are only detectable by reading the content — which
the structural linter never does.

When the user asks "can you semantically lint the wiki?", Claude has no playbook: there
is no data-export command and no instruction for how to orchestrate the analysis.

## Solution (from the user's perspective)

The user says "semantically lint the wiki" in a Claude Code session. Claude reads
`wiki/CLAUDE.md`, runs `python3 wiki/scripts/lint.py dump`, receives all concepts with
their linked source bodies as JSON, fans out one subagent per concept in parallel, and
prints aggregated findings grouped by concept title directly to the session.

## Design

**Two deliverables:**

1. `NotionWikiWriter.get_source(id) -> dict` — new public method on the existing writer
   class. Fetches a source page's title (from `GET /pages/{id}` properties) and body
   text (from `GET /blocks/{id}/children` paragraph blocks). Returns
   `{"title": str, "body": str}`. Raises `RuntimeError` on any non-200 Notion response.

2. `lint.py dump` subcommand — fetches all concepts via `list_all_concepts()`, collects
   all unique source IDs across all concepts, fetches each source once via `get_source`,
   then prints a JSON array to stdout. Each element:

   ```json
   {
     "title": "Agent Harnesses",
     "body": "...",
     "all_concept_titles": ["Agent Harnesses", "Skill Design", ...],
     "sources": [
       {"title": "Thin Harness, Fat Skills", "body": "..."}
     ]
   }
   ```

   `all_concept_titles` is the same list in every element so subagents know what
   concepts already exist. Concepts with zero linked sources emit `"sources": []`.

3. `wiki/CLAUDE.md § Semantic Analysis` — instructions (not code) telling Claude to run
   `dump`, parse the JSON, fan out one subagent per concept asking two questions (sub-
   concept suggestion + contradiction detection), then aggregate findings to stdout.

**Key decisions:**

- No `anthropic` SDK in wiki scripts. The LLM work is Claude Code's job; Python is
  purely a data fetcher.
- `get_source` makes two Notion API calls per unique source ID (one for title, one for
  body). With ~150 sources, deduplication keeps this to at most 300 calls — acceptable
  for a one-off laptop operation.
- `dump` fails loudly if any source fetch fails. A partial dump is worse than no dump
  because subagents would silently receive incomplete data.
- `all_concept_titles` is embedded per concept element (redundant but correct): each
  subagent is self-contained and must not need to see other elements to do its job.

## Modules

**`NotionWikiWriter.get_source(id: str) -> dict`**
- Interface: one string in, one `{"title": str, "body": str}` dict out
- Hides: two Notion API calls (pages endpoint + blocks endpoint), rich_text concatenation, 429 retry logic (inherited from `_request`)
- Tested through: `NotionWikiWriter.get_source(id)` public method, mocking at `_OPENER`

**`lint.py dump` (cmd_dump function)**
- Interface: CLI subcommand, no arguments, prints JSON to stdout
- Hides: concept fetch, source deduplication, per-source fetches, JSON assembly
- Tested through: `lint.main(["dump"])` with mocked `NotionWikiWriter`

## File Changes

| File | Change | Type |
|------|--------|------|
| `wiki/scripts/notion_client.py` | Add `get_source(id: str) -> dict` public method | Modify |
| `wiki/scripts/lint.py` | Add `dump` subcommand + `cmd_dump` function | Modify |
| `wiki/tests/test_notion_client.py` | Add `TestGetSource` test class | Modify |
| `wiki/tests/test_lint.py` | Add `TestDumpCommand` test class | Modify |
| `wiki/CLAUDE.md` | Add `## Semantic Analysis` section | Modify |

## Open Questions

- Q: Should `dump` also include the concept body (so subagents can see the concept's
  existing synthesis alongside its sources)?
  Default: Yes — included as `"body"` field; subagents need it to judge whether a
  sub-concept is truly missing vs. already implicitly covered.
