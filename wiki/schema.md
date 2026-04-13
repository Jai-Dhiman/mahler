# Notion Wiki Schema

The notion-wiki skill relies on three Notion databases in the user's workspace. This file is the source of truth for their shape. Both the local write-side client (`mahler/wiki/scripts/notion_client.py`) and the Hermes read-side client (`assistant/config/skills/notion-wiki/scripts/notion_client.py`) assume these exact property names and types.

If you change a property name here, you must change it in both clients and update both test suites.

## Database 1: Wiki Sources

Stores one page per ingested raw source (paper, article, blog post). Raw content stays on the laptop; this DB only holds metadata and a summary.

| Property | Type | Required | Notes |
|---|---|---|---|
| `Title` | title | yes | Human-readable title of the source |
| `URL` | url | yes | Canonical URL; used as the idempotency key |
| `Type` | select | yes | One of: `paper`, `article`, `post`, `video`, `other` |
| `Tags` | multi_select | no | Comma-free tags; validated at write time |
| `Concepts` | relation | no | Two-way relation to the `Wiki Concepts` database |
| `Ingested` | date | yes | Date the source was ingested (YYYY-MM-DD) |

Page body: the summary, written as paragraph blocks (one block per summary paragraph). No headings, no lists, no code blocks — keep sources flat so they render consistently in the Hermes reader.

## Database 2: Wiki Concepts

Stores one page per curated concept. Concepts are created by hand by the user (or by Claude in a local session, but never automatically during ingest). A concept is the Karpathy-style "wiki page" that survives across sources and evolves as the user learns more.

| Property | Type | Required | Notes |
|---|---|---|---|
| `Title` | title | yes | Unique (case-insensitive). Duplicate titles are a lint error. |
| `Tags` | multi_select | no | Comma-free |
| `Sources` | relation | no | Two-way relation back to `Wiki Sources`. Auto-populated by Notion when a Source sets its `Concepts` relation. |

Page body: free-form Markdown. Obsidian-style `[[wikilinks]]` may reference other concept titles; the lint tool checks these.

## Database 3: Wiki Log

Append-only activity log. Ingest and lint runs each append one row.

| Property | Type | Required | Notes |
|---|---|---|---|
| `Kind` | select | yes | One of: `INGEST`, `LINT`, `ERROR` |
| `Detail` | rich_text | yes | Free-form message |
| `When` | date | yes | Date of the event (YYYY-MM-DD) |

Page body: empty.

## Tag grammar

Tags are `multi_select` values. Comma characters are forbidden in tag values because Notion splits multi-select options on commas when pasted from a string. Write-side client raises `RuntimeError` if any tag contains a comma. Recommended convention: lowercase, kebab-case, short — e.g., `llm`, `inference`, `speculative-decoding`.

## Concept title uniqueness

Concept titles are unique case-insensitively. The lint tool flags duplicates. The ingest tool's `find_concept_by_title` uses case-insensitive comparison in Python (Notion's filter is case-sensitive, so we fetch the candidate set and compare locally).

## Relation semantics

The `Source → Concepts` relation is the write side. The `Concept → Sources` relation is Notion's auto-synthesized reverse view of the same edge. When creating a Source with `concept_ids`, the write client sets `Source.Concepts` only; the reverse side appears automatically in the Concepts DB without a second API call.
