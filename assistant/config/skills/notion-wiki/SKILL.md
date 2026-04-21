---
name: notion-wiki
description: Read-only access to the user's Notion-backed knowledge wiki. Look up concepts, read source summaries, and follow cross-references. Use this whenever the user asks what they know about a topic, what they've read about it, or asks to cite sources from their own reading.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [wiki, knowledge, notion, research]
    related_skills: [notion-tasks]
---

## When to use

- The user asks what they know about a topic ("what do I know about X?")
- The user asks for a summary of something they've read or ingested
- The user asks for sources or citations from their own wiki
- You need grounded context about a domain the user has written notes on

Do NOT use this skill to *write* to the wiki. Ingestion is laptop-only; this skill is read-only. If the user asks to add something to the wiki, tell them to ingest it from a local Claude Code session on their laptop.


## Operations

### Search across both databases

```bash
python3 ~/.hermes/skills/notion-wiki/scripts/wiki.py search \
  --query "TEXT" \
  [--limit N]
```

Output format per hit:
```
[page-id] (sources|concepts) Title
```

Use `search` first when the user asks an open-ended question. Pick the top 1-3 hits and follow up with `read`.

### Read a page by ID

```bash
python3 ~/.hermes/skills/notion-wiki/scripts/wiki.py read \
  --id PAGE_ID
```

Prints the page as markdown with title, type/URL metadata, body, and a `Related:` section listing linked concepts or sources. Use this to get grounded context for your reply.

### List an index

```bash
python3 ~/.hermes/skills/notion-wiki/scripts/wiki.py index \
  --db sources|concepts \
  [--limit N]
```

Prints one line per entry in the format `[page-id] title`. Use this when the user asks what's in the wiki ("what have I read about?") without a specific topic.

## Response discipline

- Always cite the page title when you use content from the wiki. You may include the page ID in brackets.
- If a search returns no hits, say so plainly ("I don't have anything on that in your wiki"). Do NOT invent content or fall back to general knowledge and present it as if it came from the wiki.
- Wiki content is the user's own notes. Treat it as authoritative over your general knowledge when the two conflict.
- Raw source files are not in Notion — only summaries. If the user wants the full text, tell them it's on their laptop in `mahler/wiki/raw/`.

## Failure modes

- Missing env var → `RuntimeError` with the variable name. Surface it verbatim; the user likely needs to run `flyctl secrets set`.
- Notion API error → `RuntimeError` with the status code. Surface it verbatim.
- Page ID not found → `RuntimeError` 404. Tell the user the page may have been deleted or archived.
