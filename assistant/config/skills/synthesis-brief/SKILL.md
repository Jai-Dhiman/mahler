---
name: synthesis-brief
description: Daily synthesis push at ~6am PT weekdays. Loads Honcho conclusions, project_log wins, and local_capture (memory + git) deltas, asks the LLM for 3 non-obvious connections + a weekly pattern + a question to sit with, validates citations and length, then writes the result to D1 (synthesis_brief table) and mahler_kv (key synthesis_brief:latest) for the 8am morning-brief to prepend.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [memory, synthesis, brief, discord, productivity, cron]
    related_skills: [morning-brief, memory-kaizen, project-synthesis]
---

## When to use

- Cron-triggered Mon-Fri at 13:00 UTC (5am PDT / 6am PST)
- When the user asks "run synthesis brief" or "what's today's synthesis"

## Procedure

Run the synthesis brief script:

    python3 ~/.hermes/skills/synthesis-brief/scripts/synthesize.py --run

Dry-run (prints the resulting brief JSON, does not write to D1 or KV):

    python3 ~/.hermes/skills/synthesis-brief/scripts/synthesize.py --run --dry-run

## Output

On success: writes a row to `synthesis_brief` and updates `mahler_kv` at key `synthesis_brief:latest` with the JSON `{posted_at, connections, pattern, question}`. Prints `Synthesis brief written.`

On thin context / validator failure: prints `Synthesis brief skipped: <reason>` and exits 0. No Discord post.

The 8am morning-brief reads `mahler_kv:synthesis_brief:latest` and prepends a Synthesis field if the row is fresher than 24h.
