---
name: memory-kaizen
description: Weekly Honcho memory distillation. Reads the last 30 days of conclusions, identifies 2-4 high-signal patterns that appear across multiple entries, and writes each as a new conclusion. Runs one hour after project-synthesis.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [memory, honcho, kaizen, synthesis, productivity, cron]
    related_skills: [project-synthesis, reflection-journal, kaizen-reflection]
---

## When to use

- Cron-triggered every Sunday at 19:00 UTC (one hour after project-synthesis)
- When the user asks "run memory kaizen" or "distill my Honcho memories"

## Procedure

```bash
python3 ~/.hermes/skills/memory-kaizen/scripts/kaizen.py --run
```

Prints `"Memory kaizen: N patterns written to Honcho."` on success.
Prints `"Insufficient data for memory kaizen (N conclusions, need 5)."` if fewer than 5 conclusions exist.
Prints `"Memory kaizen: no multi-entry patterns found."` if LLM finds no cross-entry patterns.

## Cron flow

1. Every Sunday at 19:00 UTC, Mahler runs `kaizen.py --run`
2. Mahler posts the result message to Discord
