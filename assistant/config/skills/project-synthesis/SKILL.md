---
name: project-synthesis
description: Weekly coding session synthesis. Reads the past 7 days of project_log wins and blockers from D1, synthesizes one cross-project paragraph covering attention, trajectory, and friction, and writes it to Honcho memory.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [memory, honcho, synthesis, coding, productivity, cron]
    related_skills: [memory-kaizen, kaizen-reflection, reflection-journal]
---

## When to use

- Cron-triggered every Sunday at 18:00 UTC
- When the user asks "synthesize my coding week" or "update project memory"

## Procedure

```bash
python3 ~/.hermes/skills/project-synthesis/scripts/synthesize.py --run
```

Prints `"Project synthesis: N entries synthesized."` on success, or `"No project activity this week."` if D1 is empty.

## Cron flow

1. Every Sunday at 18:00 UTC, Mahler runs `synthesize.py --run`
2. If entries exist, Mahler posts the confirmation message to Discord
3. If empty, Mahler posts "No project activity this week" to Discord
