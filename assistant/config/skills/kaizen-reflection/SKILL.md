---
name: kaizen-reflection
description: Weekly email triage reflection. Analyzes email_triage_log patterns from the past 7 days, proposes priority-map reclassifications, and applies approved changes to the D1 priority_map table.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [email, triage, kaizen, priority-map, reflection, productivity]
    related_skills: [email-triage, morning-brief]
---

## When to use

- Invoked automatically by Hermes cron every Sunday at 18:00 UTC
- When the user asks any of: "run kaizen reflection", "check triage patterns", "update priority map", "what should we reclassify"

## One-time setup

Seed the `priority_map` table once after the first deploy:

```bash
python3 ~/.hermes/skills/kaizen-reflection/scripts/migrate.py \
    --file ~/.hermes/workspace/priority-map.md
```

## Procedure

### Generate weekly proposals

```bash
python3 ~/.hermes/skills/kaizen-reflection/scripts/reflect.py --run
```

Output is a JSON array of proposals, or `"No proposals this week."` if no patterns qualify.

### Apply an approved proposal

```bash
python3 ~/.hermes/skills/kaizen-reflection/scripts/reflect.py \
    --apply '{"sender": "news@acme.com", "current_tier": "NEEDS_ACTION", "proposed_tier": "FYI", "evidence": "5 occurrences in 7 days with no follow-up"}'
```

Prints `"Priority map updated. Moved <sender> to <tier>."` on success.

## Cron flow

1. Every Sunday at 18:00 UTC, Mahler runs `reflect.py --run`
2. If proposals are returned, Mahler posts each one to Discord as a separate message with approve/deny buttons
3. On approval of a specific proposal, Mahler calls `reflect.py --apply PROPOSAL_JSON` for that proposal
4. If `"No proposals this week."` is returned, Mahler reports this to Discord and takes no further action
