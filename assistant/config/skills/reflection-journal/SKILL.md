---
name: reflection-journal
description: Weekly reflection journal. Posts three structured questions to Discord on Sunday evenings, records the user's freeform reply in D1, and concludes 2-3 synthesized facts into Honcho memory.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [reflection, journal, honcho, memory, productivity, cron]
    related_skills: [kaizen-reflection, evening-sweep]
---

## When to use

- Cron-triggered at 02:00 UTC every Sunday (6pm Pacific Saturday evening)
- When the user says "start reflection", "weekly check-in", or "reflection time"

## Prerequisites

| Variable | Purpose |
|---|---|
| `CF_ACCOUNT_ID` | Cloudflare account ID for D1 API calls |
| `CF_D1_DATABASE_ID` | D1 database ID |
| `CF_API_TOKEN` | Cloudflare API token with D1 read/write permission |
| `OPENROUTER_API_KEY` | API key for LLM synthesis |
| `HONCHO_API_KEY` | Honcho API key for durable memory storage |

The `reflection_log` table must be created before first use. Run once after deploy:

```bash
python3 -c "
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / '.hermes' / 'skills' / 'reflection-journal' / 'scripts'))
from d1_client import D1Client
d1 = D1Client(os.environ['CF_ACCOUNT_ID'], os.environ['CF_D1_DATABASE_ID'], os.environ['CF_API_TOKEN'])
d1.ensure_table()
print('reflection_log table ready.')
"
```

## Procedure

### Post reflection questions (cron or manual)

```bash
python3 ~/.hermes/skills/reflection-journal/scripts/journal.py --prompt
```

Output is the three-question block for Mahler to post to Discord.

### Record user's reply

```bash
python3 ~/.hermes/skills/reflection-journal/scripts/journal.py --record "USER_REPLY_TEXT"
```

Stores raw reply in D1 `reflection_log`, synthesizes 2-3 facts, concludes each into Honcho. Prints `"Reflection recorded."` on success.

## Cron flow

1. Every Sunday at 02:00 UTC, Mahler runs `journal.py --prompt`
2. Mahler posts the question block to Discord
3. User replies in a single message
4. Mahler calls `journal.py --record "USER_REPLY"` with the reply text
5. Mahler confirms: "Reflection recorded."
