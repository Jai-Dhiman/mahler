# Chief of Staff AI Research Articles

---

## Article 1: How I built a chief of staff on OpenClaw that's better than any human I've hired

*VC in fundraise context, boards, angel investing.*

### What makes a great chief of staff
- Filters noise so only right things reach you
- Ensures meeting prep and nothing falls through after
- Keeps full picture of what's in flight, flags what's slipping
- Tracks relationships and status with every important person
- Creates daily/weekly rhythm

### Memory architecture (two layers)
- **Daily notes:** one markdown file per day (`memory/YYYY-MM-DD.md`). Script pulls from sessions throughout the day and writes these automatically.
- **Long-term memory in MEMORY.md:** curated by the agent. Key people, active projects, lessons learned, decisions made. Agent periodically synthesizes from daily notes. Read on startup to orient.
- All flat markdown files — readable, editable, git-backable. No abstraction layer between user and assistant's understanding.

### Fundraise use case
- 100+ LP contacts across countries tracked as pipeline
- Pre-meeting brief: researches fund, recent content, maps to thesis, tailored talking points
- Ongoing relationships: pipeline stage, deck version, questions from last time, commitments made

### Kaizen loop
- Every Friday cron: scans community, checks new patterns, saves findings to `memory/kaizen-research-YYYY-MM-DD.md`
- Sunday review: summarizes week's research, surfaces top ideas, discuss what to change
- System learns from daily interactions — if user keeps correcting something, it's captured and surfaced as suggestion
- First versions always too complicated/noisy — Kaizen makes refactoring disciplined rather than ad hoc

### Meeting prep and follow-through
- Brief 60 min before external meetings via WhatsApp: prior meeting notes, recent email threads, open action items
- Post-meeting: processes notes via Granola API, extracts action items, sends to Todoist, tracks commitments in per-person markdown files
- Everything feeds back into memory for next cycle

### Task management
- Source of truth: structured markdown with all context and history
- Near-term items synced to Todoist (split is important)
- Evening task sweep: what's due, overdue, sitting too long, pattern of rolling forward flagged

### Stakeholder / relationship context
- Persistent structured context on every person/company/project
- Relationship history, last touchpoint, open commitments, what they care about
- Every meeting processed + email triaged feeds this picture

### Information filtering
- Email and calendar triage regularly across both personal and work Gmail
- Auto-pulls expense receipts → quarterly tracking
- Generates travel itineraries from booking confirmations, flags gaps
- Drafts follow-up emails as part of fundraising tasks

### Operational rhythm
- Morning brief at 9am, evening wrap at 6pm via WhatsApp
- Morning: top priorities, overdue tasks, today's calendar, items needing attention
- Evening: what happened, what stalled, what to prep for tomorrow
- If nothing to say, silence

### Architecture principle
**LLMs handle judgment. Scripts handle everything else.**
- Deterministic work (file reads, API calls, sending messages, comparing timestamps) → Python
- LLM layer: synthesis, prioritization, drafting, anything requiring reasoning
- Pushing deterministic work through LLM = unpredictable breakage, loss of trust

---

## Article 2: How to turn your OpenClaw into the world's best assistant

*Repo: github.com/snarktank/clawchief*

### Capabilities
- Schedule meetings, parse booking links
- Check inbox every 15 minutes, surface only what matters
- Proactively follow up on emails without reply
- Watch calendar, flag conflicts, warn about upcoming events
- Run day from one canonical markdown task list
- Prep task list before waking up
- Keep tasks clean (no duplicate entries)
- Update outreach tracker/CRM from email activity
- Research suppliers/partners and reach out
- Send short, high-signal updates only when action needed
- Work from durable context in files, memory, Gmail, Calendar, Sheets
- Adapt to business, preferences, operating style

### Repo structure
```
clawchief/
  clawchief/
    priority-map.md
    auto-resolver.md
    meeting-notes.md
    tasks.md
    tasks-completed.md
  skills/
    business-development/SKILL.md
    daily-task-manager/SKILL.md
    daily-task-prep/SKILL.md
    executive-assistant/SKILL.md
  workspace/
    HEARTBEAT.md
    TOOLS.md
    memory/meeting-notes-state.json
    tasks/
  cron/
    jobs.template.json
```

### Key files
- **HEARTBEAT.md:** tells assistant how to be proactive. Read priority map, auto-resolver, meeting-notes policy + ledger, live task file, run right workflow, only message when something matters.
- **TOOLS.md:** environment-specific notes (preferred email accounts, tracker notes, local quirks, target market, tactical rules).
- **tasks.md:** one canonical markdown task list — single live source of truth for what matters today.
- **priority-map.md:** defines what URGENT/NEEDS_ACTION/FYI/NOISE means
- **auto-resolver.md:** rules for automatically resolving/routing emails

### Setup
1. Install OpenClaw, get GOG (Google OAuth Gateway) working for Gmail/Calendar/Sheets/Docs
2. Copy skill directories to `~/.openclaw/skills/`
3. Copy workspace files to `~/.openclaw/workspace/`
4. Customize AGENTS.md, SOUL.md, USER.md, IDENTITY.md, MEMORY.md
5. Replace all placeholders (owner name, emails, business, timezone, channel, Sheet ID)
6. Set up cron jobs (executive assistant sweep, daily task prep, BD sourcing)

### The proactive shift
The assistant becomes dramatically more useful when it wakes itself up to do recurring work. Cron shifts it from reactive to proactive.

### Validation criteria
- Reads source-of-truth files correctly
- Routes proactive updates to right place
- Uses Gmail message-level search
- Checks all relevant calendars before booking
- Treats tracker/sheet as live outreach source of truth
- Promotes due-today items into Today
- Archives prior-day completions
- Ingests meeting notes into real tasks and follow-ups

---

## Article 3: I Gave My Hermes + OpenClaw Agents a Subconscious, and Now They Self-Improve 24/7

### Problem
Most agent systems break in the same ways:
- Need babysitting, drift over time, burn tokens on vague exploration, produce output but not momentum

### The subconscious loop
1. Gather evidence from latest run
2. Generate candidate ideas
3. Debate ideas against hard objections with a smarter agent
4. Synthesize one recommendation
5. Write result into state
6. Next run starts from updated state (not from zero)

### Required components
1. **Runner:** coordinates the full cycle (load brief → fetch state → ideation → critique → synthesis → write artifacts → hand off)
2. **Persistent state:** JSON for summaries/governance, JSONL for append-only history, markdown for human-readable output, stable directory structure
3. **Scheduler/trigger:** cron, new metrics, live signal, manual review. Be realistic about run frequency — too many runs → excessive divergence from original principles.
4. **Transport/delivery:** Discord, Telegram, file path, dashboard, task queue. Keep separate from reasoning layer.
5. **Model router:** cheap/local model for ideation, stronger model for challenge/synthesis. Keeps costs sane and quality high.
6. **Review/approval gate:** human check at end. Keeps it an assistant loop, not autopilot. You decide which workflows to auto-approve.
7. **Artifact writers:** write back into filesystem predictably (`ideas/`, `debate/`, `winning-concept.md`, `improvement-backlog.md`, `run-summary.json`)

### Artifact flow
- `ideas/` — candidate directions (thinking in options, not conclusions)
- `debate/` — challenge and defence turns (forces ideas to be more concrete)
- `winning-concept.md` — final approved direction
- `improvement-backlog.md` — what the next run should sharpen

### Minimal folder structure
```
your-system/
  runner/
    run.js, cron.js, transport.js
  state/
    governance.json, memory.jsonl, outcomes.jsonl, latest-summary.json
  runs/current-run/
    ideas/, debate/, recommendation/
    winning-concept.md, improvement-backlog.md, run-summary.json
  targets/
  briefs/
```

### Minimal pseudocode
```
load brief
load recent state
load recent memory
load governance rules
ideas = generate_candidate_directions(brief, state)
curated = select_strongest(ideas)
for idea in curated:
    debate = challenge_and_defend(idea, brief, state)
    if debate converges:
        synthesize = produce_final_recommendation(debate, brief, state)
        if human_approval_required:
            write_artifacts(synthesize)
            persist_learning(synthesize, state)
            deliver_output(synthesize)
```

### Model stack
- Local: qwen3.5 9B for fast ideation
- Frontier: ChatGPT 5.4 mini for challenge/synthesis
- Alternative: OpenRouter for routing, MiniMax M2.7

### Guardrails
- Evidence first
- Explicit states instead of fuzzy opinions
- One human approval gate at the end
- No automatic promotion from zero-confirmation clusters
- Next run must write learning back into state
