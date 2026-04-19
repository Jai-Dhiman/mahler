# E2a: Honcho Memory Backend Design

**Goal:** Mahler accumulates persistent memory about Jai's preferences, habits, communication style, and open commitments across Discord sessions — and can explicitly write durable facts for downstream phases (E5 CRM, E3+ reflection journal) to build on.

**Not in scope:**
- Any E5 CRM contact table or relationship-manager skill
- E3+ reflection journal or expanded kaizen scope
- Per-user scoping (Jai is the only user)
- Any changes to how existing plugins (kaizen-context, conversation-history) operate
- Self-hosted Honcho; cloud instance only

---

## Problem

Every Mahler session starts with zero memory of prior conversations. Grok only knows what the kaizen priority map and the last 45 minutes of Discord history provide. When Jai states a preference, names a commitment, or establishes a habit pattern, that signal disappears when the session ends. Downstream phases (E5: CRM, E3+: reflection journal, E10: life tracking) are all predicated on Honcho accumulating this signal from day one — the longer Honcho is absent, the less compounding value those phases get when they ship.

---

## Solution (from Jai's perspective)

After this phase ships, Mahler remembers things across sessions without being told twice. If Jai says "I prefer async communication over calls" in one session, Mahler factors that into advice in all future sessions without re-prompting. When Mahler learns something durable mid-turn — a preference, an open commitment, a habit — it calls `honcho_conclude` to write it before the turn ends. Honcho's Dreaming process runs asynchronously in the background, deriving additional insights from conversation history between turns.

---

## Design

**Chosen approach: Hybrid recall mode.** Honcho auto-injects a context block into every system prompt turn (base user profile + dialectic-synthesized reasoning about current state), and the agent also has explicit tools (`honcho_profile`, `honcho_search`, `honcho_context`, `honcho_conclude`) for reading and writing structured facts.

Hybrid over context-only: E5 and E3+ require `honcho_conclude` for explicit structured writes (relationship insights, reflection facts). Context-only makes Honcho a read-only oracle and breaks those downstream phases.

Hybrid over tools-only: "compounds automatically" requires passive accumulation via Dreaming + auto-injection on every turn. Tools-only bets compounding on the model's discipline to call `honcho_context` at turn start — an unreliable bet.

**dialecticCadence: 3** — dialectic reasoning re-synthesizes every 3 turns. Balances memory freshness against per-turn token overhead. Tunable post-deploy without rebuild.

**dialecticDepth: 2** — two reasoning passes per synthesis. One pass for preference/habit extraction, one pass for contextual state. Sufficient for a single-user personal assistant.

This phase is pure wiring. No custom plugin. No forked Hermes code. The entire implementation is: one new JSON config file, three file edits (entrypoint.sh, Dockerfile, SOUL.md, .env.example), one Fly secret.

---

## Modules

**config/honcho.json**
- Interface: 7-field JSON config read by Hermes at startup
- Hides: All Honcho dialectic reasoning scheduling, context injection timing, tool registration, Dreaming background process management, API authentication
- Tested through: Hermes startup (tools appear in agent), behavioral smoke tests post-deploy

*Depth verdict: DEEP — a 7-field file drives the entire Honcho integration. All complexity lives inside Hermes's Honcho provider.*

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `assistant/config/honcho.json` | New Honcho config: recallMode hybrid, dialecticCadence 3, dialecticDepth 2, baseUrl/workspace/aiPeer/peerName | New |
| `assistant/Dockerfile` | Add `COPY --chown=hermes:hermes config/honcho.json /home/hermes/.hermes/honcho.json` | Modify |
| `assistant/entrypoint.sh` | Add `HONCHO_API_KEY` to the env var block written to `~/.hermes/.env` | Modify |
| `assistant/config/SOUL.md` | Add one paragraph to operating rules: call `honcho_conclude` when learning durable facts | Modify |
| `assistant/.env.example` | Add `HONCHO_API_KEY=` with comment referencing honcho.dev provisioning | Modify |

---

## Open Questions

- Q: What is Jai's Honcho workspace ID after provisioning?  Default: placeholder `<HONCHO_WORKSPACE_ID>` in honcho.json — must be replaced before deploying.
- Q: What is the Honcho cloud baseUrl?  Resolved: `https://api.honcho.dev` is the production endpoint for paying customers. `https://demo.honcho.dev` is for unauthenticated testing only.
