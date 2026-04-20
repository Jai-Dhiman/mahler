# E7 Meeting Follow-Through Design

**Goal:** After any recorded meeting, Mahler automatically generates context-aware action items, creates Notion tasks, and updates CRM contacts — closing the loop that meeting-prep opens.

**Not in scope:** Manual trigger for past meetings; completion tracking of generated tasks; multi-attendee action item assignment per-person (owner is always the user); transcript-level analysis (chronological summary is sufficient); Fathom action item extraction (Mahler generates its own using full context).

## Problem

Meeting-prep (E4) briefs before meetings but does nothing after. Action items from calls are lost — no Notion tasks created, no CRM `last_contact` updated, no record that a meeting happened unless the user manually runs `talked-to`. The follow-through loop does not close.

## Solution (from the user's perspective)

A meeting with Alice ends. Fathom finishes processing (typically 2–5 minutes after). Mahler posts to the home Discord channel:

> "Post-meeting: 1:1 with Alice Chen — 3 action items created. `[Alice Chen] Send Q2 IC memo` · `[Alice Chen] Intro to Marcus` · `Follow up on Series A timeline`. CRM updated."

The tasks exist in Notion. Alice's `last_contact` is today's date. No user action required.

## Design

Fathom fires a webhook to a new Cloudflare Worker (`fathom-webhook`) when a meeting finishes. The Worker:

1. Verifies the HMAC-SHA256 signature (rejects invalid; 401 — Fathom does not retry 4xx)
2. Deduplicates on `recording_id` via KV with a 24-hour TTL (silently returns 200 on dup)
3. Extracts the chronological summary from `default_summary.markdown_formatted`; falls back to `GET /recordings/{id}/summary` if absent
4. Posts a structured `@Mahler [FATHOM_MEETING]` Discord message to the home channel

Hermes receives the @mention. Context plugins inject automatically (project-log, kaizen priorities, Honcho memory). The `meeting-followthrough` skill procedure drives Mahler to:

- Fetch CRM context for each external attendee via `contacts.py summarize`
- Reason over the summary + context to generate action items (dedup against open tasks)
- Create Notion tasks via `tasks.py create` with `[Contact Name]` prefix
- Update each matched contact via `contacts.py talked-to`
- Post a summary embed to Discord

**Key decisions:**
- LLM reasoning stays in Hermes so all existing context plugins inject without duplication. The CF Worker is a pure HTTP bridge with no business logic.
- Fully automatic (no confirmation gate). Quality tuning happens by observing real output.
- Discord failure → 500 so Fathom retries. `tasks.py create` failure → surface and stop (partial task list is worse than none). `contacts.py talked-to` failure → surface but do not block task creation.

## Modules

**CF Worker `fathom-webhook`**
- Interface: single `POST /` HTTP handler; four exported pure/async functions: `verifySignature`, `checkAndSetDedup`, `extractSummary`, `buildDiscordMessage`
- Hides: HMAC-SHA256 via WebCrypto, 5-minute timestamp replay guard, KV read/write, Fathom REST API fallback, Discord webhook POST
- Depth: DEEP — trivial calling surface hiding multi-step verification and I/O
- Tested through: exported functions in unit tests; full handler via `SELF.fetch` in integration test

**Hermes skill `meeting-followthrough`**
- Interface: SKILL.md trigger phrase `[FATHOM_MEETING]`; reuses existing scripts
- Hides: CRM lookup + open-task cross-reference + action item generation procedure
- Depth: N/A (instruction set, not a code module)
- Tested through: manual end-to-end with a real Fathom recording

## File Changes

| File | Change | Type |
|------|--------|------|
| `assistant/workers/fathom-webhook/src/index.ts` | Webhook handler + four exported functions | New |
| `assistant/workers/fathom-webhook/src/index.test.ts` | Vitest unit + integration tests | New |
| `assistant/workers/fathom-webhook/wrangler.toml` | CF Worker deploy config, KV binding | New |
| `assistant/workers/fathom-webhook/package.json` | wrangler, vitest, CF worker types | New |
| `assistant/workers/fathom-webhook/vitest.config.ts` | vitest-pool-workers config with test bindings | New |
| `assistant/config/skills/meeting-followthrough/SKILL.md` | Hermes skill trigger + step-by-step procedure | New |

## Open Questions

- Q: Does Fathom's Chronological template populate `default_summary.markdown_formatted` when the webhook is registered with `include_summary: true`?
  Default: yes — the fallback `GET /recordings/{id}/summary` covers the case where it does not.
