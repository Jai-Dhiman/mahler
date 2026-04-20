# E2 Reply-Attribution Feedback Loop Design

**Goal:** Track which Outlook emails Jai replies to, deposit relational memory into Honcho, and enrich kaizen-reflection's priority-map proposals with reply-rate signal.

**Not in scope:**
- Gmail sent-box reply tracking (Outlook is Jai's primary sending client)
- Negative signal from silence (no-reply does not indicate wrong classification)
- Inline per-email Honcho lookup at classification time (Model B)
- Changes to classification pipeline, prefilter, or LLM prompt template
- Email drafting or sending capabilities

## Problem

`kaizen-reflection/scripts/reflect.py` uses `get_triage_patterns` which counts sender frequency at each classification tier. There is no outcome signal: a sender appearing 10 times as URGENT is indistinguishable from one Jai consistently ignored. The classifier cannot self-correct.

`email_triage_log` rows are written once and never updated. There is no column tracking whether a triaged email was acted on.

## Solution

A reply-attribution pass is added to every `triage.py` run, after the main pipeline:
1. Query D1 for recent URGENT/NEEDS_ACTION Outlook rows with a `conversation_id` and no `replied_at`
2. Refresh the Outlook access token
3. Query Outlook Sent Items for replies to those conversation threads within 3 days
4. For each match: call `honcho_client.conclude` (deposit relational fact), then `d1.mark_replied`

`kaizen-reflection`'s `reflect.py` is extended to use `get_triage_patterns_with_reply_rate` and include reply count and rate per sender in the LLM proposal prompt.

## Design

**Atomicity:** `honcho_client.conclude` must succeed before `d1.mark_replied`. If Honcho fails, the row stays unattributed and is retried on the next triage run (15-min cadence). Prevents silent fact loss.

**Best-effort:** The entire attribution pass is wrapped in a top-level `try/except` inside `_run_attribution_pass`. Any failure logs a warning to stderr; triage exits 0. The main pipeline is never blocked.

**Outlook only:** Gmail emails do not receive a `conversation_id`. Attribution queries filter to `source = 'outlook'` rows with a non-null, non-empty `conversation_id`.

**Schema migration:** `ensure_tables` in email-triage's `d1_client.py` gains a `_add_column_if_missing` helper that runs `ALTER TABLE email_triage_log ADD COLUMN`, catching the duplicate-column error silently. The two new columns are `conversation_id TEXT` and `replied_at TEXT`.

## Modules

**`honcho_client.py`** (new, `email-triage/scripts/`)
- Interface: `conclude(text, api_key, base_url, app_name, user_id) -> None`
- Hides: session ensure + 409-handling, POST to metamessages endpoint, HTTP error mapping
- Depth: DEEP

**`outlook_client.fetch_sent_replies`** (extension)
- Interface: `fetch_sent_replies(conversation_ids, access_token, since_days=3) -> dict[str, str]`
- Hides: OData `$filter` construction, `/mailFolders/sentItems/messages` path, datetime normalization
- Depth: DEEP

**`d1_client` extensions (email-triage)**
- Interface: `get_unattributed_recent(since_days)`, `mark_replied(message_id, replied_at)`, private `_add_column_if_missing`
- Also: update `insert_triage_result` for `conversation_id`; call migration from `ensure_tables`
- Depth: DEEP

**`d1_client.get_triage_patterns_with_reply_rate` (kaizen-reflection)**
- Interface: `get_triage_patterns_with_reply_rate(since_days, min_count) -> list[dict]`
- Hides: `COUNT(replied_at)` non-null counting, grouping SQL
- Depth: DEEP

**`triage._run_attribution_pass`** (internal)
- Interface: `_run_attribution_pass(env, d1, dry_run) -> None`
- Hides: token refresh, sent-items fetch, honcho+D1 write sequence, all error handling
- Depth: DEEP

## File Changes

| File | Change | Type |
|------|--------|------|
| `assistant/config/skills/email-triage/scripts/email_types.py` | Add `conversation_id: NotRequired[str]` | Modify |
| `assistant/config/skills/email-triage/scripts/outlook_client.py` | Add `conversationId` to `_fetch_from_folder`; add `fetch_sent_replies` | Modify |
| `assistant/config/skills/email-triage/scripts/d1_client.py` | Add `_add_column_if_missing`, `get_unattributed_recent`, `mark_replied`; update `insert_triage_result`; call migration in `ensure_tables` | Modify |
| `assistant/config/skills/email-triage/scripts/honcho_client.py` | New module with `conclude` | New |
| `assistant/config/skills/email-triage/scripts/triage.py` | Add `import honcho_client`, `_run_attribution_pass`, call in `main` | Modify |
| `assistant/config/skills/kaizen-reflection/scripts/d1_client.py` | Add `get_triage_patterns_with_reply_rate` | Modify |
| `assistant/config/skills/kaizen-reflection/scripts/reflect.py` | Use `get_triage_patterns_with_reply_rate`; extend proposal prompt | Modify |
| `assistant/config/skills/email-triage/tests/test_honcho_client.py` | New test module | New |
| `assistant/config/skills/email-triage/tests/test_outlook_client.py` | Add `fetch_sent_replies` + conversation_id tests | Modify |
| `assistant/config/skills/email-triage/tests/test_d1_client.py` | Add new method tests | Modify |
| `assistant/config/skills/email-triage/tests/test_triage_integration.py` | Add attribution pass tests | Modify |
| `assistant/config/skills/kaizen-reflection/tests/test_d1_client.py` | Add `get_triage_patterns_with_reply_rate` test | Modify |
| `assistant/config/skills/kaizen-reflection/tests/test_reflect.py` | Update existing mocks; add reply-rate test | Modify |

## Open Questions

None. All decisions resolved during brainstorm.
