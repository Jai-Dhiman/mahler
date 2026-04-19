# E2a: Honcho Memory Backend Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Mahler accumulates persistent memory across Discord sessions via Honcho in Hybrid recall mode, with explicit `honcho_conclude` tooling available for downstream phases.
**Spec:** assistant/docs/specs/2026-04-19-e2a-honcho-memory-design.md
**Style:** Follow the project's coding standards (assistant/CLAUDE.md)

---

## Prerequisites (manual — do before running tasks)

1. Create a Honcho account at https://honcho.dev
2. Note your workspace ID and confirm the cloud `baseUrl` (expected: `https://demo.honcho.dev`)
3. Generate an API key
4. Replace the two placeholders in Task 1's `honcho.json` content before committing
5. Run `flyctl secrets set HONCHO_API_KEY=<your-key>` from the `assistant/` directory

---

## Task Groups

Group A (parallel): Task 1, Task 2, Task 3
Group B (sequential, depends on Group A): Task 4

---

### Task 1: Create honcho.json and wire it into the Dockerfile
**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** Hermes starts with `honcho.json` present at `~/.hermes/honcho.json` inside the Docker image, making the Honcho memory provider available at runtime.

**Interface under test:** Docker image filesystem — `~/.hermes/honcho.json` exists with the correct content after `docker build`.

**Files:**
- Create: `assistant/config/honcho.json`
- Modify: `assistant/Dockerfile`

---

- [ ] **Step 1: Confirm the failing pre-condition**

```bash
# From the repo root. Verify honcho.json does not yet exist.
ls assistant/config/honcho.json 2>/dev/null && echo "EXISTS" || echo "MISSING (expected)"
```

Expected: `MISSING (expected)`

---

- [ ] **Step 2: Create `assistant/config/honcho.json`**

Replace `<HONCHO_WORKSPACE_ID>` with the actual workspace ID from your Honcho dashboard before saving.

```json
{
  "baseUrl": "https://demo.honcho.dev",
  "workspace": "<HONCHO_WORKSPACE_ID>",
  "aiPeer": "mahler",
  "peerName": "jai",
  "recallMode": "hybrid",
  "dialecticCadence": 3,
  "dialecticDepth": 2
}
```

---

- [ ] **Step 3: Add the COPY instruction to `assistant/Dockerfile`**

Insert after the existing `COPY --chown=hermes:hermes config/priority-map.md` line (line 45):

```dockerfile
COPY --chown=hermes:hermes config/honcho.json /home/hermes/.hermes/honcho.json
```

The relevant section of Dockerfile after the edit:

```dockerfile
# Copy priority map to Hermes workspace
RUN mkdir -p /home/hermes/.hermes/workspace
COPY --chown=hermes:hermes config/priority-map.md /home/hermes/.hermes/workspace/priority-map.md
COPY --chown=hermes:hermes config/honcho.json /home/hermes/.hermes/honcho.json
```

---

- [ ] **Step 4: Verify the COPY lands correctly in the image**

```bash
cd assistant && docker build --build-arg HERMES_VERSION=v2026.4.13 -t mahler-honcho-test . \
  && docker run --rm mahler-honcho-test cat /home/hermes/.hermes/honcho.json
```

Expected: The JSON content of `honcho.json` is printed, including `"recallMode": "hybrid"` and your workspace ID (not the placeholder).

If `<HONCHO_WORKSPACE_ID>` appears in the output, stop and replace it before continuing.

---

- [ ] **Step 5: Commit**

```bash
git add assistant/config/honcho.json assistant/Dockerfile \
  && git commit -m "feat(e2a): add honcho.json config and wire into Dockerfile"
```

---

### Task 2: Bridge HONCHO_API_KEY through entrypoint.sh
**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** `HONCHO_API_KEY` set as a Fly.io secret is written to `~/.hermes/.env` at container startup, making it available to Hermes's Honcho provider.

**Interface under test:** `entrypoint.sh` env-write block — `HONCHO_API_KEY` appears in `~/.hermes/.env` after the script runs.

**Files:**
- Modify: `assistant/entrypoint.sh`
- Modify: `assistant/.env.example`

---

- [ ] **Step 1: Confirm the failing pre-condition**

```bash
grep "HONCHO_API_KEY" assistant/entrypoint.sh && echo "EXISTS" || echo "MISSING (expected)"
```

Expected: `MISSING (expected)`

---

- [ ] **Step 2: Add `HONCHO_API_KEY` to the env-write block in `entrypoint.sh`**

The first `{ ... } > "$HERMES_ENV"` block (lines 7–23) writes secrets to `~/.hermes/.env`. Add `HONCHO_API_KEY` as the last line inside that block:

Old block ending:
```bash
  echo "OPENROUTER_MODEL=${OPENROUTER_MODEL:-x-ai/grok-4.1-fast}"
} > "$HERMES_ENV"
```

New block ending:
```bash
  echo "OPENROUTER_MODEL=${OPENROUTER_MODEL:-x-ai/grok-4.1-fast}"
  echo "HONCHO_API_KEY=${HONCHO_API_KEY:-}"
} > "$HERMES_ENV"
```

---

- [ ] **Step 3: Add `HONCHO_API_KEY` to `.env.example`**

Append a new section at the end of `assistant/.env.example`:

```bash
# Honcho memory backend (Phase E2a — provision at honcho.dev)
HONCHO_API_KEY=
```

---

- [ ] **Step 4: Verify the secret bridge works**

```bash
# Simulate what entrypoint.sh does for HONCHO_API_KEY
HONCHO_API_KEY=test-key-abc bash -c "
  source assistant/entrypoint.sh 2>/dev/null || true
" 2>/dev/null
# OR: check the script directly
HONCHO_API_KEY=test-key-abc bash -c '
  HERMES_ENV=/tmp/test-hermes.env
  echo "HONCHO_API_KEY=${HONCHO_API_KEY:-}" >> "$HERMES_ENV"
  grep HONCHO_API_KEY "$HERMES_ENV"
'
```

Expected: `HONCHO_API_KEY=test-key-abc`

---

- [ ] **Step 5: Commit**

```bash
git add assistant/entrypoint.sh assistant/.env.example \
  && git commit -m "feat(e2a): bridge HONCHO_API_KEY to hermes env"
```

---

### Task 3: Add honcho_conclude guidance to SOUL.md
**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** Mahler proactively calls `honcho_conclude` when it learns a durable fact during a turn — not only when explicitly asked.

**Interface under test:** `config/SOUL.md` system prompt — the agent's behavior in production sessions.

**Files:**
- Modify: `assistant/config/SOUL.md`

---

- [ ] **Step 1: Confirm the failing pre-condition**

```bash
grep "honcho" assistant/config/SOUL.md && echo "EXISTS" || echo "MISSING (expected)"
```

Expected: `MISSING (expected)`

---

- [ ] **Step 2: Add the `honcho_conclude` paragraph to SOUL.md**

Append to the end of `assistant/config/SOUL.md`:

```markdown

Your memory rules:
- When you learn a durable fact about Jai during a turn — a stated preference, an open commitment, a named habit, or a relationship context — call `honcho_conclude` to write it before the turn ends.
- Do not wait to be asked. If Jai says "I prefer async over calls" or "I owe Sarah a follow-up by Friday," that is worth persisting immediately.
- Use `honcho_search` when answering questions where prior context from past sessions would materially improve the answer.
```

---

- [ ] **Step 3: Verify SOUL.md is well-formed**

```bash
cat assistant/config/SOUL.md
```

Expected: Original personality content followed by the new "Your memory rules:" section. No duplicate content. No stray formatting.

---

- [ ] **Step 4: Commit**

```bash
git add assistant/config/SOUL.md \
  && git commit -m "feat(e2a): add honcho_conclude guidance to SOUL.md"
```

---

### Task 4: Deploy to Fly.io and run behavioral verification
**Group:** B (depends on Group A — all three tasks above must be committed first)

**Behavior being verified:** Three behaviors, tested in order:
1. **Smoke:** Mahler starts and Honcho tools are available in the agent (session created in Honcho dashboard)
2. **Conclude:** Stating a preference in Discord causes `honcho_conclude` to be called within the same turn
3. **Persistence:** A new Discord session recalls the preference without re-stating it

**Interface under test:** Live Mahler bot in Discord + Honcho dashboard + Hermes agent logs.

**Files:**
- No file changes — deployment and verification only

---

- [ ] **Step 1: Set the Fly secret (if not already done in prerequisites)**

```bash
cd assistant && flyctl secrets set HONCHO_API_KEY=<your-actual-key>
```

Expected: `Release v<N> created` (Fly acknowledges the secret).

---

- [ ] **Step 2: Deploy**

```bash
cd assistant && flyctl deploy --remote-only
```

Expected: Build completes, deployment succeeds, health checks pass. Watch for any `honcho` errors in the build output.

---

- [ ] **Step 3: Smoke test — verify Honcho session is created**

In Discord, @mention Mahler with any message:
```
@Mahler what's on my plate today?
```

Then check Hermes logs:
```bash
flyctl ssh console --user hermes -C "hermes logs -n 50"
```

Expected in logs: No `honcho` connection errors. In the Honcho dashboard (honcho.dev), a new session should appear for peer `jai` / app `mahler` within 30 seconds of the message.

If logs show `honcho: connection refused` or `invalid api key`: check `flyctl secrets list` to confirm `HONCHO_API_KEY` is set, and verify `workspace` in `honcho.json` matches the dashboard value.

---

- [ ] **Step 4: Conclude test — verify honcho_conclude is called for durable facts**

In Discord, state a clear preference:
```
@Mahler I prefer async communication over calls — never schedule a call without asking me first.
```

Check agent logs for the tool call:
```bash
flyctl ssh console --user hermes -C "hermes logs -n 30"
```

Expected: Log entry showing `honcho_conclude` was called with a fact containing the preference. In the Honcho dashboard, a new insight should appear under peer `jai`.

---

- [ ] **Step 5: Persistence test — verify cross-session recall**

Wait at least 2 minutes (allow Honcho Dreaming to process the turn). Start a completely new Discord conversation (close the thread, start a fresh @mention):
```
@Mahler should I jump on a quick call with Marcus to sync on the project?
```

Expected: Mahler's response factors in the preference stated in the previous session (e.g., suggests async alternatives, flags that Jai prefers not to be scheduled without being asked) — without being re-told the preference in this session.

---

- [ ] **Step 6: Confirm and close**

All three behavioral tests pass. No deployment rollback needed. E2a is live and accumulating signal.

```
# No commit needed — this task is deploy + verify only.
# Update CLAUDE.md roadmap to mark E2a as shipped (separate commit after verification).
```
