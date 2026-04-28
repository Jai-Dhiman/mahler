# Mahler

Personal AI infrastructure. Three systems, one monorepo.

## What's here

**`assistant/`** — AI chief of staff (Hermes Agent on Fly.io + Discord). Handles email triage, meeting prep, morning briefs, task management, and memory. Connects to Notion, Google Calendar, Gmail, and Cloudflare D1/KV for persistence.

**`traderjoe/`** — Autonomous options credit-spread trading system. Two codebases: a Rust backtesting engine (`traderjoe-backtest/`) and a Cloudflare Worker (`trader-joe/`) that runs live scans and manages positions via Alpaca. All trade decisions are algorithmic — no LLM in the loop.

**`finance/`** — Cloudflare Worker that snapshots daily balances (Plaid + Alpaca) to D1 and posts a weekly net-worth summary to Discord. Exposes a bearer-auth read API.

## Stack

- Workers runtime: Cloudflare (WASM via worker-rs, D1, KV)
- Assistant runtime: Fly.io (Docker)
- Languages: Rust, Python, TypeScript
- Package managers: `uv` (Python), `bun` (JS)

See each subdirectory's `CLAUDE.md` for detailed conventions and architecture.
