# Always-On Scan Summary + Diagnostics

## Problem

After Phase 2.4 (autonomous mode), the morning scan notification flow broke:

1. Scan summary only fires when `recommendations_sent == 0` (line 1147 of morning_scan.py)
2. When opportunities exist but all fail (position size zero, order errors, agent rate limits), no summary is sent
3. No visibility into WHY trades aren't happening -- errors are caught and printed to Cloudflare Workers logs but never surface to Discord
4. Agent shadow decisions (approve/skip) aren't reported, making AI calibration hard to track

## Design

### 1. Always-send scan summary

Evolve `send_no_trade_notification()` into `send_scan_summary()`. Call it at the end of every morning scan, regardless of `recommendations_sent`.

- When trades placed: title "Morning Scan Complete - X Trades Placed", green embed
- When no trades: title "Morning Scan Complete - No Viable Trades", yellow/orange embed
- Individual trade alerts (`send_autonomous_notification`) continue firing as trades are placed; the summary comes at the end

### 2. Enhanced skip reason detail

Replace generic "Position Size Zero: 3" with the binding constraint from `size_result.reason`:
- "Equity correlation limit (SPY, QQQ): 2"
- "5% single position limit: 1"

Track errors separately from skip reasons:
- `agent_errors`: count of Claude API failures (rate limit, timeout)
- `order_errors`: count of broker order failures

### 3. Agent shadow decision tracking

Count shadow approve vs skip decisions in the opportunity loop. Report in summary:
- "Agent Decisions: 2 approve, 1 skip (shadow only)"

### 4. Scan timing

Record timing at key points using `time.time()`:
- Total scan duration
- Per-underlying scan time (options chain fetch + screening)
- Agent pipeline average duration per opportunity

Report in summary:
- "Total: 45s | SPY: 8s | QQQ: 12s | ..."
- "Agent pipeline: avg 15s/opportunity"

## Changes Required

### morning_scan.py

- Add timing instrumentation (`scan_start`, per-underlying, per-agent-call)
- Add `agent_shadow_stats` dict tracking approve/skip counts
- Add `scan_errors` dict tracking agent_errors and order_errors
- Use `size_result.reason` in skip_reasons instead of generic "position_size_zero"
- Move scan summary call outside the `if recommendations_sent == 0` block
- Pass new data (trades_placed, shadow_stats, timing, errors) to summary

### discord.py

- Rename `send_no_trade_notification()` to `send_scan_summary()`
- Add parameters: `trades_placed`, `agent_shadow_stats`, `scan_timing`, `errors`
- Adapt embed title/color based on whether trades were placed
- Add fields for errors, shadow decisions, timing
- Keep backward-compatible: all new params have defaults

## Notification Format

```
Morning Scan Complete - No Viable Trades
Found 140 opportunities, 8 passed filters, 0 approved.

VIX: 26.9
Market Regime: Bull High Vol
Size Multiplier: 50%
IV Percentile: SPY: 83% | QQQ: 73% | IWM: 79%

Skip Reasons:
  Equity correlation limit (SPY, QQQ): 3
  No Opportunities: 2

Agent Decisions (shadow):
  Would approve: 1 | Would skip: 2

Errors:
  Claude rate limit: 1

Timing:
  Total: 45s
  SPY: 8s | QQQ: 12s | IWM: 5s | TLT: 3s | GLD: 7s
  Agent pipeline: avg 15s

Per-Underlying:
  SPY: 28 found, 2 passed
  QQQ: 45 found, 2 passed
  ...
```
