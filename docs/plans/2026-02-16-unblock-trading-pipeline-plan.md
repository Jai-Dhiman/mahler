# Unblock Trading Pipeline -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unblock the deadlocked trading pipeline by relaxing position limits, tightening exits, adding monitoring visibility, and validating all changes with backtests.

**Architecture:** Parameter changes to 4 Python files (position_sizer, validators, regime, position_monitor) plus EOD summary enhancement. Backtests run first via the Rust engine to validate parameter choices. Deploy via `npx wrangler deploy`.

**Tech Stack:** Python (Cloudflare Workers/Pyodide), Rust (backtester), pytest, wrangler CLI

---

### Task 1: Run Backtest Sweep 1 -- Max Positions

**Files:**
- None (read-only backtest runs)

**Step 1: Run baseline backtest (current params: max 10 positions, 50% correlated exposure)**

Run:
```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && ./target/release/mahler-backtest run \
  --ticker QQQ --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 50 --stop-loss 125 --slippage orats \
  --max-correlated-exposure 50.0
```
Expected: Outputs CAGR, Sharpe, max drawdown, trade count. Record these as baseline.

**Step 2: Run with 100% correlated exposure (simulates allowing 3 equity positions)**

Run:
```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest && ./target/release/mahler-backtest run \
  --ticker QQQ --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 50 --stop-loss 125 --slippage orats \
  --max-correlated-exposure 100.0
```
Expected: Higher trade count. Compare CAGR, Sharpe, drawdown to baseline.

**Step 3: Record results**

Save the output from both runs. Note trade count, CAGR, Sharpe, max drawdown for comparison.

---

### Task 2: Run Backtest Sweep 2 -- Exit Thresholds

**Files:**
- None (read-only backtest runs)

**Step 1: Run 16-combo grid search of profit target x stop loss on QQQ**

Run each combination. The 4 most important combos to prioritize:

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest

# Proposed params: PT=35, SL=125
./target/release/mahler-backtest run \
  --ticker QQQ --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 35 --stop-loss 125 --slippage orats

# Aggressive: PT=25, SL=100
./target/release/mahler-backtest run \
  --ticker QQQ --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 25 --stop-loss 100 --slippage orats

# Moderate: PT=35, SL=150
./target/release/mahler-backtest run \
  --ticker QQQ --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 35 --stop-loss 150 --slippage orats

# Current baseline: PT=50, SL=125
./target/release/mahler-backtest run \
  --ticker QQQ --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 50 --stop-loss 125 --slippage orats
```

Then run the remaining combos (PT=25,35,50,65 x SL=100,125,150,200) to fill the grid.

Expected: Table of 16 results showing win rate, profit factor, CAGR for each combo.

**Step 2: Record best performing combo**

Note: The Jan 30 walkforward already found PT=65, SL=125 was optimal for QQQ at 6.25% CAGR. We're testing tighter targets (35%) for faster turnover during paper trading. Expect lower CAGR but more trades.

---

### Task 3: Run Backtest Sweep 3 -- Combined Validation

**Files:**
- None (read-only backtest runs)

**Step 1: Run best params from sweeps 1-2 across all tickers**

Use the winning combo from Sweep 2 (likely PT=35, SL=125) with full correlated exposure:

```bash
cd /Users/jdhiman/Documents/mahler/mahler-backtest

# SPY
./target/release/mahler-backtest run \
  --ticker SPY --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 35 --stop-loss 125 --slippage orats \
  --max-correlated-exposure 100.0

# QQQ
./target/release/mahler-backtest run \
  --ticker QQQ --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 35 --stop-loss 125 --slippage orats \
  --max-correlated-exposure 100.0

# IWM
./target/release/mahler-backtest run \
  --ticker IWM --start 2006-01-01 --end 2024-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 35 --stop-loss 125 --slippage orats \
  --max-correlated-exposure 100.0
```

Expected: All three show positive CAGR and Sharpe > 0.5.

**Step 2: Stress test -- 2020 COVID crash**

```bash
./target/release/mahler-backtest run \
  --ticker QQQ --start 2020-01-01 --end 2020-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 35 --stop-loss 125 --slippage orats \
  --max-correlated-exposure 100.0
```

**Step 3: Stress test -- 2022 bear market**

```bash
./target/release/mahler-backtest run \
  --ticker QQQ --start 2022-01-01 --end 2022-12-31 \
  --data data/orats --equity 100000 \
  --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 \
  --profit-target 35 --stop-loss 125 --slippage orats \
  --max-correlated-exposure 100.0
```

Expected: Drawdown stays manageable (< 15%). Strategy should survive even if negative returns.

**Step 4: Review all results and decide final parameters**

Compare backtest results to proposed params. If data strongly disagrees (e.g., PT=35 is clearly worse than PT=50), adjust the code changes in subsequent tasks accordingly.

**Step 5: Commit backtest findings**

No code changes -- just record results in the terminal output for reference during implementation.

---

### Task 4: Update Position Sizing Limits

**Files:**
- Modify: `src/core/risk/position_sizer.py:46-58`
- Test: `tests/test_validators.py` (existing tests should still pass)

**Step 1: Update RiskLimits defaults in position_sizer.py**

In `src/core/risk/position_sizer.py`, change three values in the `RiskLimits` dataclass:

Line 46: `max_portfolio_heat_pct: float = 0.10` -> `max_portfolio_heat_pct: float = 0.20`
Line 49: `max_per_underlying_pct: float = 0.033` -> `max_per_underlying_pct: float = 0.10`
Line 58: `max_positions_per_equity_class: int = 1` -> `max_positions_per_equity_class: int = 3`

Update the comments to reflect paper trading mode:
- Line 46 comment: `# 20% total open risk (paper trading: higher throughput)`
- Line 49 comment: `# ~10% max in any single underlying (paper trading)`
- Line 56-57 comment: `# Paper trading: allow 3 equity positions for learning & data collection`

**Step 2: Run existing tests to verify no regressions**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/ -v --timeout=30 -x`
Expected: All existing tests pass. The tests create validators with default configs, so they use the new defaults.

**Step 3: Commit**

```bash
git add src/core/risk/position_sizer.py
git commit -m "Relax position sizing limits for paper trading

Raise equity correlation limit from 1 to 3, portfolio heat from 10%
to 20%, per-underlying from 3.3% to 10%. Unblocks the deadlocked
pipeline where 2 open equity positions blocked all new trades."
```

---

### Task 5: Update Exit Conditions

**Files:**
- Modify: `src/core/risk/validators.py:181-198`
- Test: `tests/test_validators.py` (tests need updating for new thresholds)

**Step 1: Update ExitConfig defaults in validators.py**

In `src/core/risk/validators.py`, change four values in the `ExitConfig` dataclass:

Line 181: `profit_target_pct: float = 0.50` -> `profit_target_pct: float = 0.35`
Line 182: `stop_loss_pct: float = 2.00` -> `stop_loss_pct: float = 1.25`
Line 183: `time_exit_dte: int = 21` -> `time_exit_dte: int = 14`
Line 197: `gamma_protection_pnl: float = 0.70` -> `gamma_protection_pnl: float = 0.50`

Update the research notes docstring (lines 173-179) to:
```
Research notes:
- 35% profit target for paper trading (faster turnover, more data)
- 125% stop loss cuts losers at 25% beyond credit (backtest-validated)
- 14 DTE time exit lets theta work longer while avoiding gamma
- 50% gamma protection is more aggressive near-expiry profit taking
```

**Step 2: Update existing tests for new thresholds**

In `tests/test_validators.py`:

Update `test_profit_target_reached` (line 179):
- Comment: "Test profit target detection at 35% of max"
- Keep the same test: entry_credit=1.00, current_value=0.40 = 60% profit. Still passes at 35% target.

Update `test_profit_target_not_reached` (line 190):
- Comment: "Test profit target not reached"
- Change: entry_credit=1.00, current_value=0.80 = 20% profit. Still fails at 35% target. No change needed.

Update `test_stop_loss_triggered` (line 200):
- Comment: "Test stop loss detection at 125% of credit"
- Change: entry_credit=1.00, current_value=2.50 = 150% loss -> entry_credit=1.00, current_value=2.50 still works (150% > 125%)

Update `test_stop_loss_not_triggered` (line 211):
- Comment: "Test stop loss not triggered"
- Change: entry_credit=1.00, current_value=2.00 = 100% loss. This USED to be below 200%, now it's below 125%. Still passes.
- WAIT -- 100% loss is NOT below 125%. current_value=2.00 means loss = 2.00 - 1.00 = 1.00, which is 100% of credit. 100% < 125%, so still not triggered. Correct.

Update `test_time_exit_triggered` (line 221):
- near_expiry_date fixture returns 15 DTE. Old threshold was 21, new is 14. 15 DTE > 14 DTE, so this test will NOW FAIL.
- Fix: Change the `near_expiry_date` fixture in `tests/conftest.py` (line 257) from `timedelta(days=15)` to `timedelta(days=10)` to be below the new 14 DTE threshold.

Update `test_time_exit_not_triggered` (line 231):
- future_date fixture returns 35 DTE. 35 > 14, so still not triggered. No change needed.

**Step 3: Run tests**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/test_validators.py -v --timeout=30`
Expected: All tests pass with updated thresholds.

**Step 4: Commit**

```bash
git add src/core/risk/validators.py tests/test_validators.py tests/conftest.py
git commit -m "Tighten exit conditions for faster trade turnover

Profit target 50% -> 35%, stop loss 200% -> 125%, time exit 21 -> 14
DTE, gamma protection 70% -> 50%. Positions will close faster,
generating the trade data needed for AI calibration."
```

---

### Task 6: Update Regime Multipliers

**Files:**
- Modify: `src/core/analysis/regime.py:94-98`
- Test: `tests/unit/analysis/test_regime.py` (check if multiplier tests exist)

**Step 1: Update REGIME_MULTIPLIERS in regime.py**

In `src/core/analysis/regime.py`, change two values:

Line 96: `MarketRegime.BULL_HIGH_VOL: 0.5,` -> `MarketRegime.BULL_HIGH_VOL: 0.75,`
Line 98: `MarketRegime.BEAR_HIGH_VOL: 0.25,` -> `MarketRegime.BEAR_HIGH_VOL: 0.40,`

**Step 2: Run regime tests**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/unit/analysis/test_regime.py -v --timeout=30`
Expected: Pass (or update any tests that assert specific multiplier values).

**Step 3: Commit**

```bash
git add src/core/analysis/regime.py
git commit -m "Relax regime multipliers for paper trading

Bull High Vol: 0.5 -> 0.75, Bear High Vol: 0.25 -> 0.40. The 50%
cut in Bull High Vol was too aggressive combined with other limits."
```

---

### Task 7: Add Exit Monitoring Logs to Position Monitor

**Files:**
- Modify: `src/handlers/position_monitor.py:291-333`

**Step 1: Add else-branch with exit status logging**

In `src/handlers/position_monitor.py`, after the `if should_exit:` block (which ends at line 332), add an else branch before the `except` on line 334.

After line 332 (the end of the `else: await discord.send_exit_alert(...)` block), add:

```python
            else:
                # Log position status for debugging (worker logs only, not Discord)
                profit = trade.entry_credit - current_value
                profit_pct = profit / trade.entry_credit if trade.entry_credit > 0 else 0
                print(
                    f"Holding {trade.underlying}: "
                    f"profit={profit_pct:.0%}, DTE={dte}, "
                    f"style={trading_style.value if trading_style else 'N/A'}, "
                    f"entry={trade.entry_credit:.2f}, current={current_value:.2f}"
                )
```

**Step 2: Run position monitor tests**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/unit/handlers/test_position_monitor.py -v --timeout=30`
Expected: All tests pass (new code only adds print statements in an else branch).

**Step 3: Commit**

```bash
git add src/handlers/position_monitor.py
git commit -m "Add exit monitoring logs for open positions

Logs profit %, DTE, trading style, and current value to worker logs
when positions don't trigger an exit. Provides visibility into why
positions are holding."
```

---

### Task 8: Add Position Exit Status to EOD Summary

**Files:**
- Modify: `src/handlers/eod_summary.py:930-938`
- Modify: `src/core/notifications/discord.py:518-637`

**Step 1: Build position status data in eod_summary.py**

In `src/handlers/eod_summary.py`, after line 564 (`open_trades = await db.get_open_trades()`), add code to build position exit status:

```python
    # Build position exit status for daily summary
    from core.risk.validators import days_to_expiry
    position_details = []
    for trade in open_trades:
        # Find matching position for current value
        matching_pos = next((p for p in positions if p.trade_id == trade.id), None)
        current_value = matching_pos.current_value if matching_pos else None
        dte = days_to_expiry(trade.expiration)
        profit_pct = None
        if current_value is not None and trade.entry_credit > 0:
            profit_pct = (trade.entry_credit - current_value) / trade.entry_credit
        position_details.append({
            "underlying": trade.underlying,
            "dte": dte,
            "profit_pct": profit_pct,
            "entry_credit": trade.entry_credit,
            "current_value": current_value,
        })
```

Then update the `send_daily_summary` call (line 931) to pass position_details:

```python
    await discord.send_daily_summary(
        performance=performance,
        open_positions=len(open_trades),
        trade_stats=trade_stats,
        trade_stats_today=trade_stats_today,
        screening_summary=screening_summary,
        market_context=market_context_for_summary,
        position_details=position_details,
    )
```

**Step 2: Update send_daily_summary in discord.py to accept and display position details**

In `src/core/notifications/discord.py`, update the `send_daily_summary` method signature (line 518) to add:

```python
    async def send_daily_summary(
        self,
        performance: DailyPerformance,
        open_positions: int,
        trade_stats: dict,
        trade_stats_today: dict | None = None,
        screening_summary: dict | None = None,
        market_context: dict | None = None,
        position_details: list[dict] | None = None,
    ) -> str:
```

Then after the screening summary section (after line 620), add:

```python
        # Add position exit status if provided
        if position_details:
            pos_lines = []
            for pos in position_details:
                profit_str = f"{pos['profit_pct']:.0%}" if pos['profit_pct'] is not None else "N/A"
                pos_lines.append(f"{pos['underlying']}: {profit_str} profit, {pos['dte']} DTE")
            if pos_lines:
                fields.append({
                    "name": "Open Position Status",
                    "value": "\n".join(pos_lines),
                    "inline": False,
                })
```

**Step 3: Run tests**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/ -v --timeout=30 -x`
Expected: All tests pass. The new `position_details` parameter defaults to `None` so existing callers are unaffected.

**Step 4: Commit**

```bash
git add src/handlers/eod_summary.py src/core/notifications/discord.py
git commit -m "Add open position exit status to daily summary

Shows each position's profit %, DTE, and entry/current value in the
EOD Discord summary. Provides daily visibility into position health."
```

---

### Task 9: Run Full Test Suite

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/ -v --timeout=30`
Expected: All tests pass.

**Step 2: Run dynamic exit tests specifically**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/unit/risk/test_dynamic_exits.py -v --timeout=30`
Expected: All pass. Dynamic exit tests may need threshold updates if they hardcode 50% profit target or 200% stop loss.

---

### Task 10: Deploy to Cloudflare Workers

**Files:**
- None (deployment only)

**Step 1: Deploy**

Run: `cd /Users/jdhiman/Documents/mahler && npx wrangler deploy`
Expected: Successful deployment message with worker URL.

**Step 2: Verify deployment**

Check wrangler logs to confirm the worker is running:
Run: `cd /Users/jdhiman/Documents/mahler && npx wrangler tail --format pretty` (watch for a few seconds)
Expected: Logs showing position monitor runs every 5 minutes with the new "Holding" log lines.
