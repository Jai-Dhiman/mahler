# Phase 1: Unblock Live Trading - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get the paper trading system actively trading with backtest-validated parameters by removing unvalidated throttling layers.

**Architecture:** Modify 4 files in the live system -- bypass regime multiplier for paper trading, loosen portfolio constraints, make AI agents advisory-only, and align exits with backtest-validated PT=65%/SL=125%.

**Tech Stack:** Python (Cloudflare Workers), D1 SQLite, Alpaca API

---

### Task 1: Bypass Regime Multiplier for Paper Trading

**Files:**
- Modify: `src/handlers/morning_scan.py:584-588`

**Context:** The regime multiplier (0.5x for bull_high_vol) was never backtested. It halves every position. For paper trading, set it to 1.0 so the regime still logs but doesn't throttle.

**Step 1: Write the change**

At line 584-588 in `src/handlers/morning_scan.py`, the current code is:

```python
# Combined size multiplier (risk + regime)
# Use the more conservative (lower) of the two multipliers
combined_size_multiplier = min(risk_state.size_multiplier, regime_multiplier)
if combined_size_multiplier < 1.0:
    print(f"Size multiplier: {combined_size_multiplier:.2f} (risk: {risk_state.size_multiplier:.2f}, regime: {regime_multiplier:.2f})")
```

Change to:

```python
# Combined size multiplier (risk + regime)
# Paper trading: bypass regime multiplier (not backtested) while still logging it
# Regime detection still runs and stores data for future validation
is_paper = env.get("ENVIRONMENT", "paper") == "paper"
if is_paper:
    combined_size_multiplier = risk_state.size_multiplier
    if regime_multiplier < 1.0:
        print(f"Size multiplier: {combined_size_multiplier:.2f} (risk: {risk_state.size_multiplier:.2f}, regime: {regime_multiplier:.2f} [bypassed - paper trading])")
else:
    combined_size_multiplier = min(risk_state.size_multiplier, regime_multiplier)
    if combined_size_multiplier < 1.0:
        print(f"Size multiplier: {combined_size_multiplier:.2f} (risk: {risk_state.size_multiplier:.2f}, regime: {regime_multiplier:.2f})")
```

Note: `env` is already available in this scope from the handler's environment parameter.

**Step 2: Verify the env variable is accessible**

Check that `env` is available at line 584. It should be -- it's a parameter of the handler function. Search for its definition in the function signature.

**Step 3: Commit**

```bash
git add src/handlers/morning_scan.py
git commit -m "Bypass regime multiplier for paper trading

Regime detection still runs and logs for future backtest validation,
but no longer throttles position sizing. The 0.5x multiplier was
never backtested and was killing all trading activity."
```

---

### Task 2: Loosen Portfolio Constraints for Paper Trading

**Files:**
- Modify: `src/core/risk/position_sizer.py:29-62`

**Context:** The current limits (max 3 equity positions, 10% per-underlying, 20% portfolio heat) cause `position_size_zero` on 7 of 8 trading days after 2-3 positions are open. For paper trading where learning is the goal, loosen these.

**Step 1: Write the change**

In the `RiskLimits` dataclass at lines 29-62, change these specific values:

```python
# Before
max_per_underlying_pct: float = 0.10  # ~10% max in any single underlying
max_positions_per_equity_class: int = 3

# After
max_per_underlying_pct: float = 0.20  # 20% max in any single underlying (paper: allow more concentration for learning)
max_positions_per_equity_class: int = 5  # Paper: 5 equity positions for data collection
```

Leave all other constraints unchanged (2% per-trade, 5% single position, 20% portfolio heat, 50% equity class).

**Step 2: Run existing tests**

Run: `cd /Users/jdhiman/Documents/mahler && python -m pytest tests/ -v -k "position_sizer or risk" --no-header 2>&1 | head -60`

Note any failures. If tests hardcode the old values (3 positions, 10%), update them to match.

**Step 3: Commit**

```bash
git add src/core/risk/position_sizer.py
git commit -m "Loosen portfolio constraints for paper trading

Increase max equity positions 3->5 and per-underlying 10%->20%.
Fixes position_size_zero blocking all trades after 2-3 open positions."
```

---

### Task 3: Make AI Agents Advisory (Remove Low-Confidence Veto)

**Files:**
- Modify: `src/handlers/morning_scan.py:927-952`

**Context:** The fund manager returns 30% confidence, classified as LOW. The current logic skips LOW confidence trades unless the fund manager explicitly approves. But the fund manager's own prompt says "the cost of rejecting every trade far exceeds the cost of a small paper loss." The system contradicts itself. Fix: if the rules-based screener passed the trade and it survived position sizing, execute it. Agents still run and log for learning.

**Step 1: Write the change**

At lines 927-952 in `src/handlers/morning_scan.py`, the current code is:

```python
            # Skip low confidence trades unless fund manager explicitly approved
            fm_approved = (
                v2_result.fund_manager_message
                and v2_result.fund_manager_message.structured_data
                and v2_result.fund_manager_message.structured_data.get("final_contracts", 0) > 0
            )
            if analysis.confidence == Confidence.LOW and not fm_approved:
                print(f"Skipping low confidence trade: {spread.underlying}")
                skip_reasons["low_confidence"] = skip_reasons.get("low_confidence", 0) + 1
                # Notify about skipped trade
                await discord.send_trade_decision(
                    ...
                )
                continue
            elif analysis.confidence == Confidence.LOW and fm_approved:
                print(f"Fund manager override: accepting low-confidence trade for {spread.underlying}")
```

Change to:

```python
            # Paper trading: AI agents are advisory, not gatekeepers
            # Log confidence level but don't veto -- the rules-based screener
            # already validated this trade. Agent data is stored for future
            # backtest validation of whether AI filtering adds alpha.
            is_paper = env.get("ENVIRONMENT", "paper") == "paper"
            fm_approved = (
                v2_result.fund_manager_message
                and v2_result.fund_manager_message.structured_data
                and v2_result.fund_manager_message.structured_data.get("final_contracts", 0) > 0
            )
            if analysis.confidence == Confidence.LOW:
                if is_paper:
                    print(f"Low confidence ({v2_result.confidence:.0%}) for {spread.underlying} - proceeding (paper trading, agents advisory)")
                elif not fm_approved:
                    print(f"Skipping low confidence trade: {spread.underlying}")
                    skip_reasons["low_confidence"] = skip_reasons.get("low_confidence", 0) + 1
                    await discord.send_trade_decision(
                        underlying=spread.underlying,
                        spread_type=spread.spread_type.value,
                        short_strike=spread.short_strike,
                        long_strike=spread.long_strike,
                        expiration=spread.expiration,
                        credit=spread.credit,
                        decision="skipped",
                        reason="Low confidence from AI analysis",
                        ai_summary=v2_result.thesis,
                        confidence=v2_result.confidence,
                        iv_rank=iv_metrics.iv_rank,
                    )
                    continue
                else:
                    print(f"Fund manager override: accepting low-confidence trade for {spread.underlying}")
```

**Step 2: Commit**

```bash
git add src/handlers/morning_scan.py
git commit -m "Make AI agents advisory for paper trading

Low confidence trades no longer vetoed in paper mode. Agent analysis
still runs and logs for future validation. In live mode, the veto
logic is preserved."
```

---

### Task 4: Align Exit Parameters with Backtest (PT=65%, SL=125%)

**Files:**
- Modify: `src/core/inference/exit_inference.py:29-34`
- Investigate: `src/core/risk/dynamic_exit.py` (the TradingGroup formula)
- Investigate: `src/handlers/morning_scan.py` or position monitor handler (where exits are triggered)

**Context:** The backtest validated PT=65%, SL=125%. The live system has two exit systems:
1. `PrecomputedExitProvider` with PT=50%, SL=200% (in exit_inference.py)
2. `DynamicExit` using TradingGroup paper formula with sigma_d_10 volatility scaling (in dynamic_exit.py)

The dynamic exit is what's actually being used (Discord shows "dynamic_profit" and "dynamic_stop_loss" as exit reasons). The dynamic system exits at 50-80% profit and 50-63% stop depending on volatility. The backtest says 65% profit and 125% stop are optimal.

**Step 1: Investigate how exits are actually triggered**

Read the position monitor handler to find where `dynamic_exit` or `exit_inference` is called. Look for:
- Which exit system is active (dynamic vs precomputed)
- Where the style ("conservative") is set
- How to override with fixed backtest-validated thresholds

Run: Search for `dynamic_profit` and `dynamic_stop_loss` in `src/handlers/` to find the exit trigger code.

**Step 2: Update default exit parameters**

In `src/core/inference/exit_inference.py:29-34`, change:

```python
# Before
DEFAULT_PARAMS = ExitParams(
    profit_target=0.50,  # Exit at 50% of max profit
    stop_loss=0.20,  # Stop at 200% of credit received
    time_exit_dte=21,  # Exit at 21 DTE
)

# After
DEFAULT_PARAMS = ExitParams(
    profit_target=0.65,  # Exit at 65% of max profit (backtest validated: 6.25% CAGR)
    stop_loss=1.25,  # Stop at 125% of credit received (backtest validated)
    time_exit_dte=21,  # Exit at 21 DTE
)
```

**Step 3: Investigate dynamic exit override**

The dynamic exit system (TradingGroup formula) may be overriding these defaults. If so, you need to either:
- Disable dynamic exits for paper trading and use fixed thresholds
- Or adjust the dynamic exit style multipliers to approximate PT=65%/SL=125%

Read `src/handlers/position_monitor.py` (or wherever the 5-minute position check runs) to understand which exit path is active.

**Step 4: Test the exit change**

Verify by checking what exit parameters would be calculated for a typical position (e.g., $0.50 credit on QQQ bear call):
- PT=65%: exit when spread value drops to $0.175 (keep 65% of $0.50)
- SL=125%: exit when spread value rises to $1.125 ($0.50 * 2.25)

**Step 5: Commit**

```bash
git add src/core/inference/exit_inference.py
git commit -m "Align exit parameters with backtest: PT=65%, SL=125%

Backtest validated these values across 19 years (2007-2025) with
6.25% CAGR, 69.9% win rate, 4.35% max drawdown. Previous dynamic
exits were stopping out at 50-63% which is too tight."
```

---

### Task 5: Fix greeks_analyst Errors

**Files:**
- Investigate: `src/core/agents/` (find the greeks analyst module)

**Context:** Discord logs show "WARNING: 1 analyst(s) failed due to errors (greeks_analyst)" on most scans, and on Feb 18 ALL 4 analysts failed. Even though agents are now advisory, clean data matters for future validation.

**Step 1: Find the greeks_analyst code**

Search: `grep -r "greeks_analyst" src/` to find the module and any error handling.

**Step 2: Reproduce the error**

Check wrangler logs or add logging to identify the specific exception. Common causes:
- Missing data (options chain not loaded)
- Calculation errors (division by zero in Greeks)
- API timeout

**Step 3: Fix the root cause**

This depends on what Step 2 reveals. Apply explicit exception handling per the user's preference (no fallback mechanisms).

**Step 4: Commit**

```bash
git add src/core/agents/
git commit -m "Fix greeks_analyst errors in agent pipeline

[describe the specific fix based on investigation]"
```

---

### Task 6: Deploy and Verify

**Step 1: Run full test suite**

```bash
cd /Users/jdhiman/Documents/mahler
python -m pytest tests/ -v --no-header 2>&1 | tail -20
```

Fix any failures.

**Step 2: Deploy to Cloudflare**

```bash
npx wrangler deploy
```

**Step 3: Verify next morning scan**

After the next 10:00 AM ET scan, check Discord for:
- Regime multiplier logged as "bypassed - paper trading"
- Trades not blocked by `position_size_zero` (should have capacity for 5 equity positions)
- Low confidence trades proceeding with "agents advisory" log
- New trades being placed

**Step 4: Monitor exit behavior**

Over the next 1-2 days, verify:
- Profit targets trigger at ~65% (not 80%)
- Stop losses trigger at ~125% (not 50-63%)

**Step 5: Commit any deployment fixes**

```bash
git add -A
git commit -m "Deployment fixes for Phase 1 unblock trading"
```
