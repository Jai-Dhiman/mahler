# Phase 2-4: Backtest Validation of All System Layers

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate every layer of the trading system (regime multipliers, portfolio limits, AI agent filtering) against 19 years of ORATS data. Find the optimal full-system configuration.

**Architecture:** Extend the Rust backtest engine with regime-aware sizing and multi-ticker support. Build a separate Python LLM backtesting framework for agent validation. Merge findings into a single optimized configuration.

**Tech Stack:** Rust (backtest engine), Python (LLM framework), OpenRouter API (xai/grok), ORATS data (2007-2025)

---

## Phase 2: Backtest Regime & Portfolio Layers

### Task 1: Align Backtest Regime Model with Production

**Files:**
- Modify: `mahler-backtest/src/regime/classifier.rs`
- Reference: `src/core/analysis/regime.py:93-99` (production regime multipliers)

**Context:** The backtest engine already has a regime classifier, but it uses different categories than production:
- Backtest: BullCalm, BullUncertain, BearMild, BearVolatile, Crisis, Recovery, Unknown
- Production: bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol

These need to be aligned so backtest results are directly comparable to production.

**Step 1: Map production regimes to backtest engine**

In `mahler-backtest/src/regime/classifier.rs`, update the `MarketRegime` enum to match production:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    BullLowVol,     // Was BullCalm
    BullHighVol,    // Was BullUncertain
    BearLowVol,     // Was BearMild
    BearHighVol,    // Was BearVolatile + Crisis
}
```

**Step 2: Update position_size_multiplier() to match production defaults**

```rust
pub fn position_size_multiplier(&self) -> f64 {
    match self {
        MarketRegime::BullLowVol => 1.0,
        MarketRegime::BullHighVol => 0.75,
        MarketRegime::BearLowVol => 0.5,
        MarketRegime::BearHighVol => 0.4,
    }
}
```

**Step 3: Update classification logic**

The production system uses these features: realized_vol_20, momentum_20, trend, iv, iv_rv_spread. Port the same classification logic. Read `src/core/analysis/regime.py:296-325` for the exact feature computation and thresholds.

**Step 4: Add CLI flags for custom multipliers**

In `mahler-backtest/src/main.rs`, add:

```rust
/// Regime multipliers as comma-separated values: bull_low,bull_high,bear_low,bear_high
#[arg(long, default_value = "1.0,0.75,0.5,0.4")]
regime_multipliers: String,
```

Parse into a `Vec<f64>` and pass to the engine config.

**Step 5: Compile and test**

```bash
cd mahler-backtest
cargo build 2>&1 | tail -5
cargo run -- run --ticker QQQ --start 2007-01-01 --end 2025-12-31 --use-regime-sizing --data data/orats
```

**Step 6: Commit**

```bash
git add mahler-backtest/src/
git commit -m "Align backtest regime model with production system

4-state regime model (bull/bear x low/high vol) with matching
multipliers. CLI flags for custom multiplier testing."
```

---

### Task 2: Grid Search Regime Multipliers

**Files:**
- Create: `mahler-backtest/src/bin/regime_grid.rs` (or add a new CLI subcommand)

**Context:** Test whether regime-aware sizing beats flat sizing across the full dataset.

**Step 1: Define the multiplier grid**

Test these configurations:

| Config | BullLow | BullHigh | BearLow | BearHigh | Description |
|--------|---------|----------|---------|----------|-------------|
| flat | 1.0 | 1.0 | 1.0 | 1.0 | No regime (baseline) |
| prod | 1.0 | 0.75 | 0.5 | 0.4 | Current production |
| mild | 1.0 | 0.9 | 0.75 | 0.5 | Less aggressive reduction |
| aggressive | 1.0 | 0.5 | 0.25 | 0.1 | Heavy reduction |
| bear_boost | 1.0 | 0.75 | 1.25 | 1.5 | Size UP in bear (premium selling edge) |
| inverse | 0.5 | 0.75 | 1.0 | 1.25 | Counter-intuitive: bear calls work better in bears |

The "bear_boost" and "inverse" configs are important -- the backtest showed bear call spreads THRIVE in bear markets (2022: +23%, 90% WR). The production system reduces size in bears, which may be backwards for this strategy.

**Step 2: Run the grid on QQQ (2007-2025)**

For each config, run with best base params (delta 0.05-0.15, IV=0, PT=65):

```bash
for multipliers in "1.0,1.0,1.0,1.0" "1.0,0.75,0.5,0.4" "1.0,0.9,0.75,0.5" "1.0,0.5,0.25,0.1" "1.0,0.75,1.25,1.5" "0.5,0.75,1.0,1.25"; do
    echo "Testing: $multipliers"
    cargo run -- run --ticker QQQ --start 2007-01-01 --end 2025-12-31 \
        --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 --profit-target 65 \
        --use-regime-sizing --regime-multipliers "$multipliers" \
        --data data/orats
done
```

**Step 3: Repeat on SPY**

Same grid, same base params. Compare SPY vs QQQ regime sensitivity.

**Step 4: Walk-forward validate top configs**

Take the top 2-3 configs and run walk-forward optimization to check for overfitting:

```bash
cargo run -- optimize --ticker QQQ --start 2007-01-01 --end 2025-12-31 --data data/orats
```

**Step 5: Record results and commit**

Update `mahler-backtest/analysis/walkforward_findings_2026-01-30.log` (or create a new findings file) with regime grid results.

```bash
git add mahler-backtest/
git commit -m "Regime multiplier grid search results

[summarize: which config won, by how much, and whether regime
sizing adds value over flat sizing]"
```

---

### Task 3: Multi-Ticker Simultaneous Backtest

**Files:**
- Modify: `mahler-backtest/src/backtest/engine.rs`
- Modify: `mahler-backtest/src/main.rs`

**Context:** Portfolio limits and correlation constraints only matter when holding positions across multiple underlyings simultaneously. The current engine runs single-ticker. We need multi-ticker with shared portfolio state.

**Step 1: Add multi-ticker support to CLI**

```rust
/// Ticker symbols, comma-separated (e.g., "SPY,QQQ")
#[arg(short, long)]
ticker: String,  // Now accepts "SPY,QQQ,IWM"
```

**Step 2: Modify engine to accept multiple tickers**

Add a method:

```rust
pub fn run_multi(
    &mut self,
    tickers: &[&str],
    start_date: NaiveDate,
    end_date: NaiveDate,
) -> BacktestResult
```

Key difference from single-ticker: on each trading day, iterate through all tickers and screen for opportunities. Use the SAME portfolio state (positions, equity, heat) for sizing across all tickers. This naturally tests correlation and portfolio heat limits.

**Step 3: Handle date alignment**

Different tickers may have different trading days in ORATS data. Load all snapshots, merge by date, and only process days where all tickers have data.

**Step 4: Test baseline**

```bash
cargo run -- run --ticker "SPY,QQQ" --start 2007-01-01 --end 2025-12-31 \
    --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 --profit-target 65 \
    --data data/orats
```

Compare to sum of individual single-ticker runs. Portfolio limits should reduce returns vs. unconstrained.

**Step 5: Commit**

```bash
git add mahler-backtest/src/
git commit -m "Add multi-ticker simultaneous backtesting

Shared portfolio state across tickers enables proper testing of
correlation limits, portfolio heat, and per-underlying constraints."
```

---

### Task 4: Grid Search Portfolio Limits

**Files:**
- Modify: `mahler-backtest/src/main.rs` (add CLI flags)
- Modify: `mahler-backtest/src/risk/position_sizer.rs` (make limits configurable)

**Context:** Find the optimal portfolio constraint set that balances trade frequency against catastrophic correlation risk.

**Step 1: Add CLI flags for all constraints**

```rust
#[arg(long, default_value = "5")]
max_positions: usize,

#[arg(long, default_value = "20.0")]
max_portfolio_heat: f64,

#[arg(long, default_value = "10.0")]
max_per_underlying: f64,
```

**Step 2: Define the grid**

| Config | Max Pos | Portfolio Heat | Per-Underlying | Description |
|--------|---------|---------------|----------------|-------------|
| tight | 3 | 10% | 10% | Current-ish production |
| moderate | 5 | 20% | 15% | Phase 1 paper values |
| loose | 8 | 30% | 20% | Aggressive learning |
| wide_open | 10 | 50% | 30% | Near-unconstrained |
| single_only | 1 | 5% | 5% | One position at a time |

**Step 3: Run multi-ticker grid on SPY+QQQ (2007-2025)**

```bash
for config in "3,10,10" "5,20,15" "8,30,20" "10,50,30" "1,5,5"; do
    IFS=',' read -r pos heat und <<< "$config"
    echo "Testing: max_pos=$pos, heat=$heat%, und=$und%"
    cargo run -- run --ticker "SPY,QQQ" --start 2007-01-01 --end 2025-12-31 \
        --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 --profit-target 65 \
        --use-scaled-sizing --max-positions "$pos" --max-portfolio-heat "$heat" \
        --max-per-underlying "$und" --data data/orats
done
```

**Step 4: Analyze results**

Key metrics to compare:
- CAGR and Sharpe (did constraints help or hurt?)
- Max drawdown (did constraints protect during crashes?)
- Trade count (are constraints starving the system?)
- 2020 COVID and 2022 bear performance (do constraints help in stress?)

**Step 5: Commit**

```bash
git add mahler-backtest/
git commit -m "Portfolio limit grid search results

[summarize: optimal constraint set and whether limits add value]"
```

---

## Phase 3: Backtest the AI Agent Layer

### Task 5: Build LLM Backtesting Framework

**Files:**
- Create: `scripts/llm_backtest/backtest_agents.py`
- Create: `scripts/llm_backtest/market_context.py`
- Create: `scripts/llm_backtest/results.py`

**Context:** For each historical trade the rules-based screener would have taken, feed the same context to a cheap LLM and record its decision. Compare LLM-filtered results to rules-only results.

**Step 1: Extract historical trade decisions from Rust backtest**

Add an output mode to the backtest engine that dumps all trade entry/exit decisions to JSON:

In `mahler-backtest/src/main.rs`, add a `--export-trades` flag:

```rust
#[arg(long)]
export_trades: Option<String>,  // Path to output JSON file
```

Output format:
```json
[
    {
        "date": "2022-03-15",
        "ticker": "QQQ",
        "short_strike": 350.0,
        "long_strike": 345.0,
        "expiration": "2022-04-14",
        "credit": 0.85,
        "max_loss": 4.15,
        "delta": 0.12,
        "dte": 30,
        "iv_percentile": 82.5,
        "vix": 28.3,
        "underlying_price": 340.0,
        "outcome": "profit",
        "pnl": 55.25,
        "exit_reason": "profit_target",
        "hold_days": 12
    }
]
```

**Step 2: Build market context constructor**

Create `scripts/llm_backtest/market_context.py`:

```python
def build_agent_context(trade: dict) -> str:
    """Build the same context the live agents receive for a historical trade."""
    return f"""
Market Context:
- Underlying: {trade['ticker']} at ${trade['underlying_price']:.2f}
- VIX: {trade['vix']:.1f}
- IV Percentile: {trade['iv_percentile']:.1f}%

Trade Details:
- Strategy: Bear Call Spread
- Short Strike: ${trade['short_strike']:.2f} (delta: {trade['delta']:.3f})
- Long Strike: ${trade['long_strike']:.2f}
- Expiration: {trade['expiration']} ({trade['dte']} DTE)
- Credit: ${trade['credit']:.2f}
- Max Loss: ${trade['max_loss']:.2f}
- Risk/Reward: {trade['max_loss']/trade['credit']:.1f}:1
"""
```

**Step 3: Build LLM evaluation harness**

Create `scripts/llm_backtest/backtest_agents.py`:

```python
import json
import httpx
import asyncio
from pathlib import Path

OPENROUTER_API_KEY = "..."  # From env
MODEL = "x-ai/grok-2"  # Cheap, fast

FUND_MANAGER_PROMPT = """You are a fund manager evaluating a credit spread trade.
Given the market context, decide whether to approve or reject this trade.
Return JSON: {"decision": "approve"|"reject", "confidence": 0.0-1.0, "contracts": 1-5, "reason": "..."}
"""

async def evaluate_trade(trade: dict) -> dict:
    context = build_agent_context(trade)
    response = await call_llm(FUND_MANAGER_PROMPT, context)
    return {
        "trade_date": trade["date"],
        "ticker": trade["ticker"],
        "llm_decision": response["decision"],
        "llm_confidence": response["confidence"],
        "llm_contracts": response["contracts"],
        "actual_outcome": trade["outcome"],
        "actual_pnl": trade["pnl"],
    }

async def run_backtest(trades_path: str, output_path: str):
    trades = json.loads(Path(trades_path).read_text())
    results = []
    for trade in trades:
        result = await evaluate_trade(trade)
        results.append(result)
    Path(output_path).write_text(json.dumps(results, indent=2))
```

**Step 4: Commit**

```bash
git add scripts/llm_backtest/ mahler-backtest/src/
git commit -m "Add LLM backtesting framework

Export historical trades from Rust engine and evaluate with cheap
LLM (xai/grok) to measure whether AI filtering adds alpha."
```

---

### Task 6: Run LLM Backtest and Analyze Results

**Step 1: Export trades from best rules-based config**

```bash
cd mahler-backtest
cargo run -- run --ticker QQQ --start 2020-01-01 --end 2025-12-31 \
    --min-delta 0.05 --max-delta 0.15 --min-iv-percentile 0 --profit-target 65 \
    --export-trades ../scripts/llm_backtest/data/qqq_trades_2020_2025.json \
    --data data/orats
```

**Step 2: Run LLM evaluation**

```bash
cd /Users/jdhiman/Documents/mahler
uv run scripts/llm_backtest/backtest_agents.py \
    --trades scripts/llm_backtest/data/qqq_trades_2020_2025.json \
    --output scripts/llm_backtest/data/qqq_llm_results.json
```

**Step 3: Analyze marginal value**

Create `scripts/llm_backtest/analyze.py`:

```python
def analyze_results(results_path: str):
    results = json.loads(Path(results_path).read_text())

    # Split into LLM-approved and LLM-rejected
    approved = [r for r in results if r["llm_decision"] == "approve"]
    rejected = [r for r in results if r["llm_decision"] == "reject"]

    # Compare outcomes
    approved_wr = sum(1 for r in approved if r["actual_pnl"] > 0) / len(approved)
    rejected_wr = sum(1 for r in rejected if r["actual_pnl"] > 0) / len(rejected)

    approved_avg_pnl = sum(r["actual_pnl"] for r in approved) / len(approved)
    rejected_avg_pnl = sum(r["actual_pnl"] for r in rejected) / len(rejected)

    print(f"Total trades: {len(results)}")
    print(f"LLM approved: {len(approved)} ({len(approved)/len(results):.0%})")
    print(f"LLM rejected: {len(rejected)} ({len(rejected)/len(results):.0%})")
    print(f"Approved win rate: {approved_wr:.1%}")
    print(f"Rejected win rate: {rejected_wr:.1%}")
    print(f"Approved avg P/L: ${approved_avg_pnl:.2f}")
    print(f"Rejected avg P/L: ${rejected_avg_pnl:.2f}")
    print()
    if rejected_wr < approved_wr:
        print("LLM ADDS VALUE: rejected trades have lower win rate")
    else:
        print("LLM ADDS NO VALUE: rejected trades perform same or better")
```

**Step 4: Test confidence threshold calibration**

```python
def calibrate_thresholds(results_path: str):
    results = json.loads(Path(results_path).read_text())

    for threshold in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        filtered = [r for r in results if r["llm_confidence"] >= threshold]
        if not filtered:
            continue
        wr = sum(1 for r in filtered if r["actual_pnl"] > 0) / len(filtered)
        avg_pnl = sum(r["actual_pnl"] for r in filtered) / len(filtered)
        total_pnl = sum(r["actual_pnl"] for r in filtered)
        print(f"Threshold {threshold:.0%}: {len(filtered)} trades, {wr:.1%} WR, ${avg_pnl:.2f} avg, ${total_pnl:.2f} total")
```

**Step 5: Test different prompts**

Run the same trades through 3 prompt variants:
- `aggressive_prompt`: "Approve unless there's a clear structural flaw"
- `balanced_prompt`: Current fund manager prompt
- `conservative_prompt`: "Reject unless the setup is exceptional"

Compare which prompt style best predicts trade outcomes.

**Step 6: Commit results**

```bash
git add scripts/llm_backtest/
git commit -m "LLM backtest results and analysis

[summarize: does AI add alpha, at what threshold, with what prompt]"
```

---

## Phase 4: Deploy Validated Full System

### Task 7: Merge Optimal Configuration

**Files:**
- Modify: `src/core/analysis/regime.py` (validated multipliers)
- Modify: `src/core/risk/position_sizer.py` (validated limits)
- Modify: `src/handlers/morning_scan.py` (validated AI threshold)
- Modify: `src/core/inference/exit_inference.py` (validated exits)

**Step 1: Compile findings from Phases 2-3**

Create a summary table:

| Layer | Validated Config | Evidence |
|-------|-----------------|----------|
| Regime multipliers | [from Task 2] | [CAGR, Sharpe, walk-forward] |
| Portfolio limits | [from Task 4] | [CAGR, Sharpe, max DD] |
| AI threshold | [from Task 6] | [marginal alpha, optimal cutoff] |
| Exit params | PT=65%, SL=125% | Already validated (Phase 1) |

**Step 2: Update production code with validated values**

Replace the Phase 1 paper-trading overrides with permanent validated values. Remove the `is_paper` conditionals and use the validated config for all environments.

**Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

**Step 4: Deploy**

```bash
npx wrangler deploy
```

**Step 5: Commit**

```bash
git add src/
git commit -m "Deploy backtest-validated full system configuration

All layers (regime, portfolio limits, AI threshold, exits) validated
against 19 years of data. Configuration: [summarize final values]"
```

---

### Task 8: Monitor and Validate Paper Trading

**Duration:** 1-2 months post-deployment

**Step 1: Daily checks (first week)**

After each morning scan, verify:
- Trade count: should average 1-3 per day (not 0)
- Position sizes: reasonable given constraints
- Exit triggers: at ~65% profit and ~125% stop
- No `position_size_zero` blocking all opportunities

**Step 2: Weekly comparison**

Compare paper results to backtest expectations:
- Win rate: backtest says ~70%, paper should be 55-80%
- Avg trade P/L: backtest says ~$65, paper should be in range
- Slippage: measure actual fill quality vs ORATS 66% assumption

**Step 3: Monthly assessment**

After 1 month:
- Is the equity curve tracking within backtest confidence intervals?
- Are there systematic deviations that suggest regime-specific issues?
- Does the AI layer (if retained) show consistent filtering quality?

**Step 4: Graduate to live**

If paper results track backtest for 1-2 months:
- Switch to live with 50% of validated sizing
- Monitor for 2 weeks
- Scale to 100% if performance holds
