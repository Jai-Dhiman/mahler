# Mahler System Optimization Design

**Date:** 2026-02-25
**Status:** Approved
**Goal:** Transform Mahler from a barely-trading paper system into a validated, actively-trading options system with every layer backtested.

## Problem Statement

The system has a $100k paper trading account earning $4 in 16 days. Root causes:

1. **Regime multiplier (0.5x)** cuts every position in half -- never backtested
2. **Portfolio limits** zero out opportunities after 2-3 positions -- `position_size_zero` kills 7 of 8 trading days
3. **AI agents return 30% confidence** on every trade -- acts as unvalidated veto layer
4. **Dynamic exits (50-63% stop)** are tighter than backtest-validated 125% stop loss
5. **greeks_analyst fails** on most scans, degrading analysis quality

Meanwhile, the backtest (19 years, 2007-2025) validated a clear winning configuration:

| Parameter | Backtest Winner | Result |
|-----------|----------------|--------|
| Delta | 0.05-0.15 | 6.25% CAGR, 1.59 Sharpe |
| IV Filter | None (0%) | +59% CAGR vs 50% filter |
| PT/SL | 65% / 125% | 69.9% win rate, PF 6.10 |
| Sizing | 1 contract | Max DD 4.35% |

The live system's screener matches the backtest, but three unvalidated layers prevent trading.

## Design

### Phase 1: Unblock Live Trading (Immediate)

Get the paper system generating data while we backtest the advanced layers.

#### 1a. Disable regime multiplier for paper trading
- Set `position_multiplier = 1.0` when `ENVIRONMENT=paper`
- Regime detection still runs and logs -- just doesn't throttle
- File: `src/core/analysis/regime.py` (multiplier application)
- File: `src/handlers/morning_scan.py` (combined_multiplier logic)

#### 1b. Loosen portfolio constraints for paper trading
- Max equity positions: 3 -> 5
- Per-underlying limit: 10% -> 20%
- Portfolio heat stays at 20%
- File: `src/core/risk/position_sizer.py`

#### 1c. Make AI agents advisory, not gatekeepers
- Remove LOW confidence skip logic
- If rules-based screener approves, trade executes
- Agents still run, log, and store analysis for learning
- Fund manager sets contract count but cannot veto to zero
- File: `src/handlers/morning_scan.py` (confidence threshold logic)

#### 1d. Align exit parameters with backtest
- Profit target: 65% (was dynamic ~80%)
- Stop loss: 125% (was dynamic 50-63%)
- File: `src/core/risk/exit_manager.py` or equivalent exit logic

#### 1e. Fix greeks_analyst errors
- Investigate and fix greeks_analyst failures
- File: `src/core/agents/` (greeks analyst module)

### Phase 2: Backtest Regime & Portfolio Layers

Extend the Rust backtest engine to validate these layers against 19 years of data.

#### 2a. Add regime detection to backtest engine
- Port Python regime detection to Rust
- Classify each historical day: bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol
- Features: realized_vol_20, momentum_20, trend, iv, iv_rv_spread
- CLI flags: `--use-regime-sizing`, `--regime-multipliers "1.0,0.75,0.5,0.4"`

#### 2b. Grid search regime multiplier configurations
- Test: all-1.0 vs graduated vs aggressive reduction
- Walk-forward validate to avoid overfitting
- Key question: does regime-aware sizing beat flat sizing?

#### 2c. Grid search portfolio limit configurations
- Parameters: max_positions (1-8), portfolio_heat (10-50%), per_underlying (10-30%)
- Phase 1 backtest showed scaled sizing killed trade count (1626 -> 15)
- Find sweet spot: protection without starvation

#### 2d. Multi-ticker simultaneous backtest
- Current engine runs single ticker; need SPY+QQQ with shared portfolio state
- Required to properly test correlation limits

**Deliverable:** Walk-forward validated parameters for regime multipliers and portfolio limits.

### Phase 3: Backtest the AI Agent Layer

Answer: does LLM filtering add alpha over rules-based trading?

#### 3a. LLM backtesting framework
- For each historical rules-based trade, construct agent context (IV, technicals, VIX, macro)
- Feed to cheap LLM (xai/grok on OpenRouter)
- Record: approve/reject, confidence, suggested contracts
- Compare outcomes vs. rules-only

#### 3b. Measure marginal value
- Sample period: 2020-2025 (COVID crash, 2022 bear, 2023-24 bull, 2025 vol)
- Metric: does LLM filtering improve Sharpe/win rate/profit factor?
- If rejected trades had same win rate as accepted, LLM adds no value

#### 3c. Calibrate confidence thresholds
- Test cutoffs: 0%, 20%, 30%, 40%, 50%
- Find threshold where filtering improves risk-adjusted returns
- Valid outcome: optimal threshold is 0% (take everything)

#### 3d. Test prompt engineering
- Aggressive vs conservative vs balanced fund manager prompts
- Single-agent vs multi-agent debate format
- Cost estimate: ~2000 decisions at $0.001-0.01/call = $2-20 total

**Deliverable:** Evidence-based decision on AI layer value and optimal configuration.

### Phase 4: Deploy Validated Full System

#### 4a. Integrate backtest-optimized parameters
- Replace hardcoded values with validated config
- Set AI threshold to validated cutoff (or remove if no alpha)

#### 4b. Paper trade 1-2 months
- Monitor: results within backtest confidence intervals?
- Track realized slippage vs ORATS 66% assumption
- Compare AI decisions vs rules-only

#### 4c. Build feedback loop
- Wire episodic_memory and trade_trajectories for prediction-vs-outcome tracking
- Weekly calibration reports

#### 4d. Graduate to live
- After 1-2 months paper validation
- Start at 50% of validated sizing, scale up

## Execution Order

Phase 1 is immediate (live system code changes).
Phases 2 and 3 run in parallel (independent workstreams).
Phase 4 depends on both Phase 2 and Phase 3.

## Key Backtest Data Reference

From `mahler-backtest/analysis/walkforward_findings_2026-01-30.log`:

Best configs (QQQ, 2007-2025):
- Wealth building: Delta 0.05-0.15, IV=0, PT=65 -> 6.25% CAGR, 1.59 Sharpe, 4.35% MaxDD
- Risk-adjusted: Delta 0.10-0.20, IV=0, DTE=21-35 -> 5.84% CAGR, 1.85 Sharpe, 2.91% MaxDD

Stress tests (aggressive config):
- 2008 crisis: 0% (no trades, survived)
- 2020 COVID: +0.61% (survived)
- 2022 bear: +22.97%, 89.7% WR (thrived)

Slippage sensitivity: strategy breaks even at ~90% slippage, viable at 75%.
