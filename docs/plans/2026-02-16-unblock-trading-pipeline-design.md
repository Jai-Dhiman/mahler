# Unblock Trading Pipeline -- Design

## Problem

The system is deadlocked. Two open equity positions trigger the equity correlation limit (max 1), blocking all new equity trades. TLT and GLD have insufficient premiums. Positions never exit because profit targets are too strict (50%) and stop losses too loose (200%). No logging exists for non-exit monitoring checks, so the system appears dead.

## Root Causes

1. `max_positions_per_equity_class = 1` blocks all new equity trades when any equity position is open
2. Profit target at 50% with dynamic vol adjustment can push effective target to 60%+
3. Stop loss at 200% allows positions to bleed to near-max loss before exiting
4. Time exit at DTE 21 means positions sit for 2-3 weeks before forced close
5. No logging when exit conditions are checked but not met -- zero visibility
6. Bull High Vol regime multiplier (0.5) halves all position sizes on top of other constraints

## Design

### 1. Position Sizing & Risk Limits

File: `src/core/risk/position_sizer.py`

| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| `max_positions_per_equity_class` | 1 | 3 | Allow SPY, QQQ, IWM positions simultaneously |
| `max_portfolio_heat_pct` | 0.10 | 0.20 | Room for 4-5 concurrent trades |
| `max_per_underlying_pct` | 0.033 | 0.10 | Allow multiple positions per underlying |

### 2. Exit Conditions

File: `src/core/risk/validators.py`

| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| `profit_target_pct` | 0.50 | 0.35 | Faster exits, more turnover |
| `stop_loss_pct` | 2.00 | 1.25 | Cut losers faster (matches backtest findings) |
| `time_exit_dte` | 21 | 14 | Let theta work longer, still exit before gamma |
| `gamma_protection_pnl` | 0.70 | 0.50 | More aggressive profit-taking near expiry |

### 3. Regime Multipliers

File: `src/core/analysis/regime.py`

| Regime | Current | New |
|--------|---------|-----|
| BULL_HIGH_VOL | 0.5 | 0.75 |
| BEAR_HIGH_VOL | 0.25 | 0.40 |

### 4. Exit Monitoring Logs

File: `src/handlers/position_monitor.py`

Add else-branch after `if should_exit:` that logs:
- Current profit %, target %, DTE, trading style
- Printed to worker logs (not Discord) every 5 min

File: `src/handlers/eod_summary.py`

Add per-position exit status to daily Discord summary:
- Each open position's profit %, DTE, nearest exit trigger

### 5. Backtesting Validation

Run before deploying code changes.

Sweep 1 -- Equity correlation limit:
- Max concurrent equity positions: 1, 2, 3
- Ticker: QQQ
- Metrics: CAGR, Sharpe, max drawdown, trade count

Sweep 2 -- Exit thresholds:
- Profit target: 25%, 35%, 50%, 65%
- Stop loss: 100%, 125%, 150%, 200%
- 16 combinations
- Metrics: Win rate, profit factor, CAGR

Sweep 3 -- Combined validation:
- Best params from sweeps 1 and 2
- Multi-ticker: SPY, QQQ, IWM
- Include stress periods (2020, 2022)

## Execution Order

1. Run backtests (sweeps 1-3)
2. Review results, adjust params if data disagrees
3. Apply code changes
4. Deploy to Cloudflare Workers
5. Monitor next morning scan
