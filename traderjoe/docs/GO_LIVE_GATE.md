# TraderJoe Paper→Live Go-Live Gate

This is a manual review checklist. Run it to decide whether to flip `ENVIRONMENT`
from `paper` to `live`, and whether to advance the capital-fraction ramp.

All SQL queries below are run against the D1 database (`wrangler d1 execute mahler-db --command "..."`).

## Hard Pass/Fail

Every item must pass. A single failure blocks advancement.

### 1. NBBO captured for 100% of trades over last 30 days
```sql
SELECT COUNT(*) AS missing FROM trades
 WHERE created_at >= date('now','-30 days')
   AND (entry_short_bid IS NULL OR entry_short_ask IS NULL
        OR entry_long_bid IS NULL OR entry_long_ask IS NULL);
```
PASS if `missing = 0`.

### 2. No paper-fill size violations over last 30 days
```sql
SELECT COUNT(*) AS violations FROM trades
 WHERE created_at >= date('now','-30 days')
   AND nbbo_displayed_size_short IS NOT NULL
   AND contracts > CASE
       WHEN nbbo_displayed_size_short < nbbo_displayed_size_long
       THEN nbbo_displayed_size_short
       ELSE nbbo_displayed_size_long
   END;
```
PASS if `violations = 0`.

### 3. Equity curve has no gaps in last 30 trading days
```sql
SELECT COUNT(DISTINCT date(timestamp)) AS days_recorded
  FROM equity_history
 WHERE event_type = 'eod' AND timestamp >= date('now','-30 days');
```
PASS if `days_recorded >= 20` (accounting for weekends/holidays).

### 4. Weekly metrics job ran on schedule, last 4 weeks
```sql
SELECT COUNT(*) AS runs FROM metrics_weekly
 WHERE generated_at >= date('now','-28 days');
```
PASS if `runs >= 4`.

### 5. Circuit breaker has been fire-tested
Manual: at least one record in `equity_history` with `event_type = 'circuit_breaker'`, OR a dated log of a synthetic trigger test with before/after KV state captured.

### 6. Portfolio Greeks snapshot current
```sql
SELECT date FROM portfolio_greeks_eod ORDER BY date DESC LIMIT 1;
```
PASS if the most recent date is within the last 2 trading days.

### 7. Regime tags present on 100% of trades last 30 days
```sql
SELECT COUNT(*) AS missing FROM trades t
 WHERE created_at >= date('now','-30 days')
   AND NOT EXISTS (SELECT 1 FROM market_context_daily m WHERE m.date = date(t.created_at));
```
PASS if `missing = 0`.

### 8. Slippage within 2x ORATS model
```sql
SELECT slippage_vs_orats_ratio FROM metrics_weekly ORDER BY generated_at DESC LIMIT 1;
```
PASS if `slippage_vs_orats_ratio <= 2.0`.

### 9. Minimum 60 closed trades (for 10% capital)
```sql
SELECT COUNT(*) AS closed FROM trades WHERE status = 'closed';
```
PASS if `closed >= 60` for 10% capital; `>= 100` for 25%; `>= 200` for 50%+.

### 10. At least 30 trading days elapsed under all infrastructure green
Manual review of past 4 `metrics_weekly` reports; no infra-gap flags.

## Catastrophic Halts

Any single trigger halts go-live advancement.

### C1. Realized max DD exceeds theoretical per-trade max loss × 1.5
```sql
SELECT max_drawdown_pct FROM metrics_weekly ORDER BY generated_at DESC LIMIT 1;
```
Compare vs theoretical per-trade max loss percent of equity (approx 2% under default `PositionSizer`). Halt if `max_drawdown_pct > 3.0%`.

### C2. Profit factor < 1.0 over full window
```sql
SELECT profit_factor FROM metrics_weekly WHERE window_end = (SELECT MAX(window_end) FROM metrics_weekly);
```
Halt if `profit_factor < 1.0` on the full-window row.

### C3. Any single trade lost > 2× configured per-trade max loss
```sql
SELECT id, net_pnl, max_loss, contracts FROM trades
 WHERE status = 'closed' AND net_pnl < -(max_loss * contracts * 2);
```
Halt if any row returned.

## Capital Ramp

| Stage | Capital | Requirements |
|-------|---------|--------------|
| 1 | 10% | Hard items 1–8, 10 pass. ≥60 closed trades. No catastrophic halts. |
| 2 | 25% | Stage 1 + ≥100 closed trades + VIX>25 observed (≥3 trading days in `market_context_daily` with `spot_vix > 25`) on paper or live. |
| 3 | 50% | Stage 2 + ≥200 closed trades + VIX>25 survived on **live** specifically (≥3 live trading days with `spot_vix > 25` and no catastrophic halts triggered). |
| 4 | 100% | Stage 3 + 3+ months at 50% capital with live P&L tracking within ±25% of paper expectations. |

Advancing requires flipping `LIVE_CAPITAL_FRACTION` env var and redeploying.

## Beta Table Review

The static `BETA_TABLE` in `traderjoe/src/measurement/portfolio_greeks.rs` encodes:
- SPY = 1.00
- QQQ = 1.15
- IWM = 1.25

Review quarterly (1st of Jan/Apr/Jul/Oct). Recompute by regressing 252-day daily returns of each underlying against SPY. Update the constant and commit.

## VIX-Regime Observation

A "VIX >25 event" = at least 3 trading days in `market_context_daily` where `spot_vix > 25`, within a 10-trading-day window. Verify via:
```sql
SELECT date, spot_vix FROM market_context_daily
 WHERE date >= date('now','-30 days') AND spot_vix > 25
 ORDER BY date DESC;
```
