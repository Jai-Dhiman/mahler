# TraderJoe Paper→Live Measurement Infrastructure & Go-Live Gate

**Goal:** Capture the data needed over 6 months of paper trading to decide whether the credit-spread strategy is safe to run with real money, and document the explicit criteria for that decision.

**Not in scope:**
- Changing strategy logic (delta targets, DTE, profit/stop thresholds) — those are walk-forward validated and frozen.
- Automating the go-live transition — the gate is a manual review document, not an executable check.
- Migrating from VIXY to spot VIX as the circuit-breaker input — circuit breaker keeps VIXY; spot VIX is logged separately for regime analysis.
- Backfilling historical NBBO or equity data — measurement starts when this ships.

## Problem

Today the live worker stores only `fill_price` (Alpaca's `filled_avg_price`), no NBBO, no equity time series, only `short_delta`/`short_theta` at entry, and proxies VIX via VIXY. Consequences:

1. **Paper fills cannot be trusted as a live predictor.** Alpaca's own docs state paper orders are not routed to exchanges, order quantity is not checked against NBBO displayed size (fake fills are possible), and slippage/latency/liquidity are not modeled. Without capturing NBBO at placement and fill size vs. displayed size, we cannot rule out a track record built on fills that would not exist in live trading.
2. **No Sharpe / Sortino / drawdown computable** — no daily equity curve is persisted. `traderjoe/src/handlers/eod_summary.rs` reads `account.equity` for the day's P&L % but doesn't store it.
3. **No rolling portfolio Greeks.** Only per-trade `short_delta` and `short_theta` at entry are stored. Gamma, vega, and time series of portfolio aggregates are absent, so concentration drift, gamma spikes near expiry, and vega correlation are invisible.
4. **No regime tagging.** The 7-regime classifier exists only in `traderjoe-backtest`. Live trades are untagged, so "did we sample enough regime variety?" is unanswerable at month 6.
5. **No gate criteria.** There is no written document specifying what must be true to turn `ENVIRONMENT` from `paper` to `live`. Short-vol literature (LTCM, XIV Feb 2018) shows the left tail commonly arrives in year 2–4; a 6-month window cannot statistically prove profitability. The gate must therefore be operational (plumbing correctness) rather than statistical, with P&L reported but not used as the primary go/no-go.

## Solution (from the user's perspective)

After this ships:

- Every new trade records NBBO for both legs at entry and at exit, entry Greeks including gamma and vega, and the NBBO displayed size at order placement.
- The worker records an equity snapshot at EOD, at every trade open/close, and whenever the circuit breaker fires.
- At EOD the worker records portfolio-wide Greeks (beta-weighted delta vs SPY, total gamma, total vega, total theta, concentration by underlying) and market context (spot VIX from FRED, VIXY close, SPY 20-day realized vol, SPY 20-day return, SPY drawdown from 52-week high).
- Every Sunday 22:00 ET a weekly metrics job posts a Discord embed with Sharpe, Sortino, profit factor, win rate, P&L skew, max drawdown, live-vs-NBBO slippage stats, paper-fill size violations, regime bucket trade counts, and portfolio Greek ranges — each metric annotated with a sample-size confidence tag (`INSUFFICIENT` / `WEAK` / `OK`).
- An on-demand HTTP endpoint (`POST /trigger/metrics` with window query string) returns the same bundle for ad-hoc queries.
- A new markdown checklist at `traderjoe/docs/GO_LIVE_GATE.md` defines exactly what must be true to flip to live, including a capital-fraction ramp (10% → 25% → 50% → 100%) gated by trade count and VIX-regime observation.
- A `LIVE_CAPITAL_FRACTION` env var (default `0.10`) is wired into the position sizer. When running live, positions are sized against `equity × LIVE_CAPITAL_FRACTION` rather than full equity, enabling the ramp.

## Design

### Framework

Framework C — plumbing correctness is a hard pass/fail gate; P&L is informational with a small set of catastrophic halts. Rationale: 6 months yields ~50–100 trades at the system's entry cadence, which is below the Kelly-confidence threshold (thousands of trades). Statistical P&L criteria cannot honestly reject a bad strategy at that sample size. What *can* be proven in 6 months is whether the data captured is trustworthy and whether execution matches model assumptions.

### Data capture layer

All new state lives in D1. Schema is additive: new columns on `trades`, four new tables. No existing columns or tables are modified; existing writes remain correct if the new columns are omitted (SQLite permits `INSERT` without listing the new columns).

**Additions to `trades` table (15 columns):**

- NBBO at entry: `entry_short_bid`, `entry_short_ask`, `entry_long_bid`, `entry_long_ask`, `entry_net_mid`
- NBBO at exit: `exit_short_bid`, `exit_short_ask`, `exit_long_bid`, `exit_long_ask`, `exit_net_mid`
- Entry Greeks expanded: `entry_short_gamma`, `entry_short_vega`, `entry_long_delta`, `entry_long_gamma`, `entry_long_vega`
- Paper-fill sanity: `nbbo_displayed_size_short`, `nbbo_displayed_size_long`, `nbbo_snapshot_time`

(Total: 18 new nullable columns. All nullable so existing open trades remain valid.)

**New `equity_history` table:** event-driven rows with `{ id PK, timestamp, event_type (eod|trade_open|trade_close|circuit_breaker), equity, cash, open_position_mtm, realized_pnl_day, unrealized_pnl_day, open_position_count, trade_id_ref nullable }`.

**New `portfolio_greeks_eod` table:** one row per EOD with `{ date PK, beta_weighted_delta, total_gamma, total_vega, total_theta, delta_by_underlying (JSON), max_gamma_single_position, max_vega_single_position, open_position_count }`.

**New `market_context_daily` table:** one row per date with `{ date PK, spot_vix, spot_vix_source (fred|stooq|unavailable), spot_vix_source_date, vixy_close, spy_20d_realized_vol, spy_20d_return, spy_drawdown_from_52w_high }`. `spot_vix_source_date` captures FRED's T-1 lag explicitly.

**New `metrics_weekly` table:** one row per weekly run with `{ id PK, generated_at, window_start, window_end, trade_count, sharpe, sortino, profit_factor, win_rate, pnl_skew, max_drawdown_pct, mean_slippage_vs_mid, max_slippage_vs_mid, slippage_vs_orats_ratio, fill_size_violation_count, fill_size_violation_pct, regime_buckets (JSON), greek_ranges (JSON), sample_size_tag }`.

### NBBO capture flow

At order placement in `morning_scan.rs`, after the screener picks spreads but before `place_spread_order`, call `snapshot_spread_nbbo(&chain, short_strike, long_strike, option_type)` which returns an `NbboSnapshot` with the per-leg bid, ask, and size observed in the current chain. The snapshot is passed to `create_trade` and persisted.

At order exit in `position_monitor.rs`, after `get_spread_debit` resolves the current short and long contracts, build the same `NbboSnapshot` from those contracts and pass it to `close_trade`.

Alpaca's option chain response (`v1beta1/options/snapshots`) returns `latestQuote.bs` (bid size) and `latestQuote.as` (ask size) alongside `bp`/`ap`. These are already fetched — we just weren't reading them.

**Timestamp alignment:** `nbbo_snapshot_time` stores the wall-clock time when the chain was fetched. Alpaca's `filled_at` on the order detail gives the actual fill time. The spread between these two is acceptable (seconds) and the paper-fill sanity check compares `contracts` (filled quantity) against the displayed size at snapshot — the time delta is documented, not corrected for.

### Equity snapshotter

`EquitySnapshotter::snapshot(event_type, context)` in `src/measurement/equity.rs` is the single entry point. It fetches `account.equity` from Alpaca (already available via `AlpacaClient::get_account`), computes `open_position_mtm` by summing current net debits across open trades (using chain data already fetched by the calling handler where possible), and writes one row to `equity_history`.

Called from:
- `eod_summary::run` — once per day, `event_type = eod`.
- `morning_scan::run` — after successful trade open, `event_type = trade_open`, `trade_id_ref = <new trade id>`.
- `position_monitor::run` — after successful trade close, `event_type = trade_close`, `trade_id_ref = <closed trade id>`.
- `circuit_breaker` fire site in `morning_scan::run` — when risk level escalates to Halted, `event_type = circuit_breaker`.

The dedupe rule: for `event_type = eod`, the writer upserts on `(date(timestamp), event_type)` so a retried EOD run replaces rather than duplicates. Non-EOD events append.

### Portfolio Greeks aggregator

`PortfolioGreeksAggregator::compute(open_trades, per_underlying_chains) -> PortfolioGreeks` in `src/measurement/portfolio_greeks.rs`. For each open trade, looks up the short and long contract in the relevant chain, reads current delta/gamma/vega/theta, and sums. Beta-weighted delta multiplies each position's delta contribution by a static beta:

```
BETA_TABLE: SPY=1.00, QQQ=1.15, IWM=1.25
```

(Quarterly human review; noted in `GO_LIVE_GATE.md`.)

Called once per EOD from `eod_summary::run` after the existing IV snapshot block (which already fetches all three chains). Writes one row to `portfolio_greeks_eod`.

### Market context logger

`market_context::fetch_spot_vix() -> SpotVixReading` in `src/measurement/market_context.rs` tries FRED (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS`, parses the last non-empty row) and falls back to Stooq (`https://stooq.com/q/d/l/?s=^vix&i=d`) if FRED fails. Returns `{ value, source, source_date }`. No retries; a single fail on both sources stores `spot_vix = NULL, source = unavailable`.

`compute_spy_stats(bars_20d, bars_52w) -> SpyStats` computes 20-day realized vol (stdev of log returns × √252), 20-day return ((last - first)/first), and drawdown from 52-week high ((peak - last)/peak). `get_bars` already exists on `AlpacaClient`.

Called from `eod_summary::run`, writes one row to `market_context_daily`.

### Metrics aggregator

`compute_metrics(window: DateRange, data: MetricsInput) -> MetricBundle` in `src/measurement/metrics.rs` is pure — takes closed trades, equity history, portfolio Greeks history, and market context as inputs, returns a struct. No I/O.

Metrics computed:
- **Sharpe / Sortino** — from daily equity returns over the window. Daily returns are computed by pairing consecutive `equity_history` rows with `event_type = eod`.
- **Profit factor** — sum of positive `net_pnl` / abs(sum of negative `net_pnl`). Returns `f64::INFINITY` if no losers (flagged by sample-size tag).
- **Win rate** — already implemented in `eod_summary::calculate_win_rate`; reuse.
- **P&L skew** — third moment of `net_pnl` distribution.
- **Max drawdown %** — running peak on EOD equity, max (peak - current)/peak over window.
- **Mean / max slippage vs NBBO mid** — per closed trade, `(fill_price - entry_net_mid)` for entry and `(exit_net_mid - exit_price)` for exit (signs: credit received vs paid, debit paid vs mid). Reported as dollars per contract and as % of `entry_net_mid`.
- **Slippage ratio vs ORATS** — mean slippage divided by backtest's assumed slippage (derived from `traderjoe-backtest` ORATS model: ~34% of bid-ask spread for 2-leg; encoded as a constant `ORATS_ASSUMED_SLIPPAGE_PCT_OF_SPREAD = 0.34`).
- **Fill-size violation count / %** — count of trades where `contracts > min(nbbo_displayed_size_short, nbbo_displayed_size_long)`, and that count as a % of total trades.
- **Regime buckets** — classify each trade's `created_at` against the `market_context_daily` row for that date: `low_vol` (spot_vix<15), `med_vol` (15-25), `high_vol` (>25). Report trade count and aggregate P&L per bucket.
- **Greek ranges** — min/max/avg of `beta_weighted_delta`, `total_gamma`, `total_vega`, `total_theta` over window from `portfolio_greeks_eod`.

Each metric is wrapped as `TaggedMetric { value, tag }` where `tag` is one of `INSUFFICIENT` (n<30), `WEAK` (30≤n<100), `OK` (n≥100). `n` is the relevant sample count — trade count for trade-based metrics, day count for equity-based metrics.

### Weekly handler

New handler `src/handlers/weekly_metrics.rs`. Cron pattern `"0 2 * * MON"` in wrangler.toml (Monday 02:00 UTC ≈ Sunday 22:00 ET EDT / 21:00 ET EST). Behavior:

1. Compute window = trailing 30 days and full paper history.
2. Load closed trades, equity history, portfolio Greeks history, market context history.
3. Call `compute_metrics` twice (30d + full).
4. Insert rows into `metrics_weekly`.
5. Post a Discord embed (new builder in `notifications/discord.rs`).

HTTP trigger: `POST /trigger/metrics?window=30d` (or `full`) in `lib.rs`, authed via existing `check_auth`. Returns the `MetricBundle` JSON.

### Capital fraction

`src/risk/position_sizer.rs` gains an `effective_equity(raw_equity, live_capital_fraction) -> f64` helper. `calculate` and `calculate_for_underlying` accept the effective equity from the caller. Callers in `morning_scan.rs` read `LIVE_CAPITAL_FRACTION` from env (default `0.10`, parsed as `f64`), compute effective equity, pass it in.

Deliberately applied to both paper and live so paper runs reflect intended live capital from day one — avoids a "paper shows great P&L but live has 10% the sizing" surprise.

### Go-live gate doc

`traderjoe/docs/GO_LIVE_GATE.md` is a manual review checklist. Structure:

1. **Hard pass/fail items** — each with a SQL query the reviewer runs against D1 to verify.
2. **Catastrophic halt thresholds** — same format.
3. **Capital ramp schedule** — 10% / 25% / 50% / 100%, each with the criteria needed to advance.
4. **Beta table review cadence** — quarterly check with the method for recomputing.
5. **VIX-regime observation definition** — "VIX >25 observed for ≥3 trading days" logged in `market_context_daily`.

## Modules

### Deep modules

**`measurement::nbbo`** — `pub fn snapshot_spread_nbbo(chain: &OptionsChain, short_strike: f64, long_strike: f64, option_type: OptionType) -> Option<NbboSnapshot>`
- Interface: single function, one return type.
- Hides: strike-matching floating-point comparison, per-leg bid/ask/size extraction, option type filtering, snapshot timestamp generation.
- Tested through: public function returning `NbboSnapshot`; verify fields match input chain.

**`measurement::equity::EquitySnapshotter`** — `pub async fn snapshot(&self, event_type: EquityEvent, ctx: SnapshotContext) -> Result<()>`
- Interface: one method, one event enum, one context struct.
- Hides: account fetch wiring, MTM computation from open trades, D1 row construction, EOD upsert logic, event-type string serialization.
- Tested through: observing a written row via `D1Client` query after a `.snapshot()` call.

**`measurement::portfolio_greeks::PortfolioGreeksAggregator`** — `pub fn compute(open_trades: &[Trade], chains: &HashMap<String, OptionsChain>) -> PortfolioGreeks`
- Interface: one pure function, one return struct.
- Hides: per-leg lookup by strike/expiration/type, beta weighting via static table, underlying concentration aggregation, max-per-position scan.
- Tested through: pass a curated trades + chains fixture, assert on `PortfolioGreeks` fields.

**`measurement::market_context`** — `pub async fn fetch_spot_vix(http: &HttpClient) -> SpotVixReading` and `pub fn compute_spy_stats(bars_20d: &[Bar], bars_52w: &[Bar]) -> SpyStats`
- Interface: two functions.
- Hides: FRED CSV parsing, Stooq fallback routing, log-return stdev computation, drawdown-from-peak computation.
- Tested through: feed canned HTTP responses via a minimal `HttpClient` trait; assert parsed values. SPY stats tested with synthetic bars.

**`measurement::metrics::compute_metrics`** — `pub fn compute_metrics(window: DateRange, input: MetricsInput) -> MetricBundle`
- Interface: one pure function.
- Hides: Sharpe/Sortino/profit factor/skew/drawdown/slippage math, regime bucketing, sample-size tagging, ORATS-ratio constant application.
- Tested through: feed fixture trade set + equity curve, assert on `MetricBundle` values and tags.

### Shallow helpers (kept small, not separate modules)

- `sample_size_tag(n: usize) -> SampleSizeTag` — trivial threshold lookup, lives in `metrics.rs`.
- `paper_fill_violation(contracts: i64, short_size: i64, long_size: i64) -> bool` — one comparison, lives in `metrics.rs`.
- `effective_equity(raw_equity: f64, fraction: f64) -> f64` — one multiplication, lives in `position_sizer.rs`.

## File Changes

| File | Change | Type |
|------|--------|------|
| `traderjoe/src/migrations/0002_measurement_infrastructure.sql` | Add columns to `trades`, create `equity_history`, `portfolio_greeks_eod`, `market_context_daily`, `metrics_weekly`. | New |
| `traderjoe/src/measurement/mod.rs` | `pub mod nbbo; pub mod equity; pub mod portfolio_greeks; pub mod market_context; pub mod metrics;` | New |
| `traderjoe/src/measurement/nbbo.rs` | `NbboSnapshot` struct + `snapshot_spread_nbbo` function. | New |
| `traderjoe/src/measurement/equity.rs` | `EquityEvent` enum, `SnapshotContext` struct, `EquitySnapshotter`. | New |
| `traderjoe/src/measurement/portfolio_greeks.rs` | `PortfolioGreeks` struct, `PortfolioGreeksAggregator::compute`, `BETA_TABLE` const. | New |
| `traderjoe/src/measurement/market_context.rs` | `SpotVixReading`, `SpyStats`, `fetch_spot_vix`, `compute_spy_stats`, minimal `HttpClient` trait. | New |
| `traderjoe/src/measurement/metrics.rs` | `MetricBundle`, `TaggedMetric`, `SampleSizeTag`, `sample_size_tag`, `compute_metrics`, all per-metric pure helpers (Sharpe, Sortino, profit factor, skew, drawdown, slippage, regime bucketing), `paper_fill_violation`, `ORATS_ASSUMED_SLIPPAGE_PCT_OF_SPREAD`. | New |
| `traderjoe/src/handlers/weekly_metrics.rs` | Weekly cron body + HTTP endpoint body. | New |
| `traderjoe/src/handlers/mod.rs` | Add `pub mod weekly_metrics;` | Modify |
| `traderjoe/src/lib.rs` | Register weekly cron dispatch, add `/trigger/metrics` route, `mod measurement;`, read `LIVE_CAPITAL_FRACTION` env. | Modify |
| `traderjoe/wrangler.toml` | Add `"0 2 * * MON"` to crons. | Modify |
| `traderjoe/src/db/d1.rs` | Extend `create_trade` + `close_trade` signatures with NBBO + expanded Greeks params; add `insert_equity_snapshot`, `insert_portfolio_greeks_eod`, `insert_market_context_daily`, `insert_metrics_weekly`; add query helpers `get_closed_trades_in_window`, `get_equity_history_in_window`, `get_portfolio_greeks_history_in_window`, `get_market_context_in_window`; update `TradeRow::from_d1_row` to read new columns. | Modify |
| `traderjoe/src/types.rs` | Extend `Trade` with new NBBO + Greeks Option fields. | Modify |
| `traderjoe/src/broker/alpaca.rs` | Extract bid/ask size from chain snapshots in `get_options_chain`. | Modify |
| `traderjoe/src/broker/types.rs` | Extend `OptionContract` with `bid_size`, `ask_size` Option fields. | Modify |
| `traderjoe/src/handlers/morning_scan.rs` | NBBO snapshot before order, pass to `create_trade`; equity snapshot after trade open; read `LIVE_CAPITAL_FRACTION`, pass effective equity to sizer. | Modify |
| `traderjoe/src/handlers/position_monitor.rs` | NBBO snapshot before close order, pass to `close_trade`; equity snapshot after close. | Modify |
| `traderjoe/src/handlers/eod_summary.rs` | EOD equity snapshot; portfolio Greeks compute + write; market context fetch + write. | Modify |
| `traderjoe/src/risk/position_sizer.rs` | Add `effective_equity` helper; keep existing signatures (callers compute effective equity). | Modify |
| `traderjoe/src/notifications/discord.rs` | Add `build_metrics_embed` + `send_metrics_report`. | Modify |
| `traderjoe/docs/GO_LIVE_GATE.md` | Go-live checklist, SQL queries, capital ramp schedule, beta table review cadence, VIX-regime definition. | New |

## Open Questions

- Q: `traderjoe/CLAUDE.md` states migrations are at `0017`, but `src/migrations/` contains only `0001_initial_schema.sql`. Is migration history tracked elsewhere (applied directly via `wrangler d1 execute`)?  **Default:** Name the new migration `0002_measurement_infrastructure.sql`. If `0017` is real, rename before running.
- Q: FRED VIXCLS fetch from a Cloudflare Worker — any egress constraints?  **Default:** Assume outbound `fetch` is unrestricted (already used for Alpaca and Discord). If CORS/egress blocks FRED, switch to Stooq as primary.
- Q: `LIVE_CAPITAL_FRACTION` default of `0.10` — applied to paper too?  **Default:** Yes. Reason: paper track record should reflect live sizing from day one, not require a second-order mental haircut when reading metrics.
- Q: Beta table quarterly review — automated check or human reminder?  **Default:** Human reminder documented in `GO_LIVE_GATE.md`. No automation.
