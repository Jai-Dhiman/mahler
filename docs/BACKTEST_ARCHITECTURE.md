# Mahler Backtester: Rust Architecture

## Overview

A high-performance options backtesting engine written in Rust, designed for rapid iteration on credit spread strategies. The backtester produces validated parameters that feed into the Mahler V2 multi-agent trading system.

**Version**: 0.1 (Design Phase)
**Language**: Rust
**Last Updated**: 2026-01-28

---

## Table of Contents

1. [Design Goals](#1-design-goals)
2. [System Overview](#2-system-overview)
3. [Layer 1: Data Layer](#3-layer-1-data-layer)
4. [Layer 2: Pricing Layer](#4-layer-2-pricing-layer)
5. [Layer 3: Strategy Layer](#5-layer-3-strategy-layer)
6. [Layer 4: Simulation Layer](#6-layer-4-simulation-layer)
7. [Layer 5: Walk-Forward Layer](#7-layer-5-walk-forward-layer)
8. [Layer 6: Output Layer](#8-layer-6-output-layer)
9. [Data Flow](#9-data-flow)
10. [Crate Structure](#10-crate-structure)
11. [Dependencies](#11-dependencies)
12. [Integration with Mahler V2](#12-integration-with-mahler-v2)

---

## 1. Design Goals

### Primary Goals

| Goal | Rationale |
|------|-----------|
| **Speed** | Enable rapid iteration: backtest -> tweak -> backtest cycles in seconds, not minutes |
| **Accuracy** | Match ORATS methodology exactly (slippage, fills, timing) for realistic results |
| **Reproducibility** | Same inputs always produce same outputs; all randomness is seeded |
| **Rich Output** | Capture everything needed for analysis and future ML training |

### Non-Goals (for v0.1)

- Real-time trading (that's Mahler V2's job)
- Multi-asset strategies beyond equity ETF options
- Complex multi-leg strategies (iron condors, butterflies) - credit spreads only initially

### Why Rust?

1. **Performance**: Options backtesting is CPU-intensive (Greeks calculations, millions of data points)
2. **Parallelism**: Walk-forward optimization is embarrassingly parallel; Rust + rayon makes this trivial
3. **Correctness**: Type system catches errors at compile time; no runtime surprises
4. **Clean separation**: Backtester has no LLM dependencies; pure numerical computation

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKTESTER ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ORATS Data (Parquet/CSV)                                                  │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │ Data Layer  │────▶│ Pricing     │────▶│ Strategy    │                  │
│   │             │     │ Layer       │     │ Layer       │                  │
│   └─────────────┘     └─────────────┘     └─────────────┘                  │
│                                                  │                          │
│                                                  ▼                          │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│   │ Output      │◀────│ Walk-Forward│◀────│ Simulation  │                  │
│   │ Layer       │     │ Layer       │     │ Layer       │                  │
│   └─────────────┘     └─────────────┘     └─────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Input | Output | Responsibility |
|-------|-------|--------|----------------|
| Data | ORATS files | OptionsSnapshot | Parse, index, serve historical data |
| Pricing | Option parameters | Greeks, IV | Calculate/validate option metrics |
| Strategy | Snapshot + Config | Candidates | Find and score trade opportunities |
| Simulation | Candidates | Trades | Execute trades, manage positions |
| Walk-Forward | Date ranges | Optimal params | Train/validate/test optimization |
| Output | All results | Files | Metrics, logs, exports |

---

## 3. Layer 1: Data Layer

### Purpose

Load, parse, and index historical options data from ORATS format.

### Input Sources

| Source | Contents | Format |
|--------|----------|--------|
| ORATS Daily Snapshots | All options for all strikes/expirations | CSV or Parquet |
| Underlying Price History | OHLCV bars | CSV |
| VIX History | Daily VIX values | CSV |

### Core Data Structures

#### OptionQuote

Single option contract at a point in time:

- Symbol, underlying, strike, expiration
- Option type (call/put)
- Bid, ask, mid, last
- Volume, open interest
- Greeks (delta, gamma, theta, vega, rho)
- Implied volatility

#### OptionsChain

All strikes for one expiration:

- Expiration date
- Days to expiration (DTE)
- Collection of OptionQuote (calls and puts)

#### OptionsSnapshot

All chains for one underlying on one day:

- Date
- Underlying symbol
- Underlying price
- Collection of OptionsChain

#### UnderlyingBar

Daily OHLCV data:

- Date, open, high, low, close, volume

### Data Access Pattern

The data layer provides a date-indexed lookup:

```
get_snapshot(date, underlying) -> OptionsSnapshot
get_underlying_bars(underlying, start, end) -> Vec<UnderlyingBar>
get_vix(date) -> f64
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage format | Parquet | Columnar, compressed, fast reads |
| Loading strategy | Eager (all in RAM) | Dataset fits in memory; simplifies access |
| Date index | HashMap | O(1) lookup by date |
| Missing data | Skip day | Don't interpolate; holidays are holidays |

### ORATS Data Notes

- ORATS snapshots are taken 14 minutes before market close
- This timing provides stable quotes without end-of-day noise
- Greeks are pre-calculated by ORATS; we can use them directly

---

## 4. Layer 2: Pricing Layer

### Purpose

Calculate option prices and Greeks using Black-Scholes model. Primarily used for:

- Validating ORATS-provided Greeks
- Calculating Greeks mid-trade when ORATS data unavailable
- "What-if" scenarios during optimization

### Core Functions

| Function | Inputs | Output |
|----------|--------|--------|
| black_scholes_price | S, K, T, r, sigma, type | Option price |
| implied_volatility | price, S, K, T, r, type | Implied vol (sigma) |
| delta | S, K, T, r, sigma, type | Delta |
| gamma | S, K, T, r, sigma, type | Gamma |
| theta | S, K, T, r, sigma, type | Theta (per day) |
| vega | S, K, T, r, sigma, type | Vega (per 1% vol) |
| rho | S, K, T, r, sigma, type | Rho (per 1% rate) |

### Spread-Level Greeks

For credit spreads, net Greeks are calculated:

- Spread delta = short_delta - long_delta
- Spread gamma = short_gamma - long_gamma
- Spread theta = short_theta - long_theta (positive for credit spreads)
- Spread vega = short_vega - long_vega (typically negative for credit spreads)

### Algorithm Choices

| Calculation | Algorithm | Rationale |
|-------------|-----------|-----------|
| Implied Volatility | Jaeckel's "Let's Be Rational" | Machine precision in 2 iterations; same as vollib |
| Normal CDF | Abramowitz & Stegun approximation | Fast, accurate to 7 decimal places |

### Input Conventions

All inputs use consistent units:

- Time (T): Years (30 days = 30/365)
- Volatility (sigma): Decimal (20% = 0.20)
- Rate (r): Decimal (5% = 0.05)
- Prices: Dollars

---

## 5. Layer 3: Strategy Layer

### Purpose

Define entry criteria and find trade candidates matching the strategy configuration.

### Strategy Configuration

The strategy is defined declaratively via configuration:

```
[strategy]
underlying = ["SPY", "QQQ", "IWM"]
strategy_type = "bull_put_spread"  # or "bear_call_spread"

[entry]
dte_min = 30
dte_max = 45
short_delta_min = 0.10
short_delta_max = 0.15
spread_width_min = 2.0    # dollars
spread_width_max = 10.0
min_credit_pct = 0.25     # credit as % of width
iv_percentile_min = 50
min_open_interest = 100
min_volume = 10
max_bid_ask_pct = 0.08    # max spread as % of mid

[exit]
profit_target_pct = 0.50   # 50% of credit
stop_loss_pct = 1.25       # 125% of credit (loss)
dte_exit = 21              # close at 21 DTE

[position_sizing]
max_risk_per_trade_pct = 0.02    # 2% of account
max_portfolio_heat_pct = 0.10   # 10% total
max_positions = 5
max_correlated_exposure_pct = 0.50  # 50% in correlated underlyings
```

### Scanner Functions

The scanner finds candidates that match all entry criteria:

1. **Filter by DTE**: Find chains with expiration in target range
2. **Filter by delta**: Find short strikes with delta in range
3. **Construct spreads**: Pair short with long strike at target width
4. **Filter by credit**: Reject if credit < min_credit_pct of width
5. **Filter by liquidity**: Reject if OI or volume too low
6. **Filter by bid-ask**: Reject if spread too wide

### Candidate Scoring

Each valid candidate receives a score based on:

- Expected value: (credit *P(OTM)) - (max_loss* P(ITM))
- IV percentile: Higher IV = more premium
- Delta distance: Further OTM = higher probability of profit
- Credit/width ratio: Better premium capture

### Core Structures

#### CreditSpread

- Short leg (OptionQuote)
- Long leg (OptionQuote)
- Spread width
- Net credit
- Max loss
- Net Greeks
- Expiration, DTE

#### SpreadCandidate

- CreditSpread
- Score
- Entry rationale

---

## 6. Layer 4: Simulation Layer

### Purpose

Execute trades with realistic fills, track positions, manage exits.

### Sub-Components

#### 6.4.1 Execution Engine

Applies slippage and commission to simulate realistic fills.

**ORATS Slippage Model** (from ORATS methodology):

| Legs | Slippage % | Example |
|------|------------|---------|
| 1 | 75% | Single call/put |
| 2 | 66% | Vertical spread |
| 3 | 56% | Butterfly |
| 4 | 53% | Iron condor |

**Fill Price Formulas**:

- To buy: `bid + (ask - bid) * slippage_pct`
- To sell: `ask - (ask - bid) * slippage_pct`

For credit spreads (selling), we receive less than mid-price.

**Commission Model** (Alpaca-based):

- Entry: $1.00 per contract per leg
- Exit: $1.00 per contract per leg (if closed)
- Expiration OTM: $0 (no exit cost)

#### 6.4.2 Position Manager

Tracks open positions and portfolio state.

**Position Record**:

- Position ID
- Spread details (strikes, expiration)
- Entry date, entry credit
- Contracts
- Current value (mark-to-market)
- Current P&L
- Greeks at entry

**Portfolio State**:

- Open positions
- Total portfolio delta, gamma, theta, vega
- Portfolio heat (sum of max loss as % of equity)
- Daily P&L, cumulative P&L

#### 6.4.3 Exit Manager

Checks exit conditions for each open position daily.

**Exit Conditions** (checked in order):

| Condition | Trigger | Action |
|-----------|---------|--------|
| Profit Target | Current value <= entry_credit * (1 - profit_target_pct) | Close for profit |
| Stop Loss | Current value >= entry_credit * (1 + stop_loss_pct) | Close for loss |
| Time Exit | DTE <= dte_exit | Close regardless of P&L |
| Expiration | DTE = 0 | Let expire (no action needed if OTM) |

#### 6.4.4 Portfolio Constraints

Before entering a new position, verify:

| Constraint | Check |
|------------|-------|
| Position count | open_positions.len() < max_positions |
| Per-trade risk | spread.max_loss <= account_equity * max_risk_per_trade |
| Portfolio heat | current_heat + new_risk <= max_portfolio_heat |
| Correlation | correlated_exposure + new_risk <= max_correlated |

### Daily Simulation Loop

```
for each trading day:
    1. Update position values from today's snapshot
    2. Check exit conditions for all open positions
    3. Execute any exit orders (apply slippage, commission)
    4. Calculate current portfolio state
    5. If portfolio has room:
        a. Scan for entry candidates
        b. Score and rank candidates
        c. Select best candidate (if any)
        d. Execute entry (apply slippage, commission)
    6. Log daily state
```

---

## 7. Layer 5: Walk-Forward Layer

### Purpose

Optimize strategy parameters while avoiding overfitting through train/validate/test methodology.

### Window Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| train_months | 6 | In-sample optimization period |
| validate_months | 1 | Parameter selection period |
| test_months | 1 | Out-of-sample evaluation |
| roll_forward | 1 month | How far to advance each period |

### Walk-Forward Timeline

```
Period 1:
├── Train:    Jan 2020 - Jun 2020 (6 months)
├── Validate: Jul 2020 (1 month)
└── Test:     Aug 2020 (1 month)

Period 2:
├── Train:    Feb 2020 - Jul 2020 (6 months)
├── Validate: Aug 2020 (1 month)
└── Test:     Sep 2020 (1 month)

... continues rolling forward ...

Period N:
├── Train:    Jun 2024 - Nov 2024 (6 months)
├── Validate: Dec 2024 (1 month)
└── Test:     Jan 2025 (1 month)
```

### Optimization Process

#### Phase 1: Training

Grid search over parameter space:

| Parameter | Search Values |
|-----------|---------------|
| short_delta | [0.08, 0.10, 0.12, 0.15, 0.18] |
| profit_target | [0.40, 0.50, 0.60, 0.75] |
| stop_loss | [1.0, 1.25, 1.5, 2.0] |
| dte_range | [(25,35), (30,45), (35,50)] |

For each parameter combination:

1. Run full simulation on training data
2. Calculate Sharpe ratio
3. Store results

#### Phase 2: Validation

1. Take top 10 parameter sets from training (by Sharpe)
2. Run each on validation data
3. Select parameter set with best validation Sharpe
4. This prevents overfitting to training quirks

#### Phase 3: Testing

1. Run selected parameters on test data
2. Record out-of-sample performance
3. This is the unbiased estimate of expected performance

### Parallelization Strategy

The optimization is embarrassingly parallel:

- Each parameter combination is independent
- Use rayon for data parallelism across combinations
- Can also parallelize across underlyings if needed

### Output Metrics

For each walk-forward period:

- Optimal parameters selected
- Training Sharpe, validation Sharpe, test Sharpe
- Number of trades in each phase

Aggregate across all periods:

- Average out-of-sample Sharpe
- Parameter stability (do optimal params drift?)
- Alpha decay (does performance degrade over time?)

### Alpha Decay Analysis

Track out-of-sample Sharpe over time:

- Fit linear regression to test Sharpe vs. period number
- Negative slope indicates alpha decay
- Calculate half-life: time until expected Sharpe halves

This informs reoptimization frequency.

---

## 8. Layer 6: Output Layer

### Purpose

Calculate metrics, export results for analysis and future ML training.

### Trade Log Schema

Every trade is logged with full context:

| Field | Type | Description |
|-------|------|-------------|
| trade_id | string | Unique identifier |
| underlying | string | SPY, QQQ, IWM |
| spread_type | string | bull_put, bear_call |
| entry_date | date | Trade entry date |
| exit_date | date | Trade exit date |
| days_held | int | Duration in days |
| short_strike | float | Short leg strike |
| long_strike | float | Long leg strike |
| expiration | date | Option expiration |
| entry_credit | float | Credit received |
| exit_debit | float | Debit paid to close |
| commission | float | Total commission |
| pnl | float | Net P&L (dollars) |
| pnl_pct | float | P&L as % of max risk |
| exit_reason | string | profit_target, stop_loss, time_exit, expiration |
| delta_at_entry | float | Spread delta |
| theta_at_entry | float | Spread theta |
| iv_rank_at_entry | float | IV percentile |
| vix_at_entry | float | VIX level |
| underlying_price_entry | float | Stock price at entry |
| underlying_price_exit | float | Stock price at exit |
| contracts | int | Number of spreads |

### Aggregate Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Win Rate | wins / total_trades | >= 70% |
| Profit Factor | gross_profit / gross_loss | >= 1.5 |
| Sharpe Ratio | mean(returns) / std(returns) * sqrt(12) | >= 1.0 |
| Sortino Ratio | mean(returns) / downside_std * sqrt(12) | >= 1.5 |
| Max Drawdown | max peak-to-trough decline | <= 15% |
| CAGR | annualized compound return | >= 15% |
| Avg Days Held | mean(days_held) | 15-30 |
| Avg Win | mean(pnl where pnl > 0) | - |
| Avg Loss | mean(pnl where pnl < 0) | - |

### Performance by Regime

Break down metrics by market regime:

- Low VIX (< 15)
- Normal VIX (15-25)
- Elevated VIX (25-35)
- High VIX (> 35)

### Export Formats

| File | Format | Purpose |
|------|--------|---------|
| trades.parquet | Parquet | Full trade log; ML training data |
| metrics.json | JSON | Summary metrics; quick review |
| equity_curve.csv | CSV | Daily equity for plotting |
| params.toml | TOML | Best parameters found |
| walk_forward.json | JSON | Per-period optimization results |

### Why Parquet?

- Columnar format: efficient for analytical queries
- Compressed: smaller file sizes
- Native Python support: pandas, polars read directly
- Schema preservation: types are maintained
- Partitioning: can split by year/underlying if needed

---

## 9. Data Flow

### Complete Pipeline

```
                         ┌─────────────────┐
                         │  config.toml    │
                         │  (parameters)   │
                         └────────┬────────┘
                                  │
                                  ▼
┌─────────────┐           ┌─────────────────┐
│ ORATS Data  │──────────▶│   DATA LAYER    │
│ (Parquet)   │           │                 │
└─────────────┘           └────────┬────────┘
                                   │
                          OptionsSnapshot(date)
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ PRICING LAYER   │◀──── Greeks calculations
                          │                 │       (on demand)
                          └────────┬────────┘
                                   │
                          OptionsSnapshot + Greeks
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ STRATEGY LAYER  │
                          │                 │
                          └────────┬────────┘
                                   │
                          Vec<SpreadCandidate>
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ SIMULATION      │
                          │ LAYER           │──────▶ Daily Loop
                          │                 │◀──────
                          └────────┬────────┘
                                   │
                          Vec<ClosedTrade>
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ WALK-FORWARD    │
                          │ LAYER           │──────▶ Param Optimization
                          │                 │◀──────
                          └────────┬────────┘
                                   │
                          WalkForwardResults
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ OUTPUT LAYER    │
                          │                 │
                          └────────┬────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       trades.parquet       metrics.json        equity_curve.csv
```

---

## 10. Crate Structure

```
mahler-backtest/
├── Cargo.toml
├── README.md
│
├── config/
│   ├── default.toml           # Default strategy parameters
│   └── test_config.toml       # Test configuration
│
├── data/
│   └── (ORATS data goes here, gitignored)
│
├── src/
│   ├── main.rs                # CLI entry point
│   ├── lib.rs                 # Library root, re-exports
│   │
│   ├── data/
│   │   ├── mod.rs             # Module exports
│   │   ├── types.rs           # OptionQuote, Chain, Snapshot
│   │   ├── loader.rs          # ORATS file parsing
│   │   └── calendar.rs        # Trading days, holidays
│   │
│   ├── pricing/
│   │   ├── mod.rs
│   │   ├── black_scholes.rs   # BS pricing model
│   │   ├── greeks.rs          # Delta, gamma, theta, vega, rho
│   │   ├── implied_vol.rs     # Jaeckel's algorithm
│   │   └── spread.rs          # Spread-level calculations
│   │
│   ├── strategy/
│   │   ├── mod.rs
│   │   ├── config.rs          # StrategyConfig, parsing
│   │   ├── scanner.rs         # Find candidates
│   │   ├── spread.rs          # CreditSpread struct
│   │   └── scoring.rs         # Candidate ranking
│   │
│   ├── simulation/
│   │   ├── mod.rs
│   │   ├── engine.rs          # Main simulation loop
│   │   ├── execution.rs       # Slippage, fills, commission
│   │   ├── portfolio.rs       # Position tracking, state
│   │   ├── exits.rs           # PT/SL/time exit logic
│   │   └── constraints.rs     # Risk limits
│   │
│   ├── walkforward/
│   │   ├── mod.rs
│   │   ├── periods.rs         # Train/val/test date ranges
│   │   ├── optimizer.rs       # Grid search, param selection
│   │   └── stability.rs       # Param drift, alpha decay
│   │
│   └── output/
│       ├── mod.rs
│       ├── metrics.rs         # Sharpe, drawdown, etc.
│       ├── trade_log.rs       # Trade record struct
│       └── export.rs          # Parquet/JSON/CSV writers
│
├── tests/
│   ├── pricing_tests.rs       # Greeks accuracy
│   ├── execution_tests.rs     # Slippage model
│   ├── simulation_tests.rs    # End-to-end trades
│   └── golden/                # Golden file test data
│
└── benches/
    └── simulation_bench.rs    # Performance benchmarks
```

---

## 11. Dependencies

### Core Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| polars | latest | DataFrame operations, Parquet I/O |
| chrono | 0.4 | Date/time handling |
| rust_decimal | 1.x | Precise decimal arithmetic |
| rayon | 1.x | Data parallelism |
| serde | 1.x | Serialization (TOML, JSON) |
| toml | 0.8 | Config file parsing |

### Math/Stats

| Crate | Version | Purpose |
|-------|---------|---------|
| statrs | 0.16 | Normal CDF, statistical functions |

### CLI/Logging

| Crate | Version | Purpose |
|-------|---------|---------|
| clap | 4.x | Command-line argument parsing |
| tracing | 0.1 | Structured logging |
| tracing-subscriber | 0.3 | Log output formatting |
| indicatif | 0.17 | Progress bars |

### Testing

| Crate | Version | Purpose |
|-------|---------|---------|
| approx | 0.5 | Float comparison in tests |
| proptest | 1.x | Property-based testing |

---

## 12. Integration with Mahler V2

### Boundary

The backtester is a standalone tool that produces:

1. **Validated parameters** (config file)
2. **Trade logs** (Parquet files)
3. **Performance metrics** (JSON)

Mahler V2 consumes these outputs but does not call the backtester directly.

### Parameter Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Rust           │     │  Validated      │     │  Python         │
│  Backtester     │────▶│  params.toml    │────▶│  Mahler V2      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Trade Log as Training Data

Backtester trade logs become training data for:

1. Fine-tuning Qwen (Phase 2 of Mahler V2)
2. Validating agent decisions against backtest outcomes
3. Regime-specific parameter selection

### Shared Configuration

The following parameters are defined identically in both systems:

| Parameter | Backtester | Mahler V2 |
|-----------|------------|-----------|
| Slippage model | ORATS percentages | Same |
| Commission | $1.00/contract | Same |
| Exit thresholds | PT/SL/time | Same |
| Position limits | max risk, heat | Same |

This ensures backtested performance matches live expectations.

---

## Appendix A: ORATS Data Format

Expected CSV columns from ORATS daily snapshots:

```
ticker,tradeDate,expirDate,strike,optionType,
bid,ask,mid,last,volume,openInterest,
stockPrice,delta,gamma,theta,vega,rho,
impliedVolatility,underlyingBid,underlyingAsk
```

Notes:

- `tradeDate`: Date of snapshot (14 min before close)
- `optionType`: "C" for call, "P" for put
- Greeks are pre-calculated by ORATS
- IV is annualized

---

## Appendix B: CLI Interface

```
mahler-backtest

USAGE:
    mahler-backtest <COMMAND>

COMMANDS:
    run          Run a single backtest with given config
    optimize     Run walk-forward optimization
    validate     Validate parameters on test data
    export       Export results to various formats
    help         Print help information

EXAMPLES:
    # Run single backtest
    mahler-backtest run --config config/default.toml --data data/spy_2020_2024.parquet

    # Run walk-forward optimization
    mahler-backtest optimize --config config/default.toml --data data/ --output results/

    # Validate specific parameters on 2024 data
    mahler-backtest validate --params results/best_params.toml --data data/spy_2024.parquet
```

---

*Document created: 2026-01-28*
*Status: Design Phase*
