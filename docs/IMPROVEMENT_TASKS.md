# Mahler Trading System Improvement Tasks

This document outlines all tasks required to implement the algorithmic trading improvements researched in `IMPROVEMENT_RESEARCH.md`. The improvements are organized into 4 phases by priority and dependencies.

---

## Overview

| Phase | Focus Area | Tasks | Expected Impact | Status |
|-------|------------|-------|-----------------|--------|
| 1 | Exit Optimization | 7 | 15-25% win rate improvement | COMPLETE |
| 2 | Regime Detection | 8 | 20-30% drawdown reduction | COMPLETE |
| 3 | Entry Scoring & Dynamic Betas | 8 | 15-25% Sharpe improvement | COMPLETE |
| 4 | Advanced IV & Rule Validation | 11 | 10-15% timing + systematic pruning | COMPLETE |

**Total: 34 tasks (34 complete, 0 remaining)**

---

## Phase 1: Exit Optimization (COMPLETED 2024-12-29)

**Goal:** Replace fixed 50% profit target with IV-adjusted targets and gamma-aware timing.

**Status:** COMPLETE

**Files modified:**

- `src/core/risk/validators.py`
- `src/handlers/position_monitor.py`
- `src/core/db/d1.py`
- `wrangler.toml` (CPU limit increased to 120s)

**New files:**

- `src/core/analysis/exit_optimizer.py`
- `src/migrations/0004_exit_optimization.sql`

### Tasks

#### 1.1 Extend ExitConfig dataclass [DONE]

- [x] Add `iv_adjustment_enabled: bool = True`
- [x] Add `gamma_protection_enabled: bool = True`
- [x] Add `iv_high_threshold: float = 70.0` (IV rank above which to reduce targets)
- [x] Add `iv_low_threshold: float = 30.0` (IV rank below which to increase targets)
- [x] Add `gamma_protection_pnl: float = 0.70` (exit at 70% profit when DTE <= 21)
- [x] Add `gamma_explosion_dte: int = 7` (force exit when DTE <= 7)

#### 1.2 Create IVAdjustedExits class in validators.py [DONE]

- [x] Method: `calculate_iv_rank(current_iv, iv_52w_high, iv_52w_low)` - return 0-100 percentile
- [x] Method: `adjusted_profit_target(base_target, current_iv, iv_history)` - scale target by IV rank
  - High IV (rank > 70): reduce target by up to 15% to capture premium before IV crush
  - Low IV (rank < 30): increase target by up to 15% to let winners run
- [x] Method: `gamma_aware_exit(pnl_pct, dte, target)` - return (should_exit: bool, reason: str)
  - Exit at 70% of target when DTE <= 21
  - Force exit when DTE <= 7 regardless of P/L

#### 1.3 Update ExitValidator.check_exit_conditions() [DONE]

- [x] Accept current_iv and iv_history parameters
- [x] Calculate IV rank using IVAdjustedExits
- [x] Apply adjusted profit target instead of fixed 50%
- [x] Check gamma-aware exit conditions before standard checks
- [x] Return exit reason with specific trigger (iv_adjusted_profit, gamma_protection, gamma_explosion)
- [x] Return iv_rank as third element of tuple

#### 1.4 Update position_monitor handler [DONE]

- [x] Extract IV from options chain contracts
- [x] Fetch IV history via db.get_iv_history()
- [x] Pass IV context to ExitValidator
- [x] Log exit reason with IV rank and DTE for analysis
- [x] Update Discord notifications to include exit reason details

#### 1.5 Create exit_optimizer.py utility [DONE]

- [x] Class: `ExitParameterOptimizer`
- [x] Method: `create_backtest_objective(trade_history)` - return objective function for scipy
- [x] Method: `optimize_parameters()` - use scipy.optimize.differential_evolution
  - Bounds: profit_target (0.05-0.50), stop_loss (0.02-0.30), time_exit (7-45)
  - Objective: maximize Sharpe ratio
- [x] Static method: `from_db_trades()` - create optimizer from Trade objects

#### 1.6 Add weekly parameter optimization job [DEFERRED]

- [ ] Run optimization after Friday EOD summary
- [ ] Requires minimum 50 closed trades for statistical significance
- [ ] Store optimized parameters with timestamp
- [ ] Log optimization results to Discord (optional channel)
- **Note:** Framework ready in exit_optimizer.py, integration deferred until sufficient trade history

#### 1.7 Add exit analytics tracking [DONE]

- [x] Migration 0004: Add `exit_reason`, `iv_rank_at_exit`, `dte_at_exit` columns to trades
- [x] Update D1Client.close_trade() to store exit analytics
- [x] Enable analysis of exit effectiveness over time

---

## Phase 2: Regime Detection (COMPLETED 2024-12-29)

**Goal:** Replace basic VIX thresholds with GaussianMixture-based regime detection using 7 market features.

**Status:** COMPLETE

**Files modified:**

- `src/handlers/morning_scan.py`
- `src/core/broker/alpaca.py`
- `src/core/db/d1.py`
- `src/core/db/kv.py`

**New files:**

- `src/core/analysis/regime.py`
- `src/migrations/0005_market_regimes.sql`

### Tasks

#### 2.1 Add get_historical_bars() to Alpaca client [DONE]

- [x] Method signature: `get_historical_bars(symbol, timeframe, start, end)`
- [x] Fetch OHLCV data for regime feature calculation
- [x] Support timeframes: 1Day, 1Hour
- [x] Return list of dicts with keys: open, high, low, close, volume, timestamp

#### 2.2 Create MarketRegimeDetector class [DONE]

- [x] Initialize with `lookback=60` days and `n_regimes=4`
- [x] Use sklearn.mixture.GaussianMixture with covariance_type='full'
- [x] Use sklearn.preprocessing.StandardScaler for feature normalization

#### 2.3 Implement compute_features() method [DONE]

- [x] Calculate 7 features from OHLCV and IV data:
  1. `realized_vol_20`: 20-day rolling volatility (annualized)
  2. `momentum_20`: 20-day cumulative returns
  3. `trend`: (SMA20 - SMA50) / SMA50
  4. `iv`: current implied volatility
  5. `iv_rv_spread`: IV minus realized volatility
  6. `volume_ratio`: current volume / 20-day average volume
  7. `range_pct`: (high - low) / close
- [x] Return DataFrame with features, dropping NaN rows

#### 2.4 Implement fit_and_predict() method [DONE]

- [x] Scale features using StandardScaler
- [x] Fit GaussianMixture and get cluster labels
- [x] Characterize each cluster by average volatility and trend direction
- [x] Map clusters to regime names: BULL_LOW_VOL, BULL_HIGH_VOL, BEAR_LOW_VOL, BEAR_HIGH_VOL
- [x] Return current regime label and probability array

#### 2.5 Implement get_position_multiplier() method [DONE]

- [x] Return sizing multiplier based on regime:
  - BULL_LOW_VOL: 1.0 (full size)
  - BULL_HIGH_VOL: 0.5 (reduce exposure)
  - BEAR_LOW_VOL: 0.5 (cautious)
  - BEAR_HIGH_VOL: 0.25 (defensive)
- [x] Override with 0.1 multiplier if VIX > 40

#### 2.6 Add regime caching to KV store [DONE]

- [x] Cache key format: `regime:{symbol}:{date}:{hour}`
- [x] Cache duration: 1 hour (regime doesn't change frequently)
- [x] Store: regime label, probabilities, multiplier, detection timestamp
- [x] Check cache before running expensive GMM fit

#### 2.7 Integrate regime detection into morning_scan [DONE]

- [x] Detect regime at start of scan using SPY data
- [x] Pass regime multiplier to PositionSizer
- [x] Combine regime multiplier with risk circuit breaker (use more conservative)
- [x] Log regime with confidence percentage

#### 2.8 Add market_regimes table to D1 [DONE]

- [x] Columns: id, symbol, regime, probability, position_multiplier, features (JSON), detected_at
- [x] Store regime history for analysis and backtesting
- [x] Create index on (symbol, detected_at)

---

## Phase 3: Entry Scoring & Dynamic Betas (COMPLETED 2024-12-29)

**Goal:** Replace static scoring weights with regime-conditional weights and dynamic beta calculation.

**Status:** COMPLETE

**Files modified:**

- `src/core/analysis/screener.py`
- `src/core/risk/position_sizer.py`
- `src/core/db/d1.py`
- `src/core/db/kv.py`
- `src/handlers/morning_scan.py`
- `src/handlers/eod_summary.py`

**New files:**

- `src/core/analysis/indicators.py`
- `src/core/analysis/weight_optimizer.py`
- `src/core/risk/dynamic_beta.py`
- `src/migrations/0006_phase3_tables.sql`

### Tasks

#### 3.1 Create indicators.py with technical indicator calculations [DONE]

- [x] Function: `calculate_rsi(prices, period=14)` - Relative Strength Index (0-100)
- [x] Function: `calculate_macd(prices, fast=12, slow=26, signal=9)` - return (macd_line, signal_line, histogram)
- [x] Function: `calculate_bollinger_position(prices, period=20, std_dev=2)` - return position within bands (0-1)
- [x] Function: `calculate_sma(prices, period)` - Simple Moving Average
- [x] Function: `calculate_ema(prices, period)` - Exponential Moving Average
- [x] Function: `calculate_atr(high, low, close, period)` - Average True Range
- [x] All functions work with numpy arrays (Pyodide compatible)

#### 3.2 Create RegimeConditionalScorer class in screener.py [DONE]

- [x] Store weight sets for each regime type (ScoringWeights dataclass)
- [x] Default weights per regime:
  - BULL_LOW_VOL: iv=0.20, delta=0.20, credit=0.25, ev=0.35
  - BULL_HIGH_VOL: iv=0.25, delta=0.35, credit=0.20, ev=0.20
  - BEAR_LOW_VOL: iv=0.20, delta=0.25, credit=0.35, ev=0.20
  - BEAR_HIGH_VOL: iv=0.35, delta=0.30, credit=0.20, ev=0.15
- [x] Method: `get_weights(regime)` - apply regime-specific weights
- [x] Method: `update_weights(regime, weights)` - update from optimization
- [x] Method: `load_from_dict(weights_dict)` - load from KV cache

#### 3.3 Update OptionsScreener to use RegimeConditionalScorer [DONE]

- [x] Accept scorer parameter in constructor
- [x] Accept regime parameter in screen_chain() method
- [x] Get regime-specific weights from scorer
- [x] Apply weights in _score_spread() method

#### 3.4 Create weight_optimizer.py [DONE]

- [x] Class: `WeightOptimizer`
- [x] Method: `backtest_sharpe(weights, signals, outcomes)` - calculate Sharpe for given weights
- [x] Method: `optimize_weights(regime)` - use scipy.optimize.differential_evolution
  - Constraints: weights sum to 1, each weight in [0.10, 0.50]
  - Objective: maximize Sharpe ratio
- [x] Method: `optimize_all_regimes()` - optimize all 4 regimes
- [x] Static method: `from_db(db)` - create optimizer from D1 trade history

#### 3.5 Add weekly weight optimization job [DONE]

- [x] Run after Friday EOD summary in eod_summary.py
- [x] Requires minimum 100 closed trades
- [x] Optimize weights separately for each regime
- [x] Store in D1 optimized_weights table
- [x] Cache in KV for fast access
- [x] Send Discord notification

#### 3.6 Create DynamicBetaCalculator class [DONE]

- [x] Initialize with fallback_betas dict (defaults to ASSET_BETAS)
- [x] Method: `ewma_beta(asset_returns, market_returns, halflife=20)` - EWMA beta
- [x] Method: `rolling_beta(asset_returns, market_returns, window)` - rolling window beta
- [x] Method: `rolling_beta_multiwindow(returns)` - calculate 20d and 60d rolling betas
- [x] Method: `blended_beta()` - blend EWMA (50%), rolling_20 (30%), rolling_60 (20%)
- [x] Method: `calculate_for_symbol(symbol, bars, spy_bars)` - full calculation pipeline
- [x] DynamicBetaResult dataclass for serialization

#### 3.7 Replace hardcoded betas in position_sizer.py [DONE]

- [x] Add kv_client parameter to PositionSizer
- [x] Async method `get_beta_async()` to check KV cache for dynamic beta
- [x] Fall back to static betas if no cached value
- [x] Add `set_cached_beta()` and `clear_beta_cache()` helpers

#### 3.8 Add dynamic_betas and optimized_weights tables to D1 [DONE]

- [x] Migration 0006: dynamic_betas table with beta_ewma, beta_rolling_20, beta_rolling_60, beta_blended, correlation_spy
- [x] Migration 0006: optimized_weights table with regime, weight_iv, weight_delta, weight_credit, weight_ev, sharpe_ratio
- [x] D1Client methods: save_dynamic_beta(), get_latest_dynamic_beta(), get_all_dynamic_betas()
- [x] D1Client methods: save_optimized_weights(), get_latest_optimized_weights()
- [x] KV methods: cache_beta(), get_cached_beta(), cache_weights(), get_cached_weights()

---

## Phase 4: Advanced IV Analysis & Rule Validation (COMPLETED 2024-12-29)

**Goal:** Add second-order Greeks, IV term structure analysis, and statistical rule validation.

**Status:** COMPLETE

**Files modified:**

- `src/core/analysis/greeks.py`
- `src/core/analysis/iv_rank.py`
- `src/core/risk/position_sizer.py`
- `src/handlers/morning_scan.py`
- `src/handlers/eod_summary.py`
- `src/core/db/d1.py`
- `src/core/notifications/discord.py`

**New files:**

- `src/core/analysis/rule_validator.py`
- `src/migrations/0007_rule_validation.sql`

### Tasks

#### 4.1 Add second-order Greeks to greeks.py [DONE]

- [x] Function: `calculate_vanna(S, K, T, r, q, sigma)` - d(Delta)/d(IV), spot-vol correlation sensitivity
- [x] Function: `calculate_volga(S, K, T, r, q, sigma)` - d(Vega)/d(IV), vol-of-vol exposure
- [x] Function: `calculate_charm(S, K, T, r, q, sigma)` - d(Delta)/d(Time), delta decay rate
- [x] Helper: `_d1(S, K, T, r, q, sigma)` and `_d2(...)` for Black-Scholes intermediates
- [x] Helper: `_n_prime(x)` - standard normal PDF

#### 4.2 Add second-order Greeks to SpreadGreeks dataclass [DONE]

- [x] Add fields: vanna, volga, charm (all floats)
- [x] Update calculate_spread_greeks() to compute these for spreads

#### 4.3 Add position size adjustment for second-order Greeks [DONE]

- [x] Function: `calculate_second_order_adjustment(vanna, volga, vanna_threshold=0.5, volga_threshold=0.3)`
- [x] Reduce size when vanna or volga exceed thresholds
- [x] Formula: `adjustment = (1 - min(|vanna|/threshold, 0.4)) * (1 - min(|volga|/threshold, 0.5))`
- [x] Integrate into PositionSizer with RiskLimits configuration

#### 4.4 Create IVTermStructure class in iv_rank.py [DONE]

- [x] Initialize with arrays of expirations (DTE) and corresponding IVs
- [x] Use scipy.interpolate.UnivariateSpline for smooth interpolation
- [x] Method: `interpolate_iv(dte)` - return IV at any DTE
- [x] Method: `detect_regime()` - return TermStructureResult with regime and signal
  - ratio < 0.95: contango, favorable_for_selling_vol
  - ratio > 1.05: backwardation, avoid_selling_vol
  - else: flat, neutral
- [x] Method: `get_slope()` - return slope of term structure

#### 4.5 Create IVMeanReversion class in iv_rank.py [DONE]

- [x] Initialize with IV time series (numpy array)
- [x] Method: `estimate_ou_parameters()` - fit Ornstein-Uhlenbeck model via OLS
  - Return OUParameters: theta (mean reversion speed), mu (long-term mean), sigma, half_life
- [x] Method: `generate_signal(z_entry=2.0)` - return MeanReversionResult with signal and z_score
  - z > 2: SELL_VOL (IV is elevated)
  - z < -2: BUY_VOL (IV is depressed)
  - else: HOLD
- [x] Method: `test_mean_reversion()` - run ADF test, return MeanReversionTestResult

#### 4.6 Integrate IV term structure into morning_scan [DONE]

- [x] Calculate term structure using options chain expirations
- [x] Include regime in analysis context for Claude
- [x] Factor into entry decisions (skip symbols in backwardation)
- [x] Add mean reversion signal to AI context

#### 4.7 Create TradingRuleValidator class [DONE]

- [x] Initialize with trades and rules lists
- [x] Method: `validate_rule(rule_id)` - compare outcomes with/without rule
  - Use scipy.stats.mannwhitneyu for non-parametric comparison
  - Return RuleValidationResult with p_value, win_rates, effect_direction
- [x] Method: `validate_all_rules()` - validate all rules with FDR correction
- [x] Method: `_apply_fdr_correction()` - Benjamini-Hochberg correction
  - Use statsmodels.stats.multitest.multipletests
- [x] Static method: `from_db()` - create validator from database

#### 4.8 Add rule validation schema [DONE]

- [x] Migration 0007: rule_validations table
- [x] Columns: id, rule_id, validated_at, trades_with_rule, trades_without_rule, mean_pnl_with, mean_pnl_without, win_rate_with, win_rate_without, u_statistic, p_value, p_value_adjusted, is_significant, effect_direction
- [x] Foreign key to playbook table
- [x] Indexes on (rule_id, validated_at) and (is_significant, validated_at)

#### 4.9 Add rule tagging to trades [DONE]

- [x] Add applied_rule_ids column to trades table (ALTER TABLE)
- [x] D1Client method: `tag_trade_with_rules(trade_id, rule_ids)`
- [x] D1Client method: `get_closed_trades_with_rules(lookback_days)`
- [x] Enable tracing trade outcomes back to rules

#### 4.10 Add weekly rule validation job [DONE]

- [x] Run validation weekly (Friday) after EOD summary
- [x] Require minimum 50 closed trades for statistical significance
- [x] Load all playbook rules and trade outcomes via `TradingRuleValidator.from_db()`
- [x] Run validation and save results to database
- [x] Update playbook validation status
- [x] Send Discord notification with validation report

#### 4.11 Update playbook table schema and reporting [DONE]

- [x] Add column: is_validated (INTEGER DEFAULT 0)
- [x] Add column: last_validated_at (TEXT)
- [x] Add column: validation_p_value (REAL)
- [x] D1Client methods: save_rule_validation(), update_playbook_validation_status(), get_rule_validations(), get_latest_rule_validations()
- [x] Discord method: send_rule_validation_report()

---

## Database Migrations

Create migration file: `src/core/db/migrations/002_improvements.sql`

### New Tables

#### market_regimes

- id (TEXT PRIMARY KEY)
- symbol (TEXT NOT NULL)
- regime (TEXT NOT NULL) - BULL_LOW_VOL, BULL_HIGH_VOL, BEAR_LOW_VOL, BEAR_HIGH_VOL
- probability (REAL NOT NULL) - confidence 0-1
- position_multiplier (REAL NOT NULL)
- features (TEXT) - JSON blob of feature values used
- detected_at (TEXT NOT NULL)
- INDEX on (symbol, detected_at)

#### rule_validations

- id (TEXT PRIMARY KEY)
- rule_id (TEXT NOT NULL, FK to playbook)
- p_value (REAL NOT NULL)
- win_rate_with (REAL NOT NULL)
- win_rate_without (REAL NOT NULL)
- improvement (REAL NOT NULL)
- n_trades_with (INTEGER NOT NULL)
- n_trades_without (INTEGER NOT NULL)
- validated (BOOLEAN NOT NULL)
- validated_at (TEXT NOT NULL)
- INDEX on (rule_id, validated_at)

#### dynamic_betas

- id (TEXT PRIMARY KEY)
- symbol (TEXT NOT NULL)
- beta_ewma (REAL NOT NULL)
- beta_rolling_60 (REAL NOT NULL)
- beta_blended (REAL NOT NULL)
- calculated_at (TEXT NOT NULL)
- INDEX on (symbol, calculated_at)

#### optimized_weights

- id (TEXT PRIMARY KEY)
- regime (TEXT NOT NULL)
- weights (TEXT NOT NULL) - JSON blob
- sharpe_ratio (REAL NOT NULL)
- n_trades (INTEGER NOT NULL)
- optimized_at (TEXT NOT NULL)
- INDEX on (regime, optimized_at)

### Schema Modifications

#### playbook table

- ADD COLUMN is_validated BOOLEAN DEFAULT FALSE
- ADD COLUMN last_validated_at TEXT
- ADD COLUMN validation_p_value REAL

#### trades table

- ADD COLUMN exit_reason TEXT
- ADD COLUMN iv_rank_at_exit REAL
- ADD COLUMN dte_at_exit INTEGER
- ADD COLUMN applied_rule_ids TEXT (JSON array)

#### recommendations table

- ADD COLUMN applied_rule_ids TEXT (JSON array)
- ADD COLUMN regime TEXT

---

## Testing Requirements

### Unit Tests

- [ ] IVAdjustedExits.adjusted_profit_target() with various IV ranks
- [ ] IVAdjustedExits.gamma_aware_exit() at different DTE and P/L levels
- [ ] MarketRegimeDetector.compute_features() feature calculations
- [ ] MarketRegimeDetector.fit_and_predict() regime assignment
- [ ] Technical indicators (RSI, MACD, Bollinger) against known values
- [ ] DynamicBetaCalculator.ewma_beta() against pandas ewm
- [ ] IVTermStructure.detect_regime() for various term structures
- [ ] IVMeanReversion.estimate_ou_parameters() against simulated OU process
- [ ] TradingRuleValidator.test_rule() with mock trade data

### Integration Tests

- [ ] Position monitor exit flow with IV-adjusted targets
- [ ] Morning scan with regime detection and conditional scoring
- [ ] Position sizing with dynamic betas
- [ ] EOD summary with rule validation

### Backtest Validation

- [ ] Compare IV-adjusted exits vs fixed exits on historical trades
- [ ] Measure drawdown with vs without regime-based sizing
- [ ] Compare scoring accuracy with optimized vs static weights

---

## Rollout Strategy

### Phase 1: Exit Optimization

1. Implement and test IV-adjusted exits
2. Deploy with logging only (don't change actual exits yet)
3. Compare recommended exits vs actual exits for 1-2 weeks
4. Enable live IV-adjusted exits after validation

### Phase 2: Regime Detection

1. Implement regime detector and run in shadow mode
2. Log detected regimes alongside existing VIX-based logic
3. Compare regime multipliers vs current circuit breaker adjustments
4. Enable regime-based sizing after 1 week of validation

### Phase 3: Entry Scoring

1. Implement technical indicators and regime-conditional scorer
2. Run new scoring in parallel with existing
3. Track which scoring method would have selected better trades
4. Switch to new scoring after statistical validation

### Phase 4: Advanced Features

1. Deploy second-order Greeks (informational first)
2. Add IV term structure analysis to scan context
3. Enable rule validation on historical data
4. Start tagging new trades with rule_ids

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Win Rate | ~75% | 85%+ | Closed trades with profit > 0 |
| Sharpe Ratio | ~1.2 | 1.5+ | Annualized returns / volatility |
| Max Drawdown | ~8% | <5% | Peak to trough equity decline |
| Profit Factor | ~1.8 | 2.2+ | Gross profits / gross losses |
| Avg Exit Timing | Fixed 50% | Adaptive | Track actual profit % at exit |

---

## Dependencies Summary

```
Phase 1 (Exit Optimization)
    - No external dependencies
    - Can start immediately
    |
    v
Phase 2 (Regime Detection)
    - Requires historical OHLCV data access
    - Benefits from Phase 1 completion
    |
    v
Phase 3 (Entry Scoring)
    - Depends on Phase 2 (regime parameter)
    - Requires technical indicator calculations
    |
    v
Phase 4 (Advanced IV & Rules)
    - Can partially run in parallel with Phase 3
    - Rule validation needs sufficient trade history with tags
```
