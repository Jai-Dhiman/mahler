# TraderJoe Backtest — Autoresearch Loop

Autonomous parameter optimization for the put credit spread strategy using 16 years of ORATS daily options data (SPY/QQQ/IWM, 2007–2023).

---

## How to Run

```bash
cd traderjoe/traderjoe-backtest

# Single backtest with specific parameters
cargo run --release -- run \
  --ticker SPY \
  --start 2009-01-01 --end 2023-12-31 \
  --profit-target 50 --stop-loss 200 \
  --min-delta 0.20 --max-delta 0.30 \
  --min-dte 30 --max-dte 60 \
  --min-iv-percentile 25

# Walk-forward optimization (grid search over ParameterGrid)
cargo run --release -- optimize \
  --ticker SPY \
  --start 2007-01-01 --end 2023-12-31 \
  --train-months 24 \
  --validate-months 3 \
  --test-months 3
```

After building, use `./target/release/traderjoe-backtest` directly (much faster, skips compile).

---

## Walk-Forward Methodology

### Window Structure

The optimizer uses a **rolling (non-anchored) walk-forward** with three non-overlapping periods per window:

| Period | Duration | Purpose |
|---|---|---|
| Train (IS) | 24 months | Grid search — pick best parameter set by Sharpe |
| Validate | 3 months | Sanity check — verify WFE and trade count |
| Test (OOS) | 3 months | Held-out performance — what gets reported |

With 16 years of data (2007–2023 ≈ 192 months), stepping by 1 month yields ~162 rolling windows. Each window produces one OOS Sharpe data point. The reported metric is the **average OOS Sharpe** across all windows.

**Source:** VectorBT reference WFO notebook (github.com/polakowo/vectorbt) uses 730-day IS / 180-day OOS. QuantConnect's production framework (docs.quantconnect.com) uses rolling non-anchored with monthly re-optimization. Both match this structure.

### Why 24-Month IS Window

The Bailey–Lopez de Prado constraint ("Probability of Backtest Overfitting," SSRN 2326253): with 5 years of data, no more than ~45 independent configurations should be tested. With a 24-month IS window and ~7 grid dimensions, we need to keep the total combinations under ~150 to remain safe. See Parameter Space section below.

Rolling is preferred over anchored because options volatility regimes shift significantly across years (2008 crisis, 2017 calm, 2020 COVID spike). An anchored window permanently weights the 2008 crisis data; rolling gives equal weight to the current regime.

### Pass/Fail Criteria

Every run must report these metrics. A result is **rejected** if any guard fails:

| Metric | Definition | Guard |
|---|---|---|
| WFE | OOS Sharpe / IS Sharpe | **> 0.50** |
| Avg OOS Sharpe | Mean across all test windows | **> 0.0** |
| Avg trades/window | Mean trades in each OOS window | **>= 20** |
| Win rate | Closed winners / total closed | **>= 30%** |
| Max drawdown | Peak-to-trough in OOS period | **< 50%** |

**WFE (Walk-Forward Efficiency) is the primary overfitting diagnostic.** If WFE < 0.50, the IS parameter selection is not generalizing — the system is mining historical noise. A WFE of 1.0 means OOS performance equals IS performance exactly.

**Source:** WFE > 50–60% threshold is cited across QuantInsti, Unger Academy, and StrategyQuant documentation as the practitioner standard. WFE is defined in Wikipedia's walk-forward optimization article.

---

## Parameter Space

### Current Grid (ParameterGrid::default in walkforward/optimizer.rs)

The `ParameterGrid` struct holds `Vec<f64>` for each dimension. Edit it directly to define the sweep.

| Parameter | Research-Backed Range | Notes |
|---|---|---|
| `profit_target` | `[25, 35, 50, 65, 75]` | See Exit Rules research below |
| `stop_loss` | `[125, 150, 200]` | 200% is the tastytrade standard |
| `dte_min` | `[21, 30]` | 30 is the proven floor |
| `dte_max` | `[45, 60]` | 45 and 60 are the two well-studied targets |
| `delta_min` | `[0.15, 0.20, 0.25]` | Standard put credit spread range |
| `delta_max` | `[delta_min + 0.10]` | Constrain width to avoid duplicate combos |
| `iv_percentile` | `[0, 25, 50]` | 50 is the tastytrade entry filter |

**Total combinations: 5 × 3 × 2 × 2 × 3 × 3 = 540.** This exceeds the safe limit for a 24-month IS window. In practice, fix `dte_min=30`, `dte_max=60` (most empirically supported), and sweep only `profit_target × stop_loss × delta × iv_percentile = 5 × 3 × 3 × 3 = 135 combinations`. This is within the ~150 safe limit derived from Bailey–Lopez de Prado (SSRN 2326253).

### What NOT to Grid-Search (Treat as Fixed Structural Decisions)

These are not parameters — they are structural decisions that the backtest engine treats as constants:

- **Underlyings:** SPY/QQQ/IWM — most liquid options in the US, non-negotiable for credit spreads at this size
- **Spread width:** $5 wide — standard retail-accessible size for SPY/QQQ/IWM
- **Commission:** $1.00/contract/leg — Alpaca paper rate
- **Slippage model:** ORATS 66% fill model — derived from actual bid-ask data
- **Max concurrent positions:** 10 — determined by capital/margin constraints
- **Max trades per scan:** 3 — one per underlying per day

---

## Research-Backed Rationale for Each Parameter

### Profit Target

**Do not use 25%.** The Option Alpha SPY put credit spread backtest ("8 SPY Put Credit Spread Backtest Results Analyzed," optionalpha.com) found:

| Exit | Sharpe |
|---|---|
| Hold to expiration | 0.62 |
| 50% PT + 25% SL + 15 DTE | 0.77 |
| 75% PT + 25% SL + 15 DTE | 0.83 |

The 25% level appears in the literature only as a **stop-loss** threshold, not a profit target. The autoresearch iteration that selected profit_target=25 (iter 1) produced Sharpe 0.14 vs baseline 0.10 — a marginal improvement that was likely due to the old 6-month IS window being too small to distinguish signal from noise. The new 24-month IS window will re-test this.

The tastytrade standard (their "Market Measures" research, cited but not directly accessible) is 50% profit target + 21 DTE time exit. An 11-year adversarial backtest of this exact rule on SPX 2005–2016 (SJ Options, sjoptions.com) showed 61% win rate but negative returns at all allocation sizes, suggesting the 50% rule alone is insufficient without an IV entry filter.

**Recommendation:** Start sweep at 50%. Expect 50–75% to outperform 25% on OOS Sharpe.

### Stop Loss

The stop loss is effectively a **dead parameter** when profit_target=25 (iter 4–6 in the existing changelog all show zero effect at 25% PT). This is because trades exit at profit before ever hitting the stop. This confirms 25% is too tight.

At profit_target=50–75%, the stop loss becomes active and meaningful. The tastytrade standard is 200% (2x credit received). Option Alpha's 25% stop lost position too early, eliminating recovery. The 125% current default is aggressive.

**Recommendation:** Test 125%, 150%, 200%. Expect 150–200% to produce lower max drawdown without meaningfully reducing returns.

### Delta Range

The current production config uses delta 0.05–0.15. This is far below the standard range for put credit spreads. The research standard (tastytrade, Spintwig, Option Alpha) is **0.20–0.30 for the short leg**. At 0.05–0.15, you are selling very far OTM spreads with tiny premiums. The autoresearch baseline (iter 0) already used 0.20–0.30 and the production config was not updated.

**Recommendation:** Sweep 0.15–0.25 and 0.20–0.30. The production `SpreadConfig` should be updated after the next validated run.

### DTE Range

30–60 DTE is the standard range with the most empirical support:
- 45 DTE is the tastytrade standard (consistent theta decay, before gamma acceleration)
- Spintwig's primary backtest uses 45 DTE target (28–62 day range acceptable)
- CBOE PUT index uses monthly rolls (30 DTE equivalent)
- 21 DTE time exit is the tastytrade paired rule — set `gamma_exit_dte=21` in production config, not 7

### IV Percentile Entry Filter

The tastytrade rule requires IVR (IV Rank) >= 50 before entry. The SJ Options adversarial backtest showed the 50% rule fails without this filter. iv_percentile=0 (no filter) likely explains some of the poor autoresearch results.

**Recommendation:** Include iv_percentile=[0, 25, 50] in the grid. Expect iv_percentile=50 to outperform on risk-adjusted returns at the cost of fewer trades.

### VIX-Based Position Sizing

The best empirically supported regime overlay for put-writing on index options is **continuous VIX percentile rank scaling** (arXiv 2508.16598, "Sizing the Risk: Kelly, VIX, and Hybrid Approaches in Put-Writing on Index Options"):

```
contracts = floor(base_contracts × (1 - vix_percentile_rank))
```

where `vix_percentile_rank` uses a 252-day lookback. This produced 16.87–23.13% annualized returns with max drawdowns below 11% in the paper's put-writing backtest.

The current production system uses binary VIX thresholds (30/40/50). These are practitioner conventions, not empirically validated for credit spreads. The continuous scaling approach is a structural improvement that should be implemented separately from the parameter grid — it replaces the `PositionSizer.high_vix_threshold` / `high_vix_reduction` pair with a single dynamic formula.

---

## Parallelism via Git Worktrees

### Capacity

This machine has **10 logical CPUs**. The optimizer uses `rayon` for parallel grid search within a single run. A full grid of 135 combinations runs in parallel internally — one process uses all 10 cores efficiently.

For **parallel autoresearch iterations** (testing independent parameter groups simultaneously), git worktrees give each iteration an isolated repo copy with its own working tree. The constraint: with `rayon` using all available cores per process, running N worktrees means each gets ~(10/N) cores.

Recommended: **2 worktrees in parallel** (each gets ~5 cores, minimal contention). 3 is viable but starts degrading rayon efficiency.

### How to Run Parallel Worktrees

Each worktree needs its own `CARGO_TARGET_DIR` to avoid build cache conflicts:

```bash
# From mahler/ root

# Create two worktrees for independent parameter group sweeps
git worktree add /tmp/wt-exits main
git worktree add /tmp/wt-entry main

# Worktree 1: sweep profit_target × stop_loss (holding delta/DTE at best known)
cd /tmp/wt-exits/traderjoe/traderjoe-backtest
CARGO_TARGET_DIR=/tmp/wt-exits-target cargo build --release
CARGO_TARGET_DIR=/tmp/wt-exits-target ./target/release/traderjoe-backtest optimize \
  --ticker SPY --start 2007-01-01 --end 2023-12-31 \
  --train-months 24 --validate-months 3 --test-months 3 \
  > /tmp/wt-exits-results.txt &

# Worktree 2: sweep delta × iv_percentile (holding profit_target/stop_loss at best known)
cd /tmp/wt-entry/traderjoe/traderjoe-backtest
CARGO_TARGET_DIR=/tmp/wt-entry-target cargo build --release
CARGO_TARGET_DIR=/tmp/wt-entry-target ./target/release/traderjoe-backtest optimize \
  --ticker SPY --start 2007-01-01 --end 2023-12-31 \
  --train-months 24 --validate-months 3 --test-months 3 \
  > /tmp/wt-entry-results.txt &

wait
# Clean up
git worktree remove /tmp/wt-exits
git worktree remove /tmp/wt-entry
```

**Partition strategy for parallel coordinate descent:**

| Worktree | Sweeps | Fixed at |
|---|---|---|
| wt-exits | `profit_target × stop_loss` | delta=0.20–0.30, dte=30–60, iv=0 |
| wt-entry | `delta_min × iv_percentile` | pt=50, sl=200, dte=30–60 |

These two groups are quasi-independent: the optimal profit target is largely independent of the optimal delta range. Run them in parallel, take the best result from each, then run a final single-process sweep combining the best values.

### Autoresearch Iteration Protocol (Sequential)

For the `/autoresearch` skill (coordinate descent, one change at a time):

```
1. Run baseline: current ParameterGrid::default() → record OOS Sharpe + WFE
2. Propose change to one parameter group (e.g., profit_target values)
3. Update ParameterGrid, run optimize, compare OOS Sharpe
4. If OOS Sharpe improves AND WFE > 0.50: KEEP, commit to changelog
5. If OOS Sharpe drops OR WFE < 0.50: REVERT, try next group
6. Repeat for each parameter dimension
7. After all dimensions converged: update SpreadConfig in trader-joe/src/config.rs
```

The verify command for `/autoresearch`:

```bash
cd traderjoe/traderjoe-backtest && \
  cargo build --release 2>&1 | tail -5 && \
  ./target/release/traderjoe-backtest optimize \
    --ticker SPY \
    --start 2007-01-01 --end 2023-12-31 \
    --train-months 24 --validate-months 3 --test-months 3 \
  | grep -E "OOS Sharpe|WFE|win_rate|trades"
```

---

## Applying Results to Production

After each validated autoresearch run, update `traderjoe/src/config.rs` (`SpreadConfig::default()`) with the winning parameters. The production Worker reads these values at deploy time — there is no runtime config fetch.

Parameters to sync after each completed autoresearch sweep:

| Backtest param | Production field | Current value | Target |
|---|---|---|---|
| `profit_target` | `SpreadConfig.profit_target_pct` | 0.25 | ~0.50 |
| `stop_loss` | `SpreadConfig.stop_loss_pct` | 1.25 | ~2.00 |
| `delta_min` | `SpreadConfig.min_delta` | 0.05 | ~0.20 |
| `delta_max` | `SpreadConfig.max_delta` | 0.15 | ~0.30 |
| `time_exit_dte` | `SpreadConfig.gamma_exit_dte` | 7 | 21 |

The VIX percentile position sizing requires a code change to `position_sizer.rs` (replace binary thresholds with continuous formula) — this is a structural change, not a config value.

---

## Known Issues with Current Autoresearch Results

The existing 6 iterations in `autoresearch_changelog.md` used:
- **6-month IS window** (too short — should be 24 months)
- **Profit target as the optimization target, not Sharpe** (unclear from changelog)
- **WFE not tracked** (no overfitting diagnostic)

Iterations 4–6 (stop_loss variations) showed identical Sharpe because the stop loss is a dead parameter when profit_target=25. This is a signal that the 25% profit target was the wrong starting point, not that stop loss doesn't matter.

The next autoresearch run should start from a **new baseline**:
```
profit_target=50, stop_loss=200, dte=30-60, delta=0.20-0.30, iv_percentile=50
```
This is the empirically supported standard from the Option Alpha and tastytrade research. The current production config (profit_target=0.25, delta=0.05–0.15) needs to be brought in line with this baseline before further optimization.
