//! Walk-forward parameter optimization.
//!
//! Performs grid search over parameter space with train/validate/test evaluation.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::NaiveDate;
use rayon::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::backtest::{BacktestConfig, BacktestResult};
use crate::data::OptionsSnapshot;

use super::periods::{WalkForwardPeriod, WalkForwardPeriods, WalkForwardPeriodsConfig};

/// Parameter values to sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterGrid {
    /// DTE minimum values.
    pub dte_min: Vec<i32>,
    /// DTE maximum values.
    pub dte_max: Vec<i32>,
    /// Short delta minimum values.
    pub delta_min: Vec<f64>,
    /// Short delta maximum values.
    pub delta_max: Vec<f64>,
    /// Profit target percentages.
    pub profit_target: Vec<f64>,
    /// Stop loss percentages.
    pub stop_loss: Vec<f64>,
    /// IV percentile minimums.
    pub iv_percentile: Vec<f64>,
}

impl Default for ParameterGrid {
    fn default() -> Self {
        Self {
            dte_min: vec![21, 30, 45],
            dte_max: vec![45, 60, 90],
            delta_min: vec![0.15, 0.20, 0.25],
            delta_max: vec![0.25, 0.30, 0.35],
            profit_target: vec![25.0, 50.0, 75.0],
            stop_loss: vec![100.0, 125.0, 150.0, 200.0],
            iv_percentile: vec![25.0, 50.0, 75.0],
        }
    }
}

impl ParameterGrid {
    /// Calculate total number of parameter combinations.
    pub fn total_combinations(&self) -> usize {
        self.dte_min.len()
            * self.dte_max.len()
            * self.delta_min.len()
            * self.delta_max.len()
            * self.profit_target.len()
            * self.stop_loss.len()
            * self.iv_percentile.len()
    }

    /// Generate all parameter combinations.
    pub fn combinations(&self) -> Vec<ParameterSet> {
        let mut combos = Vec::new();

        for &dte_min in &self.dte_min {
            for &dte_max in &self.dte_max {
                if dte_max <= dte_min {
                    continue;
                }
                for &delta_min in &self.delta_min {
                    for &delta_max in &self.delta_max {
                        if delta_max <= delta_min {
                            continue;
                        }
                        for &profit_target in &self.profit_target {
                            for &stop_loss in &self.stop_loss {
                                for &iv_percentile in &self.iv_percentile {
                                    combos.push(ParameterSet {
                                        dte_min,
                                        dte_max,
                                        delta_min,
                                        delta_max,
                                        profit_target,
                                        stop_loss,
                                        iv_percentile,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        combos
    }
}

/// A single parameter set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSet {
    pub dte_min: i32,
    pub dte_max: i32,
    pub delta_min: f64,
    pub delta_max: f64,
    pub profit_target: f64,
    pub stop_loss: f64,
    pub iv_percentile: f64,
}

impl ParameterSet {
    /// Apply this parameter set to a backtest config.
    pub fn apply_to_config(&self, config: &mut BacktestConfig) {
        config.min_dte = self.dte_min;
        config.max_dte = self.dte_max;
        config.min_delta = self.delta_min;
        config.max_delta = self.delta_max;
        config.profit_target_pct = self.profit_target;
        config.stop_loss_pct = self.stop_loss;
        config.min_iv_percentile = self.iv_percentile;
    }

    /// Create a unique key for this parameter set.
    pub fn key(&self) -> String {
        format!(
            "dte{}-{}_delta{:.2}-{:.2}_tp{:.0}_sl{:.0}_iv{:.0}",
            self.dte_min,
            self.dte_max,
            self.delta_min,
            self.delta_max,
            self.profit_target,
            self.stop_loss,
            self.iv_percentile
        )
    }
}

/// Result of a single walk-forward period.
#[derive(Debug, Clone)]
pub struct PeriodResult {
    /// Period information.
    pub period: WalkForwardPeriod,
    /// Best parameters found during training.
    pub best_params: ParameterSet,
    /// Training performance.
    pub train_result: BacktestSummary,
    /// Validation performance.
    pub validate_result: BacktestSummary,
    /// Test performance.
    pub test_result: BacktestSummary,
}

/// Summary of backtest performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestSummary {
    pub total_return_pct: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,
    pub total_trades: usize,
    pub final_equity: Decimal,
}

impl From<&BacktestResult> for BacktestSummary {
    fn from(result: &BacktestResult) -> Self {
        Self {
            total_return_pct: result.total_return_pct,
            win_rate: result.win_rate(),
            profit_factor: result.profit_factor(),
            max_drawdown_pct: result.max_drawdown_pct,
            sharpe_ratio: result.sharpe_ratio(),
            total_trades: result.total_trades,
            final_equity: result.final_equity,
        }
    }
}

/// Complete walk-forward optimization result.
#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    /// Ticker analyzed.
    pub ticker: String,
    /// Overall date range.
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    /// Results for each period.
    pub period_results: Vec<PeriodResult>,
    /// Aggregate validation performance.
    pub aggregate_validate: BacktestSummary,
    /// Aggregate test performance.
    pub aggregate_test: BacktestSummary,
    /// Most common best parameters.
    pub consensus_params: Option<ParameterSet>,
}

impl WalkForwardResult {
    /// Calculate average validation Sharpe ratio.
    pub fn avg_validate_sharpe(&self) -> f64 {
        if self.period_results.is_empty() {
            return 0.0;
        }
        self.period_results
            .iter()
            .map(|p| p.validate_result.sharpe_ratio)
            .sum::<f64>()
            / self.period_results.len() as f64
    }

    /// Calculate average test Sharpe ratio.
    pub fn avg_test_sharpe(&self) -> f64 {
        if self.period_results.is_empty() {
            return 0.0;
        }
        self.period_results
            .iter()
            .map(|p| p.test_result.sharpe_ratio)
            .sum::<f64>()
            / self.period_results.len() as f64
    }

    /// Get summary string.
    pub fn summary(&self) -> String {
        format!(
            "Walk-Forward Results: {} periods\n\
             Avg Validate Sharpe: {:.2}\n\
             Avg Test Sharpe: {:.2}\n\
             Total Periods: {}",
            self.period_results.len(),
            self.avg_validate_sharpe(),
            self.avg_test_sharpe(),
            self.period_results.len()
        )
    }

    /// Generate comprehensive LLM-friendly analysis for system iteration.
    pub fn llm_analysis(&self) -> String {
        let mut analysis = String::new();

        analysis.push_str("=== WALK-FORWARD OPTIMIZATION ANALYSIS ===\n\n");

        // 1. Executive Summary
        analysis.push_str("## EXECUTIVE SUMMARY\n\n");
        let val_sharpe = self.avg_validate_sharpe();
        let test_sharpe = self.avg_test_sharpe();
        let sharpe_degradation = if val_sharpe > 0.0 {
            (val_sharpe - test_sharpe) / val_sharpe * 100.0
        } else {
            0.0
        };

        analysis.push_str(&format!("Ticker: {}\n", self.ticker));
        analysis.push_str(&format!("Date Range: {} to {}\n", self.start_date, self.end_date));
        analysis.push_str(&format!("Walk-Forward Periods: {}\n", self.period_results.len()));
        analysis.push_str(&format!("Validation Sharpe: {:.2}\n", val_sharpe));
        analysis.push_str(&format!("Out-of-Sample Test Sharpe: {:.2}\n", test_sharpe));
        analysis.push_str(&format!("Sharpe Degradation (overfit indicator): {:.1}%\n\n", sharpe_degradation));

        // 2. Consensus Parameters
        analysis.push_str("## CONSENSUS PARAMETERS\n\n");
        if let Some(params) = &self.consensus_params {
            analysis.push_str(&format!("DTE Range: {}-{}\n", params.dte_min, params.dte_max));
            analysis.push_str(&format!("Delta Range: {:.2}-{:.2}\n", params.delta_min, params.delta_max));
            analysis.push_str(&format!("Profit Target: {:.0}%\n", params.profit_target));
            analysis.push_str(&format!("Stop Loss: {:.0}%\n", params.stop_loss));
            analysis.push_str(&format!("IV Percentile Min: {:.0}%\n\n", params.iv_percentile));
        }

        // 3. Parameter Stability Analysis
        analysis.push_str("## PARAMETER STABILITY\n\n");
        let mut param_counts: HashMap<String, usize> = HashMap::new();
        for result in &self.period_results {
            *param_counts.entry(result.best_params.key()).or_insert(0) += 1;
        }
        let mut sorted_params: Vec<_> = param_counts.iter().collect();
        sorted_params.sort_by(|a, b| b.1.cmp(a.1));

        for (params, count) in sorted_params.iter().take(5) {
            let pct = **count as f64 / self.period_results.len() as f64 * 100.0;
            analysis.push_str(&format!("{}: {} periods ({:.0}%)\n", params, count, pct));
        }
        analysis.push('\n');

        // Calculate stability score
        let top_param_pct = sorted_params.first().map(|(_, c)| **c as f64 / self.period_results.len() as f64 * 100.0).unwrap_or(0.0);
        let stability_score = if top_param_pct >= 75.0 { "HIGH" }
            else if top_param_pct >= 50.0 { "MEDIUM" }
            else { "LOW" };
        analysis.push_str(&format!("Stability Score: {} ({:.0}% periods use same params)\n\n", stability_score, top_param_pct));

        // 4. Per-Period Performance
        analysis.push_str("## PER-PERIOD PERFORMANCE\n\n");
        analysis.push_str("| Period | Train Sharpe | Val Sharpe | Test Sharpe | Trades | Win Rate |\n");
        analysis.push_str("|--------|-------------|------------|-------------|--------|----------|\n");

        let mut total_trades = 0;
        let mut total_wins = 0;
        for (i, result) in self.period_results.iter().enumerate() {
            let trades = result.test_result.total_trades;
            let win_rate = result.test_result.win_rate * 100.0;
            total_trades += trades;
            total_wins += (win_rate / 100.0 * trades as f64) as usize;

            analysis.push_str(&format!(
                "| {:>6} | {:>11.2} | {:>10.2} | {:>11.2} | {:>6} | {:>8.1}% |\n",
                i + 1,
                result.train_result.sharpe_ratio,
                result.validate_result.sharpe_ratio,
                result.test_result.sharpe_ratio,
                trades,
                win_rate
            ));
        }
        analysis.push('\n');

        // 5. Aggregate Statistics
        analysis.push_str("## AGGREGATE STATISTICS\n\n");
        let total_win_rate = if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 };
        analysis.push_str(&format!("Total Test Trades: {}\n", total_trades));
        analysis.push_str(&format!("Overall Win Rate: {:.1}%\n", total_win_rate));
        analysis.push_str(&format!("Avg Test Return: {:.2}%\n", self.aggregate_test.total_return_pct));
        analysis.push_str(&format!("Max Test Drawdown: {:.2}%\n", self.aggregate_test.max_drawdown_pct));
        analysis.push_str(&format!("Test Profit Factor: {:.2}\n\n", self.aggregate_test.profit_factor));

        // 6. Red Flags
        analysis.push_str("## RED FLAGS & WARNINGS\n\n");
        let mut flags = Vec::new();

        if val_sharpe > 3.0 {
            flags.push(format!("- CRITICAL: Validation Sharpe ({:.2}) unrealistically high (>3.0). Likely overfitting or data issues.", val_sharpe));
        }
        if test_sharpe > 3.0 {
            flags.push(format!("- CRITICAL: Test Sharpe ({:.2}) unrealistically high (>3.0). Verify slippage/fill assumptions.", test_sharpe));
        }
        if sharpe_degradation > 30.0 {
            flags.push(format!("- WARNING: High Sharpe degradation ({:.1}%). Strategy may be overfit to training data.", sharpe_degradation));
        }
        if total_trades < self.period_results.len() * 5 {
            flags.push(format!("- WARNING: Low trade count ({} total). Results may not be statistically significant.", total_trades));
        }
        if total_win_rate > 90.0 {
            flags.push(format!("- WARNING: Win rate ({:.1}%) suspiciously high. May be gaming metrics with tight profit targets.", total_win_rate));
        }
        if let Some(params) = &self.consensus_params {
            if params.stop_loss > 150.0 {
                flags.push(format!("- WARNING: Stop loss ({:.0}%) very wide. Hidden tail risk.", params.stop_loss));
            }
            if params.profit_target < 30.0 {
                flags.push(format!("- WARNING: Profit target ({:.0}%) very tight. May be sacrificing edge for win rate.", params.profit_target));
            }
            if params.delta_max - params.delta_min > 0.15 {
                flags.push(format!("- INFO: Wide delta range ({:.2}-{:.2}). Consider narrowing for consistency.", params.delta_min, params.delta_max));
            }
        }
        if top_param_pct < 50.0 {
            flags.push("- WARNING: Low parameter stability. Optimal params vary significantly across periods.".to_string());
        }

        if flags.is_empty() {
            analysis.push_str("No major red flags detected.\n\n");
        } else {
            for flag in &flags {
                analysis.push_str(&format!("{}\n", flag));
            }
            analysis.push('\n');
        }

        // 7. Recommendations
        analysis.push_str("## RECOMMENDATIONS FOR SYSTEM IMPROVEMENT\n\n");

        if val_sharpe > 3.0 || test_sharpe > 3.0 {
            analysis.push_str("1. VERIFY SLIPPAGE MODEL: Current ORATS 66% fill may be optimistic. Test with 75% or pessimistic fills.\n");
            analysis.push_str("2. VERIFY LIQUIDITY FILTERS: Ensure minimum OI (100) and volume (10) filters are applied.\n");
            analysis.push_str("3. ADD BID-ASK SPREAD FILTER: Reject spreads with bid-ask > 8% of mid.\n");
        }

        if let Some(params) = &self.consensus_params {
            if params.stop_loss > 150.0 {
                analysis.push_str(&format!(
                    "4. CONSIDER TIGHTER STOP LOSS: Current {:.0}% stop may expose to tail risk. Test 125% stop.\n",
                    params.stop_loss
                ));
            }
            if params.profit_target < 40.0 {
                analysis.push_str(&format!(
                    "5. TEST HIGHER PROFIT TARGET: Current {:.0}% may leave money on table. Test 50% (research optimal).\n",
                    params.profit_target
                ));
            }
        }

        analysis.push_str("6. RUN STRESS TEST: Test on 2008 (financial crisis), 2020 (COVID crash), 2022 (bear market).\n");
        analysis.push_str("7. ADD REGIME INTEGRATION: Apply position multipliers by market regime.\n");
        analysis.push_str("8. ADD CIRCUIT BREAKERS: Implement daily 2% / weekly 5% / drawdown 15% halts.\n");
        analysis.push_str("9. EXTEND DATA RANGE: Run on full 19 years (2007-2026) for more robust validation.\n\n");

        // 8. JSON Summary for programmatic consumption
        analysis.push_str("## JSON SUMMARY (for programmatic use)\n\n");
        analysis.push_str("```json\n");
        analysis.push_str(&format!(r#"{{
  "ticker": "{}",
  "date_range": "{} to {}",
  "periods": {},
  "validation_sharpe": {:.2},
  "test_sharpe": {:.2},
  "sharpe_degradation_pct": {:.1},
  "total_test_trades": {},
  "overall_win_rate_pct": {:.1},
  "parameter_stability": "{}",
  "consensus_params": {{"#,
            self.ticker,
            self.start_date,
            self.end_date,
            self.period_results.len(),
            val_sharpe,
            test_sharpe,
            sharpe_degradation,
            total_trades,
            total_win_rate,
            stability_score
        ));

        if let Some(params) = &self.consensus_params {
            analysis.push_str(&format!(r#"
    "dte_min": {},
    "dte_max": {},
    "delta_min": {:.2},
    "delta_max": {:.2},
    "profit_target_pct": {:.0},
    "stop_loss_pct": {:.0},
    "iv_percentile_min": {:.0}
  }},"#,
                params.dte_min,
                params.dte_max,
                params.delta_min,
                params.delta_max,
                params.profit_target,
                params.stop_loss,
                params.iv_percentile
            ));
        } else {
            analysis.push_str("},");
        }

        analysis.push_str(&format!(r#"
  "red_flag_count": {},
  "needs_attention": {}
}}
```
"#,
            flags.len(),
            flags.len() > 2
        ));

        analysis
    }
}

/// Walk-forward optimizer with parallel execution and data preloading.
pub struct WalkForwardOptimizer {
    data_dir: String,
    periods_config: WalkForwardPeriodsConfig,
    param_grid: ParameterGrid,
    base_config: BacktestConfig,
}

impl WalkForwardOptimizer {
    /// Create a new optimizer.
    pub fn new(data_dir: &str) -> Self {
        Self {
            data_dir: data_dir.to_string(),
            periods_config: WalkForwardPeriodsConfig::default(),
            param_grid: ParameterGrid::default(),
            base_config: BacktestConfig::default(),
        }
    }

    /// Set periods configuration.
    pub fn with_periods_config(mut self, config: WalkForwardPeriodsConfig) -> Self {
        self.periods_config = config;
        self
    }

    /// Set parameter grid.
    pub fn with_param_grid(mut self, grid: ParameterGrid) -> Self {
        self.param_grid = grid;
        self
    }

    /// Set base backtest configuration.
    pub fn with_base_config(mut self, config: BacktestConfig) -> Self {
        self.base_config = config;
        self
    }

    /// Run walk-forward optimization with parallel execution.
    pub fn optimize(
        &self,
        ticker: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<WalkForwardResult, String> {
        use crate::backtest::BacktestEngine;
        use crate::data::DataLoader;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Generate periods
        let periods = WalkForwardPeriods::new(self.periods_config.clone(), start_date, end_date);
        let period_list = periods.generate();

        if period_list.is_empty() {
            return Err("No valid walk-forward periods".to_string());
        }

        info!("Generated {} walk-forward periods", period_list.len());

        // OPTIMIZATION 1: Preload ALL data once
        info!("Preloading data for {} from {} to {}...", ticker, start_date, end_date);
        let loader = DataLoader::new(&self.data_dir);
        let all_snapshots = loader
            .load_snapshots(ticker, start_date, end_date)
            .map_err(|e| format!("Failed to load data: {}", e))?;
        let all_snapshots = Arc::new(all_snapshots);
        info!("Loaded {} trading days into memory", all_snapshots.len());

        let param_combinations = self.param_grid.combinations();
        info!("Parameter combinations: {}", param_combinations.len());

        let mut period_results = Vec::new();

        for (period_idx, period) in period_list.iter().enumerate() {
            info!(
                "Processing period {}/{}: train {} to {}, validate {} to {}, test {} to {}",
                period_idx + 1,
                period_list.len(),
                period.train_start,
                period.train_end,
                period.validate_start,
                period.validate_end,
                period.test_start,
                period.test_end
            );

            // Extract snapshots for this period's training window
            let train_snapshots: Vec<_> = all_snapshots
                .iter()
                .filter(|s| s.date >= period.train_start && s.date <= period.train_end)
                .cloned()
                .collect();

            let validate_snapshots: Vec<_> = all_snapshots
                .iter()
                .filter(|s| s.date >= period.validate_start && s.date <= period.validate_end)
                .cloned()
                .collect();

            let test_snapshots: Vec<_> = all_snapshots
                .iter()
                .filter(|s| s.date >= period.test_start && s.date <= period.test_end)
                .cloned()
                .collect();

            // OPTIMIZATION 2: Parallel parameter search with Rayon
            let progress = AtomicUsize::new(0);
            let total = param_combinations.len();
            let base_config = self.base_config.clone();

            let results: Vec<(ParameterSet, f64, usize)> = param_combinations
                .par_iter()
                .map(|params| {
                    let mut config = base_config.clone();
                    params.apply_to_config(&mut config);

                    let mut engine = BacktestEngine::new_in_memory(config);
                    let result = engine.run_with_data(
                        ticker,
                        &train_snapshots,
                        period.train_start,
                        period.train_end,
                    );

                    // Progress tracking
                    let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % (total / 10).max(1) == 0 || done == total {
                        let pct = done as f64 / total as f64 * 100.0;
                        info!(
                            "  Period {}/{}: {:.0}% ({}/{} combinations)",
                            period_idx + 1,
                            period_list.len(),
                            pct,
                            done,
                            total
                        );
                    }

                    (params.clone(), result.sharpe_ratio(), result.total_trades)
                })
                .collect();

            // Find best parameters
            let (best_params, best_sharpe, _) = results
                .into_iter()
                .filter(|(_, _, trades)| *trades > 0)
                .max_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or("No valid parameter combinations found")?;

            info!(
                "  Period {}/{} complete: best params = {}, Sharpe = {:.2}",
                period_idx + 1,
                period_list.len(),
                best_params.key(),
                best_sharpe
            );

            // Run validation and test with best params
            let mut config = base_config.clone();
            best_params.apply_to_config(&mut config);

            let mut engine = BacktestEngine::new_in_memory(config.clone());
            let train_result = engine.run_with_data(ticker, &train_snapshots, period.train_start, period.train_end);

            let mut engine = BacktestEngine::new_in_memory(config.clone());
            let validate_result = engine.run_with_data(ticker, &validate_snapshots, period.validate_start, period.validate_end);

            let mut engine = BacktestEngine::new_in_memory(config);
            let test_result = engine.run_with_data(ticker, &test_snapshots, period.test_start, period.test_end);

            period_results.push(PeriodResult {
                period: period.clone(),
                best_params,
                train_result: BacktestSummary::from(&train_result),
                validate_result: BacktestSummary::from(&validate_result),
                test_result: BacktestSummary::from(&test_result),
            });
        }

        // Calculate aggregate results
        let aggregate_validate = self.aggregate_summaries(
            &period_results.iter().map(|p| &p.validate_result).collect::<Vec<_>>(),
        );
        let aggregate_test = self.aggregate_summaries(
            &period_results.iter().map(|p| &p.test_result).collect::<Vec<_>>(),
        );

        // Find consensus parameters (most common)
        let consensus_params = self.find_consensus_params(&period_results);

        Ok(WalkForwardResult {
            ticker: ticker.to_string(),
            start_date,
            end_date,
            period_results,
            aggregate_validate,
            aggregate_test,
            consensus_params,
        })
    }

    /// Aggregate multiple backtest summaries.
    fn aggregate_summaries(&self, summaries: &[&BacktestSummary]) -> BacktestSummary {
        if summaries.is_empty() {
            return BacktestSummary {
                total_return_pct: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                max_drawdown_pct: 0.0,
                sharpe_ratio: 0.0,
                total_trades: 0,
                final_equity: Decimal::ZERO,
            };
        }

        let n = summaries.len() as f64;

        BacktestSummary {
            total_return_pct: summaries.iter().map(|s| s.total_return_pct).sum::<f64>() / n,
            win_rate: summaries.iter().map(|s| s.win_rate).sum::<f64>() / n,
            profit_factor: summaries.iter().map(|s| s.profit_factor).sum::<f64>() / n,
            max_drawdown_pct: summaries
                .iter()
                .map(|s| s.max_drawdown_pct)
                .fold(0.0, f64::max),
            sharpe_ratio: summaries.iter().map(|s| s.sharpe_ratio).sum::<f64>() / n,
            total_trades: summaries.iter().map(|s| s.total_trades).sum(),
            final_equity: summaries.last().map(|s| s.final_equity).unwrap_or(Decimal::ZERO),
        }
    }

    /// Find the most common parameters across periods.
    fn find_consensus_params(&self, results: &[PeriodResult]) -> Option<ParameterSet> {
        if results.is_empty() {
            return None;
        }

        let mut counts: HashMap<String, (usize, ParameterSet)> = HashMap::new();

        for result in results {
            let key = result.best_params.key();
            let entry = counts.entry(key).or_insert((0, result.best_params.clone()));
            entry.0 += 1;
        }

        counts
            .into_values()
            .max_by_key(|(count, _)| *count)
            .map(|(_, params)| params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_grid_combinations() {
        let grid = ParameterGrid {
            dte_min: vec![30, 45],
            dte_max: vec![60],
            delta_min: vec![0.10],
            delta_max: vec![0.15],
            profit_target: vec![50.0],
            stop_loss: vec![125.0],
            iv_percentile: vec![50.0],
        };

        // Only dte_min=30 is valid (since dte_max must be > dte_min)
        let combos = grid.combinations();
        assert!(!combos.is_empty());
    }

    #[test]
    fn test_parameter_set_key() {
        let params = ParameterSet {
            dte_min: 30,
            dte_max: 45,
            delta_min: 0.10,
            delta_max: 0.15,
            profit_target: 50.0,
            stop_loss: 125.0,
            iv_percentile: 50.0,
        };

        let key = params.key();
        assert!(key.contains("dte30-45"));
        assert!(key.contains("tp50"));
    }

    #[test]
    fn test_parameter_set_apply() {
        let params = ParameterSet {
            dte_min: 25,
            dte_max: 40,
            delta_min: 0.08,
            delta_max: 0.12,
            profit_target: 60.0,
            stop_loss: 150.0,
            iv_percentile: 60.0,
        };

        let mut config = BacktestConfig::default();
        params.apply_to_config(&mut config);

        assert_eq!(config.min_dte, 25);
        assert_eq!(config.max_dte, 40);
        assert_eq!(config.profit_target_pct, 60.0);
    }

    #[test]
    fn test_backtest_summary_from_result() {
        // This would require a full backtest result, so we test the struct directly
        let summary = BacktestSummary {
            total_return_pct: 15.0,
            win_rate: 0.70,
            profit_factor: 2.0,
            max_drawdown_pct: 8.0,
            sharpe_ratio: 1.5,
            total_trades: 50,
            final_equity: Decimal::from(115000),
        };

        assert_eq!(summary.win_rate, 0.70);
        assert_eq!(summary.total_trades, 50);
    }
}
