//! Core backtesting engine.
//!
//! Runs the simulation loop:
//! 1. Load daily market data
//! 2. Check circuit breakers
//! 3. Classify market regime
//! 4. Check for exit conditions on open positions
//! 5. Update mark-to-market values
//! 6. Screen for new trade candidates (with IV percentile filter)
//! 7. Enter new positions (with regime position sizing)
//! 8. Record daily equity

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::analytics::{IVTermStructureAnalyzer, SpreadScreener, SpreadScreenerConfig};
use crate::data::{DataLoader, LoaderError, OptionsSnapshot};
use crate::regime::{DailyMarketData, MarketRegime, RegimeClassifier, RegimeClassifierConfig};
use crate::risk::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStatus};

use super::commission::CommissionModel;
use super::slippage::SlippageModel;
use super::trade::{CreditSpreadBuilder, ExitReason, Position, Trade};

/// Configuration for backtest execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Starting equity.
    pub initial_equity: Decimal,

    /// Profit target as percentage of credit (e.g., 50.0 = 50%).
    pub profit_target_pct: f64,

    /// Stop loss as percentage of credit (e.g., 125.0 = 125%).
    pub stop_loss_pct: f64,

    /// Time-based exit DTE (e.g., 21 = exit at 21 DTE).
    pub time_exit_dte: i32,

    /// Minimum DTE for new trades.
    pub min_dte: i32,

    /// Maximum DTE for new trades.
    pub max_dte: i32,

    /// Minimum delta for short strike (absolute value).
    pub min_delta: f64,

    /// Maximum delta for short strike (absolute value).
    pub max_delta: f64,

    /// Minimum IV percentile for entry.
    pub min_iv_percentile: f64,

    /// Maximum trades per day.
    pub max_trades_per_day: usize,

    /// Maximum concurrent positions.
    pub max_positions: usize,

    /// Maximum risk per trade as percentage of equity.
    pub max_risk_per_trade_pct: f64,

    /// Maximum total portfolio risk as percentage of equity.
    pub max_portfolio_risk_pct: f64,

    /// Slippage model.
    #[serde(default)]
    pub slippage: SlippageModel,

    /// Commission model.
    #[serde(default)]
    pub commission: CommissionModel,

    /// Enable IV percentile entry filter.
    #[serde(default = "default_true")]
    pub use_iv_percentile_filter: bool,

    /// Enable market regime position sizing.
    #[serde(default = "default_true")]
    pub use_regime_sizing: bool,

    /// Enable circuit breakers.
    #[serde(default = "default_true")]
    pub use_circuit_breakers: bool,

    /// Circuit breaker configuration.
    #[serde(default)]
    pub circuit_breaker_config: CircuitBreakerConfig,
}

fn default_true() -> bool {
    true
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_equity: Decimal::from(100_000),
            profit_target_pct: 50.0,
            stop_loss_pct: 125.0,
            time_exit_dte: 21,
            min_dte: 30,
            max_dte: 45,
            min_delta: 0.20,
            max_delta: 0.30,
            min_iv_percentile: 50.0,
            max_trades_per_day: 1,
            max_positions: 10,
            max_risk_per_trade_pct: 2.0,
            max_portfolio_risk_pct: 10.0,
            slippage: SlippageModel::default(),
            commission: CommissionModel::default(),
            use_iv_percentile_filter: true,
            use_regime_sizing: true,
            use_circuit_breakers: true,
            circuit_breaker_config: CircuitBreakerConfig::default(),
        }
    }
}

/// Daily equity snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub date: NaiveDate,
    pub equity: Decimal,
    pub cash: Decimal,
    pub positions_value: Decimal,
    pub open_positions: usize,
    pub daily_pnl: Decimal,
}

/// Result of a completed backtest.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Configuration used.
    pub config: BacktestConfig,

    /// Ticker(s) tested.
    pub tickers: Vec<String>,

    /// Start date.
    pub start_date: NaiveDate,

    /// End date.
    pub end_date: NaiveDate,

    /// All completed trades.
    pub trades: Vec<Trade>,

    /// Daily equity curve.
    pub equity_curve: Vec<EquityPoint>,

    /// Final equity.
    pub final_equity: Decimal,

    /// Total return percentage.
    pub total_return_pct: f64,

    /// Number of trading days.
    pub trading_days: usize,

    /// Peak equity (for drawdown calculation).
    pub peak_equity: Decimal,

    /// Maximum drawdown.
    pub max_drawdown: Decimal,

    /// Maximum drawdown percentage.
    pub max_drawdown_pct: f64,

    /// Total trades.
    pub total_trades: usize,

    /// Winning trades.
    pub winning_trades: usize,

    /// Losing trades.
    pub losing_trades: usize,

    /// Total P&L.
    pub total_pnl: Decimal,

    /// Gross profit.
    pub gross_profit: Decimal,

    /// Gross loss.
    pub gross_loss: Decimal,

    /// Total commission paid.
    pub total_commission: Decimal,
}

impl BacktestResult {
    /// Calculate win rate.
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        self.winning_trades as f64 / self.total_trades as f64
    }

    /// Calculate profit factor.
    pub fn profit_factor(&self) -> f64 {
        let loss: f64 = self.gross_loss.abs().try_into().unwrap_or(0.0);
        if loss == 0.0 {
            return f64::INFINITY;
        }
        let profit: f64 = self.gross_profit.try_into().unwrap_or(0.0);
        profit / loss
    }

    /// Calculate average trade P&L.
    pub fn avg_trade_pnl(&self) -> Decimal {
        if self.total_trades == 0 {
            return Decimal::ZERO;
        }
        self.total_pnl / Decimal::from(self.total_trades as i64)
    }

    /// Calculate average winner.
    pub fn avg_winner(&self) -> Decimal {
        if self.winning_trades == 0 {
            return Decimal::ZERO;
        }
        self.gross_profit / Decimal::from(self.winning_trades as i64)
    }

    /// Calculate average loser.
    pub fn avg_loser(&self) -> Decimal {
        if self.losing_trades == 0 {
            return Decimal::ZERO;
        }
        self.gross_loss / Decimal::from(self.losing_trades as i64)
    }

    /// Calculate Sharpe ratio (simplified, assuming risk-free rate = 0).
    pub fn sharpe_ratio(&self) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        // Calculate daily returns
        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| {
                let prev: f64 = w[0].equity.try_into().unwrap_or(1.0);
                let curr: f64 = w[1].equity.try_into().unwrap_or(1.0);
                (curr - prev) / prev
            })
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Annualize: 252 trading days
        (mean * 252.0_f64.sqrt()) / std_dev
    }

    /// Generate summary string.
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results ({} to {})\n\
             ----------------------------------------\n\
             Total Return: {:.2}%\n\
             Final Equity: ${:.2}\n\
             Max Drawdown: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             \n\
             Trades: {} (W: {}, L: {})\n\
             Win Rate: {:.1}%\n\
             Profit Factor: {:.2}\n\
             Avg Trade: ${:.2}\n\
             Avg Winner: ${:.2}\n\
             Avg Loser: ${:.2}\n\
             \n\
             Total Commission: ${:.2}",
            self.start_date,
            self.end_date,
            self.total_return_pct,
            self.final_equity,
            self.max_drawdown_pct,
            self.sharpe_ratio(),
            self.total_trades,
            self.winning_trades,
            self.losing_trades,
            self.win_rate() * 100.0,
            self.profit_factor(),
            self.avg_trade_pnl(),
            self.avg_winner(),
            self.avg_loser(),
            self.total_commission,
        )
    }
}

/// The main backtesting engine.
pub struct BacktestEngine {
    config: BacktestConfig,
    loader: DataLoader,
    equity: Decimal,
    cash: Decimal,
    positions: Vec<Position>,
    closed_trades: Vec<Trade>,
    equity_curve: Vec<EquityPoint>,
    peak_equity: Decimal,
    max_drawdown: Decimal,
    total_commission: Decimal,
    // Production system integrations
    iv_analyzer: IVTermStructureAnalyzer,
    regime_classifier: RegimeClassifier,
    circuit_breaker: CircuitBreaker,
    current_regime: MarketRegime,
    current_iv_percentile: Option<f64>,
    // Tracking
    circuit_breaker_halts: usize,
    regime_reductions: usize,
    iv_filter_skips: usize,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(config: BacktestConfig, data_dir: &str) -> Self {
        let equity = config.initial_equity;
        let circuit_breaker_config = config.circuit_breaker_config.clone();
        Self {
            config,
            loader: DataLoader::new(data_dir),
            equity,
            cash: equity,
            positions: Vec::new(),
            closed_trades: Vec::new(),
            equity_curve: Vec::new(),
            peak_equity: equity,
            max_drawdown: Decimal::ZERO,
            total_commission: Decimal::ZERO,
            iv_analyzer: IVTermStructureAnalyzer::new(),
            regime_classifier: RegimeClassifier::new(RegimeClassifierConfig::default()),
            circuit_breaker: CircuitBreaker::new(circuit_breaker_config, equity),
            current_regime: MarketRegime::Unknown,
            current_iv_percentile: None,
            circuit_breaker_halts: 0,
            regime_reductions: 0,
            iv_filter_skips: 0,
        }
    }

    /// Run backtest for a single ticker.
    pub fn run(
        &mut self,
        ticker: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<BacktestResult, LoaderError> {
        // Load snapshots
        let snapshots = self.loader.load_snapshots(ticker, start_date, end_date)?;
        Ok(self.run_with_data(ticker, &snapshots, start_date, end_date))
    }

    /// Run backtest with pre-loaded data (for optimization - avoids re-reading files).
    pub fn run_with_data(
        &mut self,
        ticker: &str,
        snapshots: &[OptionsSnapshot],
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> BacktestResult {
        // Reset state
        self.equity = self.config.initial_equity;
        self.cash = self.config.initial_equity;
        self.positions.clear();
        self.closed_trades.clear();
        self.equity_curve.clear();
        self.peak_equity = self.config.initial_equity;
        self.max_drawdown = Decimal::ZERO;
        self.total_commission = Decimal::ZERO;

        // Reset production integration state
        self.iv_analyzer = IVTermStructureAnalyzer::new();
        self.regime_classifier = RegimeClassifier::new(RegimeClassifierConfig::default());
        self.circuit_breaker = CircuitBreaker::new(
            self.config.circuit_breaker_config.clone(),
            self.config.initial_equity,
        );
        self.current_regime = MarketRegime::Unknown;
        self.current_iv_percentile = None;
        self.circuit_breaker_halts = 0;
        self.regime_reductions = 0;
        self.iv_filter_skips = 0;

        for snapshot in snapshots {
            self.process_day(snapshot);
        }

        // Close any remaining positions at end of period
        self.close_remaining_positions(end_date);

        // Build result
        self.build_result(vec![ticker.to_string()], start_date, end_date)
    }

    /// Create engine without data loader (for in-memory backtesting).
    pub fn new_in_memory(config: BacktestConfig) -> Self {
        let equity = config.initial_equity;
        let circuit_breaker_config = config.circuit_breaker_config.clone();
        Self {
            config,
            loader: DataLoader::new(""), // Unused for in-memory
            equity,
            cash: equity,
            positions: Vec::new(),
            closed_trades: Vec::new(),
            equity_curve: Vec::new(),
            peak_equity: equity,
            max_drawdown: Decimal::ZERO,
            total_commission: Decimal::ZERO,
            iv_analyzer: IVTermStructureAnalyzer::new(),
            regime_classifier: RegimeClassifier::new(RegimeClassifierConfig::default()),
            circuit_breaker: CircuitBreaker::new(circuit_breaker_config, equity),
            current_regime: MarketRegime::Unknown,
            current_iv_percentile: None,
            circuit_breaker_halts: 0,
            regime_reductions: 0,
            iv_filter_skips: 0,
        }
    }

    /// Process a single trading day.
    fn process_day(&mut self, snapshot: &OptionsSnapshot) {
        let date = snapshot.date;

        // 1. Update IV analysis and percentile
        let iv_structure = self.iv_analyzer.analyze(snapshot);
        if let Some(atm_iv) = iv_structure.atm_iv {
            self.current_iv_percentile = self.iv_analyzer.iv_percentile(atm_iv);
        }

        // 2. Update market regime classification
        // Note: In production, VIX comes from separate data source
        // Here we approximate from ATM IV (multiply by 100 for VIX-like scale)
        let estimated_vix = iv_structure.atm_iv.unwrap_or(0.20) * 100.0;
        let market_data = DailyMarketData {
            date,
            vix: estimated_vix,
            price: snapshot.underlying_price,
        };
        self.current_regime = self.regime_classifier.classify(market_data);

        // 3. Update circuit breaker with current equity and VIX
        if self.config.use_circuit_breakers {
            self.circuit_breaker.update(date, self.equity, Some(estimated_vix));
        }

        // 4. Update MTM for all open positions
        self.update_positions_mtm(snapshot);

        // 5. Check exit conditions
        self.check_exits(snapshot);

        // 6. Screen for new trades (with IV percentile, regime, and circuit breaker checks)
        self.screen_and_enter(snapshot);

        // 7. Calculate daily equity
        let positions_value: Decimal = self
            .positions
            .iter()
            .filter(|p| p.is_open())
            .map(|p| p.unrealized_pnl())
            .sum();

        self.equity = self.cash + positions_value;

        // Track peak and drawdown
        if self.equity > self.peak_equity {
            self.peak_equity = self.equity;
        }
        let drawdown = self.peak_equity - self.equity;
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }

        // Calculate daily P&L
        let prev_equity = self
            .equity_curve
            .last()
            .map(|e| e.equity)
            .unwrap_or(self.config.initial_equity);
        let daily_pnl = self.equity - prev_equity;

        // Record equity point
        self.equity_curve.push(EquityPoint {
            date,
            equity: self.equity,
            cash: self.cash,
            positions_value,
            open_positions: self.positions.iter().filter(|p| p.is_open()).count(),
            daily_pnl,
        });
    }

    /// Update mark-to-market for all open positions.
    fn update_positions_mtm(&mut self, snapshot: &OptionsSnapshot) {
        // Collect all quotes into a flat list
        let quotes: Vec<_> = snapshot
            .chains
            .iter()
            .flat_map(|c| c.calls.iter().chain(c.puts.iter()))
            .collect();

        for position in &mut self.positions {
            if position.is_open() {
                position.update_mtm(&quotes);
            }
        }
    }

    /// Check and process exit conditions.
    fn check_exits(&mut self, snapshot: &OptionsSnapshot) {
        let date = snapshot.date;

        for position in &mut self.positions {
            if !position.is_open() {
                continue;
            }

            // Check exit conditions
            let exit_reason = if position.is_expired(date) {
                // Check if ITM or OTM at expiration
                if let Some(expiration) = position.expiration() {
                    if expiration <= date {
                        // Simplified: check if short strike is breached
                        let short_leg = position.legs.iter().find(|l| l.is_short());
                        if let Some(leg) = short_leg {
                            if snapshot.underlying_price < leg.strike {
                                Some(ExitReason::ExpiredITM)
                            } else {
                                Some(ExitReason::ExpiredWorthless)
                            }
                        } else {
                            Some(ExitReason::ExpiredWorthless)
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else if position.is_profit_target_hit(self.config.profit_target_pct) {
                Some(ExitReason::ProfitTarget)
            } else if position.is_stop_loss_hit(self.config.stop_loss_pct) {
                Some(ExitReason::StopLoss)
            } else if position.is_time_exit(date, self.config.time_exit_dte) {
                Some(ExitReason::TimeExit)
            } else {
                None
            };

            if let Some(reason) = exit_reason {
                // Calculate exit commission
                let exit_commission = self
                    .config
                    .commission
                    .calculate(position.total_contracts(), position.num_legs())
                    .total;
                self.total_commission += exit_commission;

                // Realize the P&L
                let pnl = position.unrealized_pnl() - exit_commission;
                self.cash += pnl;

                // Close the position
                position.close(date, reason, exit_commission);

                // Record the trade
                if let Some(trade) = Trade::from_position(position.clone()) {
                    self.closed_trades.push(trade);
                }
            }
        }

        // Remove closed positions
        self.positions.retain(|p| p.is_open());
    }

    /// Screen for new trades and enter positions.
    /// Integrates: circuit breakers, IV percentile filter, regime position sizing
    fn screen_and_enter(&mut self, snapshot: &OptionsSnapshot) {
        // === CIRCUIT BREAKER CHECK (Production: hard halt) ===
        if self.config.use_circuit_breakers {
            let status = self.circuit_breaker.status();
            if !status.allows_new_positions() {
                self.circuit_breaker_halts += 1;
                return;
            }
        }

        // === IV PERCENTILE FILTER (Production: >= 50%) ===
        if self.config.use_iv_percentile_filter {
            if let Some(iv_pct) = self.current_iv_percentile {
                if iv_pct < self.config.min_iv_percentile {
                    self.iv_filter_skips += 1;
                    return;
                }
            }
            // If IV percentile not available (insufficient history), allow trading
        }

        // === REGIME CHECK (Production: Crisis = no new positions) ===
        if self.config.use_regime_sizing {
            let multiplier = self.current_regime.position_size_multiplier();
            if multiplier == 0.0 {
                // Crisis regime - no new positions
                self.regime_reductions += 1;
                return;
            }
        }

        // Check if we can open new positions
        let open_count = self.positions.iter().filter(|p| p.is_open()).count();
        if open_count >= self.config.max_positions {
            return;
        }

        // Check how many trades we've done today
        let today_trades = self
            .closed_trades
            .iter()
            .filter(|t| t.position.entry_date == snapshot.date)
            .count()
            + self
                .positions
                .iter()
                .filter(|p| p.entry_date == snapshot.date)
                .count();

        if today_trades >= self.config.max_trades_per_day {
            return;
        }

        // Create spread screener from config
        let screener_config = SpreadScreenerConfig {
            min_dte: self.config.min_dte,
            max_dte: self.config.max_dte,
            min_short_delta: self.config.min_delta,
            max_short_delta: self.config.max_delta,
            min_iv_percentile: self.config.min_iv_percentile,
            ..Default::default()
        };

        let screener = SpreadScreener::new(screener_config);

        // Screen for put credit spreads (primary strategy)
        // Pass IV percentile for additional scoring
        let candidates = screener.screen_put_spreads(snapshot, self.current_iv_percentile);

        if let Some(best) = screener.best_candidate(&candidates) {
            // Check risk limits
            let max_loss: f64 = best.max_loss.try_into().unwrap_or(0.0);
            let equity: f64 = self.equity.try_into().unwrap_or(1.0);
            let risk_pct = max_loss / equity * 100.0;

            if risk_pct > self.config.max_risk_per_trade_pct {
                return;
            }

            // Calculate total portfolio risk
            let current_risk: f64 = self
                .positions
                .iter()
                .filter(|p| p.is_open())
                .map(|p| {
                    let ml: f64 = p.max_loss.try_into().unwrap_or(0.0);
                    ml
                })
                .sum();

            if (current_risk + max_loss) / equity * 100.0 > self.config.max_portfolio_risk_pct {
                return;
            }

            // === REGIME-BASED POSITION SIZING (Production: 0.25x to 1.0x) ===
            let mut contracts = 1;
            if self.config.use_regime_sizing {
                let multiplier = self.current_regime.position_size_multiplier();
                if multiplier < 1.0 {
                    self.regime_reductions += 1;
                }
                // For now, we use 1 contract minimum. In production, this would scale.
                // A more sophisticated approach would adjust contracts based on multiplier.
                contracts = if multiplier >= 0.5 { 1 } else { 1 }; // Minimum 1 contract
            }

            // Apply slippage to get fill prices (2-leg spread)
            let slippage = self.config.slippage.for_legs(2);

            // For the short leg, we're selling so use sell_fill
            // Approximate bid-ask spread from available data
            let short_bid = best.short_bid;
            let short_ask = short_bid * Decimal::from(110) / Decimal::from(100); // ~10% spread estimate
            let short_fill = slippage.sell_fill(short_bid, short_ask);

            // For the long leg, we're buying so use buy_fill
            let long_ask = best.long_ask;
            let long_bid = long_ask * Decimal::from(90) / Decimal::from(100); // ~10% spread estimate
            let long_fill = slippage.buy_fill(long_bid, long_ask);

            // Calculate entry commission
            let commission = self.config.commission.calculate(contracts * 2, 2); // contracts * 2 legs
            let entry_commission = commission.total;

            // Build position
            let position = CreditSpreadBuilder::put_credit_spread(&snapshot.ticker, snapshot.date)
                .stock_price(snapshot.underlying_price)
                .short_leg(
                    best.short_strike,
                    Decimal::try_from(short_fill).unwrap_or(best.short_bid),
                    -best.short_delta,
                )
                .long_leg(
                    best.long_strike,
                    Decimal::try_from(long_fill).unwrap_or(best.long_ask),
                    -best.long_delta,
                )
                .expiration(best.expiration, best.dte)
                .contracts(contracts)
                .commission(entry_commission)
                .build();

            // Add position
            self.add_position(position);
        }
    }

    /// Close any remaining positions at end of backtest.
    fn close_remaining_positions(&mut self, end_date: NaiveDate) {
        for position in &mut self.positions {
            if position.is_open() {
                let exit_commission = self
                    .config
                    .commission
                    .calculate(position.total_contracts(), position.num_legs())
                    .total;
                self.total_commission += exit_commission;

                let pnl = position.unrealized_pnl() - exit_commission;
                self.cash += pnl;

                position.close(end_date, ExitReason::EndOfPeriod, exit_commission);

                if let Some(trade) = Trade::from_position(position.clone()) {
                    self.closed_trades.push(trade);
                }
            }
        }
        self.positions.clear();
    }

    /// Build the final backtest result.
    fn build_result(
        &self,
        tickers: Vec<String>,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> BacktestResult {
        let initial: f64 = self.config.initial_equity.try_into().unwrap_or(1.0);
        let final_eq: f64 = self.equity.try_into().unwrap_or(1.0);
        let total_return_pct = (final_eq - initial) / initial * 100.0;

        let peak: f64 = self.peak_equity.try_into().unwrap_or(1.0);
        let max_dd: f64 = self.max_drawdown.try_into().unwrap_or(0.0);
        let max_drawdown_pct = if peak > 0.0 { max_dd / peak * 100.0 } else { 0.0 };

        let winning_trades = self.closed_trades.iter().filter(|t| t.is_winner()).count();
        let losing_trades = self.closed_trades.len() - winning_trades;

        let total_pnl: Decimal = self.closed_trades.iter().map(|t| t.pnl()).sum();
        let gross_profit: Decimal = self
            .closed_trades
            .iter()
            .filter(|t| t.is_winner())
            .map(|t| t.pnl())
            .sum();
        let gross_loss: Decimal = self
            .closed_trades
            .iter()
            .filter(|t| !t.is_winner())
            .map(|t| t.pnl())
            .sum();

        BacktestResult {
            config: self.config.clone(),
            tickers,
            start_date,
            end_date,
            trades: self.closed_trades.clone(),
            equity_curve: self.equity_curve.clone(),
            final_equity: self.equity,
            total_return_pct,
            trading_days: self.equity_curve.len(),
            peak_equity: self.peak_equity,
            max_drawdown: self.max_drawdown,
            max_drawdown_pct,
            total_trades: self.closed_trades.len(),
            winning_trades,
            losing_trades,
            total_pnl,
            gross_profit,
            gross_loss,
            total_commission: self.total_commission,
        }
    }

    /// Add a position manually (for testing or external entry logic).
    pub fn add_position(&mut self, position: Position) {
        self.total_commission += position.entry_commission;
        self.cash -= position.entry_commission;
        self.positions.push(position);
    }

    /// Get current open positions.
    pub fn open_positions(&self) -> Vec<&Position> {
        self.positions.iter().filter(|p| p.is_open()).collect()
    }

    /// Get current equity.
    pub fn current_equity(&self) -> Decimal {
        self.equity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_default_config() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_equity, dec!(100_000));
        assert_eq!(config.profit_target_pct, 50.0);
        assert_eq!(config.stop_loss_pct, 125.0);
        assert_eq!(config.time_exit_dte, 21);
    }

    #[test]
    fn test_backtest_result_calculations() {
        let config = BacktestConfig::default();
        let result = BacktestResult {
            config,
            tickers: vec!["SPY".to_string()],
            start_date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
            trades: vec![],
            equity_curve: vec![],
            final_equity: dec!(110_000),
            total_return_pct: 10.0,
            trading_days: 252,
            peak_equity: dec!(115_000),
            max_drawdown: dec!(5_000),
            max_drawdown_pct: 4.35,
            total_trades: 100,
            winning_trades: 70,
            losing_trades: 30,
            total_pnl: dec!(10_000),
            gross_profit: dec!(15_000),
            gross_loss: dec!(-5_000),
            total_commission: dec!(400),
        };

        assert_eq!(result.win_rate(), 0.7);
        assert_eq!(result.profit_factor(), 3.0); // 15000 / 5000
        assert_eq!(result.avg_trade_pnl(), dec!(100)); // 10000 / 100
    }
}
