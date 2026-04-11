use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::analytics::{IVTermStructureAnalyzer, SpreadScreener, SpreadScreenerConfig};
use crate::backtest::{ExitReason, Position, SpreadType};
use crate::engine_core::events::{FillEvent, LegIntent, OrderAction, OrderIntent, OrderSide, PortfolioView};
use crate::data::{OptionType, OptionsSnapshot};

use super::Strategy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutSpreadConfig {
    pub profit_target_pct: f64,
    pub stop_loss_pct: f64,
    pub time_exit_dte: i32,
    pub screener: SpreadScreenerConfig,
    pub max_positions: usize,
    pub max_trades_per_day: usize,
    pub min_iv_percentile: f64,
    pub use_iv_filter: bool,
}

impl Default for PutSpreadConfig {
    fn default() -> Self {
        Self {
            profit_target_pct: 50.0,
            stop_loss_pct: 200.0,
            time_exit_dte: 21,
            screener: SpreadScreenerConfig::default(),
            max_positions: 10,
            max_trades_per_day: 3,
            min_iv_percentile: 50.0,
            use_iv_filter: true,
        }
    }
}

pub struct PutCreditSpreadStrategy {
    config: PutSpreadConfig,
    screener: SpreadScreener,
    iv_analyzer: IVTermStructureAnalyzer,
    trades_today: usize,
    last_trade_date: Option<NaiveDate>,
}

impl PutCreditSpreadStrategy {
    pub fn new(config: PutSpreadConfig) -> Self {
        let screener = SpreadScreener::new(config.screener.clone());
        Self {
            config,
            screener,
            iv_analyzer: IVTermStructureAnalyzer::new(),
            trades_today: 0,
            last_trade_date: None,
        }
    }

    fn check_exits(&self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) -> Vec<OrderIntent> {
        let mut orders = Vec::new();
        for position in &portfolio.open_positions {
            if let Some(reason) = self.should_exit(position, snapshot.date) {
                orders.push(OrderIntent::exit_position(position.id, reason));
            }
        }
        orders
    }

    fn should_exit(&self, position: &Position, current_date: NaiveDate) -> Option<ExitReason> {
        if position.is_expired(current_date) {
            if position.unrealized_pnl() < Decimal::ZERO {
                return Some(ExitReason::ExpiredITM);
            }
            return Some(ExitReason::ExpiredWorthless);
        }

        if position.is_profit_target_hit(self.config.profit_target_pct) {
            return Some(ExitReason::ProfitTarget);
        }

        if position.is_stop_loss_hit(self.config.stop_loss_pct) {
            return Some(ExitReason::StopLoss);
        }

        if position.is_time_exit(current_date, self.config.time_exit_dte) {
            return Some(ExitReason::TimeExit);
        }

        None
    }

    fn check_entries(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) -> Vec<OrderIntent> {
        if self.last_trade_date != Some(snapshot.date) {
            self.trades_today = 0;
            self.last_trade_date = Some(snapshot.date);
        }

        if portfolio.open_positions.len() >= self.config.max_positions {
            return vec![];
        }

        if self.trades_today >= self.config.max_trades_per_day {
            return vec![];
        }

        let iv_structure = self.iv_analyzer.analyze(snapshot);

        // Only apply IV filter once we have enough history for a meaningful percentile.
        // With < 20 days of history, iv_percentile() returns 0.0 which would incorrectly
        // block all entries during the warm-up period.
        if self.config.use_iv_filter && self.iv_analyzer.history_length() >= 20 {
            if let Some(atm_iv) = iv_structure.atm_iv {
                if let Some(pct) = self.iv_analyzer.iv_percentile(atm_iv) {
                    if pct < self.config.min_iv_percentile {
                        return vec![];
                    }
                }
            }
        }

        // Only pass IV percentile if we have enough history for a meaningful estimate.
        // With sparse history, iv_percentile() returns a misleadingly low value that
        // would incorrectly block entries.
        let iv_percentile = if self.iv_analyzer.history_length() >= 20 {
            iv_structure.atm_iv.and_then(|iv| self.iv_analyzer.iv_percentile(iv))
        } else {
            None
        };
        let candidates = self.screener.screen_put_spreads(snapshot, iv_percentile);

        if let Some(best) = self.screener.best_candidate(&candidates) {
            self.trades_today += 1;

            let order = OrderIntent::enter_spread(
                snapshot.ticker.clone(),
                SpreadType::PutCreditSpread,
                vec![
                    LegIntent {
                        option_type: OptionType::Put,
                        strike: best.short_strike,
                        expiration: best.expiration,
                        contracts: -1,
                        side: OrderSide::Sell,
                    },
                    LegIntent {
                        option_type: OptionType::Put,
                        strike: best.long_strike,
                        expiration: best.expiration,
                        contracts: 1,
                        side: OrderSide::Buy,
                    },
                ],
            );

            return vec![order];
        }

        vec![]
    }
}

impl Strategy for PutCreditSpreadStrategy {
    fn on_snapshot(&mut self, snapshot: &OptionsSnapshot, portfolio: &PortfolioView) -> Vec<OrderIntent> {
        let mut orders = Vec::new();
        orders.extend(self.check_exits(snapshot, portfolio));
        orders.extend(self.check_entries(snapshot, portfolio));
        orders
    }

    fn on_fill(&mut self, _fill: &FillEvent) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::engine_core::events::{OrderAction, PortfolioView};
    use crate::data::{OptionType, OptionQuote, OptionsChain, OptionsSnapshot, Greeks};
    use crate::risk::PortfolioGreeks;

    fn make_tradeable_snapshot() -> OptionsSnapshot {
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();
        let mut snapshot = OptionsSnapshot::new(date, "SPY".to_string(), dec!(480));
        let mut chain = OptionsChain::new(exp, 32);

        // 470 short: delta -0.25 (in 0.20-0.30 range), bid-ask 3.8% of mid, passes all filters.
        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(), trade_date: date, expiration: exp, dte: 32,
            strike: dec!(470), option_type: OptionType::Put, stock_price: dec!(480),
            bid: dec!(2.55), ask: dec!(2.65), mid: dec!(2.60),
            theoretical_value: dec!(2.60), volume: 1000, open_interest: 5000,
            bid_iv: 0.18, mid_iv: 0.19, ask_iv: 0.20, smv_vol: 0.19,
            greeks: Greeks { delta: -0.25, gamma: 0.02, theta: -0.05, vega: 0.10, rho: 0.01 },
            residual_rate: 0.05,
        });

        // 465 long: bid-ask 7.1% of mid (< 8%), credit = 2.55 - 1.45 = 1.10 = 22% of $5 width.
        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(), trade_date: date, expiration: exp, dte: 32,
            strike: dec!(465), option_type: OptionType::Put, stock_price: dec!(480),
            bid: dec!(1.35), ask: dec!(1.45), mid: dec!(1.40),
            theoretical_value: dec!(1.40), volume: 800, open_interest: 4000,
            bid_iv: 0.19, mid_iv: 0.20, ask_iv: 0.21, smv_vol: 0.20,
            greeks: Greeks { delta: -0.15, gamma: 0.015, theta: -0.03, vega: 0.08, rho: 0.008 },
            residual_rate: 0.05,
        });

        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(), trade_date: date, expiration: exp, dte: 32,
            strike: dec!(460), option_type: OptionType::Put, stock_price: dec!(480),
            bid: dec!(0.88), ask: dec!(0.94), mid: dec!(0.91),
            theoretical_value: dec!(0.91), volume: 600, open_interest: 3000,
            bid_iv: 0.20, mid_iv: 0.21, ask_iv: 0.22, smv_vol: 0.21,
            greeks: Greeks { delta: -0.08, gamma: 0.01, theta: -0.02, vega: 0.06, rho: 0.005 },
            residual_rate: 0.05,
        });

        snapshot.chains.push(chain);
        snapshot
    }

    fn make_empty_portfolio() -> PortfolioView {
        PortfolioView {
            equity: dec!(100_000),
            cash: dec!(100_000),
            open_positions: vec![],
            aggregate_greeks: PortfolioGreeks::default(),
            current_drawdown_pct: 0.0,
            peak_equity: dec!(100_000),
        }
    }

    #[test]
    fn test_strategy_generates_entry_order() {
        let config = PutSpreadConfig::default();
        let mut strategy = PutCreditSpreadStrategy::new(config);
        let snapshot = make_tradeable_snapshot();
        let portfolio = make_empty_portfolio();

        let orders = strategy.on_snapshot(&snapshot, &portfolio);

        assert!(!orders.is_empty(), "Strategy should find a tradeable spread");
        assert!(matches!(orders[0].action, OrderAction::Open));
        assert_eq!(orders[0].legs.len(), 2);
    }

    #[test]
    fn test_strategy_respects_max_positions() {
        let mut config = PutSpreadConfig::default();
        config.max_positions = 0;
        let mut strategy = PutCreditSpreadStrategy::new(config);
        let snapshot = make_tradeable_snapshot();
        let portfolio = make_empty_portfolio();

        let orders = strategy.on_snapshot(&snapshot, &portfolio);
        assert!(orders.iter().all(|o| !matches!(o.action, OrderAction::Open)));
    }

    #[test]
    fn test_strategy_generates_exit_on_profit_target() {
        let config = PutSpreadConfig::default();
        let mut strategy = PutCreditSpreadStrategy::new(config);
        let snapshot = make_tradeable_snapshot();

        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();
        let mut position = crate::backtest::CreditSpreadBuilder::put_credit_spread("SPY", date)
            .stock_price(dec!(480))
            .short_leg(dec!(470), dec!(2.50), -0.25)
            .long_leg(dec!(465), dec!(1.50), -0.15)
            .expiration(exp, 32)
            .contracts(1)
            .build();
        // Simulate 60% profit (spread decayed from $1.00 to $0.40)
        position.current_value = dec!(40);

        let portfolio = PortfolioView {
            equity: dec!(100_060),
            cash: dec!(100_000),
            open_positions: vec![position],
            aggregate_greeks: PortfolioGreeks::default(),
            current_drawdown_pct: 0.0,
            peak_equity: dec!(100_060),
        };

        let orders = strategy.on_snapshot(&snapshot, &portfolio);

        let exit_orders: Vec<_> = orders.iter().filter(|o| matches!(o.action, OrderAction::Close { .. })).collect();
        assert!(!exit_orders.is_empty(), "Strategy should exit at profit target");
    }
}
