use chrono::NaiveDate;
use rust_decimal::Decimal;

use crate::backtest::{EquityPoint, ExitReason, Position, Trade};
use crate::engine_core::events::PortfolioView;
use crate::data::OptionsSnapshot;
use crate::risk::portfolio_greeks::{PortfolioGreeks, PositionWithGreeks};

pub struct PortfolioTracker {
    initial_equity: Decimal,
    equity: Decimal,
    cash: Decimal,
    peak_equity: Decimal,
    positions: Vec<Position>,
    closed_trades: Vec<Trade>,
    equity_curve: Vec<EquityPoint>,
    total_commission: Decimal,
}

impl PortfolioTracker {
    pub fn new(initial_equity: Decimal) -> Self {
        Self {
            initial_equity,
            equity: initial_equity,
            cash: initial_equity,
            peak_equity: initial_equity,
            positions: Vec::new(),
            closed_trades: Vec::new(),
            equity_curve: Vec::new(),
            total_commission: Decimal::ZERO,
        }
    }

    pub fn add_position(&mut self, position: Position) {
        self.total_commission += position.entry_commission;
        self.cash -= position.entry_commission;
        self.positions.push(position);
    }

    pub fn close_position(
        &mut self,
        position_id: u64,
        exit_date: NaiveDate,
        reason: ExitReason,
        current_value: Decimal,
        exit_commission: Decimal,
    ) {
        if let Some(pos) = self.positions.iter_mut().find(|p| p.id == position_id && p.is_open()) {
            pos.current_value = current_value;
            self.total_commission += exit_commission;

            let pnl = pos.unrealized_pnl() - exit_commission;
            self.cash += pnl;

            pos.close(exit_date, reason, exit_commission);

            if let Some(trade) = Trade::from_position(pos.clone()) {
                self.closed_trades.push(trade);
            }
        }
    }

    pub fn update_mtm(&mut self, snapshot: &OptionsSnapshot) {
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

        let positions_value: Decimal = self.positions.iter()
            .filter(|p| p.is_open())
            .map(|p| p.unrealized_pnl())
            .sum();

        self.equity = self.cash + positions_value;

        if self.equity > self.peak_equity {
            self.peak_equity = self.equity;
        }
    }

    pub fn record_equity(&mut self, date: NaiveDate) {
        let positions_value: Decimal = self.positions.iter()
            .filter(|p| p.is_open())
            .map(|p| p.unrealized_pnl())
            .sum();

        let prev_equity = self.equity_curve.last().map(|e| e.equity).unwrap_or(self.initial_equity);
        let daily_pnl = self.equity - prev_equity;

        self.equity_curve.push(EquityPoint {
            date,
            equity: self.equity,
            cash: self.cash,
            positions_value,
            open_positions: self.positions.iter().filter(|p| p.is_open()).count(),
            daily_pnl,
        });
    }

    pub fn view(&self) -> PortfolioView {
        let open_positions: Vec<Position> = self.positions.iter()
            .filter(|p| p.is_open())
            .cloned()
            .collect();

        let greeks_positions: Vec<PositionWithGreeks> = open_positions.iter()
            .map(|p| PositionWithGreeks::from_position(p))
            .collect();

        let aggregate_greeks = PortfolioGreeks::from_positions_with_greeks(&greeks_positions);

        let current_drawdown_pct = if !self.peak_equity.is_zero() {
            let peak: f64 = self.peak_equity.try_into().unwrap_or(1.0);
            let current: f64 = self.equity.try_into().unwrap_or(peak);
            (peak - current) / peak * 100.0
        } else {
            0.0
        };

        PortfolioView {
            equity: self.equity,
            cash: self.cash,
            open_positions,
            aggregate_greeks,
            current_drawdown_pct,
            peak_equity: self.peak_equity,
        }
    }

    pub fn cleanup_closed(&mut self) {
        self.positions.retain(|p| p.is_open());
    }

    pub fn closed_trades(&self) -> &[Trade] {
        &self.closed_trades
    }

    pub fn equity_curve(&self) -> &[EquityPoint] {
        &self.equity_curve
    }

    pub fn equity(&self) -> Decimal {
        self.equity
    }

    pub fn peak_equity(&self) -> Decimal {
        self.peak_equity
    }

    pub fn total_commission(&self) -> Decimal {
        self.total_commission
    }

    pub fn initial_equity(&self) -> Decimal {
        self.initial_equity
    }

    pub fn current_risk(&self) -> Decimal {
        self.positions.iter().filter(|p| p.is_open()).map(|p| p.max_loss).sum()
    }

    pub fn equity_correlated_risk(&self) -> Decimal {
        self.positions.iter()
            .filter(|p| p.is_open())
            .filter(|p| matches!(p.ticker.to_uppercase().as_str(), "SPY" | "QQQ" | "IWM"))
            .map(|p| p.max_loss)
            .sum()
    }

    pub fn close_all_remaining(&mut self, end_date: NaiveDate, exit_commission_per_trade: Decimal) {
        let open_ids: Vec<u64> = self.positions.iter().filter(|p| p.is_open()).map(|p| p.id).collect();
        for id in open_ids {
            if let Some(pos) = self.positions.iter_mut().find(|p| p.id == id) {
                self.total_commission += exit_commission_per_trade;
                let pnl = pos.unrealized_pnl() - exit_commission_per_trade;
                self.cash += pnl;
                pos.close(end_date, ExitReason::EndOfPeriod, exit_commission_per_trade);
                if let Some(trade) = Trade::from_position(pos.clone()) {
                    self.closed_trades.push(trade);
                }
            }
        }
        self.positions.retain(|p| p.is_open());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::backtest::{CreditSpreadBuilder, ExitReason, SpreadType, CommissionModel, SlippageModel};
    use crate::engine_core::events::{FillEvent, OrderIntent, OrderAction, LegIntent, OrderSide};
    use crate::data::{OptionType, OptionQuote, OptionsChain, OptionsSnapshot, Greeks};

    #[test]
    fn test_initial_state() {
        let tracker = PortfolioTracker::new(dec!(100_000));
        let view = tracker.view();
        assert_eq!(view.equity, dec!(100_000));
        assert_eq!(view.cash, dec!(100_000));
        assert!(view.open_positions.is_empty());
        assert_eq!(view.current_drawdown_pct, 0.0);
    }

    #[test]
    fn test_open_position_via_fill() {
        let mut tracker = PortfolioTracker::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();

        let position = CreditSpreadBuilder::put_credit_spread("SPY", date)
            .stock_price(dec!(480))
            .short_leg(dec!(470), dec!(2.50), -0.15)
            .long_leg(dec!(465), dec!(1.50), -0.10)
            .expiration(exp, 32)
            .contracts(1)
            .commission(dec!(2))
            .build();

        tracker.add_position(position);

        let view = tracker.view();
        assert_eq!(view.open_positions.len(), 1);
        assert_eq!(view.cash, dec!(100_000) - dec!(2));
    }

    #[test]
    fn test_close_position() {
        let mut tracker = PortfolioTracker::new(dec!(100_000));
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();

        let position = CreditSpreadBuilder::put_credit_spread("SPY", date)
            .stock_price(dec!(480))
            .short_leg(dec!(470), dec!(2.50), -0.15)
            .long_leg(dec!(465), dec!(1.50), -0.10)
            .expiration(exp, 32)
            .contracts(1)
            .commission(dec!(2))
            .build();

        tracker.add_position(position);
        let pos_id = tracker.view().open_positions[0].id;

        tracker.close_position(pos_id, date + chrono::Duration::days(10), ExitReason::ProfitTarget, dec!(50), dec!(2));

        let view = tracker.view();
        assert_eq!(view.open_positions.len(), 0);
        assert_eq!(tracker.closed_trades().len(), 1);
        assert!(tracker.closed_trades()[0].is_winner());
    }

    #[test]
    fn test_equity_curve_recording() {
        let mut tracker = PortfolioTracker::new(dec!(100_000));
        let date1 = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let date2 = NaiveDate::from_ymd_opt(2024, 1, 16).unwrap();

        tracker.record_equity(date1);
        tracker.record_equity(date2);

        assert_eq!(tracker.equity_curve().len(), 2);
        assert_eq!(tracker.equity_curve()[0].date, date1);
    }
}
