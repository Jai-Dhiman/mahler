//! Trade and position management for backtesting.
//!
//! Handles the complete trade lifecycle:
//! - Entry (position creation)
//! - Position tracking (mark-to-market)
//! - Exit conditions (profit target, stop loss, time exit, expiration)
//! - P&L calculation

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::data::{OptionQuote, OptionType};

/// Direction of the trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    /// Sell premium (credit spreads).
    Short,
    /// Buy premium (debit spreads).
    Long,
}

/// Type of spread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpreadType {
    /// Vertical put credit spread (bull put).
    PutCreditSpread,
    /// Vertical call credit spread (bear call).
    CallCreditSpread,
    /// Iron condor (both put and call credit spreads).
    IronCondor,
    /// Single option.
    Single,
    /// Custom multi-leg.
    Custom,
}

/// Reason for exiting a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExitReason {
    /// Hit profit target.
    ProfitTarget,
    /// Hit stop loss.
    StopLoss,
    /// Time-based exit (e.g., 21 DTE).
    TimeExit,
    /// Expired worthless (max profit).
    ExpiredWorthless,
    /// Expired in-the-money (assignment risk).
    ExpiredITM,
    /// Manual/forced exit.
    Manual,
    /// End of backtest period.
    EndOfPeriod,
}

/// Status of a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionStatus {
    /// Position is open.
    Open,
    /// Position has been closed.
    Closed,
}

/// A single leg of a multi-leg position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLeg {
    /// Option type (call/put).
    pub option_type: OptionType,
    /// Strike price.
    pub strike: Decimal,
    /// Expiration date.
    pub expiration: NaiveDate,
    /// Number of contracts (positive = long, negative = short).
    pub contracts: i32,
    /// Entry price per contract.
    pub entry_price: Decimal,
    /// Current market price per contract.
    pub current_price: Decimal,
    /// Entry delta.
    pub entry_delta: f64,
    /// Current delta.
    pub current_delta: f64,
}

impl PositionLeg {
    /// Check if this leg is short.
    pub fn is_short(&self) -> bool {
        self.contracts < 0
    }

    /// Calculate unrealized P&L for this leg (per contract, before multiplier).
    pub fn unrealized_pnl(&self) -> Decimal {
        // For short positions: profit when price decreases
        // For long positions: profit when price increases
        if self.is_short() {
            self.entry_price - self.current_price
        } else {
            self.current_price - self.entry_price
        }
    }

    /// Calculate total unrealized P&L (with multiplier).
    pub fn total_unrealized_pnl(&self) -> Decimal {
        self.unrealized_pnl() * Decimal::from(self.contracts.abs()) * Decimal::from(100)
    }
}

/// A complete options position (potentially multi-leg).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Unique position ID.
    pub id: u64,
    /// Underlying ticker.
    pub ticker: String,
    /// Date position was opened.
    pub entry_date: NaiveDate,
    /// Date position was closed (if closed).
    pub exit_date: Option<NaiveDate>,
    /// Spread type.
    pub spread_type: SpreadType,
    /// Trade direction.
    pub direction: TradeDirection,
    /// Position legs.
    pub legs: Vec<PositionLeg>,
    /// Total net credit (positive) or debit (negative) received.
    pub net_credit: Decimal,
    /// Maximum potential loss (for risk calculations).
    pub max_loss: Decimal,
    /// Maximum potential profit.
    pub max_profit: Decimal,
    /// Current position value (for MTM).
    pub current_value: Decimal,
    /// Entry commission paid.
    pub entry_commission: Decimal,
    /// Exit commission paid (if closed).
    pub exit_commission: Decimal,
    /// Position status.
    pub status: PositionStatus,
    /// Exit reason (if closed).
    pub exit_reason: Option<ExitReason>,
    /// Realized P&L (if closed).
    pub realized_pnl: Option<Decimal>,
    /// DTE at entry.
    pub entry_dte: i32,
    /// Stock price at entry.
    pub entry_stock_price: Decimal,
}

impl Position {
    /// Create a new position ID.
    pub fn new_id() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    /// Get total number of contracts (absolute).
    pub fn total_contracts(&self) -> i32 {
        self.legs.iter().map(|l| l.contracts.abs()).sum()
    }

    /// Get number of legs.
    pub fn num_legs(&self) -> usize {
        self.legs.len()
    }

    /// Check if position is open.
    pub fn is_open(&self) -> bool {
        self.status == PositionStatus::Open
    }

    /// Calculate current unrealized P&L.
    pub fn unrealized_pnl(&self) -> Decimal {
        // For credit spreads: profit = credit received - cost to close
        // current_value is the cost to close (negative = credit, positive = debit)
        self.net_credit - self.current_value - self.entry_commission
    }

    /// Calculate unrealized P&L as percentage of credit received.
    pub fn unrealized_pnl_pct(&self) -> f64 {
        if self.net_credit.is_zero() {
            return 0.0;
        }
        let pnl: f64 = self.unrealized_pnl().try_into().unwrap_or(0.0);
        let credit: f64 = self.net_credit.try_into().unwrap_or(1.0);
        pnl / credit * 100.0
    }

    /// Calculate unrealized P&L as percentage of max loss (for risk).
    pub fn unrealized_pnl_of_risk(&self) -> f64 {
        if self.max_loss.is_zero() {
            return 0.0;
        }
        let pnl: f64 = self.unrealized_pnl().try_into().unwrap_or(0.0);
        let risk: f64 = self.max_loss.try_into().unwrap_or(1.0);
        pnl / risk * 100.0
    }

    /// Get current DTE.
    pub fn current_dte(&self, current_date: NaiveDate) -> i32 {
        if let Some(leg) = self.legs.first() {
            (leg.expiration - current_date).num_days() as i32
        } else {
            0
        }
    }

    /// Get expiration date.
    pub fn expiration(&self) -> Option<NaiveDate> {
        self.legs.first().map(|l| l.expiration)
    }

    /// Get aggregate delta (sum of all legs).
    pub fn aggregate_delta(&self) -> f64 {
        self.legs
            .iter()
            .map(|l| l.current_delta * l.contracts as f64 * 100.0)
            .sum()
    }

    /// Update position with new market data.
    pub fn update_mtm(&mut self, quotes: &[&OptionQuote]) {
        let mut total_value = Decimal::ZERO;

        for leg in &mut self.legs {
            // Find matching quote
            if let Some(quote) = quotes.iter().find(|q| {
                q.strike == leg.strike
                    && q.option_type == leg.option_type
                    && q.expiration == leg.expiration
            }) {
                leg.current_price = quote.mid;
                leg.current_delta = quote.greeks.delta;

                // Value = price * contracts * 100
                // For short positions, we need to pay to close (negative value)
                // For long positions, we receive when closing (positive value)
                let leg_value = quote.mid * Decimal::from(leg.contracts) * Decimal::from(100);
                total_value += leg_value;
            }
        }

        // For credit spreads: current_value is what we'd pay to close
        // If we're short the spread, negative total_value means credit to close
        self.current_value = -total_value;
    }

    /// Check if profit target is hit.
    pub fn is_profit_target_hit(&self, target_pct: f64) -> bool {
        if self.net_credit.is_zero() {
            return false;
        }
        self.unrealized_pnl_pct() >= target_pct
    }

    /// Check if stop loss is hit.
    pub fn is_stop_loss_hit(&self, stop_pct: f64) -> bool {
        if self.net_credit.is_zero() {
            return false;
        }
        // Stop loss = loss exceeds X% of credit received
        // e.g., 125% stop means if we've lost 125% of credit, we exit
        self.unrealized_pnl_pct() <= -stop_pct
    }

    /// Check if time exit is triggered.
    pub fn is_time_exit(&self, current_date: NaiveDate, exit_dte: i32) -> bool {
        self.current_dte(current_date) <= exit_dte
    }

    /// Check if position has expired.
    pub fn is_expired(&self, current_date: NaiveDate) -> bool {
        self.current_dte(current_date) <= 0
    }

    /// Close the position.
    pub fn close(
        &mut self,
        exit_date: NaiveDate,
        exit_reason: ExitReason,
        exit_commission: Decimal,
    ) {
        self.status = PositionStatus::Closed;
        self.exit_date = Some(exit_date);
        self.exit_reason = Some(exit_reason);
        self.exit_commission = exit_commission;

        // Calculate realized P&L
        let pnl = self.net_credit - self.current_value - self.entry_commission - exit_commission;
        self.realized_pnl = Some(pnl);
    }
}

/// A completed trade (closed position) for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Position details.
    pub position: Position,
    /// Days in trade.
    pub days_held: i32,
    /// Return on risk (P&L / max_loss).
    pub return_on_risk: f64,
}

impl Trade {
    /// Create a trade from a closed position.
    pub fn from_position(position: Position) -> Option<Self> {
        if position.status != PositionStatus::Closed {
            return None;
        }

        let entry = position.entry_date;
        let exit = position.exit_date?;
        let days_held = (exit - entry).num_days() as i32;

        let pnl: f64 = position.realized_pnl?.try_into().unwrap_or(0.0);
        let risk: f64 = position.max_loss.try_into().unwrap_or(1.0);
        let return_on_risk = if risk != 0.0 { pnl / risk } else { 0.0 };

        Some(Self {
            position,
            days_held,
            return_on_risk,
        })
    }

    /// Check if trade was profitable.
    pub fn is_winner(&self) -> bool {
        self.position
            .realized_pnl
            .map(|p| p > Decimal::ZERO)
            .unwrap_or(false)
    }

    /// Get realized P&L.
    pub fn pnl(&self) -> Decimal {
        self.position.realized_pnl.unwrap_or(Decimal::ZERO)
    }
}

/// Builder for creating credit spread positions.
pub struct CreditSpreadBuilder {
    ticker: String,
    entry_date: NaiveDate,
    entry_stock_price: Decimal,
    spread_type: SpreadType,
    short_strike: Decimal,
    long_strike: Decimal,
    expiration: NaiveDate,
    contracts: i32,
    short_price: Decimal,
    long_price: Decimal,
    short_delta: f64,
    long_delta: f64,
    entry_dte: i32,
    entry_commission: Decimal,
}

impl CreditSpreadBuilder {
    pub fn put_credit_spread(ticker: &str, entry_date: NaiveDate) -> Self {
        Self {
            ticker: ticker.to_string(),
            entry_date,
            entry_stock_price: Decimal::ZERO,
            spread_type: SpreadType::PutCreditSpread,
            short_strike: Decimal::ZERO,
            long_strike: Decimal::ZERO,
            expiration: entry_date,
            contracts: 1,
            short_price: Decimal::ZERO,
            long_price: Decimal::ZERO,
            short_delta: 0.0,
            long_delta: 0.0,
            entry_dte: 0,
            entry_commission: Decimal::ZERO,
        }
    }

    pub fn call_credit_spread(ticker: &str, entry_date: NaiveDate) -> Self {
        Self {
            spread_type: SpreadType::CallCreditSpread,
            ..Self::put_credit_spread(ticker, entry_date)
        }
    }

    pub fn stock_price(mut self, price: Decimal) -> Self {
        self.entry_stock_price = price;
        self
    }

    pub fn short_leg(mut self, strike: Decimal, price: Decimal, delta: f64) -> Self {
        self.short_strike = strike;
        self.short_price = price;
        self.short_delta = delta;
        self
    }

    pub fn long_leg(mut self, strike: Decimal, price: Decimal, delta: f64) -> Self {
        self.long_strike = strike;
        self.long_price = price;
        self.long_delta = delta;
        self
    }

    pub fn expiration(mut self, exp: NaiveDate, dte: i32) -> Self {
        self.expiration = exp;
        self.entry_dte = dte;
        self
    }

    pub fn contracts(mut self, contracts: i32) -> Self {
        self.contracts = contracts;
        self
    }

    pub fn commission(mut self, commission: Decimal) -> Self {
        self.entry_commission = commission;
        self
    }

    pub fn build(self) -> Position {
        let option_type = match self.spread_type {
            SpreadType::PutCreditSpread => OptionType::Put,
            SpreadType::CallCreditSpread => OptionType::Call,
            _ => OptionType::Put,
        };

        // Credit received = sell high, buy low
        let net_credit =
            (self.short_price - self.long_price) * Decimal::from(self.contracts) * Decimal::from(100);

        // Max loss = spread width - credit
        let width = (self.short_strike - self.long_strike).abs();
        let max_loss = width * Decimal::from(self.contracts) * Decimal::from(100) - net_credit;

        // Max profit = net credit (minus commissions)
        let max_profit = net_credit;

        let short_leg = PositionLeg {
            option_type,
            strike: self.short_strike,
            expiration: self.expiration,
            contracts: -self.contracts, // Negative = short
            entry_price: self.short_price,
            current_price: self.short_price,
            entry_delta: self.short_delta,
            current_delta: self.short_delta,
        };

        let long_leg = PositionLeg {
            option_type,
            strike: self.long_strike,
            expiration: self.expiration,
            contracts: self.contracts, // Positive = long
            entry_price: self.long_price,
            current_price: self.long_price,
            entry_delta: self.long_delta,
            current_delta: self.long_delta,
        };

        Position {
            id: Position::new_id(),
            ticker: self.ticker,
            entry_date: self.entry_date,
            exit_date: None,
            spread_type: self.spread_type,
            direction: TradeDirection::Short,
            legs: vec![short_leg, long_leg],
            net_credit,
            max_loss,
            max_profit,
            current_value: Decimal::ZERO, // Cost to close at entry = 0
            entry_commission: self.entry_commission,
            exit_commission: Decimal::ZERO,
            status: PositionStatus::Open,
            exit_reason: None,
            realized_pnl: None,
            entry_dte: self.entry_dte,
            entry_stock_price: self.entry_stock_price,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_position_leg_pnl() {
        let mut leg = PositionLeg {
            option_type: OptionType::Put,
            strike: dec!(400),
            expiration: NaiveDate::from_ymd_opt(2024, 3, 15).unwrap(),
            contracts: -1, // Short
            entry_price: dec!(2.00),
            current_price: dec!(1.00),
            entry_delta: -0.15,
            current_delta: -0.10,
        };

        // Short position: profit when price decreases
        assert_eq!(leg.unrealized_pnl(), dec!(1.00));
        assert_eq!(leg.total_unrealized_pnl(), dec!(100)); // 1 * 1 * 100

        // Price increases = loss for short
        leg.current_price = dec!(3.00);
        assert_eq!(leg.unrealized_pnl(), dec!(-1.00));
    }

    #[test]
    fn test_credit_spread_builder() {
        let entry = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();

        let position = CreditSpreadBuilder::put_credit_spread("SPY", entry)
            .stock_price(dec!(480))
            .short_leg(dec!(470), dec!(2.50), -0.15)
            .long_leg(dec!(465), dec!(1.50), -0.10)
            .expiration(exp, 32)
            .contracts(10)
            .commission(dec!(20))
            .build();

        assert_eq!(position.ticker, "SPY");
        assert_eq!(position.spread_type, SpreadType::PutCreditSpread);
        assert_eq!(position.legs.len(), 2);
        assert_eq!(position.net_credit, dec!(1000)); // (2.50 - 1.50) * 10 * 100
        assert_eq!(position.max_loss, dec!(4000)); // (5 * 10 * 100) - 1000
        assert!(position.is_open());
    }

    #[test]
    fn test_profit_target_detection() {
        let entry = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();

        let mut position = CreditSpreadBuilder::put_credit_spread("SPY", entry)
            .short_leg(dec!(470), dec!(2.50), -0.15)
            .long_leg(dec!(465), dec!(1.50), -0.10)
            .expiration(exp, 32)
            .contracts(1)
            .build();

        // Net credit = $100
        // Current value (cost to close) = 0 at entry
        // Unrealized P&L = 100 - 0 - 0 = 100 (100% of credit)

        // Simulate spread decaying to 50% of original value
        position.current_value = dec!(50); // Would cost $50 to close
        // Unrealized P&L = 100 - 50 - 0 = 50 (50% of credit)

        assert!(position.is_profit_target_hit(50.0));
        assert!(!position.is_profit_target_hit(51.0));
    }
}
