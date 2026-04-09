//! Slippage model based on ORATS methodology.
//!
//! The ORATS slippage model adjusts fill prices based on the number of legs
//! in the trade. More legs = worse fills due to complexity.
//!
//! | Legs | Fill % of spread |
//! |------|------------------|
//! | 1    | 75%              |
//! | 2    | 66%              |
//! | 3    | 56%              |
//! | 4    | 53%              |

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Slippage percentage based on number of legs.
#[derive(Debug, Clone, Copy)]
pub struct Slippage {
    pub legs: usize,
    pub fill_pct: f64,
}

impl Slippage {
    /// Get slippage for a given number of legs using ORATS methodology.
    pub fn for_legs(legs: usize) -> Self {
        let fill_pct = match legs {
            1 => 0.75,
            2 => 0.66,
            3 => 0.56,
            4 => 0.53,
            _ => 0.50, // Conservative default for 5+ legs
        };
        Self { legs, fill_pct }
    }

    /// Calculate fill price for buying (paying the ask, getting fill between bid and ask).
    /// fill = bid + (ask - bid) * fill_pct
    pub fn buy_fill(&self, bid: Decimal, ask: Decimal) -> Decimal {
        let spread = ask - bid;
        let fill_pct = Decimal::from_f64_retain(self.fill_pct).unwrap_or(Decimal::ONE);
        bid + spread * fill_pct
    }

    /// Calculate fill price for selling (hitting the bid, getting fill between bid and ask).
    /// For selling, we get (1 - fill_pct) of the spread from the ask.
    /// fill = ask - (ask - bid) * fill_pct = bid + (ask - bid) * (1 - fill_pct)
    pub fn sell_fill(&self, bid: Decimal, ask: Decimal) -> Decimal {
        let spread = ask - bid;
        let fill_pct = Decimal::from_f64_retain(1.0 - self.fill_pct).unwrap_or(Decimal::ZERO);
        bid + spread * fill_pct
    }
}

/// Configurable slippage model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageModel {
    /// Custom fill percentages by leg count (1-indexed).
    pub fill_pcts: Vec<f64>,
    /// Whether to use ORATS defaults.
    pub use_orats_defaults: bool,
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self {
            fill_pcts: vec![0.75, 0.66, 0.56, 0.53],
            use_orats_defaults: true,
        }
    }
}

impl SlippageModel {
    /// Create a new slippage model with custom fill percentages.
    pub fn new(fill_pcts: Vec<f64>) -> Self {
        Self {
            fill_pcts,
            use_orats_defaults: false,
        }
    }

    /// Get slippage for a given number of legs.
    pub fn for_legs(&self, legs: usize) -> Slippage {
        if self.use_orats_defaults {
            return Slippage::for_legs(legs);
        }

        let fill_pct = if legs > 0 && legs <= self.fill_pcts.len() {
            self.fill_pcts[legs - 1]
        } else if !self.fill_pcts.is_empty() {
            *self.fill_pcts.last().unwrap()
        } else {
            0.50
        };

        Slippage { legs, fill_pct }
    }

    /// Create an ORATS-standard slippage model.
    pub fn orats() -> Self {
        Self::default()
    }

    /// Create a zero-slippage model (fills at mid).
    pub fn zero() -> Self {
        Self {
            fill_pcts: vec![0.50, 0.50, 0.50, 0.50],
            use_orats_defaults: false,
        }
    }

    /// Create a pessimistic slippage model (fills at bid/ask).
    pub fn pessimistic() -> Self {
        Self {
            fill_pcts: vec![1.0, 1.0, 1.0, 1.0],
            use_orats_defaults: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_orats_slippage_values() {
        assert_eq!(Slippage::for_legs(1).fill_pct, 0.75);
        assert_eq!(Slippage::for_legs(2).fill_pct, 0.66);
        assert_eq!(Slippage::for_legs(3).fill_pct, 0.56);
        assert_eq!(Slippage::for_legs(4).fill_pct, 0.53);
    }

    #[test]
    fn test_buy_fill() {
        let slippage = Slippage::for_legs(2);
        let bid = dec!(1.00);
        let ask = dec!(1.10);
        let fill = slippage.buy_fill(bid, ask);
        // fill = 1.00 + 0.10 * 0.66 = 1.066
        assert!(fill > bid && fill < ask);
        assert!((fill - dec!(1.066)).abs() < dec!(0.001));
    }

    #[test]
    fn test_sell_fill() {
        let slippage = Slippage::for_legs(2);
        let bid = dec!(1.00);
        let ask = dec!(1.10);
        let fill = slippage.sell_fill(bid, ask);
        // fill = 1.00 + 0.10 * 0.34 = 1.034
        assert!(fill > bid && fill < ask);
        assert!((fill - dec!(1.034)).abs() < dec!(0.001));
    }

    #[test]
    fn test_slippage_model_default() {
        let model = SlippageModel::default();
        assert!(model.use_orats_defaults);
        assert_eq!(model.for_legs(2).fill_pct, 0.66);
    }

    #[test]
    fn test_slippage_model_zero() {
        let model = SlippageModel::zero();
        assert_eq!(model.for_legs(2).fill_pct, 0.50);
    }
}
