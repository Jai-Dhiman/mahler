//! Core data types for options backtesting.
//!
//! These types represent the fundamental data structures used throughout
//! the backtester, designed to match ORATS data format while being
//! efficient for Rust processing.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Option type (call or put).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "C" | "CALL" => Some(Self::Call),
            "P" | "PUT" => Some(Self::Put),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Call => "C",
            Self::Put => "P",
        }
    }
}

/// Greeks for an option contract.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// A single option quote at a point in time.
///
/// This is the fundamental unit of options data. Contains all the
/// information needed for backtesting a single option contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionQuote {
    /// Underlying symbol (e.g., "SPY")
    pub ticker: String,

    /// Date of the quote (trade date)
    pub trade_date: NaiveDate,

    /// Option expiration date
    pub expiration: NaiveDate,

    /// Days to expiration
    pub dte: i32,

    /// Strike price
    pub strike: Decimal,

    /// Option type (call or put)
    pub option_type: OptionType,

    /// Underlying stock price at quote time
    pub stock_price: Decimal,

    /// Bid price
    pub bid: Decimal,

    /// Ask price
    pub ask: Decimal,

    /// Mid price (calculated)
    pub mid: Decimal,

    /// Theoretical value (ORATS SMV)
    pub theoretical_value: Decimal,

    /// Trading volume
    pub volume: i64,

    /// Open interest
    pub open_interest: i64,

    /// Bid implied volatility
    pub bid_iv: f64,

    /// Mid implied volatility
    pub mid_iv: f64,

    /// Ask implied volatility
    pub ask_iv: f64,

    /// ORATS smoothed implied volatility
    pub smv_vol: f64,

    /// Greeks
    pub greeks: Greeks,

    /// Implied interest rate from ORATS
    pub residual_rate: f64,
}

impl OptionQuote {
    /// Calculate bid-ask spread as percentage of mid.
    pub fn spread_pct(&self) -> f64 {
        if self.mid.is_zero() {
            return 0.0;
        }
        let spread = self.ask - self.bid;
        (spread / self.mid).try_into().unwrap_or(0.0)
    }

    /// Check if the option has reasonable liquidity.
    pub fn is_liquid(&self, min_oi: i64, min_volume: i64, max_spread_pct: f64) -> bool {
        self.open_interest >= min_oi
            && self.volume >= min_volume
            && self.spread_pct() <= max_spread_pct
    }
}

/// All options for a single expiration date.
#[derive(Debug, Clone, Default)]
pub struct OptionsChain {
    /// Expiration date for this chain
    pub expiration: NaiveDate,

    /// Days to expiration
    pub dte: i32,

    /// Call options indexed by strike
    pub calls: Vec<OptionQuote>,

    /// Put options indexed by strike
    pub puts: Vec<OptionQuote>,
}

impl OptionsChain {
    /// Create a new empty chain.
    pub fn new(expiration: NaiveDate, dte: i32) -> Self {
        Self {
            expiration,
            dte,
            calls: Vec::new(),
            puts: Vec::new(),
        }
    }

    /// Add a quote to the appropriate side.
    pub fn add_quote(&mut self, quote: OptionQuote) {
        match quote.option_type {
            OptionType::Call => self.calls.push(quote),
            OptionType::Put => self.puts.push(quote),
        }
    }

    /// Get all strikes available in this chain.
    pub fn strikes(&self) -> Vec<Decimal> {
        let mut strikes: Vec<_> = self
            .calls
            .iter()
            .chain(self.puts.iter())
            .map(|q| q.strike)
            .collect();
        strikes.sort();
        strikes.dedup();
        strikes
    }

    /// Find a call at a specific strike.
    pub fn call_at_strike(&self, strike: Decimal) -> Option<&OptionQuote> {
        self.calls.iter().find(|q| q.strike == strike)
    }

    /// Find a put at a specific strike.
    pub fn put_at_strike(&self, strike: Decimal) -> Option<&OptionQuote> {
        self.puts.iter().find(|q| q.strike == strike)
    }

    /// Find options by delta range.
    pub fn puts_by_delta(&self, min_delta: f64, max_delta: f64) -> Vec<&OptionQuote> {
        self.puts
            .iter()
            .filter(|q| {
                let d = q.greeks.delta.abs();
                d >= min_delta && d <= max_delta
            })
            .collect()
    }

    /// Find calls by delta range.
    pub fn calls_by_delta(&self, min_delta: f64, max_delta: f64) -> Vec<&OptionQuote> {
        self.calls
            .iter()
            .filter(|q| {
                let d = q.greeks.delta.abs();
                d >= min_delta && d <= max_delta
            })
            .collect()
    }
}

/// Complete options snapshot for one underlying on one date.
///
/// Contains all option chains (expirations) for a single underlying
/// symbol on a single trading day.
#[derive(Debug, Clone, Default)]
pub struct OptionsSnapshot {
    /// Trading date
    pub date: NaiveDate,

    /// Underlying symbol
    pub ticker: String,

    /// Underlying price
    pub underlying_price: Decimal,

    /// All option chains keyed by expiration
    pub chains: Vec<OptionsChain>,
}

impl OptionsSnapshot {
    /// Create a new empty snapshot.
    pub fn new(date: NaiveDate, ticker: String, underlying_price: Decimal) -> Self {
        Self {
            date,
            ticker,
            underlying_price,
            chains: Vec::new(),
        }
    }

    /// Get chain for a specific expiration.
    pub fn chain_at_expiration(&self, expiration: NaiveDate) -> Option<&OptionsChain> {
        self.chains.iter().find(|c| c.expiration == expiration)
    }

    /// Get chains within a DTE range.
    pub fn chains_by_dte(&self, min_dte: i32, max_dte: i32) -> Vec<&OptionsChain> {
        self.chains
            .iter()
            .filter(|c| c.dte >= min_dte && c.dte <= max_dte)
            .collect()
    }

    /// Total number of option quotes in this snapshot.
    pub fn total_quotes(&self) -> usize {
        self.chains
            .iter()
            .map(|c| c.calls.len() + c.puts.len())
            .sum()
    }
}

/// Daily bar data for underlying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderlyingBar {
    pub date: NaiveDate,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_type_parsing() {
        assert_eq!(OptionType::from_str("C"), Some(OptionType::Call));
        assert_eq!(OptionType::from_str("P"), Some(OptionType::Put));
        assert_eq!(OptionType::from_str("call"), Some(OptionType::Call));
        assert_eq!(OptionType::from_str("PUT"), Some(OptionType::Put));
        assert_eq!(OptionType::from_str("X"), None);
    }
}
