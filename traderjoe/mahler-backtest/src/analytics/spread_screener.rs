//! Credit spread screening and candidate selection.
//!
//! Screens for put credit spread candidates based on:
//! - DTE filter (30-45 DTE)
//! - Delta filter (0.10-0.15 for short strike)
//! - IV percentile filter (above 50th)
//! - Spread width validation
//! - Credit calculation and scoring

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::data::{OptionQuote, OptionType, OptionsChain, OptionsSnapshot};

/// Configuration for spread screening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadScreenerConfig {
    /// Minimum DTE for entry.
    pub min_dte: i32,
    /// Maximum DTE for entry.
    pub max_dte: i32,
    /// Minimum delta for short strike (absolute value).
    pub min_short_delta: f64,
    /// Maximum delta for short strike (absolute value).
    pub max_short_delta: f64,
    /// Minimum IV percentile for entry.
    pub min_iv_percentile: f64,
    /// Minimum open interest for liquidity.
    pub min_open_interest: i64,
    /// Minimum volume for liquidity.
    pub min_volume: i64,
    /// Maximum bid-ask spread percentage.
    pub max_spread_pct: f64,
    /// Spread widths to consider (in dollars).
    pub spread_widths: Vec<Decimal>,
    /// Minimum credit as percentage of width.
    pub min_credit_pct: f64,
}

impl Default for SpreadScreenerConfig {
    fn default() -> Self {
        Self {
            min_dte: 30,
            max_dte: 45,
            min_short_delta: 0.20,
            max_short_delta: 0.30,
            min_iv_percentile: 50.0,
            min_open_interest: 100,
            min_volume: 10,
            max_spread_pct: 0.20, // 20% max spread
            spread_widths: vec![
                Decimal::from(5),
                Decimal::from(10),
            ],
            min_credit_pct: 20.0, // Minimum 20% of width as credit
        }
    }
}

/// A potential credit spread candidate.
#[derive(Debug, Clone)]
pub struct SpreadCandidate {
    /// Underlying ticker.
    pub ticker: String,
    /// Trade date.
    pub date: NaiveDate,
    /// Short strike.
    pub short_strike: Decimal,
    /// Long strike.
    pub long_strike: Decimal,
    /// Expiration date.
    pub expiration: NaiveDate,
    /// Days to expiration.
    pub dte: i32,
    /// Short strike delta (absolute value).
    pub short_delta: f64,
    /// Long strike delta (absolute value).
    pub long_delta: f64,
    /// Short strike bid price.
    pub short_bid: Decimal,
    /// Long strike ask price.
    pub long_ask: Decimal,
    /// Net credit (short bid - long ask).
    pub credit: Decimal,
    /// Spread width.
    pub width: Decimal,
    /// Credit as percentage of width.
    pub credit_pct: f64,
    /// Maximum loss per contract.
    pub max_loss: Decimal,
    /// Short strike mid IV.
    pub short_iv: f64,
    /// Expected value score.
    pub ev_score: f64,
    /// Underlying price.
    pub underlying_price: Decimal,
    /// Option type (Put for put credit spread).
    pub option_type: OptionType,
}

impl SpreadCandidate {
    /// Calculate return on risk if max profit achieved.
    pub fn return_on_risk(&self) -> f64 {
        let credit: f64 = self.credit.try_into().unwrap_or(0.0);
        let max_loss: f64 = self.max_loss.try_into().unwrap_or(1.0);
        if max_loss > 0.0 {
            credit / max_loss * 100.0
        } else {
            0.0
        }
    }

    /// Distance from current price to short strike as percentage.
    pub fn distance_pct(&self) -> f64 {
        let price: f64 = self.underlying_price.try_into().unwrap_or(1.0);
        let short: f64 = self.short_strike.try_into().unwrap_or(0.0);
        if price > 0.0 {
            (price - short).abs() / price * 100.0
        } else {
            0.0
        }
    }
}

/// Spread screener for finding credit spread candidates.
pub struct SpreadScreener {
    config: SpreadScreenerConfig,
}

impl SpreadScreener {
    pub fn new(config: SpreadScreenerConfig) -> Self {
        Self { config }
    }

    /// Screen for put credit spread candidates.
    pub fn screen_put_spreads(
        &self,
        snapshot: &OptionsSnapshot,
        iv_percentile: Option<f64>,
    ) -> Vec<SpreadCandidate> {
        let mut candidates = Vec::new();

        // Check IV percentile filter
        if let Some(pct) = iv_percentile {
            if pct < self.config.min_iv_percentile {
                return candidates;
            }
        }

        // Get chains in DTE range
        let chains = snapshot.chains_by_dte(self.config.min_dte, self.config.max_dte);

        for chain in chains {
            let chain_candidates = self.screen_chain_puts(snapshot, chain);
            candidates.extend(chain_candidates);
        }

        // Sort by EV score descending
        candidates.sort_by(|a, b| {
            b.ev_score
                .partial_cmp(&a.ev_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Screen a single chain for put credit spread candidates.
    fn screen_chain_puts(
        &self,
        snapshot: &OptionsSnapshot,
        chain: &OptionsChain,
    ) -> Vec<SpreadCandidate> {
        let mut candidates = Vec::new();

        // Find puts in delta range for short strike
        let short_candidates: Vec<&OptionQuote> = chain
            .puts
            .iter()
            .filter(|q| {
                let delta = q.greeks.delta.abs();
                delta >= self.config.min_short_delta && delta <= self.config.max_short_delta
            })
            .filter(|q| self.passes_liquidity_filter(q))
            .collect();

        for short in short_candidates {
            // For each short strike, find valid long strikes
            for width in &self.config.spread_widths {
                let long_strike = short.strike - *width;

                // Find the long put at this strike
                if let Some(long) = chain.puts.iter().find(|q| q.strike == long_strike) {
                    if !self.passes_liquidity_filter(long) {
                        continue;
                    }

                    // Calculate credit
                    let credit = short.bid - long.ask;
                    if credit <= Decimal::ZERO {
                        continue;
                    }

                    // Check minimum credit percentage
                    let credit_f64: f64 = credit.try_into().unwrap_or(0.0);
                    let width_f64: f64 = (*width).try_into().unwrap_or(1.0);
                    let credit_pct = credit_f64 / width_f64 * 100.0;

                    if credit_pct < self.config.min_credit_pct {
                        continue;
                    }

                    // Calculate max loss (width - credit)
                    let max_loss = *width - credit;

                    // Calculate EV score
                    // Simple model: credit_pct * (1 - delta) - (1 - credit_pct) * delta
                    let delta = short.greeks.delta.abs();
                    let prob_otm = 1.0 - delta;
                    let ev_score = credit_pct * prob_otm - (100.0 - credit_pct) * delta;

                    candidates.push(SpreadCandidate {
                        ticker: snapshot.ticker.clone(),
                        date: snapshot.date,
                        short_strike: short.strike,
                        long_strike: long.strike,
                        expiration: chain.expiration,
                        dte: chain.dte,
                        short_delta: short.greeks.delta.abs(),
                        long_delta: long.greeks.delta.abs(),
                        short_bid: short.bid,
                        long_ask: long.ask,
                        credit,
                        width: *width,
                        credit_pct,
                        max_loss,
                        short_iv: short.mid_iv,
                        ev_score,
                        underlying_price: snapshot.underlying_price,
                        option_type: OptionType::Put,
                    });
                }
            }
        }

        candidates
    }

    /// Check if a quote passes the liquidity filter.
    fn passes_liquidity_filter(&self, quote: &OptionQuote) -> bool {
        if quote.open_interest < self.config.min_open_interest {
            return false;
        }

        if quote.volume < self.config.min_volume {
            return false;
        }

        if quote.spread_pct() > self.config.max_spread_pct {
            return false;
        }

        true
    }

    /// Screen for call credit spread candidates.
    pub fn screen_call_spreads(
        &self,
        snapshot: &OptionsSnapshot,
        iv_percentile: Option<f64>,
    ) -> Vec<SpreadCandidate> {
        let mut candidates = Vec::new();

        // Check IV percentile filter
        if let Some(pct) = iv_percentile {
            if pct < self.config.min_iv_percentile {
                return candidates;
            }
        }

        // Get chains in DTE range
        let chains = snapshot.chains_by_dte(self.config.min_dte, self.config.max_dte);

        for chain in chains {
            let chain_candidates = self.screen_chain_calls(snapshot, chain);
            candidates.extend(chain_candidates);
        }

        // Sort by EV score descending
        candidates.sort_by(|a, b| {
            b.ev_score
                .partial_cmp(&a.ev_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Screen a single chain for call credit spread candidates.
    fn screen_chain_calls(
        &self,
        snapshot: &OptionsSnapshot,
        chain: &OptionsChain,
    ) -> Vec<SpreadCandidate> {
        let mut candidates = Vec::new();

        // Find calls in delta range for short strike
        let short_candidates: Vec<&OptionQuote> = chain
            .calls
            .iter()
            .filter(|q| {
                let delta = q.greeks.delta.abs();
                delta >= self.config.min_short_delta && delta <= self.config.max_short_delta
            })
            .filter(|q| self.passes_liquidity_filter(q))
            .collect();

        for short in short_candidates {
            // For each short strike, find valid long strikes
            for width in &self.config.spread_widths {
                let long_strike = short.strike + *width;

                // Find the long call at this strike
                if let Some(long) = chain.calls.iter().find(|q| q.strike == long_strike) {
                    if !self.passes_liquidity_filter(long) {
                        continue;
                    }

                    // Calculate credit
                    let credit = short.bid - long.ask;
                    if credit <= Decimal::ZERO {
                        continue;
                    }

                    // Check minimum credit percentage
                    let credit_f64: f64 = credit.try_into().unwrap_or(0.0);
                    let width_f64: f64 = (*width).try_into().unwrap_or(1.0);
                    let credit_pct = credit_f64 / width_f64 * 100.0;

                    if credit_pct < self.config.min_credit_pct {
                        continue;
                    }

                    // Calculate max loss
                    let max_loss = *width - credit;

                    // Calculate EV score
                    let delta = short.greeks.delta.abs();
                    let prob_otm = 1.0 - delta;
                    let ev_score = credit_pct * prob_otm - (100.0 - credit_pct) * delta;

                    candidates.push(SpreadCandidate {
                        ticker: snapshot.ticker.clone(),
                        date: snapshot.date,
                        short_strike: short.strike,
                        long_strike: long.strike,
                        expiration: chain.expiration,
                        dte: chain.dte,
                        short_delta: short.greeks.delta.abs(),
                        long_delta: long.greeks.delta.abs(),
                        short_bid: short.bid,
                        long_ask: long.ask,
                        credit,
                        width: *width,
                        credit_pct,
                        max_loss,
                        short_iv: short.mid_iv,
                        ev_score,
                        underlying_price: snapshot.underlying_price,
                        option_type: OptionType::Call,
                    });
                }
            }
        }

        candidates
    }

    /// Get the best candidate (highest EV score).
    pub fn best_candidate<'a>(&self, candidates: &'a [SpreadCandidate]) -> Option<&'a SpreadCandidate> {
        candidates.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SpreadScreenerConfig::default();
        assert_eq!(config.min_dte, 30);
        assert_eq!(config.max_dte, 45);
        assert_eq!(config.min_short_delta, 0.20);
        assert_eq!(config.max_short_delta, 0.30);
    }

    #[test]
    fn test_spread_candidate_return_on_risk() {
        let candidate = SpreadCandidate {
            ticker: "SPY".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            short_strike: Decimal::from(470),
            long_strike: Decimal::from(465),
            expiration: NaiveDate::from_ymd_opt(2024, 2, 16).unwrap(),
            dte: 32,
            short_delta: 0.25,
            long_delta: 0.15,
            short_bid: Decimal::from(2),
            long_ask: Decimal::from(1),
            credit: Decimal::from(1),
            width: Decimal::from(5),
            credit_pct: 20.0,
            max_loss: Decimal::from(4),
            short_iv: 0.18,
            ev_score: 15.0,
            underlying_price: Decimal::from(480),
            option_type: OptionType::Put,
        };

        // $1 credit / $4 max loss = 25% RoR
        assert_eq!(candidate.return_on_risk(), 25.0);
    }

    #[test]
    fn test_spread_candidate_distance() {
        let candidate = SpreadCandidate {
            ticker: "SPY".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            short_strike: Decimal::from(460),
            long_strike: Decimal::from(455),
            expiration: NaiveDate::from_ymd_opt(2024, 2, 16).unwrap(),
            dte: 32,
            short_delta: 0.25,
            long_delta: 0.15,
            short_bid: Decimal::from(2),
            long_ask: Decimal::from(1),
            credit: Decimal::from(1),
            width: Decimal::from(5),
            credit_pct: 20.0,
            max_loss: Decimal::from(4),
            short_iv: 0.18,
            ev_score: 15.0,
            underlying_price: Decimal::from(480),
            option_type: OptionType::Put,
        };

        // (480 - 460) / 480 * 100 = 4.17%
        let dist = candidate.distance_pct();
        assert!((dist - 4.17).abs() < 0.1);
    }
}
