use crate::analysis::greeks::{black_scholes_delta, days_to_expiry, years_to_expiry};
use crate::analysis::iv_rank::IVMetrics;
use crate::broker::types::{OptionContract, OptionType, OptionsChain};
use crate::config::SpreadConfig;
use crate::types::{CreditSpread, SpreadType};

#[derive(Debug, Clone)]
pub struct ScreenerConfig {
    pub min_dte: i64,
    pub max_dte: i64,
    pub min_delta: f64,
    pub max_delta: f64,
    pub min_credit_pct: f64,
    pub min_spread_width: f64,
    pub max_spread_width: f64,
    pub min_open_interest: i64,
    pub min_volume: i64,
    pub max_bid_ask_spread_pct: f64,
}

impl Default for ScreenerConfig {
    fn default() -> Self {
        let cfg = SpreadConfig::default();
        ScreenerConfig {
            min_dte: cfg.min_dte,
            max_dte: cfg.max_dte,
            min_delta: cfg.min_delta,
            max_delta: cfg.max_delta,
            min_credit_pct: cfg.min_credit_pct,
            min_spread_width: cfg.min_spread_width,
            max_spread_width: 10.0,
            min_open_interest: 100,
            min_volume: 1,
            max_bid_ask_spread_pct: 0.08,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredSpread {
    pub spread: CreditSpread,
    pub score: f64,
    pub probability_otm: f64,
    pub expected_value: f64,
}

pub struct OptionsScreener {
    config: ScreenerConfig,
}

impl Default for OptionsScreener {
    fn default() -> Self {
        OptionsScreener { config: ScreenerConfig::default() }
    }
}

impl OptionsScreener {
    pub fn new(config: ScreenerConfig) -> Self {
        OptionsScreener { config }
    }

    /// Screen an options chain for credit spread opportunities.
    ///
    /// Returns spreads sorted by score descending.
    pub fn screen_chain(&self, chain: &OptionsChain, iv_metrics: &IVMetrics) -> Vec<ScoredSpread> {
        let valid_expirations: Vec<&String> = chain
            .expirations
            .iter()
            .filter(|exp| {
                if let Some(dte) = days_to_expiry(exp) {
                    dte >= self.config.min_dte && dte <= self.config.max_dte
                } else {
                    false
                }
            })
            .collect();

        let mut results = Vec::new();

        for exp in valid_expirations {
            let tte = years_to_expiry(exp);

            let mut puts = self.filter_liquidity(chain.get_puts(exp));
            puts.sort_by(|a, b| b.strike.partial_cmp(&a.strike).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..puts.len() {
                let short = &puts[i];
                let short_delta = self.get_delta(short, chain.underlying_price, tte, iv_metrics.current_iv, "put");
                let abs_delta = short_delta.abs();
                if abs_delta < self.config.min_delta || abs_delta > self.config.max_delta {
                    continue;
                }

                for j in (i + 1)..puts.len() {
                    let long = &puts[j];
                    let width = short.strike - long.strike;
                    if width < self.config.min_spread_width || width > self.config.max_spread_width {
                        continue;
                    }

                    let credit = short.mid_price() - long.mid_price();
                    if credit <= 0.0 {
                        continue;
                    }
                    if credit / width < self.config.min_credit_pct {
                        continue;
                    }

                    let spread = CreditSpread {
                        underlying: chain.underlying.clone(),
                        spread_type: SpreadType::BullPut,
                        short_strike: short.strike,
                        long_strike: long.strike,
                        expiration: exp.clone(),
                        entry_credit: credit,
                        short_delta: Some(-abs_delta),
                        short_theta: short.delta.map(|_| 0.0),
                        short_iv: short.implied_volatility,
                        long_iv: long.implied_volatility,
                    };

                    let scored = self.score_spread(&spread, iv_metrics, abs_delta);
                    results.push(scored);
                    break;
                }
            }

            let mut calls = self.filter_liquidity(chain.get_calls(exp));
            calls.sort_by(|a, b| a.strike.partial_cmp(&b.strike).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..calls.len() {
                let short = &calls[i];
                let short_delta = self.get_delta(short, chain.underlying_price, tte, iv_metrics.current_iv, "call");
                let abs_delta = short_delta.abs();
                if abs_delta < self.config.min_delta || abs_delta > self.config.max_delta {
                    continue;
                }

                for j in (i + 1)..calls.len() {
                    let long = &calls[j];
                    let width = long.strike - short.strike;
                    if width < self.config.min_spread_width || width > self.config.max_spread_width {
                        continue;
                    }

                    let credit = short.mid_price() - long.mid_price();
                    if credit <= 0.0 {
                        continue;
                    }
                    if credit / width < self.config.min_credit_pct {
                        continue;
                    }

                    let spread = CreditSpread {
                        underlying: chain.underlying.clone(),
                        spread_type: SpreadType::BearCall,
                        short_strike: short.strike,
                        long_strike: long.strike,
                        expiration: exp.clone(),
                        entry_credit: credit,
                        short_delta: Some(abs_delta),
                        short_theta: None,
                        short_iv: short.implied_volatility,
                        long_iv: long.implied_volatility,
                    };

                    let scored = self.score_spread(&spread, iv_metrics, abs_delta);
                    results.push(scored);
                    break;
                }
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn filter_liquidity(&self, contracts: Vec<OptionContract>) -> Vec<OptionContract> {
        contracts
            .into_iter()
            .filter(|c| {
                c.open_interest >= self.config.min_open_interest
                    && c.volume >= self.config.min_volume
                    && c.bid > 0.0
                    && c.ask > 0.0
                    && c.bid_ask_spread_pct() <= self.config.max_bid_ask_spread_pct
            })
            .collect()
    }

    fn get_delta(&self, contract: &OptionContract, spot: f64, tte: f64, iv: f64, option_type: &str) -> f64 {
        if let Some(d) = contract.delta {
            return d;
        }
        black_scholes_delta(option_type, spot, contract.strike, tte, iv, 0.05)
    }

    fn score_spread(&self, spread: &CreditSpread, iv_metrics: &IVMetrics, short_delta: f64) -> ScoredSpread {
        let prob_otm = 1.0 - short_delta;
        let credit_per_contract = spread.entry_credit * 100.0;
        let max_loss = spread.max_loss_per_contract();
        let expected_value = credit_per_contract * prob_otm - max_loss * (1.0 - prob_otm);

        let iv_score = iv_metrics.iv_percentile / 100.0;
        let delta_score = (1.0 - (short_delta - 0.10).abs() * 10.0).max(0.0).min(1.0);
        let credit_score = (spread.entry_credit / spread.width()).min(0.5) * 2.0;
        let ev_score = (expected_value / (spread.width() * 100.0)).max(0.0).min(1.0);

        let score = (iv_score + delta_score + credit_score + ev_score) / 4.0;

        ScoredSpread { spread: spread.clone(), score, probability_otm: prob_otm, expected_value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::iv_rank::IVMetrics;
    use crate::broker::types::{OptionContract, OptionType, OptionsChain};

    fn make_put(strike: f64, delta: f64, bid: f64, ask: f64, expiration: &str) -> OptionContract {
        OptionContract {
            symbol: format!("SPY{}", strike),
            underlying: "SPY".to_string(),
            expiration: expiration.to_string(),
            strike,
            option_type: OptionType::Put,
            bid,
            ask,
            last: (bid + ask) / 2.0,
            volume: 200,
            open_interest: 1000,
            implied_volatility: Some(0.22),
            delta: Some(-delta),
            gamma: None,
            theta: None,
            vega: None,
        }
    }

    fn good_iv() -> IVMetrics {
        IVMetrics { current_iv: 0.25, iv_rank: 60.0, iv_percentile: 65.0, iv_high: 0.40, iv_low: 0.12 }
    }

    fn exp_40dte() -> String {
        use chrono::{Duration, Utc};
        (Utc::now() + Duration::days(40)).format("%Y-%m-%d").to_string()
    }

    #[test]
    fn rejects_options_with_delta_outside_range() {
        // Deltas 0.40 and 0.35 are above the max_delta of 0.30
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(460.0, 0.40, 2.00, 2.10, &exp),
                make_put(455.0, 0.35, 1.80, 1.90, &exp),
            ],
        };
        let results = OptionsScreener::default().screen_chain(&chain, &good_iv());
        assert!(results.is_empty(), "should reject options with delta above max_delta 0.30");
    }

    #[test]
    fn accepts_valid_bull_put_spread() {
        // Short at delta 0.27 (in 0.25-0.30 range), long at lower strike
        // Short: mid 2.55, bid-ask 3.9%; Long: mid 1.35, bid-ask 7.4%
        // Width = 5, credit = 1.20, credit/width = 24% > 10% minimum
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(470.0, 0.27, 2.50, 2.60, &exp),
                make_put(465.0, 0.15, 1.30, 1.40, &exp),
            ],
        };
        let results = OptionsScreener::default().screen_chain(&chain, &good_iv());
        assert!(!results.is_empty(), "should find a valid bull put spread");
        assert!(results[0].score > 0.0);
        assert!(results[0].score <= 1.0);
    }

    #[test]
    fn higher_iv_percentile_scores_higher() {
        let exp = exp_40dte();
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec![exp.clone()],
            contracts: vec![
                make_put(470.0, 0.27, 2.50, 2.60, &exp),
                make_put(465.0, 0.15, 1.30, 1.40, &exp),
            ],
        };
        let screener = OptionsScreener::default();
        let low_iv = IVMetrics { current_iv: 0.15, iv_rank: 20.0, iv_percentile: 20.0, iv_high: 0.40, iv_low: 0.12 };
        let high_iv = IVMetrics { current_iv: 0.35, iv_rank: 80.0, iv_percentile: 85.0, iv_high: 0.40, iv_low: 0.12 };

        let low_results = screener.screen_chain(&chain, &low_iv);
        let high_results = screener.screen_chain(&chain, &high_iv);

        assert!(!low_results.is_empty() && !high_results.is_empty(), "both should find spreads");
        assert!(
            high_results[0].score > low_results[0].score,
            "high IV should score higher: {} vs {}",
            high_results[0].score,
            low_results[0].score
        );
    }
}
