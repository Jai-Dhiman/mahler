use std::collections::HashMap;
use crate::broker::types::{OptionType, OptionsChain};
use crate::types::{SpreadType, Trade};

/// Static beta vs SPY. Quarterly human review — see GO_LIVE_GATE.md.
pub const BETA_TABLE: &[(&str, f64)] = &[
    ("SPY", 1.00),
    ("QQQ", 1.15),
    ("IWM", 1.25),
];

/// Look up beta for an underlying; returns 1.0 when not in table.
pub fn lookup_beta(underlying: &str) -> f64 {
    BETA_TABLE.iter()
        .find(|(u, _)| *u == underlying)
        .map(|(_, b)| *b)
        .unwrap_or(1.0)
}

/// Sum of (delta x beta) across underlyings.
pub fn beta_weighted_delta(contribs: &[(String, f64)]) -> f64 {
    contribs.iter().map(|(u, d)| d * lookup_beta(u)).sum()
}

#[derive(Debug, Clone)]
pub struct PortfolioGreeks {
    pub beta_weighted_delta: f64,
    pub total_gamma: f64,
    pub total_vega: f64,
    pub total_theta: f64,
    pub delta_by_underlying: HashMap<String, f64>,
    pub max_gamma_single_position: f64,
    pub max_vega_single_position: f64,
    pub open_position_count: usize,
}

pub struct PortfolioGreeksAggregator;

impl PortfolioGreeksAggregator {
    pub fn compute(trades: &[Trade], chains: &HashMap<String, OptionsChain>) -> PortfolioGreeks {
        let mut delta_contribs: HashMap<String, f64> = HashMap::new();
        let mut total_gamma = 0.0;
        let mut total_vega = 0.0;
        let mut total_theta = 0.0;
        let mut max_gamma = 0.0f64;
        let mut max_vega = 0.0f64;

        for trade in trades {
            let chain = match chains.get(&trade.underlying) {
                Some(c) => c, None => continue,
            };
            let option_type = match trade.spread_type {
                SpreadType::BullPut => OptionType::Put,
                SpreadType::BearCall => OptionType::Call,
            };
            let short = chain.contracts.iter().find(|c| {
                (c.strike - trade.short_strike).abs() < 0.01
                    && c.option_type == option_type
                    && c.expiration == trade.expiration
            });
            let long = chain.contracts.iter().find(|c| {
                (c.strike - trade.long_strike).abs() < 0.01
                    && c.option_type == option_type
                    && c.expiration == trade.expiration
            });
            let (Some(s), Some(l)) = (short, long) else { continue; };

            // Net spread greek = short_leg_greek - long_leg_greek (short is the dominant leg).
            let mult = trade.contracts as f64 * 100.0;
            let d = (s.delta.unwrap_or(0.0) - l.delta.unwrap_or(0.0)) * mult;
            let g = (s.gamma.unwrap_or(0.0) - l.gamma.unwrap_or(0.0)) * mult;
            let v = (s.vega.unwrap_or(0.0) - l.vega.unwrap_or(0.0)) * mult;
            let th = (s.theta.unwrap_or(0.0) - l.theta.unwrap_or(0.0)) * mult;

            *delta_contribs.entry(trade.underlying.clone()).or_insert(0.0) += d;
            total_gamma += g;
            total_vega += v;
            total_theta += th;
            max_gamma = max_gamma.max(g.abs());
            max_vega = max_vega.max(v.abs());
        }

        let contribs_vec: Vec<(String, f64)> = delta_contribs.clone().into_iter().collect();
        PortfolioGreeks {
            beta_weighted_delta: beta_weighted_delta(&contribs_vec),
            total_gamma, total_vega, total_theta,
            delta_by_underlying: delta_contribs,
            max_gamma_single_position: max_gamma,
            max_vega_single_position: max_vega,
            open_position_count: trades.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::types::{OptionContract, OptionType, OptionsChain};
    use crate::types::{Trade, TradeStatus, SpreadType};
    use std::collections::HashMap;

    fn open_trade(id: &str, underlying: &str, contracts: i64, short: f64, long: f64) -> Trade {
        Trade {
            id: id.to_string(), created_at: "2026-04-22T00:00:00Z".to_string(),
            underlying: underlying.to_string(), spread_type: SpreadType::BullPut,
            short_strike: short, long_strike: long, expiration: "2026-05-15".to_string(),
            contracts, entry_credit: 0.50, max_loss: 450.0,
            broker_order_id: Some("o".to_string()), status: TradeStatus::Open,
            fill_price: Some(0.50), fill_time: Some("2026-04-22T00:00:00Z".to_string()),
            exit_price: None, exit_time: None, exit_reason: None, net_pnl: None,
            iv_rank: Some(60.0), short_delta: Some(-0.25), short_theta: Some(0.05),
            entry_short_bid: None, entry_short_ask: None, entry_long_bid: None,
            entry_long_ask: None, entry_net_mid: None,
            exit_short_bid: None, exit_short_ask: None, exit_long_bid: None,
            exit_long_ask: None, exit_net_mid: None,
            entry_short_gamma: None, entry_short_vega: None,
            entry_long_delta: None, entry_long_gamma: None, entry_long_vega: None,
            nbbo_displayed_size_short: None, nbbo_displayed_size_long: None,
            nbbo_snapshot_time: None,
        }
    }

    fn put_contract(strike: f64, d: f64, g: f64, v: f64, th: f64) -> OptionContract {
        OptionContract {
            symbol: format!("S{}", strike), underlying: "SPY".to_string(),
            expiration: "2026-05-15".to_string(), strike,
            option_type: OptionType::Put, bid: 0.74, ask: 0.76, last: 0.75,
            volume: 0, open_interest: 0, implied_volatility: Some(0.20),
            delta: Some(d), gamma: Some(g), theta: Some(th), vega: Some(v),
            bid_size: Some(10), ask_size: Some(10),
        }
    }

    #[test]
    fn compute_aggregates_open_trade_greeks_with_beta_weighting() {
        // One SPY trade, 2 contracts. Short put delta -0.28 gamma 0.01 vega 0.30 theta -0.05,
        // Long put delta -0.18 gamma 0.008 vega 0.25 theta -0.03.
        // Per-contract net delta = (-0.28) - (-0.18) = -0.10 ; x 100 x 2 contracts = -20 delta_shares.
        // Beta = 1.00 for SPY, so beta-weighted delta = -20.
        let spy_chain = OptionsChain {
            underlying: "SPY".to_string(), underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string()],
            contracts: vec![
                put_contract(460.0, -0.28, 0.010, 0.30, -0.05),
                put_contract(455.0, -0.18, 0.008, 0.25, -0.03),
            ],
        };
        let trades = vec![open_trade("t1", "SPY", 2, 460.0, 455.0)];
        let mut chains = HashMap::new();
        chains.insert("SPY".to_string(), spy_chain);

        let result = PortfolioGreeksAggregator::compute(&trades, &chains);
        assert_eq!(result.open_position_count, 1);
        assert!((result.beta_weighted_delta - (-20.0)).abs() < 1e-6,
                "got {}", result.beta_weighted_delta);
    }

    #[test]
    fn beta_weighted_delta_applies_static_table() {
        // SPY delta 10, QQQ delta 5, IWM delta 4
        // beta-weighted = 10*1.00 + 5*1.15 + 4*1.25 = 10 + 5.75 + 5.0 = 20.75
        let contribs = vec![
            ("SPY".to_string(), 10.0),
            ("QQQ".to_string(), 5.0),
            ("IWM".to_string(), 4.0),
        ];
        let bwd = beta_weighted_delta(&contribs);
        assert!((bwd - 20.75).abs() < 1e-9);
    }

    #[test]
    fn unknown_underlying_uses_beta_1() {
        let contribs = vec![("UNKNOWN".to_string(), 3.0)];
        let bwd = beta_weighted_delta(&contribs);
        assert!((bwd - 3.0).abs() < 1e-9);
    }
}
