use chrono::Utc;
use crate::broker::types::{OptionsChain, OptionType};

#[derive(Debug, Clone, PartialEq)]
pub struct NbboSnapshot {
    pub short_bid: f64,
    pub short_ask: f64,
    pub long_bid: f64,
    pub long_ask: f64,
    pub net_mid: f64,
    pub short_bid_size: Option<i64>,
    pub short_ask_size: Option<i64>,
    pub long_bid_size: Option<i64>,
    pub long_ask_size: Option<i64>,
    pub snapshot_time: String,
}

/// Snapshot bid/ask/size for both legs of a credit spread from an options chain.
///
/// Returns None if either leg cannot be found (missing strike/type/expiration match).
/// `net_mid` is defined as (short_mid - long_mid) -- positive for a credit received.
pub fn snapshot_spread_nbbo(
    chain: &OptionsChain,
    short_strike: f64,
    long_strike: f64,
    option_type: OptionType,
) -> Option<NbboSnapshot> {
    let short = chain.contracts.iter().find(|c| {
        (c.strike - short_strike).abs() < 0.01 && c.option_type == option_type
    })?;
    let long = chain.contracts.iter().find(|c| {
        (c.strike - long_strike).abs() < 0.01 && c.option_type == option_type
    })?;
    let short_mid = (short.bid + short.ask) / 2.0;
    let long_mid = (long.bid + long.ask) / 2.0;
    Some(NbboSnapshot {
        short_bid: short.bid, short_ask: short.ask,
        long_bid: long.bid, long_ask: long.ask,
        net_mid: short_mid - long_mid,
        short_bid_size: short.bid_size,
        short_ask_size: short.ask_size,
        long_bid_size: long.bid_size,
        long_ask_size: long.ask_size,
        snapshot_time: Utc::now().to_rfc3339(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::types::{OptionContract, OptionType, OptionsChain};

    fn mk_contract(strike: f64, ot: OptionType, bid: f64, ask: f64, bs: i64, r#as: i64) -> OptionContract {
        OptionContract {
            symbol: format!("X{}{:?}", strike, ot),
            underlying: "SPY".to_string(),
            expiration: "2026-05-15".to_string(),
            strike,
            option_type: ot,
            bid, ask, last: (bid+ask)/2.0,
            volume: 0, open_interest: 0,
            implied_volatility: Some(0.20),
            delta: Some(-0.25), gamma: Some(0.01), theta: Some(-0.03), vega: Some(0.20),
            bid_size: Some(bs), ask_size: Some(r#as),
        }
    }

    #[test]
    fn snapshots_both_legs_of_bull_put_spread() {
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string()],
            contracts: vec![
                mk_contract(460.0, OptionType::Put, 0.74, 0.76, 50, 55),
                mk_contract(455.0, OptionType::Put, 0.24, 0.26, 60, 75),
            ],
        };
        let snap = snapshot_spread_nbbo(&chain, 460.0, 455.0, OptionType::Put)
            .expect("snapshot exists");
        assert!((snap.short_bid - 0.74).abs() < 1e-9);
        assert!((snap.short_ask - 0.76).abs() < 1e-9);
        assert!((snap.long_bid - 0.24).abs() < 1e-9);
        assert!((snap.long_ask - 0.26).abs() < 1e-9);
        assert_eq!(snap.short_bid_size, Some(50));
        assert_eq!(snap.long_ask_size, Some(75));
        assert!((snap.net_mid - 0.50).abs() < 1e-9);
    }
}
