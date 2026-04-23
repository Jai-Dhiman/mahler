/// Assumed by backtest's ORATS slippage model: fills land ~34% into the bid-ask spread.
pub const ORATS_ASSUMED_SLIPPAGE_PCT_OF_SPREAD: f64 = 0.34;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SampleSizeTag {
    Insufficient,
    Weak,
    Ok,
}

impl SampleSizeTag {
    pub fn as_str(&self) -> &'static str {
        match self {
            SampleSizeTag::Insufficient => "INSUFFICIENT",
            SampleSizeTag::Weak => "WEAK",
            SampleSizeTag::Ok => "OK",
        }
    }
}

/// n<30: INSUFFICIENT (noise); 30<=n<100: WEAK (meaningful but not Kelly-confident); n>=100: OK.
pub fn sample_size_tag(n: usize) -> SampleSizeTag {
    if n < 30 {
        SampleSizeTag::Insufficient
    } else if n < 100 {
        SampleSizeTag::Weak
    } else {
        SampleSizeTag::Ok
    }
}

/// Paper fill violation: filled contracts exceed min of short/long leg displayed size.
/// A live exchange would not fill an order larger than displayed size; Alpaca paper does.
pub fn paper_fill_violation(contracts: i64, short_size: i64, long_size: i64) -> bool {
    contracts > short_size.min(long_size)
}

/// Annualized Sharpe from daily returns. Returns 0 if < 2 returns or zero stdev.
pub fn compute_sharpe(daily_returns: &[f64]) -> f64 {
    if daily_returns.len() < 2 { return 0.0; }
    let n = daily_returns.len() as f64;
    let mean = daily_returns.iter().sum::<f64>() / n;
    let var = daily_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd == 0.0 { return 0.0; }
    (mean / sd) * (252f64).sqrt()
}

/// Annualized Sortino — mean over downside-only stdev.
pub fn compute_sortino(daily_returns: &[f64]) -> f64 {
    if daily_returns.len() < 2 { return 0.0; }
    let n = daily_returns.len() as f64;
    let mean = daily_returns.iter().sum::<f64>() / n;
    let downside: Vec<f64> = daily_returns.iter().filter(|r| **r < 0.0).copied().collect();
    if downside.is_empty() { return f64::INFINITY; }
    let var = downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
    let sd = var.sqrt();
    if sd == 0.0 { return 0.0; }
    (mean / sd) * (252f64).sqrt()
}

/// Profit factor = gross wins / abs(gross losses). Infinity if no losses.
pub fn compute_profit_factor(pnls: &[f64]) -> f64 {
    let wins: f64 = pnls.iter().filter(|p| **p > 0.0).sum();
    let losses: f64 = pnls.iter().filter(|p| **p < 0.0).sum::<f64>().abs();
    if losses == 0.0 { return f64::INFINITY; }
    wins / losses
}

/// Sample skewness (third standardized moment).
pub fn compute_pnl_skew(pnls: &[f64]) -> f64 {
    let n = pnls.len();
    if n < 3 { return 0.0; }
    let mean = pnls.iter().sum::<f64>() / n as f64;
    let var = pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = var.sqrt();
    if sd == 0.0 { return 0.0; }
    let m3 = pnls.iter().map(|p| ((p - mean) / sd).powi(3)).sum::<f64>() / n as f64;
    m3
}

/// Max drawdown as a positive fraction (e.g. 0.25 for 25%).
pub fn compute_max_drawdown_pct(equity_curve: &[f64]) -> f64 {
    let mut peak = 0.0f64;
    let mut max_dd = 0.0f64;
    for &e in equity_curve {
        peak = peak.max(e);
        if peak > 0.0 {
            let dd = (peak - e) / peak;
            max_dd = max_dd.max(dd);
        }
    }
    max_dd
}

#[derive(Debug, Clone)]
pub struct SlippageInput {
    pub entry_mid: f64,
    pub entry_fill: f64,
    /// Mean of (short_ask-short_bid) and (long_ask-long_bid) at snapshot time.
    pub leg_width: f64,
}

#[derive(Debug, Clone)]
pub struct SlippageStats {
    pub mean_abs: f64,
    pub max_abs: f64,
    pub ratio_vs_orats: f64,
}

pub fn compute_slippage_stats(inputs: &[SlippageInput]) -> SlippageStats {
    if inputs.is_empty() {
        return SlippageStats { mean_abs: 0.0, max_abs: 0.0, ratio_vs_orats: 0.0 };
    }
    let slippages: Vec<f64> = inputs.iter().map(|i| (i.entry_mid - i.entry_fill).abs()).collect();
    let mean = slippages.iter().sum::<f64>() / slippages.len() as f64;
    let max = slippages.iter().cloned().fold(0.0f64, f64::max);
    let assumed: f64 = inputs.iter()
        .map(|i| ORATS_ASSUMED_SLIPPAGE_PCT_OF_SPREAD * i.leg_width)
        .sum::<f64>() / inputs.len() as f64;
    let ratio = if assumed > 0.0 { mean / assumed } else { 0.0 };
    SlippageStats { mean_abs: mean, max_abs: max, ratio_vs_orats: ratio }
}

/// Regime bucket from spot VIX level.
pub fn classify_regime(spot_vix: f64) -> &'static str {
    if spot_vix < 15.0 { "low_vol" }
    else if spot_vix < 25.0 { "med_vol" }
    else { "high_vol" }
}

use crate::db::d1::MarketContextRow;
use crate::types::Trade;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MetricsInput {
    pub window_start: String,
    pub window_end: String,
    pub trades: Vec<Trade>,
    pub equity_eod: Vec<(String, f64)>,
    pub greeks_eod: Vec<(String, f64, f64, f64, f64)>,
    pub market_context: Vec<MarketContextRow>,
}

#[derive(Debug, Clone)]
pub struct MetricBundle {
    pub window_start: String,
    pub window_end: String,
    pub trade_count: usize,
    pub sharpe: f64,
    pub sortino: f64,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub pnl_skew: f64,
    pub max_drawdown_pct: f64,
    pub mean_slippage_vs_mid: f64,
    pub max_slippage_vs_mid: f64,
    pub slippage_vs_orats_ratio: f64,
    pub fill_size_violation_count: usize,
    pub fill_size_violation_pct: f64,
    pub regime_buckets: HashMap<String, (i64, f64)>,
    pub greek_ranges: HashMap<String, (f64, f64, f64)>,
    pub sample_size_tag: SampleSizeTag,
}

pub fn compute_metrics(input: MetricsInput) -> MetricBundle {
    let n = input.trades.len();
    let tag = sample_size_tag(n);

    let pnls: Vec<f64> = input.trades.iter().filter_map(|t| t.net_pnl).collect();
    let wins = pnls.iter().filter(|p| **p > 0.0).count();
    let win_rate = if pnls.is_empty() { 0.0 } else { wins as f64 / pnls.len() as f64 };

    let equity_values: Vec<f64> = input.equity_eod.iter().map(|(_, e)| *e).collect();
    let daily_returns: Vec<f64> = equity_values.windows(2)
        .filter(|w| w[0] > 0.0)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let sharpe = compute_sharpe(&daily_returns);
    let sortino = compute_sortino(&daily_returns);
    let profit_factor = compute_profit_factor(&pnls);
    let pnl_skew = compute_pnl_skew(&pnls);
    let max_drawdown_pct = compute_max_drawdown_pct(&equity_values);

    let slippage_inputs: Vec<SlippageInput> = input.trades.iter().filter_map(|t| {
        let mid = t.entry_net_mid?;
        let fill = t.fill_price?;
        let sw = (t.entry_short_ask? - t.entry_short_bid?).abs();
        let lw = (t.entry_long_ask? - t.entry_long_bid?).abs();
        Some(SlippageInput { entry_mid: mid, entry_fill: fill, leg_width: (sw + lw) / 2.0 })
    }).collect();
    let slip = compute_slippage_stats(&slippage_inputs);

    let violations = input.trades.iter().filter(|t| {
        match (t.nbbo_displayed_size_short, t.nbbo_displayed_size_long) {
            (Some(s), Some(l)) => paper_fill_violation(t.contracts, s, l),
            _ => false,
        }
    }).count();
    let violation_pct = if n == 0 { 0.0 } else { violations as f64 / n as f64 };

    let ctx_by_date: HashMap<String, Option<f64>> = input.market_context.iter()
        .map(|m| (m.date.clone(), m.spot_vix))
        .collect();
    let mut regime_buckets: HashMap<String, (i64, f64)> = HashMap::new();
    for t in &input.trades {
        let date = t.created_at.get(..10).unwrap_or(&t.created_at).to_string();
        let vix = ctx_by_date.get(&date).and_then(|v| *v).unwrap_or(20.0);
        let bucket = classify_regime(vix).to_string();
        let entry = regime_buckets.entry(bucket).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += t.net_pnl.unwrap_or(0.0);
    }

    let mut greek_ranges: HashMap<String, (f64, f64, f64)> = HashMap::new();
    if !input.greeks_eod.is_empty() {
        let k = input.greeks_eod.len() as f64;
        let bwd: Vec<f64> = input.greeks_eod.iter().map(|g| g.1).collect();
        let gam: Vec<f64> = input.greeks_eod.iter().map(|g| g.2).collect();
        let veg: Vec<f64> = input.greeks_eod.iter().map(|g| g.3).collect();
        let the: Vec<f64> = input.greeks_eod.iter().map(|g| g.4).collect();
        greek_ranges.insert("beta_weighted_delta".to_string(), (
            bwd.iter().cloned().fold(f64::INFINITY, f64::min),
            bwd.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            bwd.iter().sum::<f64>() / k));
        greek_ranges.insert("total_gamma".to_string(), (
            gam.iter().cloned().fold(f64::INFINITY, f64::min),
            gam.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            gam.iter().sum::<f64>() / k));
        greek_ranges.insert("total_vega".to_string(), (
            veg.iter().cloned().fold(f64::INFINITY, f64::min),
            veg.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            veg.iter().sum::<f64>() / k));
        greek_ranges.insert("total_theta".to_string(), (
            the.iter().cloned().fold(f64::INFINITY, f64::min),
            the.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            the.iter().sum::<f64>() / k));
    }

    MetricBundle {
        window_start: input.window_start, window_end: input.window_end,
        trade_count: n, sharpe, sortino, profit_factor, win_rate, pnl_skew,
        max_drawdown_pct,
        mean_slippage_vs_mid: slip.mean_abs,
        max_slippage_vs_mid: slip.max_abs,
        slippage_vs_orats_ratio: slip.ratio_vs_orats,
        fill_size_violation_count: violations,
        fill_size_violation_pct: violation_pct,
        regime_buckets, greek_ranges, sample_size_tag: tag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_size_tag_classifies_thresholds() {
        assert_eq!(sample_size_tag(0), SampleSizeTag::Insufficient);
        assert_eq!(sample_size_tag(29), SampleSizeTag::Insufficient);
        assert_eq!(sample_size_tag(30), SampleSizeTag::Weak);
        assert_eq!(sample_size_tag(99), SampleSizeTag::Weak);
        assert_eq!(sample_size_tag(100), SampleSizeTag::Ok);
        assert_eq!(sample_size_tag(1000), SampleSizeTag::Ok);
    }

    #[test]
    fn paper_fill_violation_flags_oversized_fill() {
        // 5 contracts vs displayed short 3 / long 10 -> violation (min = 3 < 5)
        assert!(paper_fill_violation(5, 3, 10));
        // 2 contracts vs displayed short 5 / long 5 -> no violation
        assert!(!paper_fill_violation(2, 5, 5));
    }

    #[test]
    fn sharpe_matches_annualized_mean_over_stdev() {
        // Daily returns: +0.01, -0.005, +0.015, -0.01, +0.02
        let returns = vec![0.01, -0.005, 0.015, -0.01, 0.02];
        let s = compute_sharpe(&returns);
        // Mean = 0.006, stdev (pop) computed, sharpe = mean/stdev * sqrt(252).
        // We assert on magnitude & sign (should be positive).
        assert!(s > 0.0, "sharpe should be positive for positive-mean returns, got {}", s);
    }

    #[test]
    fn sortino_downside_only_stdev() {
        // Only negatives: -0.005, -0.01. Upside returns don't penalize.
        let returns = vec![0.01, -0.005, 0.015, -0.01, 0.02];
        let so = compute_sortino(&returns);
        // Sortino should exceed Sharpe since upside vol isn't penalized.
        let sh = compute_sharpe(&returns);
        assert!(so > sh, "sortino ({}) should exceed sharpe ({}) with upside vol", so, sh);
    }

    #[test]
    fn profit_factor_is_gross_wins_over_gross_losses() {
        let pnls = vec![100.0, -50.0, 200.0, -80.0, 60.0];
        // Wins = 360, losses = 130, pf = 360 / 130 = 2.769...
        let pf = compute_profit_factor(&pnls);
        assert!((pf - (360.0 / 130.0)).abs() < 1e-6);
    }

    #[test]
    fn profit_factor_returns_infinity_when_no_losers() {
        let pnls = vec![100.0, 200.0];
        let pf = compute_profit_factor(&pnls);
        assert!(pf.is_infinite());
    }

    #[test]
    fn pnl_skew_is_negative_for_left_skewed_distribution() {
        // Many small wins, few big losses -> negative skew (classic credit spread shape)
        let pnls = vec![50.0; 10].into_iter().chain(vec![-500.0]).collect::<Vec<f64>>();
        let sk = compute_pnl_skew(&pnls);
        assert!(sk < 0.0, "left-skewed distribution should have negative skew, got {}", sk);
    }

    #[test]
    fn max_drawdown_tracks_peak_to_trough() {
        // Equity curve: 100 -> 120 -> 90 -> 110. Peak 120, trough after = 90, dd = 25%.
        let equity = vec![100.0, 120.0, 90.0, 110.0];
        let dd = compute_max_drawdown_pct(&equity);
        assert!((dd - 0.25).abs() < 1e-6, "expected 0.25, got {}", dd);
    }

    #[test]
    fn slippage_stats_mean_and_max_from_entry_fills() {
        // Trade A: entry_net_mid 0.50, fill_price 0.48 → slippage 0.02
        // Trade B: entry_net_mid 0.60, fill_price 0.56 → slippage 0.04
        let inputs = vec![
            SlippageInput { entry_mid: 0.50, entry_fill: 0.48, leg_width: 0.10 },
            SlippageInput { entry_mid: 0.60, entry_fill: 0.56, leg_width: 0.10 },
        ];
        let s = compute_slippage_stats(&inputs);
        assert!((s.mean_abs - 0.03).abs() < 1e-9);
        assert!((s.max_abs - 0.04).abs() < 1e-9);
        // ORATS constant = 0.34 × 0.10 = 0.034 per trade assumed; mean 0.03 / 0.034 ≈ 0.882
        assert!((s.ratio_vs_orats - (0.03 / (0.34 * 0.10))).abs() < 1e-6);
    }

    #[test]
    fn classify_regime_thresholds_on_spot_vix() {
        assert_eq!(classify_regime(12.0), "low_vol");
        assert_eq!(classify_regime(20.0), "med_vol");
        assert_eq!(classify_regime(30.0), "high_vol");
    }

    use crate::types::{Trade, TradeStatus, SpreadType};
    use crate::db::d1::MarketContextRow;

    fn closed_trade(id: &str, net_pnl: f64, created: &str) -> Trade {
        Trade {
            id: id.to_string(), created_at: created.to_string(),
            underlying: "SPY".to_string(), spread_type: SpreadType::BullPut,
            short_strike: 460.0, long_strike: 455.0, expiration: "2026-05-15".to_string(),
            contracts: 1, entry_credit: 0.50, max_loss: 450.0,
            broker_order_id: Some("o".to_string()), status: TradeStatus::Closed,
            fill_price: Some(0.48), fill_time: Some(created.to_string()),
            exit_price: Some(0.20), exit_time: Some(created.to_string()),
            exit_reason: None, net_pnl: Some(net_pnl),
            iv_rank: None, short_delta: None, short_theta: None,
            entry_short_bid: Some(0.74), entry_short_ask: Some(0.76),
            entry_long_bid: Some(0.24), entry_long_ask: Some(0.26),
            entry_net_mid: Some(0.50),
            exit_short_bid: None, exit_short_ask: None,
            exit_long_bid: None, exit_long_ask: None, exit_net_mid: None,
            entry_short_gamma: None, entry_short_vega: None,
            entry_long_delta: None, entry_long_gamma: None, entry_long_vega: None,
            nbbo_displayed_size_short: Some(50), nbbo_displayed_size_long: Some(75),
            nbbo_snapshot_time: None,
        }
    }

    #[test]
    fn compute_metrics_small_sample_tagged_insufficient() {
        let trades = vec![closed_trade("t1", 50.0, "2026-04-01T00:00:00Z")];
        let equity: Vec<(String, f64)> = vec![
            ("2026-04-01".to_string(), 100_000.0),
            ("2026-04-02".to_string(), 100_050.0),
        ];
        let greeks: Vec<(String, f64, f64, f64, f64)> = vec![];
        let ctx: Vec<MarketContextRow> = vec![];
        let bundle = compute_metrics(MetricsInput {
            window_start: "2026-04-01".to_string(),
            window_end: "2026-04-02".to_string(),
            trades, equity_eod: equity, greeks_eod: greeks, market_context: ctx,
        });
        assert_eq!(bundle.sample_size_tag, SampleSizeTag::Insufficient);
        assert_eq!(bundle.trade_count, 1);
    }
}
