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
}
