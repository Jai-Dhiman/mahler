/// IV metrics for an underlying, derived from historical data.
#[derive(Debug, Clone)]
pub struct IVMetrics {
    pub current_iv: f64,
    /// IV Rank: (current - 52wk_low) / (52wk_high - 52wk_low) * 100
    pub iv_rank: f64,
    /// IV Percentile: % of historical days where IV was lower than current
    pub iv_percentile: f64,
    pub iv_high: f64,
    pub iv_low: f64,
}

/// Calculates IV Rank and IV Percentile from historical IV data.
///
/// Returns neutral defaults (rank=50, percentile=50) when history is empty.
/// Uses the full history slice — caller is responsible for windowing to 252 days.
pub fn calculate_iv_metrics(current_iv: f64, history: &[f64]) -> IVMetrics {
    if history.is_empty() {
        return IVMetrics {
            current_iv,
            iv_rank: 50.0,
            iv_percentile: 50.0,
            iv_high: current_iv,
            iv_low: current_iv,
        };
    }

    let iv_high = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let iv_low = history.iter().cloned().fold(f64::INFINITY, f64::min);

    let iv_rank = if (iv_high - iv_low).abs() < 1e-12 {
        50.0
    } else {
        ((current_iv - iv_low) / (iv_high - iv_low) * 100.0)
            .max(0.0)
            .min(100.0)
    };

    let days_lower = history.iter().filter(|&&iv| iv < current_iv).count();
    let iv_percentile = (days_lower as f64 / history.len() as f64) * 100.0;

    IVMetrics {
        current_iv,
        iv_rank,
        iv_percentile,
        iv_high,
        iv_low,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iv_rank_at_midpoint_of_range() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.30, &history);
        // rank = (0.30 - 0.10) / (0.50 - 0.10) * 100 = 50.0
        assert!((metrics.iv_rank - 50.0).abs() < 1e-9);
    }

    #[test]
    fn iv_percentile_counts_days_below_current() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.30, &history);
        // 2 values (0.10, 0.20) are < 0.30 → 2/5 = 40.0%
        assert!((metrics.iv_percentile - 40.0).abs() < 1e-9);
    }

    #[test]
    fn iv_rank_at_maximum_returns_100() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.50, &history);
        assert!((metrics.iv_rank - 100.0).abs() < 1e-9);
    }

    #[test]
    fn iv_rank_at_minimum_returns_0() {
        let history = vec![0.10, 0.20, 0.30, 0.40, 0.50];
        let metrics = calculate_iv_metrics(0.10, &history);
        assert!((metrics.iv_rank - 0.0).abs() < 1e-9);
    }

    #[test]
    fn empty_history_returns_neutral_defaults() {
        let metrics = calculate_iv_metrics(0.25, &[]);
        assert!((metrics.iv_rank - 50.0).abs() < 1e-9);
        assert!((metrics.iv_percentile - 50.0).abs() < 1e-9);
        assert!((metrics.iv_high - 0.25).abs() < 1e-9);
        assert!((metrics.iv_low - 0.25).abs() < 1e-9);
    }
}
