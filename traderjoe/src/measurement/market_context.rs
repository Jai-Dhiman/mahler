use crate::broker::types::Bar;

#[derive(Debug, Clone, PartialEq)]
pub struct SpyStats {
    pub realized_vol_20d: f64,
    pub return_20d: f64,
    pub drawdown_from_52w_high: f64,
}

/// 20-day realized vol = stdev of log returns x sqrt(252).
/// 20-day return = (last - first) / first.
/// Drawdown from 52w high = (peak - last) / peak, peak taken over bars_52w close values.
pub fn compute_spy_stats(bars_20d: &[Bar], bars_52w: &[Bar]) -> SpyStats {
    let return_20d = if bars_20d.len() >= 2 && bars_20d.first().map(|b| b.close).unwrap_or(0.0) > 0.0 {
        let first = bars_20d.first().unwrap().close;
        let last = bars_20d.last().unwrap().close;
        (last - first) / first
    } else { 0.0 };

    let realized_vol_20d = if bars_20d.len() >= 2 {
        let log_returns: Vec<f64> = bars_20d.windows(2)
            .filter(|w| w[0].close > 0.0 && w[1].close > 0.0)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();
        if log_returns.is_empty() { 0.0 } else {
            let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
            let var = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / log_returns.len() as f64;
            var.sqrt() * (252f64).sqrt()
        }
    } else { 0.0 };

    let drawdown_from_52w_high = if !bars_52w.is_empty() {
        let peak = bars_52w.iter().map(|b| b.close).fold(0.0f64, f64::max);
        let last = bars_52w.last().map(|b| b.close).unwrap_or(0.0);
        if peak > 0.0 { (peak - last) / peak } else { 0.0 }
    } else { 0.0 };

    SpyStats { realized_vol_20d, return_20d, drawdown_from_52w_high }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpotVixReading {
    pub value: f64,
    pub source: &'static str,
    pub source_date: String,
}

/// Parse FRED VIXCLS CSV. Lines are `DATE,VALUE`. Missing values are `.`.
/// Returns the most recent row with a numeric VIX, or None.
pub fn parse_fred_vix_csv(body: &str) -> Option<SpotVixReading> {
    body.lines().rev()
        .filter(|l| !l.is_empty() && !l.starts_with("DATE"))
        .filter_map(|l| {
            let mut parts = l.split(',');
            let date = parts.next()?.trim();
            let val = parts.next()?.trim();
            let v: f64 = val.parse().ok()?;
            Some(SpotVixReading { value: v, source: "fred", source_date: date.to_string() })
        })
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::types::Bar;

    fn bar(close: f64) -> Bar {
        Bar { timestamp: "2026-04-22".to_string(), open: close, high: close, low: close, close, volume: 0 }
    }

    #[test]
    fn spy_stats_computes_drawdown_from_52w_peak() {
        // Peak 500, current 450 -> drawdown 10%
        let mut bars = vec![bar(400.0); 260];
        bars[100] = bar(500.0);
        bars[259] = bar(450.0);
        let stats = compute_spy_stats(&bars[240..260], &bars);
        assert!((stats.drawdown_from_52w_high - 0.10).abs() < 1e-6,
                "got {}", stats.drawdown_from_52w_high);
    }

    #[test]
    fn spy_stats_computes_20d_return() {
        // Start 480, end 480*1.02 = 489.6 -> return 0.02
        let mut bars20 = vec![bar(480.0); 20];
        bars20[19] = bar(489.6);
        let bars52 = bars20.clone();
        let stats = compute_spy_stats(&bars20, &bars52);
        assert!((stats.return_20d - 0.02).abs() < 1e-6);
    }

    #[test]
    fn parse_fred_vix_csv_returns_most_recent_nonempty_row() {
        // FRED CSV format: "DATE,VIXCLS\n2026-04-18,16.23\n2026-04-21,17.45\n2026-04-22,.\n"
        // Value "." is FRED's marker for "no data yet" -- skip to prior row.
        let csv = "DATE,VIXCLS\n2026-04-18,16.23\n2026-04-21,17.45\n2026-04-22,.\n";
        let parsed = parse_fred_vix_csv(csv).expect("parse");
        assert!((parsed.value - 17.45).abs() < 1e-6);
        assert_eq!(parsed.source_date, "2026-04-21");
    }

    #[test]
    fn parse_fred_vix_csv_returns_none_when_all_rows_empty() {
        let csv = "DATE,VIXCLS\n2026-04-21,.\n2026-04-22,.\n";
        assert!(parse_fred_vix_csv(csv).is_none());
    }
}
