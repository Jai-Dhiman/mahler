use chrono::{NaiveDate, Utc};

/// Standard normal CDF via Hart (1968) rational approximation.
/// Error < 1.5e-7 for all x.
fn standard_normal_cdf(x: f64) -> f64 {
    const A1: f64 = 0.319381530;
    const A2: f64 = -0.356563782;
    const A3: f64 = 1.781477937;
    const A4: f64 = -1.821255978;
    const A5: f64 = 1.330274429;
    const P: f64 = 0.2316419;

    let t = 1.0 / (1.0 + P * x.abs());
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * poly;

    if x >= 0.0 { cdf } else { 1.0 - cdf }
}

/// Compute Black-Scholes delta for a European option.
///
/// # Arguments
/// - `option_type`: "put" or "call" (case-insensitive)
/// - `spot`: current underlying price
/// - `strike`: option strike price
/// - `tte`: time to expiry in years (e.g. 45.0/365.0)
/// - `vol`: implied volatility (e.g. 0.20 for 20%)
/// - `risk_free`: risk-free rate (e.g. 0.05 for 5%)
///
/// Returns delta in [-1, 1]. Returns 0.0 if inputs are invalid (tte <= 0, vol <= 0).
pub fn black_scholes_delta(
    option_type: &str,
    spot: f64,
    strike: f64,
    tte: f64,
    vol: f64,
    risk_free: f64,
) -> f64 {
    if tte <= 0.0 || vol <= 0.0 || spot <= 0.0 || strike <= 0.0 {
        return 0.0;
    }

    let d1 = ((spot / strike).ln() + (risk_free + 0.5 * vol * vol) * tte)
        / (vol * tte.sqrt());

    match option_type.to_lowercase().as_str() {
        "call" => standard_normal_cdf(d1),
        "put" => standard_normal_cdf(d1) - 1.0,
        _ => 0.0,
    }
}

/// Returns calendar days until expiration from today (UTC).
///
/// Returns None if the date string is not valid YYYY-MM-DD format.
pub fn days_to_expiry(expiration: &str) -> Option<i64> {
    let exp_date = NaiveDate::parse_from_str(expiration, "%Y-%m-%d").ok()?;
    let today = Utc::now().date_naive();
    Some((exp_date - today).num_days())
}

/// Returns time to expiry in years. Returns 0.0 if expiry is in the past.
pub fn years_to_expiry(expiration: &str) -> f64 {
    days_to_expiry(expiration)
        .map(|d| (d.max(0) as f64) / 365.0)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_delta_is_negative_for_otm_put() {
        // SPY 480, put at 460, 45 DTE, 20% IV, 5% risk-free
        let delta = black_scholes_delta("put", 480.0, 460.0, 45.0 / 365.0, 0.20, 0.05);
        assert!(delta < 0.0, "put delta must be negative, got {}", delta);
        assert!(delta > -1.0, "put delta must be > -1.0, got {}", delta);
        // 460 put is ~4% OTM on SPY 480 with 20% IV at 45 DTE — should be small delta (< 0.35)
        assert!(delta.abs() < 0.35, "OTM put delta abs should be < 0.35, got {}", delta.abs());
    }

    #[test]
    fn call_delta_is_positive_for_otm_call() {
        // SPY 480, call at 500, 45 DTE, 20% IV, 5% risk-free
        let delta = black_scholes_delta("call", 480.0, 500.0, 45.0 / 365.0, 0.20, 0.05);
        assert!(delta > 0.0, "call delta must be positive, got {}", delta);
        assert!(delta < 1.0, "call delta must be < 1.0, got {}", delta);
        assert!(delta < 0.40, "OTM call delta should be < 0.40, got {}", delta);
    }

    #[test]
    fn atm_put_delta_is_approximately_negative_half() {
        // ATM option (spot == strike) delta should be close to -0.50 for put
        let delta = black_scholes_delta("put", 480.0, 480.0, 30.0 / 365.0, 0.20, 0.00);
        assert!(
            (delta - (-0.50)).abs() < 0.05,
            "ATM put delta should be ~-0.50, got {}",
            delta
        );
    }

    #[test]
    fn days_to_expiry_returns_correct_positive_count() {
        let far_future = "2099-12-31";
        let dte = days_to_expiry(far_future);
        assert!(dte.is_some());
        assert!(dte.unwrap() > 0, "future date must have positive DTE");
    }

    #[test]
    fn days_to_expiry_returns_none_for_invalid_date() {
        let dte = days_to_expiry("not-a-date");
        assert!(dte.is_none());
    }
}
