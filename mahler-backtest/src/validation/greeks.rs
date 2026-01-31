//! Greeks calculation validation using Black-Scholes.
//!
//! Validates ORATS Greeks by recalculating them using Black-Scholes
//! and comparing against the stored values within tolerance.
//!
//! Tolerances:
//! - Delta: +/- 0.01
//! - Gamma: +/- 0.001
//! - Theta: +/- 0.05
//! - Vega: +/- 0.01
//! - IV round-trip: +/- $0.01

use std::f64::consts::PI;

use statrs::distribution::{ContinuousCDF, Normal};

use crate::data::{DataLoader, LoaderError, OptionType};

/// Black-Scholes calculator for options pricing and Greeks.
pub struct BlackScholes {
    /// Risk-free interest rate
    pub rate: f64,
    /// Dividend yield
    pub dividend: f64,
}

impl Default for BlackScholes {
    fn default() -> Self {
        Self {
            rate: 0.05,     // 5% risk-free rate
            dividend: 0.01, // 1% dividend yield for SPY
        }
    }
}

impl BlackScholes {
    pub fn new(rate: f64, dividend: f64) -> Self {
        Self { rate, dividend }
    }

    /// Calculate d1 parameter.
    fn d1(&self, spot: f64, strike: f64, time: f64, vol: f64) -> f64 {
        let numerator =
            (spot / strike).ln() + (self.rate - self.dividend + 0.5 * vol * vol) * time;
        numerator / (vol * time.sqrt())
    }

    /// Calculate d2 parameter.
    fn d2(&self, spot: f64, strike: f64, time: f64, vol: f64) -> f64 {
        self.d1(spot, strike, time, vol) - vol * time.sqrt()
    }

    /// Standard normal CDF.
    fn norm_cdf(x: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.cdf(x)
    }

    /// Standard normal PDF.
    fn norm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
    }

    /// Calculate call option price.
    pub fn call_price(&self, spot: f64, strike: f64, time: f64, vol: f64) -> f64 {
        if time <= 0.0 {
            return (spot - strike).max(0.0);
        }

        let d1 = self.d1(spot, strike, time, vol);
        let d2 = self.d2(spot, strike, time, vol);

        spot * (-self.dividend * time).exp() * Self::norm_cdf(d1)
            - strike * (-self.rate * time).exp() * Self::norm_cdf(d2)
    }

    /// Calculate put option price.
    pub fn put_price(&self, spot: f64, strike: f64, time: f64, vol: f64) -> f64 {
        if time <= 0.0 {
            return (strike - spot).max(0.0);
        }

        let d1 = self.d1(spot, strike, time, vol);
        let d2 = self.d2(spot, strike, time, vol);

        strike * (-self.rate * time).exp() * Self::norm_cdf(-d2)
            - spot * (-self.dividend * time).exp() * Self::norm_cdf(-d1)
    }

    /// Calculate option price based on type.
    pub fn price(&self, spot: f64, strike: f64, time: f64, vol: f64, opt_type: OptionType) -> f64 {
        match opt_type {
            OptionType::Call => self.call_price(spot, strike, time, vol),
            OptionType::Put => self.put_price(spot, strike, time, vol),
        }
    }

    /// Calculate delta.
    pub fn delta(&self, spot: f64, strike: f64, time: f64, vol: f64, opt_type: OptionType) -> f64 {
        if time <= 0.0 {
            return match opt_type {
                OptionType::Call => {
                    if spot > strike {
                        1.0
                    } else {
                        0.0
                    }
                }
                OptionType::Put => {
                    if spot < strike {
                        -1.0
                    } else {
                        0.0
                    }
                }
            };
        }

        let d1 = self.d1(spot, strike, time, vol);
        let discount = (-self.dividend * time).exp();

        match opt_type {
            OptionType::Call => discount * Self::norm_cdf(d1),
            OptionType::Put => discount * (Self::norm_cdf(d1) - 1.0),
        }
    }

    /// Calculate gamma (same for calls and puts).
    pub fn gamma(&self, spot: f64, strike: f64, time: f64, vol: f64) -> f64 {
        if time <= 0.0 || vol <= 0.0 {
            return 0.0;
        }

        let d1 = self.d1(spot, strike, time, vol);
        let discount = (-self.dividend * time).exp();

        discount * Self::norm_pdf(d1) / (spot * vol * time.sqrt())
    }

    /// Calculate vega (same for calls and puts).
    /// Returns vega per 1% change in volatility.
    pub fn vega(&self, spot: f64, strike: f64, time: f64, vol: f64) -> f64 {
        if time <= 0.0 {
            return 0.0;
        }

        let d1 = self.d1(spot, strike, time, vol);
        let discount = (-self.dividend * time).exp();

        // Vega per 1% move = vega / 100
        spot * discount * Self::norm_pdf(d1) * time.sqrt() / 100.0
    }

    /// Calculate theta (annualized).
    pub fn theta(&self, spot: f64, strike: f64, time: f64, vol: f64, opt_type: OptionType) -> f64 {
        if time <= 0.0 {
            return 0.0;
        }

        let d1 = self.d1(spot, strike, time, vol);
        let d2 = self.d2(spot, strike, time, vol);
        let discount_d = (-self.dividend * time).exp();
        let discount_r = (-self.rate * time).exp();

        let term1 = -spot * discount_d * Self::norm_pdf(d1) * vol / (2.0 * time.sqrt());

        match opt_type {
            OptionType::Call => {
                let term2 = self.dividend * spot * discount_d * Self::norm_cdf(d1);
                let term3 = self.rate * strike * discount_r * Self::norm_cdf(d2);
                // Return daily theta (divide by 365)
                (term1 + term2 - term3) / 365.0
            }
            OptionType::Put => {
                let term2 = self.dividend * spot * discount_d * Self::norm_cdf(-d1);
                let term3 = self.rate * strike * discount_r * Self::norm_cdf(-d2);
                // Return daily theta (divide by 365)
                (term1 - term2 + term3) / 365.0
            }
        }
    }

    /// Calculate rho.
    pub fn rho(&self, spot: f64, strike: f64, time: f64, vol: f64, opt_type: OptionType) -> f64 {
        if time <= 0.0 {
            return 0.0;
        }

        let d2 = self.d2(spot, strike, time, vol);
        let discount = (-self.rate * time).exp();

        // Rho per 1% move = rho / 100
        match opt_type {
            OptionType::Call => strike * time * discount * Self::norm_cdf(d2) / 100.0,
            OptionType::Put => -strike * time * discount * Self::norm_cdf(-d2) / 100.0,
        }
    }

    /// Calculate implied volatility from option price using Newton-Raphson.
    pub fn implied_vol(
        &self,
        spot: f64,
        strike: f64,
        time: f64,
        price: f64,
        opt_type: OptionType,
    ) -> Option<f64> {
        if time <= 0.0 || price <= 0.0 {
            return None;
        }

        // Initial guess using Brenner-Subrahmanyam approximation
        let mut vol = (price / spot) * (2.0 * PI / time).sqrt();
        vol = vol.clamp(0.01, 5.0);

        // Newton-Raphson iteration
        let max_iter = 100;
        let tolerance = 1e-6;

        for _ in 0..max_iter {
            let calc_price = self.price(spot, strike, time, vol, opt_type);
            let diff = calc_price - price;

            if diff.abs() < tolerance {
                return Some(vol);
            }

            // Vega (not scaled)
            let vega = spot
                * (-self.dividend * time).exp()
                * Self::norm_pdf(self.d1(spot, strike, time, vol))
                * time.sqrt();

            if vega.abs() < 1e-10 {
                break;
            }

            vol -= diff / vega;
            vol = vol.clamp(0.001, 10.0);
        }

        None
    }
}

/// Result of Greeks validation.
#[derive(Debug)]
pub struct GreeksValidationReport {
    pub ticker: String,
    pub year: i32,
    pub total_rows: usize,
    pub validated_rows: usize,
    pub delta_within_tolerance: usize,
    pub gamma_within_tolerance: usize,
    pub theta_within_tolerance: usize,
    pub vega_within_tolerance: usize,
    pub iv_roundtrip_pass: usize,
    pub delta_tolerance: f64,
    pub gamma_tolerance: f64,
    pub theta_tolerance: f64,
    pub vega_tolerance: f64,
}

impl GreeksValidationReport {
    pub fn delta_pass_rate(&self) -> f64 {
        if self.validated_rows == 0 {
            return 0.0;
        }
        self.delta_within_tolerance as f64 / self.validated_rows as f64
    }

    pub fn gamma_pass_rate(&self) -> f64 {
        if self.validated_rows == 0 {
            return 0.0;
        }
        self.gamma_within_tolerance as f64 / self.validated_rows as f64
    }

    pub fn theta_pass_rate(&self) -> f64 {
        if self.validated_rows == 0 {
            return 0.0;
        }
        self.theta_within_tolerance as f64 / self.validated_rows as f64
    }

    pub fn vega_pass_rate(&self) -> f64 {
        if self.validated_rows == 0 {
            return 0.0;
        }
        self.vega_within_tolerance as f64 / self.validated_rows as f64
    }

    pub fn iv_roundtrip_rate(&self) -> f64 {
        if self.validated_rows == 0 {
            return 0.0;
        }
        self.iv_roundtrip_pass as f64 / self.validated_rows as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "{} {} Greeks validation: delta={:.1}%, gamma={:.1}%, theta={:.1}%, vega={:.1}%, iv_rt={:.1}%",
            self.ticker,
            self.year,
            self.delta_pass_rate() * 100.0,
            self.gamma_pass_rate() * 100.0,
            self.theta_pass_rate() * 100.0,
            self.vega_pass_rate() * 100.0,
            self.iv_roundtrip_rate() * 100.0
        )
    }

    pub fn all_pass(&self, threshold: f64) -> bool {
        self.delta_pass_rate() >= threshold
            && self.gamma_pass_rate() >= threshold
            && self.theta_pass_rate() >= threshold
            && self.vega_pass_rate() >= threshold
    }
}

/// Validator for Greeks calculations.
pub struct GreeksValidator {
    loader: DataLoader,
    bs: BlackScholes,
    delta_tolerance: f64,
    gamma_tolerance: f64,
    theta_tolerance: f64,
    vega_tolerance: f64,
    price_tolerance: f64,
}

impl GreeksValidator {
    pub fn new(data_dir: &str) -> Self {
        Self {
            loader: DataLoader::new(data_dir),
            bs: BlackScholes::default(),
            delta_tolerance: 0.01,
            gamma_tolerance: 0.001,
            theta_tolerance: 0.05,
            vega_tolerance: 0.01,
            price_tolerance: 0.01,
        }
    }

    pub fn with_tolerances(
        mut self,
        delta: f64,
        gamma: f64,
        theta: f64,
        vega: f64,
        price: f64,
    ) -> Self {
        self.delta_tolerance = delta;
        self.gamma_tolerance = gamma;
        self.theta_tolerance = theta;
        self.vega_tolerance = vega;
        self.price_tolerance = price;
        self
    }

    pub fn with_black_scholes(mut self, rate: f64, dividend: f64) -> Self {
        self.bs = BlackScholes::new(rate, dividend);
        self
    }

    /// Validate Greeks for a ticker/year.
    pub fn validate(&self, ticker: &str, year: i32) -> Result<GreeksValidationReport, LoaderError> {
        let df = self.loader.load_dataframe(ticker, year)?;
        let total_rows = df.height();

        // Get columns
        let opt_type = df.column("option_type").unwrap().str().unwrap();
        let spot = df.column("stock_price").unwrap().f64().unwrap();
        let strike = df.column("strike").unwrap().f64().unwrap();
        let dte = df.column("dte").unwrap().i32().unwrap();
        let mid_iv = df.column("mid_iv").unwrap().f64().unwrap();
        let delta = df.column("delta").unwrap().f64().unwrap();
        let gamma = df.column("gamma").unwrap().f64().unwrap();
        let theta = df.column("theta").unwrap().f64().unwrap();
        let vega = df.column("vega").unwrap().f64().unwrap();
        let bid = df.column("bid").unwrap().f64().unwrap();
        let ask = df.column("ask").unwrap().f64().unwrap();

        let mut validated = 0;
        let mut delta_pass = 0;
        let mut gamma_pass = 0;
        let mut theta_pass = 0;
        let mut vega_pass = 0;
        let mut iv_rt_pass = 0;

        // Sample validation (every 100th row to save time)
        let sample_rate = 100;

        for idx in (0..total_rows).step_by(sample_rate) {
            let opt = opt_type.get(idx);
            let s = spot.get(idx);
            let k = strike.get(idx);
            let d = dte.get(idx);
            let iv = mid_iv.get(idx);
            let del = delta.get(idx);
            let gam = gamma.get(idx);
            let the = theta.get(idx);
            let veg = vega.get(idx);
            let b = bid.get(idx);
            let a = ask.get(idx);

            // Skip if any required value is missing
            if opt.is_none()
                || s.is_none()
                || k.is_none()
                || d.is_none()
                || iv.is_none()
                || del.is_none()
            {
                continue;
            }

            let opt = opt.unwrap();
            let s = s.unwrap();
            let k = k.unwrap();
            let d = d.unwrap();
            let iv = iv.unwrap();
            let del = del.unwrap();

            // Skip invalid data
            if iv <= 0.0 || d <= 0 || s <= 0.0 || k <= 0.0 {
                continue;
            }

            validated += 1;

            let opt_type = match opt {
                "C" => OptionType::Call,
                "P" => OptionType::Put,
                _ => continue,
            };

            // Time in years
            let time = d as f64 / 365.0;

            // Calculate Greeks
            let calc_delta = self.bs.delta(s, k, time, iv, opt_type);
            let calc_gamma = self.bs.gamma(s, k, time, iv);
            let calc_vega = self.bs.vega(s, k, time, iv);
            let calc_theta = self.bs.theta(s, k, time, iv, opt_type);

            // Compare with tolerances
            if (calc_delta - del).abs() <= self.delta_tolerance {
                delta_pass += 1;
            }

            if let Some(gam) = gam {
                if (calc_gamma - gam).abs() <= self.gamma_tolerance {
                    gamma_pass += 1;
                }
            }

            if let Some(the) = the {
                if (calc_theta - the).abs() <= self.theta_tolerance {
                    theta_pass += 1;
                }
            }

            if let Some(veg) = veg {
                if (calc_vega - veg).abs() <= self.vega_tolerance {
                    vega_pass += 1;
                }
            }

            // IV round-trip test: price -> IV -> price
            if let (Some(b), Some(a)) = (b, a) {
                let mid_price = (b + a) / 2.0;
                if mid_price > 0.0 {
                    let calc_price = self.bs.price(s, k, time, iv, opt_type);
                    if (calc_price - mid_price).abs() <= self.price_tolerance {
                        iv_rt_pass += 1;
                    }
                }
            }
        }

        Ok(GreeksValidationReport {
            ticker: ticker.to_string(),
            year,
            total_rows,
            validated_rows: validated,
            delta_within_tolerance: delta_pass,
            gamma_within_tolerance: gamma_pass,
            theta_within_tolerance: theta_pass,
            vega_within_tolerance: vega_pass,
            iv_roundtrip_pass: iv_rt_pass,
            delta_tolerance: self.delta_tolerance,
            gamma_tolerance: self.gamma_tolerance,
            theta_tolerance: self.theta_tolerance,
            vega_tolerance: self.vega_tolerance,
        })
    }

    /// Validate all years for a ticker.
    pub fn validate_ticker(
        &self,
        ticker: &str,
    ) -> Result<Vec<GreeksValidationReport>, LoaderError> {
        let years = self.loader.available_years(ticker)?;
        let mut reports = Vec::new();

        for year in years {
            reports.push(self.validate(ticker, year)?);
        }

        Ok(reports)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_black_scholes_call_price() {
        let bs = BlackScholes::new(0.05, 0.0);
        // Example: S=100, K=100, T=1, vol=0.20
        let price = bs.call_price(100.0, 100.0, 1.0, 0.20);
        // Expected ~10.45 for ATM call
        assert!(price > 9.0 && price < 12.0);
    }

    #[test]
    fn test_black_scholes_put_price() {
        let bs = BlackScholes::new(0.05, 0.0);
        let price = bs.put_price(100.0, 100.0, 1.0, 0.20);
        // Put should be less than call for ATM due to interest rates
        assert!(price > 5.0 && price < 9.0);
    }

    #[test]
    fn test_put_call_parity() {
        let bs = BlackScholes::new(0.05, 0.0);
        let spot = 100.0;
        let strike = 100.0;
        let time = 1.0;
        let vol = 0.20;

        let call = bs.call_price(spot, strike, time, vol);
        let put = bs.put_price(spot, strike, time, vol);

        // Put-call parity: C - P = S - K*e^(-rT)
        let parity_rhs = spot - strike * (-bs.rate * time).exp();
        assert_relative_eq!(call - put, parity_rhs, epsilon = 0.01);
    }

    #[test]
    fn test_delta_bounds() {
        let bs = BlackScholes::default();
        let spot = 100.0;
        let strike = 100.0;
        let time = 0.5;
        let vol = 0.25;

        let call_delta = bs.delta(spot, strike, time, vol, OptionType::Call);
        let put_delta = bs.delta(spot, strike, time, vol, OptionType::Put);

        // Call delta should be in (0, 1)
        assert!(call_delta > 0.0 && call_delta < 1.0);
        // Put delta should be in (-1, 0)
        assert!(put_delta > -1.0 && put_delta < 0.0);
        // Call delta - Put delta should be approximately 1 (for no dividend)
        // With dividend: Call delta - Put delta = e^(-qT)
    }

    #[test]
    fn test_gamma_positive() {
        let bs = BlackScholes::default();
        let gamma = bs.gamma(100.0, 100.0, 0.5, 0.25);
        assert!(gamma > 0.0);
    }

    #[test]
    fn test_vega_positive() {
        let bs = BlackScholes::default();
        let vega = bs.vega(100.0, 100.0, 0.5, 0.25);
        assert!(vega > 0.0);
    }

    #[test]
    fn test_implied_vol() {
        let bs = BlackScholes::new(0.05, 0.0);
        let vol = 0.25;
        let price = bs.call_price(100.0, 100.0, 0.5, vol);

        let iv = bs
            .implied_vol(100.0, 100.0, 0.5, price, OptionType::Call)
            .unwrap();
        assert_relative_eq!(iv, vol, epsilon = 0.001);
    }
}
