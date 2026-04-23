#[derive(Debug, Clone)]
pub struct ExistingPosition {
    pub underlying: String,
    pub total_max_loss: f64,
}

#[derive(Debug, Clone)]
pub struct SizeResult {
    pub contracts: i64,
    pub reason: Option<String>,
}

impl SizeResult {
    fn zero(reason: impl Into<String>) -> Self {
        SizeResult { contracts: 0, reason: Some(reason.into()) }
    }
}

#[derive(Debug, Clone)]
pub struct PositionSizer {
    pub max_risk_per_trade_pct: f64,
    pub max_portfolio_heat_pct: f64,
    pub max_per_underlying_pct: f64,
    pub high_vix_threshold: f64,
    pub high_vix_reduction: f64,
    pub extreme_vix_threshold: f64,
}

impl Default for PositionSizer {
    fn default() -> Self {
        PositionSizer {
            max_risk_per_trade_pct: 0.02,
            max_portfolio_heat_pct: 0.20,
            max_per_underlying_pct: 0.20,
            high_vix_threshold: 40.0,
            high_vix_reduction: 0.50,
            extreme_vix_threshold: 50.0,
        }
    }
}

impl PositionSizer {
    /// Calculate position size given a spread's max_loss_per_contract and portfolio context.
    ///
    /// Applies four limits in order: per-trade risk, portfolio heat, VIX reduction.
    /// Returns the minimum of all applicable limits, floored to 1 if any contracts allowed.
    pub fn calculate(
        &self,
        max_loss_per_contract: f64,
        equity: f64,
        positions: &[ExistingPosition],
        vix: f64,
    ) -> SizeResult {
        // Check portfolio heat
        let current_heat: f64 = positions.iter().map(|p| p.total_max_loss).sum();
        let max_heat = equity * self.max_portfolio_heat_pct;
        if current_heat >= max_heat {
            return SizeResult::zero(format!(
                "Portfolio heat {:.0}% at limit {:.0}%",
                current_heat / equity * 100.0,
                self.max_portfolio_heat_pct * 100.0
            ));
        }

        if max_loss_per_contract <= 0.0 {
            return SizeResult::zero("max_loss_per_contract must be positive");
        }

        // Max contracts from per-trade risk limit
        let max_risk = equity * self.max_risk_per_trade_pct;
        let mut contracts = (max_risk / max_loss_per_contract).floor() as i64;

        // Also cap by remaining heat capacity
        let remaining_heat = max_heat - current_heat;
        let heat_cap = (remaining_heat / max_loss_per_contract).floor() as i64;
        contracts = contracts.min(heat_cap);

        // VIX reduction
        if vix >= self.extreme_vix_threshold {
            contracts = 0;
        } else if vix >= self.high_vix_threshold {
            contracts = (contracts as f64 * self.high_vix_reduction).floor() as i64;
        }

        if contracts <= 0 {
            return SizeResult::zero("Position size rounds to zero after risk limits");
        }

        // Enforce minimum of 1 when at least 1 contract fits within per-trade risk
        contracts = contracts.max(1);

        SizeResult { contracts, reason: None }
    }

    /// Calculate position size for a specific underlying, checking concentration limit.
    pub fn calculate_for_underlying(
        &self,
        underlying: &str,
        max_loss_per_contract: f64,
        equity: f64,
        positions: &[ExistingPosition],
        vix: f64,
    ) -> SizeResult {
        // Check per-underlying concentration
        let underlying_heat: f64 = positions
            .iter()
            .filter(|p| p.underlying == underlying)
            .map(|p| p.total_max_loss)
            .sum();

        let max_underlying = equity * self.max_per_underlying_pct;
        if underlying_heat >= max_underlying {
            return SizeResult::zero(format!(
                "{} position at concentration limit {:.0}%",
                underlying,
                self.max_per_underlying_pct * 100.0
            ));
        }

        self.calculate(max_loss_per_contract, equity, positions, vix)
    }
}

/// Scales equity by a live-capital fraction, clamped to [0, 1].
pub fn effective_equity(raw_equity: f64, fraction: f64) -> f64 {
    raw_equity * fraction.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_to_2_pct_account_risk() {
        let sizer = PositionSizer::default();
        // max_risk = 100_000 * 0.02 = $2,000
        // max_loss_per_contract = $500
        // max contracts = 2000 / 500 = 4
        let result = sizer.calculate(500.0, 100_000.0, &[], 20.0);
        assert_eq!(result.contracts, 4);
    }

    #[test]
    fn returns_zero_when_portfolio_heat_at_limit() {
        let sizer = PositionSizer::default();
        let positions = vec![
            ExistingPosition { underlying: "SPY".to_string(), total_max_loss: 20_000.0 },
        ];
        let result = sizer.calculate(500.0, 100_000.0, &positions, 20.0);
        assert_eq!(result.contracts, 0);
        assert!(result.reason.is_some());
    }

    #[test]
    fn reduces_size_by_50_pct_at_high_vix() {
        let sizer = PositionSizer::default();
        // Normal max = 4 contracts. High VIX (>40) should halve it to 2.
        let result = sizer.calculate(500.0, 100_000.0, &[], 42.0);
        assert_eq!(result.contracts, 2);
    }

    #[test]
    fn minimum_is_1_contract_when_any_size_allowed() {
        let sizer = PositionSizer::default();
        // max_risk = 2000, max_loss = 1999 -> 1 contract
        let result = sizer.calculate(1_999.0, 100_000.0, &[], 20.0);
        assert_eq!(result.contracts, 1);
    }

    #[test]
    fn returns_zero_for_single_underlying_at_concentration_limit() {
        let sizer = PositionSizer::default();
        let positions = vec![
            ExistingPosition { underlying: "SPY".to_string(), total_max_loss: 20_000.0 },
        ];
        let result = sizer.calculate_for_underlying("SPY", 500.0, 100_000.0, &positions, 20.0);
        assert_eq!(result.contracts, 0);
    }

    #[test]
    fn effective_equity_scales_by_fraction() {
        assert!((effective_equity(100_000.0, 0.10) - 10_000.0).abs() < 1e-9);
        assert!((effective_equity(100_000.0, 1.00) - 100_000.0).abs() < 1e-9);
    }

    #[test]
    fn effective_equity_clamps_fraction_to_zero_one() {
        assert!((effective_equity(100_000.0, 1.50) - 100_000.0).abs() < 1e-9);
        assert!((effective_equity(100_000.0, -0.5) - 0.0).abs() < 1e-9);
    }
}
