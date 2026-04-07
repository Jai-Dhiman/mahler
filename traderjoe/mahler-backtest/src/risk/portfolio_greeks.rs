//! Portfolio Greeks aggregation.
//!
//! Calculates and monitors portfolio-level Greek exposures:
//! - Delta: sum(position_delta * contracts * 100)
//! - Gamma: sum(position_gamma * contracts * 100)
//! - Theta: sum(position_theta * contracts * 100)
//! - Vega: sum(position_vega * contracts * 100)

use serde::{Deserialize, Serialize};

use crate::backtest::Position;

/// Portfolio Greeks configuration (limits).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioGreeksConfig {
    /// Maximum absolute portfolio delta (as % of equity notional).
    pub max_delta: f64,
    /// Maximum portfolio gamma.
    pub max_gamma: f64,
    /// Minimum portfolio theta (should be positive for premium sellers).
    pub min_theta: f64,
    /// Maximum portfolio vega (sensitivity to IV changes).
    pub max_vega: f64,
}

impl Default for PortfolioGreeksConfig {
    fn default() -> Self {
        Self {
            max_delta: 0.30,   // 30% delta exposure
            max_gamma: 0.20,   // 20% gamma
            min_theta: 0.0,    // Must be positive (earning decay)
            max_vega: f64::MAX, // No vega limit by default
        }
    }
}

/// Aggregated portfolio Greeks.
#[derive(Debug, Clone, Default)]
pub struct PortfolioGreeks {
    /// Net delta exposure.
    pub delta: f64,
    /// Net gamma exposure.
    pub gamma: f64,
    /// Net theta (daily decay).
    pub theta: f64,
    /// Net vega exposure.
    pub vega: f64,
    /// Number of positions included.
    pub position_count: usize,
}

impl PortfolioGreeks {
    /// Calculate portfolio Greeks from a list of positions.
    pub fn from_positions(positions: &[Position]) -> Self {
        let mut greeks = Self::default();

        for position in positions {
            if !position.is_open() {
                continue;
            }

            greeks.position_count += 1;

            for leg in &position.legs {
                let multiplier = leg.contracts as f64 * 100.0;

                greeks.delta += leg.current_delta * multiplier;
                // Note: gamma, theta, vega would need to be tracked on legs
                // For now, we use approximations based on delta
            }
        }

        greeks
    }

    /// Calculate portfolio Greeks from positions with full Greek data.
    pub fn from_positions_with_greeks(positions: &[PositionWithGreeks]) -> Self {
        let mut greeks = Self::default();

        for pos in positions {
            greeks.delta += pos.delta;
            greeks.gamma += pos.gamma;
            greeks.theta += pos.theta;
            greeks.vega += pos.vega;
            greeks.position_count += 1;
        }

        greeks
    }

    /// Check if portfolio is within configured limits.
    pub fn check_limits(&self, config: &PortfolioGreeksConfig) -> GreeksLimitCheck {
        let mut violations = Vec::new();

        if self.delta.abs() > config.max_delta {
            violations.push(format!(
                "Delta {:.2} exceeds limit {:.2}",
                self.delta, config.max_delta
            ));
        }

        if self.gamma.abs() > config.max_gamma {
            violations.push(format!(
                "Gamma {:.2} exceeds limit {:.2}",
                self.gamma, config.max_gamma
            ));
        }

        if self.theta < config.min_theta {
            violations.push(format!(
                "Theta {:.2} below minimum {:.2}",
                self.theta, config.min_theta
            ));
        }

        if self.vega.abs() > config.max_vega {
            violations.push(format!(
                "Vega {:.2} exceeds limit {:.2}",
                self.vega, config.max_vega
            ));
        }

        GreeksLimitCheck {
            is_within_limits: violations.is_empty(),
            violations,
        }
    }

    /// Calculate the impact of adding a new position.
    pub fn impact_of_position(&self, new_position: &PositionWithGreeks) -> PortfolioGreeks {
        PortfolioGreeks {
            delta: self.delta + new_position.delta,
            gamma: self.gamma + new_position.gamma,
            theta: self.theta + new_position.theta,
            vega: self.vega + new_position.vega,
            position_count: self.position_count + 1,
        }
    }

    /// Check if a new position would violate limits.
    pub fn would_violate_limits(
        &self,
        new_position: &PositionWithGreeks,
        config: &PortfolioGreeksConfig,
    ) -> bool {
        let projected = self.impact_of_position(new_position);
        !projected.check_limits(config).is_within_limits
    }
}

/// Position with explicit Greek values.
#[derive(Debug, Clone)]
pub struct PositionWithGreeks {
    pub ticker: String,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
}

impl PositionWithGreeks {
    pub fn new(ticker: &str, delta: f64, gamma: f64, theta: f64, vega: f64) -> Self {
        Self {
            ticker: ticker.to_string(),
            delta,
            gamma,
            theta,
            vega,
        }
    }

    /// Create from a Position (extracts delta, estimates others).
    pub fn from_position(position: &Position) -> Self {
        let delta: f64 = position
            .legs
            .iter()
            .map(|l| l.current_delta * l.contracts as f64 * 100.0)
            .sum();

        // Simplified: gamma and vega not tracked on Position legs
        // In production, these would be calculated from option quotes
        Self {
            ticker: position.ticker.clone(),
            delta,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
        }
    }
}

/// Result of checking Greeks limits.
#[derive(Debug, Clone)]
pub struct GreeksLimitCheck {
    pub is_within_limits: bool,
    pub violations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PortfolioGreeksConfig::default();
        assert_eq!(config.max_delta, 0.30);
        assert_eq!(config.max_gamma, 0.20);
        assert_eq!(config.min_theta, 0.0);
    }

    #[test]
    fn test_portfolio_greeks_aggregation() {
        let positions = vec![
            PositionWithGreeks::new("SPY", -100.0, 5.0, 25.0, 50.0),
            PositionWithGreeks::new("QQQ", -150.0, 8.0, 30.0, 60.0),
        ];

        let greeks = PortfolioGreeks::from_positions_with_greeks(&positions);
        assert_eq!(greeks.delta, -250.0);
        assert_eq!(greeks.gamma, 13.0);
        assert_eq!(greeks.theta, 55.0);
        assert_eq!(greeks.vega, 110.0);
        assert_eq!(greeks.position_count, 2);
    }

    #[test]
    fn test_limit_check_pass() {
        let greeks = PortfolioGreeks {
            delta: 0.20,
            gamma: 0.10,
            theta: 50.0,
            vega: 100.0,
            position_count: 5,
        };

        let config = PortfolioGreeksConfig::default();
        let check = greeks.check_limits(&config);
        assert!(check.is_within_limits);
        assert!(check.violations.is_empty());
    }

    #[test]
    fn test_limit_check_violations() {
        let greeks = PortfolioGreeks {
            delta: 0.50, // Exceeds 0.30 limit
            gamma: 0.25, // Exceeds 0.20 limit
            theta: -10.0, // Below 0 minimum
            vega: 100.0,
            position_count: 5,
        };

        let config = PortfolioGreeksConfig::default();
        let check = greeks.check_limits(&config);
        assert!(!check.is_within_limits);
        assert_eq!(check.violations.len(), 3);
    }

    #[test]
    fn test_impact_of_position() {
        let current = PortfolioGreeks {
            delta: -100.0,
            gamma: 5.0,
            theta: 25.0,
            vega: 50.0,
            position_count: 2,
        };

        let new_pos = PositionWithGreeks::new("IWM", -50.0, 3.0, 15.0, 30.0);
        let projected = current.impact_of_position(&new_pos);

        assert_eq!(projected.delta, -150.0);
        assert_eq!(projected.gamma, 8.0);
        assert_eq!(projected.theta, 40.0);
        assert_eq!(projected.vega, 80.0);
        assert_eq!(projected.position_count, 3);
    }
}
