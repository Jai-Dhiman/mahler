#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Normal,
    Elevated,
    Caution,
    High,
    Critical,
    Halted,
}

#[derive(Debug, Clone)]
pub struct RiskState {
    pub level: RiskLevel,
    pub size_multiplier: f64,
    pub reason: Option<String>,
}

impl RiskState {
    fn normal() -> Self {
        RiskState { level: RiskLevel::Normal, size_multiplier: 1.0, reason: None }
    }

    fn halted(reason: impl Into<String>) -> Self {
        RiskState { level: RiskLevel::Halted, size_multiplier: 0.0, reason: Some(reason.into()) }
    }

    fn caution(multiplier: f64, reason: impl Into<String>) -> Self {
        RiskState { level: RiskLevel::Caution, size_multiplier: multiplier, reason: Some(reason.into()) }
    }

    fn elevated(reason: impl Into<String>) -> Self {
        RiskState { level: RiskLevel::Elevated, size_multiplier: 1.0, reason: Some(reason.into()) }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluateParams {
    pub starting_daily_equity: f64,
    pub current_equity: f64,
    pub starting_weekly_equity: f64,
    pub peak_equity: f64,
    pub current_vix: f64,
}

#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub daily_halt_pct: f64,
    pub daily_reduce_pct: f64,
    pub daily_alert_pct: f64,
    pub weekly_halt_pct: f64,
    pub weekly_caution_pct: f64,
    pub drawdown_halt_pct: f64,
    pub drawdown_caution_pct: f64,
    pub vix_halt: f64,
    pub vix_high: f64,
    pub vix_caution: f64,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        CircuitBreaker {
            daily_halt_pct: 0.02,
            daily_reduce_pct: 0.015,
            daily_alert_pct: 0.01,
            weekly_halt_pct: 0.05,
            weekly_caution_pct: 0.03,
            drawdown_halt_pct: 0.15,
            drawdown_caution_pct: 0.10,
            vix_halt: 50.0,
            vix_high: 40.0,
            vix_caution: 30.0,
        }
    }
}

impl CircuitBreaker {
    /// Returns the worst RiskState across all trigger dimensions.
    ///
    /// Checks daily loss, weekly loss, drawdown, and VIX in order of severity.
    /// The most severe condition wins.
    pub fn evaluate(&self, params: &EvaluateParams) -> RiskState {
        let daily_loss_pct = (params.starting_daily_equity - params.current_equity)
            / params.starting_daily_equity;
        let weekly_loss_pct = (params.starting_weekly_equity - params.current_equity)
            / params.starting_weekly_equity;
        let drawdown_pct = (params.peak_equity - params.current_equity) / params.peak_equity;

        let mut worst = RiskState::normal();

        // Daily loss checks
        if daily_loss_pct >= self.daily_halt_pct {
            return RiskState::halted(format!(
                "Daily loss {:.1}% exceeds halt threshold {:.0}%",
                daily_loss_pct * 100.0,
                self.daily_halt_pct * 100.0
            ));
        } else if daily_loss_pct >= self.daily_reduce_pct {
            worst = RiskState::caution(0.50, format!(
                "Daily loss {:.1}% — reducing position size to 50%",
                daily_loss_pct * 100.0
            ));
        } else if daily_loss_pct >= self.daily_alert_pct {
            worst = RiskState::elevated(format!(
                "Daily loss {:.1}% — alert",
                daily_loss_pct * 100.0
            ));
        }

        // Weekly loss checks
        if weekly_loss_pct >= self.weekly_halt_pct {
            return RiskState::halted(format!(
                "Weekly loss {:.1}% exceeds halt threshold {:.0}%",
                weekly_loss_pct * 100.0,
                self.weekly_halt_pct * 100.0
            ));
        } else if weekly_loss_pct >= self.weekly_caution_pct && worst.size_multiplier > 0.50 {
            worst = RiskState::caution(0.50, format!(
                "Weekly loss {:.1}% — reducing to 50%",
                weekly_loss_pct * 100.0
            ));
        }

        // Drawdown checks
        if drawdown_pct >= self.drawdown_halt_pct {
            return RiskState::halted(format!(
                "Drawdown {:.1}% exceeds maximum {:.0}%",
                drawdown_pct * 100.0,
                self.drawdown_halt_pct * 100.0
            ));
        }

        // VIX checks
        if params.current_vix >= self.vix_halt {
            return RiskState::halted(format!("VIX {:.0} exceeds halt threshold {:.0}", params.current_vix, self.vix_halt));
        } else if params.current_vix >= self.vix_high && worst.size_multiplier > 0.25 {
            worst = RiskState::caution(0.25, format!("VIX {:.0} — reducing to 25%", params.current_vix));
        } else if params.current_vix >= self.vix_caution && worst.size_multiplier > 0.50 {
            worst = RiskState::caution(0.50, format!("VIX {:.0} — reducing to 50%", params.current_vix));
        }

        worst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safe_params() -> EvaluateParams {
        EvaluateParams {
            starting_daily_equity: 100_000.0,
            current_equity: 100_000.0,
            starting_weekly_equity: 100_000.0,
            peak_equity: 100_000.0,
            current_vix: 18.0,
        }
    }

    #[test]
    fn returns_normal_when_no_thresholds_breached() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&safe_params());
        assert_eq!(state.level, RiskLevel::Normal);
        assert!((state.size_multiplier - 1.0).abs() < 1e-9);
    }

    #[test]
    fn halts_when_daily_loss_exceeds_2_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_equity: 97_900.0, // 2.1% loss
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
        assert!((state.size_multiplier - 0.0).abs() < 1e-9);
    }

    #[test]
    fn reduces_size_when_daily_loss_between_1_5_and_2_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_equity: 98_400.0, // 1.6% loss
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Caution);
        assert!(state.size_multiplier < 1.0, "size must be reduced at caution, got {}", state.size_multiplier);
        assert!(state.size_multiplier > 0.0);
    }

    #[test]
    fn halts_when_vix_exceeds_50() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            current_vix: 55.0,
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
    }

    #[test]
    fn halts_when_weekly_loss_exceeds_5_pct() {
        let cb = CircuitBreaker::default();
        let state = cb.evaluate(&EvaluateParams {
            starting_weekly_equity: 100_000.0,
            current_equity: 94_900.0, // 5.1% weekly loss
            starting_daily_equity: 94_900.0, // no daily loss today
            ..safe_params()
        });
        assert_eq!(state.level, RiskLevel::Halted);
    }
}
