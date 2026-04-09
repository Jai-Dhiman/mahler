/// Spread configuration with backtest-validated parameters.
///
/// profit_target=0.25 and stop_loss=1.25 are from autoresearch:
/// iter 1 (profit_target=25%) -> Sharpe 0.14, win_rate 57.7% vs baseline 0.10/51%
#[derive(Debug, Clone)]
pub struct SpreadConfig {
    /// Close when profit >= this fraction of entry credit (25%)
    pub profit_target_pct: f64,
    /// Close when loss >= this fraction of entry credit (125%)
    pub stop_loss_pct: f64,
    pub min_dte: i64,
    pub max_dte: i64,
    pub min_delta: f64,
    pub max_delta: f64,
    /// Minimum credit as fraction of spread width (10%)
    pub min_credit_pct: f64,
    /// Minimum spread width in dollars
    pub min_spread_width: f64,
    /// Force-close when DTE reaches this value (gamma explosion risk)
    pub gamma_exit_dte: i64,
    /// Underlyings to scan
    pub underlyings: &'static [&'static str],
    /// Max trades to place per morning scan
    pub max_trades_per_scan: usize,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        SpreadConfig {
            profit_target_pct: 0.25,
            stop_loss_pct: 1.25,
            min_dte: 30,
            max_dte: 45,
            min_delta: 0.05,
            max_delta: 0.15,
            min_credit_pct: 0.10,
            min_spread_width: 2.0,
            gamma_exit_dte: 7,
            underlyings: &["SPY", "QQQ", "IWM"],
            max_trades_per_scan: 3,
        }
    }
}

impl SpreadConfig {
    /// Returns true if the position should be closed for profit.
    ///
    /// Profit condition: current debit to close is <= entry_credit * (1 - profit_target_pct)
    pub fn should_exit_profit(&self, entry_credit: f64, current_debit: f64) -> bool {
        let profit = entry_credit - current_debit;
        profit / entry_credit >= self.profit_target_pct
    }

    /// Returns true if the position should be closed for a stop loss.
    ///
    /// Stop condition: current debit to close is >= entry_credit * (1 + stop_loss_pct)
    pub fn should_exit_stop_loss(&self, entry_credit: f64, current_debit: f64) -> bool {
        let loss = current_debit - entry_credit;
        loss / entry_credit >= self.stop_loss_pct
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_autoresearch_validated_params() {
        let cfg = SpreadConfig::default();
        assert_eq!(cfg.profit_target_pct, 0.25);
        assert_eq!(cfg.stop_loss_pct, 1.25);
        assert_eq!(cfg.min_dte, 30);
        assert_eq!(cfg.max_dte, 45);
        assert!((cfg.min_delta - 0.05).abs() < 1e-9);
        assert!((cfg.max_delta - 0.15).abs() < 1e-9);
    }

    #[test]
    fn exits_at_profit_target_when_debit_is_25_pct_of_credit() {
        let cfg = SpreadConfig::default();
        assert!(cfg.should_exit_profit(1.00, 0.75));
    }

    #[test]
    fn does_not_exit_when_profit_is_below_target() {
        let cfg = SpreadConfig::default();
        assert!(!cfg.should_exit_profit(1.00, 0.85));
    }

    #[test]
    fn exits_at_stop_loss_when_debit_is_125_pct_of_credit() {
        let cfg = SpreadConfig::default();
        assert!(cfg.should_exit_stop_loss(1.00, 2.25));
    }

    #[test]
    fn does_not_trigger_stop_loss_below_threshold() {
        let cfg = SpreadConfig::default();
        assert!(!cfg.should_exit_stop_loss(1.00, 2.00));
    }
}
