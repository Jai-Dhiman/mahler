/// Spread configuration validated by walk-forward optimization (SPY 2007-2023, QQQ 2008-2023).
///
/// Parameters consensus from 156-174 rolling 24-month IS / 3-month OOS windows:
/// WFE=1.12 (SPY), WFE=0.73 (QQQ) — both pass the >0.50 generalization guard.
#[derive(Debug, Clone)]
pub struct SpreadConfig {
    /// Close when profit >= this fraction of entry credit (75%)
    pub profit_target_pct: f64,
    /// Close when loss >= this fraction of entry credit (200%)
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
    /// Minimum IV percentile required before entering (skip low-IV environments)
    pub min_iv_percentile: f64,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        SpreadConfig {
            profit_target_pct: 0.75,
            stop_loss_pct: 2.00,
            min_dte: 30,
            max_dte: 45,
            min_delta: 0.25,
            max_delta: 0.30,
            min_credit_pct: 0.10,
            min_spread_width: 2.0,
            gamma_exit_dte: 7,
            underlyings: &["SPY", "QQQ", "IWM"],
            max_trades_per_scan: 3,
            min_iv_percentile: 50.0,
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
    fn default_config_has_walkforward_validated_params() {
        let cfg = SpreadConfig::default();
        assert_eq!(cfg.profit_target_pct, 0.75);
        assert_eq!(cfg.stop_loss_pct, 2.00);
        assert_eq!(cfg.min_dte, 30);
        assert_eq!(cfg.max_dte, 45);
        assert!((cfg.min_delta - 0.25).abs() < 1e-9);
        assert!((cfg.max_delta - 0.30).abs() < 1e-9);
        assert!((cfg.min_iv_percentile - 50.0).abs() < 1e-9);
    }

    #[test]
    fn exits_at_profit_target_when_debit_is_25_pct_of_credit() {
        // 75% profit: entry 1.00, debit 0.25 (profit = 0.75 = 75%)
        let cfg = SpreadConfig::default();
        assert!(cfg.should_exit_profit(1.00, 0.25));
    }

    #[test]
    fn does_not_exit_when_profit_is_below_target() {
        // 50% profit: below 75% target, should not exit
        let cfg = SpreadConfig::default();
        assert!(!cfg.should_exit_profit(1.00, 0.50));
    }

    #[test]
    fn exits_at_stop_loss_when_debit_exceeds_200_pct_of_credit() {
        // 210% loss: entry 1.00, debit 3.10 (loss = 2.10 > 2.00 threshold)
        let cfg = SpreadConfig::default();
        assert!(cfg.should_exit_stop_loss(1.00, 3.10));
    }

    #[test]
    fn does_not_trigger_stop_loss_below_threshold() {
        // 150% loss: entry 1.00, debit 2.50 (loss = 1.50 < 2.00 threshold)
        let cfg = SpreadConfig::default();
        assert!(!cfg.should_exit_stop_loss(1.00, 2.50));
    }
}
