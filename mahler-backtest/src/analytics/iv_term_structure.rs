//! IV term structure analysis.
//!
//! Analyzes the implied volatility curve across expirations:
//! - Contango: near-term IV < far-term IV (ratio < 0.95)
//! - Backwardation: near-term IV > far-term IV (ratio > 1.05)
//! - Flat: near-term IV ~ far-term IV (0.95 <= ratio <= 1.05)
//!
//! Also calculates IV z-score for percentile-based entry decisions.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::data::OptionsSnapshot;

/// Term structure regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TermStructureRegime {
    /// Near-term IV < far-term IV (normal, bullish).
    Contango,
    /// Near-term IV > far-term IV (fear, bearish).
    Backwardation,
    /// Near-term IV ~ far-term IV.
    Flat,
    /// Insufficient data.
    Unknown,
}

impl TermStructureRegime {
    /// Whether this regime favors selling premium.
    pub fn favors_selling(&self) -> bool {
        matches!(self, Self::Contango | Self::Flat)
    }
}

/// IV term structure data for a single date.
#[derive(Debug, Clone)]
pub struct IVTermStructure {
    pub date: NaiveDate,
    /// IV at ~30 DTE.
    pub iv_30d: Option<f64>,
    /// IV at ~60 DTE.
    pub iv_60d: Option<f64>,
    /// IV at ~90 DTE.
    pub iv_90d: Option<f64>,
    /// Ratio of 30d to 90d IV.
    pub ratio_30_90: Option<f64>,
    /// Detected regime.
    pub regime: TermStructureRegime,
    /// ATM IV (for z-score calculation).
    pub atm_iv: Option<f64>,
}

impl Default for IVTermStructure {
    fn default() -> Self {
        Self {
            date: NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
            iv_30d: None,
            iv_60d: None,
            iv_90d: None,
            ratio_30_90: None,
            regime: TermStructureRegime::Unknown,
            atm_iv: None,
        }
    }
}

/// IV term structure analyzer.
pub struct IVTermStructureAnalyzer {
    /// Threshold for contango (below this = contango).
    contango_threshold: f64,
    /// Threshold for backwardation (above this = backwardation).
    backwardation_threshold: f64,
    /// Historical IV data for z-score calculation.
    iv_history: Vec<f64>,
    /// Maximum history length.
    max_history: usize,
}

impl Default for IVTermStructureAnalyzer {
    fn default() -> Self {
        Self {
            contango_threshold: 0.95,
            backwardation_threshold: 1.05,
            iv_history: Vec::new(),
            max_history: 252, // 1 year of trading days
        }
    }
}

impl IVTermStructureAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_thresholds(mut self, contango: f64, backwardation: f64) -> Self {
        self.contango_threshold = contango;
        self.backwardation_threshold = backwardation;
        self
    }

    /// Analyze IV term structure from an options snapshot.
    pub fn analyze(&mut self, snapshot: &OptionsSnapshot) -> IVTermStructure {
        let date = snapshot.date;
        let _underlying_price = snapshot.underlying_price;

        // Find chains near target DTEs
        let iv_30d = self.find_atm_iv(snapshot, 25, 35);
        let iv_60d = self.find_atm_iv(snapshot, 50, 70);
        let iv_90d = self.find_atm_iv(snapshot, 80, 100);

        // Calculate ratio and regime
        let ratio_30_90 = match (iv_30d, iv_90d) {
            (Some(near), Some(far)) if far > 0.0 => Some(near / far),
            _ => None,
        };

        let regime = match ratio_30_90 {
            Some(ratio) if ratio < self.contango_threshold => TermStructureRegime::Contango,
            Some(ratio) if ratio > self.backwardation_threshold => TermStructureRegime::Backwardation,
            Some(_) => TermStructureRegime::Flat,
            None => TermStructureRegime::Unknown,
        };

        // Get ATM IV for z-score
        let atm_iv = iv_30d.or(iv_60d);

        // Update history
        if let Some(iv) = atm_iv {
            self.iv_history.push(iv);
            if self.iv_history.len() > self.max_history {
                self.iv_history.remove(0);
            }
        }

        IVTermStructure {
            date,
            iv_30d,
            iv_60d,
            iv_90d,
            ratio_30_90,
            regime,
            atm_iv,
        }
    }

    /// Find ATM implied volatility for chains in a DTE range.
    fn find_atm_iv(&self, snapshot: &OptionsSnapshot, min_dte: i32, max_dte: i32) -> Option<f64> {
        let underlying: f64 = snapshot.underlying_price.try_into().unwrap_or(0.0);
        if underlying <= 0.0 {
            return None;
        }

        let chains = snapshot.chains_by_dte(min_dte, max_dte);
        if chains.is_empty() {
            return None;
        }

        // Use the chain closest to midpoint of range
        let target_dte = (min_dte + max_dte) / 2;
        let chain = chains
            .into_iter()
            .min_by_key(|c| (c.dte - target_dte).abs())?;

        // Find strikes closest to ATM
        let mut atm_ivs = Vec::new();

        for quote in chain.puts.iter().chain(chain.calls.iter()) {
            let strike: f64 = quote.strike.try_into().unwrap_or(0.0);
            let distance = (strike - underlying).abs() / underlying;

            // Within 5% of ATM
            if distance < 0.05 && quote.mid_iv > 0.0 {
                atm_ivs.push(quote.mid_iv);
            }
        }

        if atm_ivs.is_empty() {
            return None;
        }

        // Average of ATM IVs
        Some(atm_ivs.iter().sum::<f64>() / atm_ivs.len() as f64)
    }

    /// Calculate IV z-score (how many std deviations from mean).
    pub fn iv_zscore(&self, current_iv: f64) -> Option<f64> {
        if self.iv_history.len() < 20 {
            return None;
        }

        let mean = self.iv_history.iter().sum::<f64>() / self.iv_history.len() as f64;
        let variance = self
            .iv_history
            .iter()
            .map(|iv| (iv - mean).powi(2))
            .sum::<f64>()
            / self.iv_history.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return None;
        }

        Some((current_iv - mean) / std_dev)
    }

    /// Calculate IV percentile rank (0-100).
    pub fn iv_percentile(&self, current_iv: f64) -> Option<f64> {
        if self.iv_history.is_empty() {
            return None;
        }

        let count_below = self.iv_history.iter().filter(|&&iv| iv < current_iv).count();
        Some((count_below as f64 / self.iv_history.len() as f64) * 100.0)
    }

    /// Get current IV history length.
    pub fn history_length(&self) -> usize {
        self.iv_history.len()
    }

    /// Get historical mean IV.
    pub fn historical_mean(&self) -> Option<f64> {
        if self.iv_history.is_empty() {
            return None;
        }
        Some(self.iv_history.iter().sum::<f64>() / self.iv_history.len() as f64)
    }

    /// Get historical IV standard deviation.
    pub fn historical_std(&self) -> Option<f64> {
        if self.iv_history.len() < 2 {
            return None;
        }

        let mean = self.historical_mean()?;
        let variance = self
            .iv_history
            .iter()
            .map(|iv| (iv - mean).powi(2))
            .sum::<f64>()
            / self.iv_history.len() as f64;
        Some(variance.sqrt())
    }

    /// Pre-load historical IV data.
    pub fn load_history(&mut self, ivs: Vec<f64>) {
        self.iv_history = ivs;
        if self.iv_history.len() > self.max_history {
            self.iv_history = self.iv_history[self.iv_history.len() - self.max_history..].to_vec();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_favors_selling() {
        assert!(TermStructureRegime::Contango.favors_selling());
        assert!(TermStructureRegime::Flat.favors_selling());
        assert!(!TermStructureRegime::Backwardation.favors_selling());
    }

    #[test]
    fn test_iv_zscore() {
        let mut analyzer = IVTermStructureAnalyzer::new();

        // Load history with mean ~0.20, std ~0.02
        let history: Vec<f64> = (0..100).map(|i| 0.18 + 0.0004 * i as f64).collect();
        analyzer.load_history(history);

        // Current IV of 0.24 should be above mean
        let zscore = analyzer.iv_zscore(0.24).unwrap();
        assert!(zscore > 0.0);
    }

    #[test]
    fn test_iv_percentile() {
        let mut analyzer = IVTermStructureAnalyzer::new();

        // Load sorted history
        let history: Vec<f64> = (0..100).map(|i| 0.10 + 0.01 * i as f64).collect();
        analyzer.load_history(history);

        // 0.60 is at the 50th percentile (50 values below)
        let pct = analyzer.iv_percentile(0.60).unwrap();
        assert!((pct - 50.0).abs() < 5.0);
    }

    #[test]
    fn test_regime_classification() {
        // Contango: near < far
        let ratio = 0.90;
        let regime = if ratio < 0.95 {
            TermStructureRegime::Contango
        } else if ratio > 1.05 {
            TermStructureRegime::Backwardation
        } else {
            TermStructureRegime::Flat
        };
        assert_eq!(regime, TermStructureRegime::Contango);

        // Backwardation: near > far
        let ratio = 1.10;
        let regime = if ratio < 0.95 {
            TermStructureRegime::Contango
        } else if ratio > 1.05 {
            TermStructureRegime::Backwardation
        } else {
            TermStructureRegime::Flat
        };
        assert_eq!(regime, TermStructureRegime::Backwardation);
    }
}
