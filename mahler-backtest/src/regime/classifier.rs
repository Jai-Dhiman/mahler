//! Market regime classifier.
//!
//! Uses VIX levels and price trends to classify market regimes.

use std::collections::HashMap;

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Market regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// VIX < 15, trending up.
    BullCalm,
    /// VIX 15-25, trending up.
    BullUncertain,
    /// VIX 25-35, trending down.
    BearMild,
    /// VIX > 35, trending down.
    BearVolatile,
    /// VIX > 50, crash.
    Crisis,
    /// VIX 25-40, recovering from crisis.
    Recovery,
    /// Unable to classify.
    Unknown,
}

impl MarketRegime {
    /// Whether this regime favors selling premium.
    pub fn favors_premium_selling(&self) -> bool {
        matches!(
            self,
            Self::BullCalm | Self::BullUncertain | Self::Recovery
        )
    }

    /// Suggested position size multiplier.
    /// Backtest validated (2026-02-25): all_on regime outperforms selective configs.
    pub fn position_size_multiplier(&self) -> f64 {
        match self {
            Self::BullCalm => 1.0,
            Self::BullUncertain => 1.0,
            Self::BearMild => 1.0,
            Self::BearVolatile => 1.0,
            Self::Crisis => 1.0,
            Self::Recovery => 1.0,
            Self::Unknown => 1.0,
        }
    }

    /// Description of the regime.
    pub fn description(&self) -> &'static str {
        match self {
            Self::BullCalm => "Bull market, low volatility",
            Self::BullUncertain => "Bull market, elevated volatility",
            Self::BearMild => "Mild bear market",
            Self::BearVolatile => "Volatile bear market",
            Self::Crisis => "Market crisis",
            Self::Recovery => "Recovery phase",
            Self::Unknown => "Unknown regime",
        }
    }

    /// Historical examples.
    pub fn examples(&self) -> &'static str {
        match self {
            Self::BullCalm => "2017, 2019, 2021",
            Self::BullUncertain => "2013-2014",
            Self::BearMild => "Early 2018",
            Self::BearVolatile => "2008, 2022",
            Self::Crisis => "Mar 2020, Oct 2008",
            Self::Recovery => "Apr-Jun 2020",
            Self::Unknown => "N/A",
        }
    }
}

/// Daily market data for regime classification.
#[derive(Debug, Clone)]
pub struct DailyMarketData {
    pub date: NaiveDate,
    pub vix: f64,
    pub price: Decimal,
}

/// Statistics for a regime.
#[derive(Debug, Clone)]
pub struct RegimeStats {
    pub regime: MarketRegime,
    pub days: usize,
    pub pct_of_total: f64,
    pub avg_vix: f64,
    pub avg_return: f64,
    pub max_drawdown: f64,
}

/// Regime classifier configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeClassifierConfig {
    /// VIX threshold for Bull Calm.
    pub vix_calm: f64,
    /// VIX threshold for uncertain.
    pub vix_uncertain: f64,
    /// VIX threshold for volatile.
    pub vix_volatile: f64,
    /// VIX threshold for crisis.
    pub vix_crisis: f64,
    /// Lookback period for trend calculation (days).
    pub trend_lookback: usize,
    /// Trend threshold for "up" classification.
    pub trend_threshold: f64,
}

impl Default for RegimeClassifierConfig {
    fn default() -> Self {
        Self {
            vix_calm: 15.0,
            vix_uncertain: 25.0,
            vix_volatile: 35.0,
            vix_crisis: 50.0,
            trend_lookback: 20, // ~1 month
            trend_threshold: 0.0, // Positive = up
        }
    }
}

/// Market regime classifier.
pub struct RegimeClassifier {
    config: RegimeClassifierConfig,
    history: Vec<DailyMarketData>,
    regime_history: Vec<(NaiveDate, MarketRegime)>,
}

impl RegimeClassifier {
    /// Create a new classifier.
    pub fn new(config: RegimeClassifierConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            regime_history: Vec::new(),
        }
    }

    /// Classify the current market regime.
    pub fn classify(&mut self, data: DailyMarketData) -> MarketRegime {
        self.history.push(data.clone());

        // Calculate trend
        let trend = self.calculate_trend();

        // Classify based on VIX and trend
        let regime = self.classify_from_vix_trend(data.vix, trend);

        self.regime_history.push((data.date, regime));

        regime
    }

    /// Classify based on VIX level and price trend.
    fn classify_from_vix_trend(&self, vix: f64, trend: Option<f64>) -> MarketRegime {
        // Crisis threshold
        if vix >= self.config.vix_crisis {
            return MarketRegime::Crisis;
        }

        // Check if we're in recovery (was crisis, VIX falling, price rising)
        if self.was_recently_crisis() && vix < self.config.vix_crisis {
            if let Some(t) = trend {
                if t > self.config.trend_threshold {
                    return MarketRegime::Recovery;
                }
            }
        }

        // Classify by VIX level
        let trend_up = trend.map(|t| t > self.config.trend_threshold).unwrap_or(false);

        if vix < self.config.vix_calm {
            if trend_up {
                MarketRegime::BullCalm
            } else {
                MarketRegime::BullUncertain
            }
        } else if vix < self.config.vix_uncertain {
            if trend_up {
                MarketRegime::BullUncertain
            } else {
                MarketRegime::BearMild
            }
        } else if vix < self.config.vix_volatile {
            if trend_up {
                MarketRegime::Recovery
            } else {
                MarketRegime::BearMild
            }
        } else {
            // vix >= volatile
            MarketRegime::BearVolatile
        }
    }

    /// Calculate price trend (% change over lookback period).
    fn calculate_trend(&self) -> Option<f64> {
        if self.history.len() < self.config.trend_lookback {
            return None;
        }

        let start_idx = self.history.len() - self.config.trend_lookback;
        let start_price: f64 = self.history[start_idx].price.try_into().unwrap_or(0.0);
        let end_price: f64 = self.history.last()?.price.try_into().unwrap_or(0.0);

        if start_price <= 0.0 {
            return None;
        }

        Some((end_price - start_price) / start_price * 100.0)
    }

    /// Check if there was a crisis in the last 20 days.
    fn was_recently_crisis(&self) -> bool {
        let lookback = 20;
        self.regime_history
            .iter()
            .rev()
            .take(lookback)
            .any(|(_, regime)| *regime == MarketRegime::Crisis)
    }

    /// Get regime statistics.
    pub fn get_stats(&self) -> HashMap<MarketRegime, RegimeStats> {
        let mut stats: HashMap<MarketRegime, RegimeStats> = HashMap::new();
        let total_days = self.regime_history.len();

        for (_, regime) in &self.regime_history {
            let entry = stats.entry(*regime).or_insert_with(|| RegimeStats {
                regime: *regime,
                ..Default::default()
            });
            entry.days += 1;
        }

        // Calculate percentages
        for entry in stats.values_mut() {
            entry.pct_of_total = entry.days as f64 / total_days as f64 * 100.0;
        }

        stats
    }

    /// Get current regime.
    pub fn current_regime(&self) -> MarketRegime {
        self.regime_history
            .last()
            .map(|(_, regime)| *regime)
            .unwrap_or(MarketRegime::Unknown)
    }

    /// Get regime history.
    pub fn regime_history(&self) -> &[(NaiveDate, MarketRegime)] {
        &self.regime_history
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.regime_history.clear();
    }

    /// Analyze a series of market data.
    pub fn analyze(&mut self, data: Vec<DailyMarketData>) -> HashMap<MarketRegime, RegimeStats> {
        self.clear();

        for d in data {
            self.classify(d);
        }

        self.get_stats()
    }
}

impl Default for RegimeStats {
    fn default() -> Self {
        Self {
            regime: MarketRegime::Unknown,
            days: 0,
            pct_of_total: 0.0,
            avg_vix: 0.0,
            avg_return: 0.0,
            max_drawdown: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_regime_favors_selling() {
        assert!(MarketRegime::BullCalm.favors_premium_selling());
        assert!(MarketRegime::BullUncertain.favors_premium_selling());
        assert!(!MarketRegime::Crisis.favors_premium_selling());
        assert!(!MarketRegime::BearVolatile.favors_premium_selling());
    }

    #[test]
    fn test_regime_multiplier() {
        // all_on: every regime uses 1.0 (backtest validated 2026-02-25)
        assert_eq!(MarketRegime::BullCalm.position_size_multiplier(), 1.0);
        assert_eq!(MarketRegime::Crisis.position_size_multiplier(), 1.0);
        assert_eq!(MarketRegime::BearVolatile.position_size_multiplier(), 1.0);
    }

    #[test]
    fn test_classification_bull_calm() {
        let config = RegimeClassifierConfig::default();
        let mut classifier = RegimeClassifier::new(config);

        // Add enough history for trend calculation
        let base_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        for i in 0..25 {
            let data = DailyMarketData {
                date: base_date + chrono::Duration::days(i),
                vix: 12.0,
                price: dec!(450) + Decimal::from(i), // Trending up
            };
            classifier.classify(data);
        }

        assert_eq!(classifier.current_regime(), MarketRegime::BullCalm);
    }

    #[test]
    fn test_classification_crisis() {
        let config = RegimeClassifierConfig::default();
        let mut classifier = RegimeClassifier::new(config);

        let data = DailyMarketData {
            date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            vix: 55.0,
            price: dec!(400),
        };
        let regime = classifier.classify(data);

        assert_eq!(regime, MarketRegime::Crisis);
    }

    #[test]
    fn test_stats_calculation() {
        let config = RegimeClassifierConfig::default();
        let mut classifier = RegimeClassifier::new(config);

        // Simulate mixed regime history
        let base_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        for i in 0..30 {
            let vix = if i < 10 { 12.0 } else if i < 20 { 22.0 } else { 55.0 };
            let data = DailyMarketData {
                date: base_date + chrono::Duration::days(i as i64),
                vix,
                price: dec!(450),
            };
            classifier.classify(data);
        }

        let stats = classifier.get_stats();
        assert!(stats.contains_key(&MarketRegime::Crisis));
    }
}
