//! Market regime classification module.
//!
//! Categorizes market conditions based on VIX and price trends:
//! - Bull Calm: VIX < 15, trend up
//! - Bull Uncertain: VIX 15-25, trend up
//! - Bear Volatile: VIX > 30, trend down
//! - Crisis: VIX > 50, crash
//! - Recovery: VIX 25-40, trend up

pub mod classifier;

pub use classifier::{DailyMarketData, MarketRegime, RegimeClassifier, RegimeClassifierConfig, RegimeStats};
