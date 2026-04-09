pub mod analytics;
pub mod backtest;
pub mod data;
pub mod metrics;
pub mod regime;
pub mod risk;
pub mod validation;
pub mod walkforward;

// Re-export commonly used types
pub use data::{OptionQuote, OptionType, OptionsChain, OptionsSnapshot};
pub use validation::{BlackScholes, DataIntegrityValidator, GreeksValidator};
pub use backtest::{BacktestConfig, BacktestResult, Position, SlippageModel, Trade};
pub use risk::{CircuitBreaker, PortfolioGreeks, PositionSizer};
pub use analytics::{IVTermStructureAnalyzer, SpreadScreener, SpreadCandidate};
pub use walkforward::{WalkForwardOptimizer, WalkForwardResult, ParameterGrid};
pub use regime::{MarketRegime, RegimeClassifier};
pub use metrics::{PerformanceMetrics, MetricsCalculator};

pub mod engine_core;
pub mod strategy;
pub mod broker;
pub mod portfolio;

pub use engine_core::engine::Engine;
pub use strategy::Strategy;
pub use broker::Broker;
pub use data::DataSource;
pub use risk::gate::RiskGate;
pub use portfolio::PortfolioTracker;

#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_new_module_imports() {
        use crate::engine_core::engine::Engine;
        use crate::engine_core::events::{MarketEvent, OrderIntent, FillEvent, RiskDecision, PortfolioView};
        use crate::strategy::Strategy;
        use crate::strategy::put_spread::PutCreditSpreadStrategy;
        use crate::broker::Broker;
        use crate::broker::backtest::SimulatedBroker;
        use crate::portfolio::PortfolioTracker;
        use crate::risk::gate::{RiskGate, DefaultRiskGate};
        use crate::data::DataSource;

        // Verify old types still accessible
        use crate::backtest::{Position, Trade, SlippageModel, CommissionModel};
        use crate::data::{OptionQuote, OptionsChain, OptionsSnapshot};
        use crate::risk::{CircuitBreaker, PositionSizer};
        use crate::metrics::MetricsCalculator;
    }
}
