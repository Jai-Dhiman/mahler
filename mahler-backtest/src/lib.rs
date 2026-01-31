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
pub use backtest::{BacktestConfig, BacktestEngine, BacktestResult, Position, SlippageModel, Trade};
pub use risk::{CircuitBreaker, PortfolioGreeks, PositionSizer};
pub use analytics::{IVTermStructureAnalyzer, SpreadScreener, SpreadCandidate};
pub use walkforward::{WalkForwardOptimizer, WalkForwardResult, ParameterGrid};
pub use regime::{MarketRegime, RegimeClassifier};
pub use metrics::{PerformanceMetrics, MetricsCalculator};
