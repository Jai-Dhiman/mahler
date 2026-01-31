//! Risk management module.
//!
//! Provides:
//! - Position sizing (per-trade and portfolio risk limits)
//! - Portfolio Greeks aggregation
//! - Circuit breakers (daily/weekly loss limits, drawdown, VIX)

pub mod circuit_breakers;
pub mod portfolio_greeks;
pub mod position_sizer;

pub use circuit_breakers::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStatus};
pub use portfolio_greeks::{PortfolioGreeks, PortfolioGreeksConfig};
pub use position_sizer::{PositionSizer, PositionSizerConfig, SizingResult};
