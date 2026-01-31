//! Performance metrics module.
//!
//! Provides comprehensive performance calculations:
//! - Win rate, profit factor
//! - Sharpe ratio, Sortino ratio
//! - Maximum drawdown
//! - CAGR, monthly returns
//! - Risk-adjusted metrics

pub mod calculator;

pub use calculator::{PerformanceMetrics, MetricsCalculator, DrawdownAnalysis};
