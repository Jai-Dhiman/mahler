//! Mahler Backtest - High-performance options backtesting engine.
//!
//! This crate provides:
//! - Data layer for loading and managing historical options data
//! - Pricing layer for Black-Scholes calculations
//! - Strategy layer for trade candidate screening
//! - Simulation layer for realistic trade execution
//! - Walk-forward optimization framework
//!
//! # Architecture
//!
//! The backtester is designed as a series of layers:
//!
//! ```text
//! ORATS Data (Parquet) -> Data Layer -> Pricing Layer -> Strategy Layer
//!                                                             |
//! Output Layer <- Walk-Forward Layer <- Simulation Layer <----+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use mahler_backtest::data::{OptionsSnapshot, ORATSClient};
//!
//! // Load historical data
//! let snapshot = load_snapshot("SPY", date)?;
//!
//! // Find trade candidates
//! let candidates = screener.scan(&snapshot)?;
//!
//! // Simulate trades
//! let results = simulator.run(candidates)?;
//! ```

pub mod data;

// Re-export commonly used types
pub use data::{OptionQuote, OptionType, OptionsChain, OptionsSnapshot};
