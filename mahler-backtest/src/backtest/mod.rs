//! Backtesting engine for options strategies.
//!
//! This module provides a complete backtesting framework for options trading:
//! - Trade lifecycle management (entry, position tracking, exits)
//! - Slippage modeling (ORATS methodology)
//! - Commission tracking
//! - Mark-to-market valuation
//! - P&L calculation

pub mod commission;
pub mod engine;
pub mod slippage;
pub mod trade;

pub use commission::{Commission, CommissionModel};
pub use engine::{BacktestConfig, BacktestEngine, BacktestResult, EquityPoint};
pub use slippage::{Slippage, SlippageModel};
pub use trade::{
    ExitReason, Position, PositionLeg, PositionStatus, SpreadType, Trade, TradeDirection,
};
