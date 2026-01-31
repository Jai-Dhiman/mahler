//! Walk-forward validation module.
//!
//! Implements rolling window optimization with train/validate/test splits:
//! - Train: 6 months (parameter optimization)
//! - Validate: 1 month (out-of-sample check)
//! - Test: 1 month (held-out performance)
//! - Roll: Monthly

pub mod optimizer;
pub mod periods;

pub use optimizer::{ParameterGrid, WalkForwardOptimizer, WalkForwardResult};
pub use periods::{WalkForwardPeriod, WalkForwardPeriods};
