//! Validation module for ORATS data and calculations.
//!
//! This module provides comprehensive validation of:
//! - Data integrity (schema, continuity, value ranges)
//! - Greeks calculations (Black-Scholes verification)

pub mod data_integrity;
pub mod greeks;

pub use data_integrity::{
    DataIntegrityReport, DataIntegrityValidator, ValidationError, ValidationResult,
};
pub use greeks::{BlackScholes, GreeksValidator, GreeksValidationReport};
