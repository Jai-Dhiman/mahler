//! Data layer for the backtester.
//!
//! This module handles:
//! - Core data types (OptionQuote, OptionsChain, OptionsSnapshot)
//! - ORATS API client for downloading historical data
//! - Parquet file loading and storage

pub mod orats;
pub mod types;

pub use orats::{ORATSClient, ORATSError, RawStrikeRecord, TickerInfo, records_to_snapshot};
pub use types::{Greeks, OptionQuote, OptionType, OptionsChain, OptionsSnapshot, UnderlyingBar};
