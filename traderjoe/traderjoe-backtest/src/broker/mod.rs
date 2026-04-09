pub mod backtest;

use crate::engine_core::events::{FillEvent, OrderIntent};
use crate::data::OptionsSnapshot;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BrokerError {
    #[error("No matching quote for strike {strike} {option_type:?} exp {expiration}")]
    NoMatchingQuote {
        strike: rust_decimal::Decimal,
        option_type: crate::data::OptionType,
        expiration: chrono::NaiveDate,
    },

    #[error("Order rejected: {0}")]
    OrderRejected(String),
}

pub trait Broker {
    fn submit_order(
        &mut self,
        order: &OrderIntent,
        snapshot: &OptionsSnapshot,
    ) -> Result<FillEvent, BrokerError>;
}
