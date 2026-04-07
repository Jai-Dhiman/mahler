pub mod loader;
pub mod orats;
pub mod types;

pub use loader::{DataLoader, LoaderError, SnapshotIterator, EXPECTED_COLUMNS};
pub use orats::{records_to_snapshot, ORATSClient, ORATSError, RawStrikeRecord, TickerInfo};
pub use types::{Greeks, OptionQuote, OptionType, OptionsChain, OptionsSnapshot, UnderlyingBar};

use crate::engine_core::events::MarketEvent;

pub trait DataSource {
    fn next_event(&mut self) -> Option<MarketEvent>;
}

pub struct HistoricalDataSource {
    snapshots: Vec<OptionsSnapshot>,
    index: usize,
}

impl HistoricalDataSource {
    pub fn new(
        data_dir: &str,
        ticker: &str,
        start_date: chrono::NaiveDate,
        end_date: chrono::NaiveDate,
    ) -> Result<Self, loader::LoaderError> {
        let loader = DataLoader::new(data_dir);
        let snapshots = loader.load_snapshots(ticker, start_date, end_date)?;
        Ok(Self { snapshots, index: 0 })
    }

    pub fn from_snapshots(snapshots: Vec<OptionsSnapshot>) -> Self {
        Self { snapshots, index: 0 }
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

impl DataSource for HistoricalDataSource {
    fn next_event(&mut self) -> Option<MarketEvent> {
        if self.index >= self.snapshots.len() {
            return None;
        }
        let snapshot = self.snapshots[self.index].clone();
        self.index += 1;
        Some(MarketEvent::new(snapshot))
    }
}

#[cfg(test)]
mod datasource_tests {
    use super::*;

    #[test]
    fn test_historical_datasource_emits_events_in_order() {
        let source = HistoricalDataSource::new(
            "data/orats",
            "SPY",
            chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2020, 1, 10).unwrap(),
        );

        match source {
            Ok(mut src) => {
                let mut prev_ts = None;
                let mut count = 0;
                while let Some(event) = src.next_event() {
                    if let Some(prev) = prev_ts {
                        assert!(event.timestamp >= prev, "Events must be in chronological order");
                    }
                    assert!(!event.snapshot.chains.is_empty(), "Snapshot should have chains");
                    assert!(event.snapshot.underlying_price > rust_decimal::Decimal::ZERO);
                    prev_ts = Some(event.timestamp);
                    count += 1;
                }
                assert!(count >= 5, "Should have at least 5 trading days in Jan 2-10 2020, got {}", count);
            }
            Err(e) => {
                eprintln!("Skipping test: {}", e);
            }
        }
    }

    #[test]
    fn test_historical_datasource_returns_none_when_exhausted() {
        let source = HistoricalDataSource::new(
            "data/orats",
            "SPY",
            chrono::NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2020, 1, 3).unwrap(),
        );

        match source {
            Ok(mut src) => {
                while src.next_event().is_some() {}
                assert!(src.next_event().is_none());
            }
            Err(_) => {
                eprintln!("Skipping test: data not available");
            }
        }
    }
}
