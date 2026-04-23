#[derive(Debug, Clone, PartialEq)]
pub enum EquityEvent {
    Eod,
    TradeOpen,
    TradeClose,
    CircuitBreaker,
}

impl EquityEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            EquityEvent::Eod => "eod",
            EquityEvent::TradeOpen => "trade_open",
            EquityEvent::TradeClose => "trade_close",
            EquityEvent::CircuitBreaker => "circuit_breaker",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnapshotInputs {
    pub timestamp: String,
    pub equity: f64,
    pub cash: f64,
    pub open_position_mtm: f64,
    pub realized_pnl_day: f64,
    pub unrealized_pnl_day: f64,
    pub open_position_count: i64,
    pub trade_id_ref: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EquitySnapshot {
    pub timestamp: String,
    pub event_type: &'static str,
    pub equity: f64,
    pub cash: f64,
    pub open_position_mtm: f64,
    pub realized_pnl_day: f64,
    pub unrealized_pnl_day: f64,
    pub open_position_count: i64,
    pub trade_id_ref: Option<String>,
}

pub fn build_equity_snapshot(event: EquityEvent, inputs: SnapshotInputs) -> EquitySnapshot {
    EquitySnapshot {
        timestamp: inputs.timestamp,
        event_type: event.as_str(),
        equity: inputs.equity,
        cash: inputs.cash,
        open_position_mtm: inputs.open_position_mtm,
        realized_pnl_day: inputs.realized_pnl_day,
        unrealized_pnl_day: inputs.unrealized_pnl_day,
        open_position_count: inputs.open_position_count,
        trade_id_ref: inputs.trade_id_ref,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_equity_snapshot_for_eod_event() {
        let snap = build_equity_snapshot(EquityEvent::Eod, SnapshotInputs {
            timestamp: "2026-04-22T22:00:00Z".to_string(),
            equity: 100_000.0, cash: 80_000.0,
            open_position_mtm: 20_000.0,
            realized_pnl_day: 500.0, unrealized_pnl_day: 150.0,
            open_position_count: 3,
            trade_id_ref: None,
        });
        assert_eq!(snap.event_type, "eod");
        assert!((snap.equity - 100_000.0).abs() < 1e-9);
        assert_eq!(snap.open_position_count, 3);
        assert!(snap.trade_id_ref.is_none());
    }

    #[test]
    fn build_equity_snapshot_for_trade_open_carries_trade_id() {
        let snap = build_equity_snapshot(EquityEvent::TradeOpen, SnapshotInputs {
            timestamp: "2026-04-22T14:05:00Z".to_string(),
            equity: 100_500.0, cash: 79_500.0,
            open_position_mtm: 21_000.0,
            realized_pnl_day: 0.0, unrealized_pnl_day: 0.0,
            open_position_count: 4,
            trade_id_ref: Some("trade-xyz".to_string()),
        });
        assert_eq!(snap.event_type, "trade_open");
        assert_eq!(snap.trade_id_ref.as_deref(), Some("trade-xyz"));
    }
}
