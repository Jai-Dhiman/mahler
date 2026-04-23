use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpreadType {
    BullPut,
    BearCall,
}

impl SpreadType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpreadType::BullPut => "bull_put",
            SpreadType::BearCall => "bear_call",
        }
    }
}

impl std::fmt::Display for SpreadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditSpread {
    pub underlying: String,
    pub spread_type: SpreadType,
    pub short_strike: f64,
    pub long_strike: f64,
    pub expiration: String,
    pub entry_credit: f64,
    pub short_delta: Option<f64>,
    pub short_theta: Option<f64>,
    pub short_iv: Option<f64>,
    pub long_iv: Option<f64>,
}

impl CreditSpread {
    pub fn width(&self) -> f64 {
        (self.short_strike - self.long_strike).abs()
    }

    pub fn max_loss_per_contract(&self) -> f64 {
        (self.width() - self.entry_credit) * 100.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TradeStatus {
    PendingFill,
    Open,
    Closed,
    Cancelled,
}

impl TradeStatus {
    pub fn is_closed(&self) -> bool {
        matches!(self, TradeStatus::Closed | TradeStatus::Cancelled)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            TradeStatus::PendingFill => "pending_fill",
            TradeStatus::Open => "open",
            TradeStatus::Closed => "closed",
            TradeStatus::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitReason {
    ProfitTarget,
    StopLoss,
    Expiration,
    Manual,
}

impl ExitReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExitReason::ProfitTarget => "profit_target",
            ExitReason::StopLoss => "stop_loss",
            ExitReason::Expiration => "expiration",
            ExitReason::Manual => "manual",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub created_at: String,
    pub underlying: String,
    pub spread_type: SpreadType,
    pub short_strike: f64,
    pub long_strike: f64,
    pub expiration: String,
    pub contracts: i64,
    pub entry_credit: f64,
    pub max_loss: f64,
    pub broker_order_id: Option<String>,
    pub status: TradeStatus,
    pub fill_price: Option<f64>,
    pub fill_time: Option<String>,
    pub exit_price: Option<f64>,
    pub exit_time: Option<String>,
    pub exit_reason: Option<ExitReason>,
    pub net_pnl: Option<f64>,
    pub iv_rank: Option<f64>,
    pub short_delta: Option<f64>,
    pub short_theta: Option<f64>,
    pub entry_short_bid: Option<f64>,
    pub entry_short_ask: Option<f64>,
    pub entry_long_bid: Option<f64>,
    pub entry_long_ask: Option<f64>,
    pub entry_net_mid: Option<f64>,
    pub exit_short_bid: Option<f64>,
    pub exit_short_ask: Option<f64>,
    pub exit_long_bid: Option<f64>,
    pub exit_long_ask: Option<f64>,
    pub exit_net_mid: Option<f64>,
    pub entry_short_gamma: Option<f64>,
    pub entry_short_vega: Option<f64>,
    pub entry_long_delta: Option<f64>,
    pub entry_long_gamma: Option<f64>,
    pub entry_long_vega: Option<f64>,
    pub nbbo_displayed_size_short: Option<i64>,
    pub nbbo_displayed_size_long: Option<i64>,
    pub nbbo_snapshot_time: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credit_spread_max_loss_is_width_minus_credit_times_100() {
        let spread = CreditSpread {
            underlying: "SPY".to_string(),
            spread_type: SpreadType::BullPut,
            short_strike: 480.0,
            long_strike: 475.0,
            expiration: "2026-05-15".to_string(),
            entry_credit: 0.50,
            short_delta: Some(-0.10),
            short_theta: Some(0.05),
            short_iv: Some(0.22),
            long_iv: Some(0.24),
        };
        assert!((spread.max_loss_per_contract() - 450.0).abs() < f64::EPSILON);
    }

    #[test]
    fn trade_status_open_means_not_closed() {
        let status = TradeStatus::Open;
        assert!(!status.is_closed());
    }

    #[test]
    fn trade_status_closed_is_closed() {
        let status = TradeStatus::Closed;
        assert!(status.is_closed());
    }

    #[test]
    fn trade_round_trips_nbbo_and_expanded_greeks() {
        let trade = Trade {
            id: "t1".to_string(),
            created_at: "2026-04-22T10:00:00Z".to_string(),
            underlying: "SPY".to_string(),
            spread_type: SpreadType::BullPut,
            short_strike: 460.0, long_strike: 455.0,
            expiration: "2026-05-15".to_string(),
            contracts: 2,
            entry_credit: 0.75, max_loss: 425.0,
            broker_order_id: Some("o1".to_string()),
            status: TradeStatus::Open,
            fill_price: Some(0.74), fill_time: Some("2026-04-22T10:05:00Z".to_string()),
            exit_price: None, exit_time: None, exit_reason: None, net_pnl: None,
            iv_rank: Some(65.0),
            short_delta: Some(-0.28), short_theta: Some(0.05),
            entry_short_bid: Some(0.74), entry_short_ask: Some(0.76),
            entry_long_bid: Some(0.24), entry_long_ask: Some(0.26),
            entry_net_mid: Some(0.75),
            exit_short_bid: None, exit_short_ask: None,
            exit_long_bid: None, exit_long_ask: None, exit_net_mid: None,
            entry_short_gamma: Some(0.012), entry_short_vega: Some(0.32),
            entry_long_delta: Some(-0.18), entry_long_gamma: Some(0.009), entry_long_vega: Some(0.28),
            nbbo_displayed_size_short: Some(50), nbbo_displayed_size_long: Some(75),
            nbbo_snapshot_time: Some("2026-04-22T10:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&trade).unwrap();
        let back: Trade = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entry_net_mid, Some(0.75));
        assert_eq!(back.entry_short_gamma, Some(0.012));
        assert_eq!(back.nbbo_displayed_size_short, Some(50));
    }
}
