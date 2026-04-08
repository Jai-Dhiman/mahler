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
}
