use serde::{Deserialize, Serialize};
use worker::{kv::KvStore, Result};

const KEY_CIRCUIT_BREAKER: &str = "circuit_breaker:status";
const KEY_DAILY_STATS: &str = "daily_stats";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerState {
    pub halted: bool,
    pub reason: Option<String>,
    pub triggered_at: Option<String>,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        CircuitBreakerState { halted: false, reason: None, triggered_at: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyStats {
    pub date: String,
    pub starting_equity: f64,
    pub realized_pnl: f64,
    pub trades_placed: i64,
    pub trades_closed: i64,
}

impl Default for DailyStats {
    fn default() -> Self {
        DailyStats {
            date: String::new(),
            starting_equity: 0.0,
            realized_pnl: 0.0,
            trades_placed: 0,
            trades_closed: 0,
        }
    }
}

pub struct KvClient {
    kv: KvStore,
}

impl KvClient {
    pub fn new(kv: KvStore) -> Self {
        KvClient { kv }
    }

    pub async fn get_circuit_breaker(&self) -> Result<CircuitBreakerState> {
        match self.kv.get(KEY_CIRCUIT_BREAKER).json::<CircuitBreakerState>().await? {
            Some(state) => Ok(state),
            None => Ok(CircuitBreakerState::default()),
        }
    }

    pub async fn set_circuit_breaker(&self, state: &CircuitBreakerState) -> Result<()> {
        let json = serde_json::to_string(state)
            .map_err(|e| worker::Error::RustError(e.to_string()))?;
        self.kv.put(KEY_CIRCUIT_BREAKER, json)?.execute().await?;
        Ok(())
    }

    pub async fn reset_circuit_breaker(&self) -> Result<()> {
        self.set_circuit_breaker(&CircuitBreakerState::default()).await
    }

    pub async fn get_daily_stats(&self) -> Result<DailyStats> {
        use chrono::Utc;
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let key = format!("{}:{}", KEY_DAILY_STATS, today);
        match self.kv.get(&key).json::<DailyStats>().await? {
            Some(stats) => Ok(stats),
            None => Ok(DailyStats { date: today, ..DailyStats::default() }),
        }
    }

    pub async fn set_daily_stats(&self, stats: &DailyStats) -> Result<()> {
        let key = format!("{}:{}", KEY_DAILY_STATS, stats.date);
        let json = serde_json::to_string(stats)
            .map_err(|e| worker::Error::RustError(e.to_string()))?;
        // TTL: 3 days (to survive weekends)
        self.kv.put(&key, json)?.expiration_ttl(60 * 60 * 24 * 3).execute().await?;
        Ok(())
    }

    pub async fn set_with_ttl(&self, key: &str, value: &str, ttl_seconds: u64) -> Result<()> {
        self.kv.put(key, value)?.expiration_ttl(ttl_seconds).execute().await?;
        Ok(())
    }

    pub async fn get_raw(&self, key: &str) -> Result<Option<String>> {
        self.kv.get(key).text().await.map_err(|e| worker::Error::RustError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_breaker_state_round_trips_json() {
        let state = CircuitBreakerState {
            halted: true,
            reason: Some("Daily loss 2.1%".to_string()),
            triggered_at: Some("2026-04-08T10:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&state).expect("serialize");
        let deserialized: CircuitBreakerState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.halted, true);
        assert_eq!(deserialized.reason, Some("Daily loss 2.1%".to_string()));
    }

    #[test]
    fn daily_stats_default_has_zero_pnl() {
        let stats = DailyStats::default();
        assert!((stats.realized_pnl - 0.0).abs() < 1e-9);
        assert_eq!(stats.trades_closed, 0);
    }
}
