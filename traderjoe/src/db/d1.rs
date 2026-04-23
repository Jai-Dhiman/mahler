use serde_json::Value;
use worker::{D1Database, Result};

use crate::types::{ExitReason, SpreadType, Trade, TradeStatus};

/// Serialization helpers for D1 row deserialization.
pub struct TradeRow;

impl TradeRow {
    /// Deserialize a D1 row (as JSON Value) into a Trade.
    ///
    /// Returns None if any required field is missing or unparseable.
    pub fn from_d1_row(row: &Value) -> Option<Trade> {
        Some(Trade {
            id: row["id"].as_str()?.to_string(),
            created_at: row["created_at"].as_str()?.to_string(),
            underlying: row["underlying"].as_str()?.to_string(),
            spread_type: match row["spread_type"].as_str()? {
                "bull_put" => SpreadType::BullPut,
                "bear_call" => SpreadType::BearCall,
                _ => return None,
            },
            short_strike: row["short_strike"].as_f64()?,
            long_strike: row["long_strike"].as_f64()?,
            expiration: row["expiration"].as_str()?.to_string(),
            contracts: row["contracts"].as_i64()?,
            entry_credit: row["entry_credit"].as_f64()?,
            max_loss: row["max_loss"].as_f64()?,
            broker_order_id: row["broker_order_id"].as_str().map(str::to_string),
            status: match row["status"].as_str()? {
                "pending_fill" => TradeStatus::PendingFill,
                "open" => TradeStatus::Open,
                "closed" => TradeStatus::Closed,
                "cancelled" => TradeStatus::Cancelled,
                _ => return None,
            },
            fill_price: row["fill_price"].as_f64(),
            fill_time: row["fill_time"].as_str().map(str::to_string),
            exit_price: row["exit_price"].as_f64(),
            exit_time: row["exit_time"].as_str().map(str::to_string),
            exit_reason: row["exit_reason"].as_str().and_then(|s| match s {
                "profit_target" => Some(ExitReason::ProfitTarget),
                "stop_loss" => Some(ExitReason::StopLoss),
                "expiration" => Some(ExitReason::Expiration),
                "manual" => Some(ExitReason::Manual),
                _ => None,
            }),
            net_pnl: row["net_pnl"].as_f64(),
            iv_rank: row["iv_rank"].as_f64(),
            short_delta: row["short_delta"].as_f64(),
            short_theta: row["short_theta"].as_f64(),
            entry_short_bid: row["entry_short_bid"].as_f64(),
            entry_short_ask: row["entry_short_ask"].as_f64(),
            entry_long_bid: row["entry_long_bid"].as_f64(),
            entry_long_ask: row["entry_long_ask"].as_f64(),
            entry_net_mid: row["entry_net_mid"].as_f64(),
            exit_short_bid: row["exit_short_bid"].as_f64(),
            exit_short_ask: row["exit_short_ask"].as_f64(),
            exit_long_bid: row["exit_long_bid"].as_f64(),
            exit_long_ask: row["exit_long_ask"].as_f64(),
            exit_net_mid: row["exit_net_mid"].as_f64(),
            entry_short_gamma: row["entry_short_gamma"].as_f64(),
            entry_short_vega: row["entry_short_vega"].as_f64(),
            entry_long_delta: row["entry_long_delta"].as_f64(),
            entry_long_gamma: row["entry_long_gamma"].as_f64(),
            entry_long_vega: row["entry_long_vega"].as_f64(),
            nbbo_displayed_size_short: row["nbbo_displayed_size_short"].as_i64(),
            nbbo_displayed_size_long: row["nbbo_displayed_size_long"].as_i64(),
            nbbo_snapshot_time: row["nbbo_snapshot_time"].as_str().map(str::to_string),
        })
    }
}

/// Client for Cloudflare D1 database operations.
pub struct D1Client {
    db: D1Database,
}

impl D1Client {
    pub fn new(db: D1Database) -> Self {
        D1Client { db }
    }

    pub async fn create_trade(
        &self,
        id: &str,
        underlying: &str,
        spread_type: &SpreadType,
        short_strike: f64,
        long_strike: f64,
        expiration: &str,
        contracts: i64,
        entry_credit: f64,
        max_loss: f64,
        broker_order_id: Option<&str>,
        iv_rank: Option<f64>,
        short_delta: Option<f64>,
        short_theta: Option<f64>,
    ) -> Result<()> {
        self.db
            .prepare(
                "INSERT INTO trades (id, underlying, spread_type, short_strike, long_strike,
                expiration, contracts, entry_credit, max_loss, broker_order_id,
                iv_rank, short_delta, short_theta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&[
                id.into(),
                underlying.into(),
                spread_type.as_str().into(),
                short_strike.into(),
                long_strike.into(),
                expiration.into(),
                (contracts as f64).into(),
                entry_credit.into(),
                max_loss.into(),
                broker_order_id.map(|s| s.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
                iv_rank.map(|v| v.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
                short_delta.map(|v| v.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
                short_theta.map(|v| v.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
            ])?
            .run()
            .await?;
        Ok(())
    }

    pub async fn get_open_trades(&self) -> Result<Vec<Trade>> {
        let result = self
            .db
            .prepare("SELECT * FROM trades WHERE status IN ('open', 'pending_fill')")
            .all()
            .await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(TradeRow::from_d1_row).collect())
    }

    /// Close a trade. Returns Ok(true) if the row was updated, Ok(false) if it
    /// was already closed (concurrent invocation beat us — caller should skip
    /// the Discord notification to avoid double-reporting).
    pub async fn close_trade(
        &self,
        trade_id: &str,
        exit_price: f64,
        exit_reason: &ExitReason,
        net_pnl: f64,
    ) -> Result<bool> {
        use chrono::Utc;
        let now = Utc::now().to_rfc3339();
        let result = self.db
            .prepare(
                "UPDATE trades SET status = 'closed', exit_price = ?, exit_time = ?,
                exit_reason = ?, net_pnl = ? WHERE id = ? AND status = 'open'",
            )
            .bind(&[
                exit_price.into(),
                now.as_str().into(),
                exit_reason.as_str().into(),
                net_pnl.into(),
                trade_id.into(),
            ])?
            .run()
            .await?;
        let written = result.meta()
            .ok()
            .flatten()
            .and_then(|m| m.rows_written)
            .unwrap_or(0);
        Ok(written > 0)
    }

    pub async fn mark_trade_filled(&self, trade_id: &str, fill_price: f64) -> Result<()> {
        use chrono::Utc;
        let now = Utc::now().to_rfc3339();
        self.db
            .prepare(
                "UPDATE trades SET status = 'open', fill_price = ?, fill_time = ? WHERE id = ?",
            )
            .bind(&[fill_price.into(), now.as_str().into(), trade_id.into()])?
            .run()
            .await?;
        Ok(())
    }

    pub async fn get_iv_history(&self, symbol: &str, lookback_days: i64) -> Result<Vec<f64>> {
        let result = self
            .db
            .prepare(
                "SELECT iv FROM iv_history WHERE symbol = ?
                AND date >= date('now', ? || ' days')
                ORDER BY date ASC",
            )
            .bind(&[symbol.into(), format!("-{}", lookback_days).as_str().into()])?
            .all()
            .await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(|r| r["iv"].as_f64()).collect())
    }

    pub async fn upsert_iv_history(&self, symbol: &str, date: &str, iv: f64) -> Result<()> {
        self.db
            .prepare(
                "INSERT INTO iv_history (symbol, date, iv) VALUES (?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET iv = excluded.iv",
            )
            .bind(&[symbol.into(), date.into(), iv.into()])?
            .run()
            .await?;
        Ok(())
    }

    pub async fn log_scan(
        &self,
        scan_type: &str,
        underlyings_scanned: i64,
        opportunities_found: i64,
        trades_placed: i64,
        vix: Option<f64>,
        circuit_breaker_active: bool,
        duration_ms: i64,
        notes: Option<&str>,
    ) -> Result<()> {
        use chrono::Utc;
        let now = Utc::now().to_rfc3339();
        self.db
            .prepare(
                "INSERT INTO scan_log (scan_time, scan_type, underlyings_scanned,
                opportunities_found, trades_placed, vix, circuit_breaker_active,
                duration_ms, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&[
                now.as_str().into(),
                scan_type.into(),
                (underlyings_scanned as f64).into(),
                (opportunities_found as f64).into(),
                (trades_placed as f64).into(),
                vix.map(|v| v.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
                (circuit_breaker_active as i32).into(),
                (duration_ms as f64).into(),
                notes.map(|s| s.into()).unwrap_or(worker::wasm_bindgen::JsValue::NULL),
            ])?
            .run()
            .await?;
        Ok(())
    }

    /// Get all trades closed on a specific date (YYYY-MM-DD).
    pub async fn get_trades_closed_on(&self, date: &str) -> Result<Vec<Trade>> {
        let result = self
            .db
            .prepare(
                "SELECT * FROM trades WHERE status = 'closed'
                AND date(exit_time) = ?",
            )
            .bind(&[date.into()])?
            .all()
            .await?;
        let rows = result.results::<serde_json::Value>()?;
        Ok(rows.iter().filter_map(TradeRow::from_d1_row).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserializes_open_trade_from_d1_row() {
        let row = json!({
            "id": "trade-123",
            "created_at": "2026-04-08T10:00:00",
            "underlying": "SPY",
            "spread_type": "bull_put",
            "short_strike": 460.0,
            "long_strike": 455.0,
            "expiration": "2026-05-15",
            "contracts": 2,
            "entry_credit": 0.75,
            "max_loss": 425.0,
            "broker_order_id": "order-abc",
            "status": "open",
            "fill_price": 0.74,
            "fill_time": "2026-04-08T10:05:00",
            "exit_price": null,
            "exit_time": null,
            "exit_reason": null,
            "net_pnl": null,
            "iv_rank": 65.0,
            "short_delta": -0.10,
            "short_theta": 0.05
        });

        let trade = TradeRow::from_d1_row(&row).expect("should parse valid row");
        assert_eq!(trade.id, "trade-123");
        assert_eq!(trade.underlying, "SPY");
        assert!(matches!(trade.status, crate::types::TradeStatus::Open));
        assert_eq!(trade.contracts, 2);
        assert!((trade.entry_credit - 0.75).abs() < 1e-9);
    }

    #[test]
    fn returns_none_for_row_with_missing_required_field() {
        let row = json!({ "id": "trade-123" });
        let result = TradeRow::from_d1_row(&row);
        assert!(result.is_none());
    }
}
