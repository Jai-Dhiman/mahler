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
        nbbo: Option<&crate::measurement::nbbo::NbboSnapshot>,
        entry_short_gamma: Option<f64>,
        entry_short_vega: Option<f64>,
        entry_long_delta: Option<f64>,
        entry_long_gamma: Option<f64>,
        entry_long_vega: Option<f64>,
    ) -> Result<()> {
        use worker::wasm_bindgen::JsValue;
        let nb_short_bid = nbbo.map(|n| n.short_bid.into()).unwrap_or(JsValue::NULL);
        let nb_short_ask = nbbo.map(|n| n.short_ask.into()).unwrap_or(JsValue::NULL);
        let nb_long_bid = nbbo.map(|n| n.long_bid.into()).unwrap_or(JsValue::NULL);
        let nb_long_ask = nbbo.map(|n| n.long_ask.into()).unwrap_or(JsValue::NULL);
        let nb_net_mid = nbbo.map(|n| n.net_mid.into()).unwrap_or(JsValue::NULL);
        let nb_sz_short = nbbo.and_then(|n| n.short_bid_size).map(|s| (s as f64).into()).unwrap_or(JsValue::NULL);
        let nb_sz_long = nbbo.and_then(|n| n.long_ask_size).map(|s| (s as f64).into()).unwrap_or(JsValue::NULL);
        let nb_time = nbbo.map(|n| n.snapshot_time.as_str().into()).unwrap_or(JsValue::NULL);

        self.db
            .prepare(
                "INSERT INTO trades (id, underlying, spread_type, short_strike, long_strike,
                expiration, contracts, entry_credit, max_loss, broker_order_id,
                iv_rank, short_delta, short_theta,
                entry_short_bid, entry_short_ask, entry_long_bid, entry_long_ask, entry_net_mid,
                entry_short_gamma, entry_short_vega, entry_long_delta, entry_long_gamma, entry_long_vega,
                nbbo_displayed_size_short, nbbo_displayed_size_long, nbbo_snapshot_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&[
                id.into(), underlying.into(), spread_type.as_str().into(),
                short_strike.into(), long_strike.into(), expiration.into(),
                (contracts as f64).into(), entry_credit.into(), max_loss.into(),
                broker_order_id.map(|s| s.into()).unwrap_or(JsValue::NULL),
                iv_rank.map(|v| v.into()).unwrap_or(JsValue::NULL),
                short_delta.map(|v| v.into()).unwrap_or(JsValue::NULL),
                short_theta.map(|v| v.into()).unwrap_or(JsValue::NULL),
                nb_short_bid, nb_short_ask, nb_long_bid, nb_long_ask, nb_net_mid,
                entry_short_gamma.map(|v| v.into()).unwrap_or(JsValue::NULL),
                entry_short_vega.map(|v| v.into()).unwrap_or(JsValue::NULL),
                entry_long_delta.map(|v| v.into()).unwrap_or(JsValue::NULL),
                entry_long_gamma.map(|v| v.into()).unwrap_or(JsValue::NULL),
                entry_long_vega.map(|v| v.into()).unwrap_or(JsValue::NULL),
                nb_sz_short, nb_sz_long, nb_time,
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
        exit_nbbo: Option<&crate::measurement::nbbo::NbboSnapshot>,
    ) -> Result<bool> {
        use chrono::Utc;
        use worker::wasm_bindgen::JsValue;
        let now = Utc::now().to_rfc3339();
        let ex_sb = exit_nbbo.map(|n| n.short_bid.into()).unwrap_or(JsValue::NULL);
        let ex_sa = exit_nbbo.map(|n| n.short_ask.into()).unwrap_or(JsValue::NULL);
        let ex_lb = exit_nbbo.map(|n| n.long_bid.into()).unwrap_or(JsValue::NULL);
        let ex_la = exit_nbbo.map(|n| n.long_ask.into()).unwrap_or(JsValue::NULL);
        let ex_mid = exit_nbbo.map(|n| n.net_mid.into()).unwrap_or(JsValue::NULL);

        let result = self.db
            .prepare(
                "UPDATE trades SET status = 'closed', exit_price = ?, exit_time = ?,
                exit_reason = ?, net_pnl = ?,
                exit_short_bid = ?, exit_short_ask = ?, exit_long_bid = ?, exit_long_ask = ?, exit_net_mid = ?
                WHERE id = ? AND status = 'open'",
            )
            .bind(&[
                exit_price.into(), now.as_str().into(),
                exit_reason.as_str().into(), net_pnl.into(),
                ex_sb, ex_sa, ex_lb, ex_la, ex_mid,
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

#[derive(Debug, Clone, Default)]
pub struct MetricsWeeklyRow {
    pub generated_at: String,
    pub window_start: String,
    pub window_end: String,
    pub trade_count: i64,
    pub sharpe: Option<f64>,
    pub sortino: Option<f64>,
    pub profit_factor: Option<f64>,
    pub win_rate: Option<f64>,
    pub pnl_skew: Option<f64>,
    pub max_drawdown_pct: Option<f64>,
    pub mean_slippage_vs_mid: Option<f64>,
    pub max_slippage_vs_mid: Option<f64>,
    pub slippage_vs_orats_ratio: Option<f64>,
    pub fill_size_violation_count: i64,
    pub fill_size_violation_pct: Option<f64>,
    pub regime_buckets_json: String,
    pub greek_ranges_json: String,
    pub sample_size_tag: String,
}

impl D1Client {
    pub async fn insert_equity_snapshot(
        &self,
        snap: &crate::measurement::equity::EquitySnapshot,
    ) -> Result<()> {
        use worker::wasm_bindgen::JsValue;
        let trade_ref = snap.trade_id_ref.as_deref()
            .map(|s| s.into()).unwrap_or(JsValue::NULL);
        self.db.prepare(
            "INSERT INTO equity_history
             (timestamp, event_type, equity, cash, open_position_mtm,
              realized_pnl_day, unrealized_pnl_day, open_position_count, trade_id_ref)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(date(timestamp)) WHERE event_type = 'eod'
               DO UPDATE SET equity = excluded.equity, cash = excluded.cash,
                 open_position_mtm = excluded.open_position_mtm,
                 realized_pnl_day = excluded.realized_pnl_day,
                 unrealized_pnl_day = excluded.unrealized_pnl_day,
                 open_position_count = excluded.open_position_count"
        ).bind(&[
            snap.timestamp.as_str().into(),
            snap.event_type.into(),
            snap.equity.into(), snap.cash.into(),
            snap.open_position_mtm.into(),
            snap.realized_pnl_day.into(), snap.unrealized_pnl_day.into(),
            (snap.open_position_count as f64).into(),
            trade_ref,
        ])?.run().await?;
        Ok(())
    }

    pub async fn insert_portfolio_greeks_eod(
        &self, date: &str,
        pg: &crate::measurement::portfolio_greeks::PortfolioGreeks,
    ) -> Result<()> {
        let delta_json = serde_json::to_string(&pg.delta_by_underlying)
            .map_err(|e| worker::Error::RustError(e.to_string()))?;
        self.db.prepare(
            "INSERT OR REPLACE INTO portfolio_greeks_eod
             (date, beta_weighted_delta, total_gamma, total_vega, total_theta,
              delta_by_underlying, max_gamma_single_position, max_vega_single_position,
              open_position_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).bind(&[
            date.into(),
            pg.beta_weighted_delta.into(),
            pg.total_gamma.into(), pg.total_vega.into(), pg.total_theta.into(),
            delta_json.as_str().into(),
            pg.max_gamma_single_position.into(),
            pg.max_vega_single_position.into(),
            (pg.open_position_count as f64).into(),
        ])?.run().await?;
        Ok(())
    }

    pub async fn insert_market_context_daily(
        &self, date: &str,
        spot_vix: Option<f64>, source: &str, source_date: Option<&str>,
        vixy_close: Option<f64>,
        spy_20d_rv: Option<f64>, spy_20d_ret: Option<f64>, spy_dd: Option<f64>,
    ) -> Result<()> {
        use worker::wasm_bindgen::JsValue;
        let vix = spot_vix.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let src_date = source_date.map(|s| s.into()).unwrap_or(JsValue::NULL);
        let vy = vixy_close.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let rv = spy_20d_rv.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let rt = spy_20d_ret.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let dd = spy_dd.map(|v| v.into()).unwrap_or(JsValue::NULL);
        self.db.prepare(
            "INSERT OR REPLACE INTO market_context_daily
             (date, spot_vix, spot_vix_source, spot_vix_source_date, vixy_close,
              spy_20d_realized_vol, spy_20d_return, spy_drawdown_from_52w_high)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        ).bind(&[
            date.into(), vix, source.into(), src_date, vy, rv, rt, dd,
        ])?.run().await?;
        Ok(())
    }

    pub async fn insert_metrics_weekly(&self, row: MetricsWeeklyRow) -> Result<()> {
        use worker::wasm_bindgen::JsValue;
        let sh = row.sharpe.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let so = row.sortino.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let pf = row.profit_factor.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let wr = row.win_rate.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let sk = row.pnl_skew.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let md = row.max_drawdown_pct.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let ms = row.mean_slippage_vs_mid.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let mx = row.max_slippage_vs_mid.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let rt = row.slippage_vs_orats_ratio.map(|v| v.into()).unwrap_or(JsValue::NULL);
        let fvp = row.fill_size_violation_pct.map(|v| v.into()).unwrap_or(JsValue::NULL);
        self.db.prepare(
            "INSERT INTO metrics_weekly (generated_at, window_start, window_end, trade_count,
             sharpe, sortino, profit_factor, win_rate, pnl_skew, max_drawdown_pct,
             mean_slippage_vs_mid, max_slippage_vs_mid, slippage_vs_orats_ratio,
             fill_size_violation_count, fill_size_violation_pct,
             regime_buckets, greek_ranges, sample_size_tag)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).bind(&[
            row.generated_at.as_str().into(),
            row.window_start.as_str().into(),
            row.window_end.as_str().into(),
            (row.trade_count as f64).into(),
            sh, so, pf, wr, sk, md, ms, mx, rt,
            (row.fill_size_violation_count as f64).into(), fvp,
            row.regime_buckets_json.as_str().into(),
            row.greek_ranges_json.as_str().into(),
            row.sample_size_tag.as_str().into(),
        ])?.run().await?;
        Ok(())
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

    #[test]
    fn create_trade_accepts_full_parameter_list() {
        // Compile-only assertion that create_trade accepts NBBO + expanded Greeks.
        // Build an async block that would call it; never executed.
        let _unused = async {
            let db: D1Client = unreachable!();
            let nbbo: crate::measurement::nbbo::NbboSnapshot = unreachable!();
            db.create_trade(
                "id", "SPY", &SpreadType::BullPut, 460.0, 455.0, "2026-05-15",
                2, 0.75, 425.0, None, Some(65.0), Some(-0.28), Some(0.05),
                Some(&nbbo),
                Some(0.010), Some(0.30), Some(-0.18), Some(0.008), Some(0.25),
            ).await.ok();
        };
    }

    #[test]
    fn close_trade_signature_accepts_exit_nbbo() {
        let _unused = async {
            let db: D1Client = unreachable!();
            let nbbo: crate::measurement::nbbo::NbboSnapshot = unreachable!();
            let reason: crate::types::ExitReason = unreachable!();
            db.close_trade("id", 0.20, &reason, 55.0, Some(&nbbo)).await.ok();
        };
    }

    #[test]
    fn measurement_inserters_exist_with_expected_signatures() {
        use crate::measurement::equity::EquitySnapshot;
        use crate::measurement::portfolio_greeks::PortfolioGreeks;
        let _unused = async {
            let db: D1Client = unreachable!();
            let snap: EquitySnapshot = unreachable!();
            let pg: PortfolioGreeks = unreachable!();
            db.insert_equity_snapshot(&snap).await.ok();
            db.insert_portfolio_greeks_eod("2026-04-22", &pg).await.ok();
            db.insert_market_context_daily(
                "2026-04-22", Some(17.5), "fred", Some("2026-04-21"),
                Some(14.2), Some(0.12), Some(0.02), Some(0.03),
            ).await.ok();
            db.insert_metrics_weekly(MetricsWeeklyRow::default()).await.ok();
        };
    }
}
