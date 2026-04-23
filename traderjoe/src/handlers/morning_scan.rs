use worker::{console_log, Env, Result};

use crate::analysis::iv_rank::calculate_iv_metrics;
use crate::analysis::screener::OptionsScreener;
use crate::broker::alpaca::{build_occ_symbol, AlpacaClient};
use crate::broker::types::SpreadOrder;
use crate::config::SpreadConfig;
use crate::db::d1::D1Client;
use crate::db::kv::KvClient;
use crate::notifications::discord::DiscordClient;
use crate::risk::circuit_breaker::{CircuitBreaker, EvaluateParams, RiskLevel};
use crate::risk::position_sizer::{ExistingPosition, PositionSizer};
use crate::types::SpreadType;

/// Returns true if scanning should be skipped because the circuit breaker
/// was tripped today (not yesterday — yesterday gets auto-reset).
pub fn should_skip_due_to_circuit_breaker(halted: bool, triggered_today: bool) -> bool {
    halted && triggered_today
}

/// Returns true if IV conditions are sufficient to enter new positions.
///
/// Blocks entries when IV percentile is below the configured minimum.
/// When history is empty, `calculate_iv_metrics` returns 50.0 (neutral default),
/// which passes the standard 50% threshold and allows trading in new deployments.
pub fn passes_iv_filter(iv_percentile: f64, min_iv_percentile: f64) -> bool {
    iv_percentile >= min_iv_percentile
}

/// Extract the date portion (YYYY-MM-DD) from an ISO timestamp string.
fn date_from_iso(iso: &str) -> &str {
    &iso[..iso.len().min(10)]
}

pub async fn run(env: &Env) -> Result<()> {
    run_inner(env, false).await
}

pub async fn run_forced(env: &Env) -> Result<()> {
    run_inner(env, true).await
}

async fn run_inner(env: &Env, force: bool) -> Result<()> {
    use chrono::Utc;
    let start = Utc::now();
    console_log!("Morning scan starting at {} (force={})", start.to_rfc3339(), force);

    let db = D1Client::new(env.d1("DB")?);
    let kv = KvClient::new(env.kv("KV")?);
    let discord = DiscordClient::new(
        env.var("DISCORD_BOT_TOKEN")?.to_string(),
        env.var("DISCORD_CHANNEL_ID")?.to_string(),
    );

    // Check circuit breaker
    let cb_state = kv.get_circuit_breaker().await?;
    let today = Utc::now().format("%Y-%m-%d").to_string();
    let triggered_today = cb_state.triggered_at.as_deref()
        .map(|t| date_from_iso(t) == today)
        .unwrap_or(false);

    if should_skip_due_to_circuit_breaker(cb_state.halted, triggered_today) {
        console_log!("Circuit breaker active today: {}", cb_state.reason.as_deref().unwrap_or("unknown"));
        discord.send_error(
            "Morning Scan Skipped",
            &format!("Circuit breaker active: {}", cb_state.reason.as_deref().unwrap_or("unknown")),
        ).await?;
        return Ok(());
    }

    // Auto-reset circuit breaker if it was tripped on a prior day
    if cb_state.halted && !triggered_today {
        console_log!("Auto-resetting circuit breaker (previous day trigger)");
        kv.reset_circuit_breaker().await?;
    }

    let alpaca = AlpacaClient::new(
        env.var("ALPACA_API_KEY")?.to_string(),
        env.var("ALPACA_SECRET_KEY")?.to_string(),
        env.var("ENVIRONMENT")?.to_string() == "paper",
    );

    if !force && !alpaca.is_market_open().await? {
        console_log!("Market is closed, skipping scan");
        return Ok(());
    }

    let account = alpaca.get_account().await?;
    let vix_data = alpaca.get_vix().await?;
    let current_vix = vix_data.as_ref().map(|v| v.vix).unwrap_or(20.0);

    let daily_stats = kv.get_daily_stats().await?;
    let starting_daily = if daily_stats.starting_equity > 0.0 {
        daily_stats.starting_equity
    } else {
        account.equity
    };

    let cb = CircuitBreaker::default();
    let risk_state = cb.evaluate(&EvaluateParams {
        starting_daily_equity: starting_daily,
        current_equity: account.equity,
        starting_weekly_equity: starting_daily,
        peak_equity: account.equity.max(starting_daily),
        current_vix,
    });

    if risk_state.level == RiskLevel::Halted {
        let reason = risk_state.reason.as_deref().unwrap_or("risk threshold exceeded");
        console_log!("Risk evaluation halted: {}", reason);
        let new_state = crate::db::kv::CircuitBreakerState {
            halted: true,
            reason: Some(reason.to_string()),
            triggered_at: Some(Utc::now().to_rfc3339()),
        };
        kv.set_circuit_breaker(&new_state).await?;
        discord.send_error("Trading Halted", reason).await?;
        return Ok(());
    }

    let screener = OptionsScreener::default();
    let sizer = PositionSizer::default();
    let cfg = SpreadConfig::default();

    let open_trades = db.get_open_trades().await?;
    let existing_positions: Vec<ExistingPosition> = open_trades
        .iter()
        .map(|t| ExistingPosition {
            underlying: t.underlying.clone(),
            total_max_loss: t.max_loss * t.contracts as f64,
        })
        .collect();

    let mut all_opportunities = Vec::new();

    for symbol in cfg.underlyings {
        console_log!("Scanning {}...", symbol);

        let chain = match alpaca.get_options_chain(symbol).await {
            Ok(c) => c,
            Err(e) => {
                console_log!("Failed to fetch chain for {}: {:?}", symbol, e);
                continue;
            }
        };

        if chain.contracts.is_empty() {
            console_log!("{}: empty options chain, skipping", symbol);
            continue;
        }
        console_log!(
            "{}: chain has {} contracts, underlying price {:.2}",
            symbol, chain.contracts.len(), chain.underlying_price
        );

        let history = match db.get_iv_history(symbol, 252).await {
            Ok(h) => h,
            Err(e) => {
                console_log!("Failed to fetch IV history for {}: {:?}", symbol, e);
                continue;
            }
        };

        let atm_iv = chain.contracts.iter()
            .filter(|c| (c.strike - chain.underlying_price).abs() < chain.underlying_price * 0.02)
            .filter_map(|c| c.implied_volatility)
            .fold((0.0f64, 0usize), |(sum, n), iv| (sum + iv, n + 1));
        let current_iv = if atm_iv.1 > 0 { atm_iv.0 / atm_iv.1 as f64 } else { 0.20 };
        let iv_metrics = calculate_iv_metrics(current_iv, &history);

        if !passes_iv_filter(iv_metrics.iv_percentile, cfg.min_iv_percentile) {
            console_log!(
                "Skipping {}: IV percentile {:.0} below minimum {:.0}",
                symbol,
                iv_metrics.iv_percentile,
                cfg.min_iv_percentile
            );
            continue;
        }

        let mut spreads = screener.screen_chain(&chain, &iv_metrics);
        let found = spreads.len().min(2);
        console_log!(
            "{}: IV percentile {:.0}, current IV {:.3} — {} spread(s) found",
            symbol, iv_metrics.iv_percentile, iv_metrics.current_iv, found
        );
        for scored in spreads.drain(..).take(2) {
            all_opportunities.push((scored, symbol.to_string(), iv_metrics.clone()));
        }
    }

    all_opportunities.sort_by(|(a, _, _), (b, _, _)| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });

    let opportunities_found = all_opportunities.len();
    let mut trades_placed = 0usize;

    for (scored, _symbol, iv_metrics) in all_opportunities.into_iter().take(cfg.max_trades_per_scan) {
        let spread = &scored.spread;

        let option_type = match spread.spread_type {
            SpreadType::BullPut => "P",
            SpreadType::BearCall => "C",
        };

        let size_result = sizer.calculate_for_underlying(
            &spread.underlying,
            spread.max_loss_per_contract(),
            account.equity,
            &existing_positions,
            current_vix,
        );

        let final_contracts = (size_result.contracts as f64 * risk_state.size_multiplier) as i64;

        if final_contracts == 0 {
            console_log!(
                "Skipping {}: size = 0 ({})",
                spread.underlying,
                size_result.reason.as_deref().unwrap_or("risk multiplier")
            );
            continue;
        }

        let short_occ = build_occ_symbol(&spread.underlying, &spread.expiration, option_type, spread.short_strike);
        let long_occ = build_occ_symbol(&spread.underlying, &spread.expiration, option_type, spread.long_strike);

        let order = SpreadOrder {
            underlying: spread.underlying.clone(),
            short_occ_symbol: short_occ,
            long_occ_symbol: long_occ,
            contracts: final_contracts,
            limit_price: spread.entry_credit,
        };

        let placed = match alpaca.place_spread_order(&order).await {
            Ok(o) => o,
            Err(e) => {
                console_log!("Order placement failed for {}: {:?}", spread.underlying, e);
                discord.send_error(
                    &format!("Order Failed: {}", spread.underlying),
                    &format!("{:?}", e),
                ).await.ok();
                continue;
            }
        };

        let trade_id = uuid_v4();
        match db.create_trade(
            &trade_id,
            &spread.underlying,
            &spread.spread_type,
            spread.short_strike,
            spread.long_strike,
            &spread.expiration,
            final_contracts,
            spread.entry_credit,
            spread.max_loss_per_contract(),
            Some(&placed.id),
            Some(iv_metrics.iv_rank),
            spread.short_delta,
            spread.short_theta,
            None, None, None, None, None, None,  // TEMP: Task 18 wires real values
        ).await {
            Ok(_) => {}
            Err(e) => {
                // Ghost trade prevention: DB write failed after order placed — cancel immediately
                console_log!("CRITICAL: DB write failed for order {}. Cancelling.", placed.id);
                alpaca.cancel_order(&placed.id).await.ok();
                discord.send_error(
                    "GHOST TRADE ALERT",
                    &format!("Order {} placed but DB write failed: {:?}. Order cancelled.", placed.id, e),
                ).await.ok();
                continue;
            }
        }

        discord.send_trade_placed(
            &spread.underlying,
            spread.short_strike,
            spread.long_strike,
            final_contracts,
            spread.entry_credit,
            &placed.id,
        ).await.ok();

        trades_placed += 1;
    }

    let duration_ms = (Utc::now() - start).num_milliseconds();
    let vix_value = vix_data.as_ref().map(|v| v.vix);
    if let Err(e) = db.log_scan(
        "morning",
        cfg.underlyings.len() as i64,
        opportunities_found as i64,
        trades_placed as i64,
        vix_value,
        false,
        duration_ms,
        None,
    ).await {
        console_log!("WARN: scan_log write failed: {:?}", e);
        discord.send_error("scan_log write failed", &format!("{:?}", e)).await.ok();
    }

    discord.send_scan_summary(
        "morning",
        cfg.underlyings.len() as i64,
        opportunities_found as i64,
        trades_placed as i64,
        vix_value,
    ).await.ok();

    console_log!("Morning scan complete. Placed {} trade(s) in {}ms", trades_placed, duration_ms);
    Ok(())
}

/// UUID v4 using crypto.getRandomValues() for collision safety under concurrent
/// Workers invocations. Math.random() is a PRNG and can produce duplicate IDs
/// if multiple trades are placed within the same millisecond.
fn uuid_v4() -> String {
    use worker::js_sys::{global, Reflect, Uint8Array, Function};
    use worker::wasm_bindgen::JsValue;
    use worker::wasm_bindgen::JsCast;

    let buf = Uint8Array::new_with_length(16);
    let crypto = Reflect::get(&global(), &JsValue::from_str("crypto"))
        .expect("crypto not available");
    let get_random = Reflect::get(&crypto, &JsValue::from_str("getRandomValues"))
        .expect("crypto.getRandomValues not available");
    get_random
        .dyn_ref::<Function>()
        .expect("getRandomValues is not a function")
        .call1(&crypto, &buf)
        .expect("getRandomValues failed");

    let bytes = buf.to_vec();
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-4{:01x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6] & 0x0f, bytes[7],
        (bytes[8] & 0x3f) | 0x80, bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_when_circuit_breaker_is_halted_today() {
        assert!(should_skip_due_to_circuit_breaker(true, true));
    }

    #[test]
    fn does_not_skip_when_circuit_breaker_halted_on_previous_day() {
        assert!(!should_skip_due_to_circuit_breaker(true, false));
    }

    #[test]
    fn does_not_skip_when_circuit_breaker_not_halted() {
        assert!(!should_skip_due_to_circuit_breaker(false, false));
    }

    #[test]
    fn iv_filter_passes_at_minimum_threshold() {
        assert!(passes_iv_filter(50.0, 50.0));
    }

    #[test]
    fn iv_filter_passes_above_minimum_threshold() {
        assert!(passes_iv_filter(75.0, 50.0));
    }

    #[test]
    fn iv_filter_blocks_below_minimum_threshold() {
        assert!(!passes_iv_filter(49.9, 50.0));
    }

    #[test]
    fn iv_filter_blocks_low_iv_environment() {
        assert!(!passes_iv_filter(20.0, 50.0));
    }
}
