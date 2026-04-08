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

/// Extract the date portion (YYYY-MM-DD) from an ISO timestamp string.
fn date_from_iso(iso: &str) -> &str {
    &iso[..iso.len().min(10)]
}

pub async fn run(env: &Env) -> Result<()> {
    use chrono::Utc;
    let start = Utc::now();
    console_log!("Morning scan starting at {}", start.to_rfc3339());

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

    if !alpaca.is_market_open().await? {
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
            continue;
        }

        let history = db.get_iv_history(symbol, 252).await.unwrap_or_default();

        let atm_iv = chain.contracts.iter()
            .filter(|c| (c.strike - chain.underlying_price).abs() < chain.underlying_price * 0.02)
            .filter_map(|c| c.implied_volatility)
            .fold((0.0f64, 0usize), |(sum, n), iv| (sum + iv, n + 1));
        let current_iv = if atm_iv.1 > 0 { atm_iv.0 / atm_iv.1 as f64 } else { 0.20 };
        let iv_metrics = calculate_iv_metrics(current_iv, &history);

        let mut spreads = screener.screen_chain(&chain, &iv_metrics);
        for scored in spreads.drain(..).take(2) {
            all_opportunities.push((scored, symbol.to_string(), iv_metrics.clone()));
        }
    }

    all_opportunities.sort_by(|(a, _, _), (b, _, _)| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });

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
    db.log_scan(
        "morning",
        cfg.underlyings.len() as i64,
        0,
        trades_placed as i64,
        vix_value,
        false,
        duration_ms,
        None,
    ).await.ok();

    discord.send_scan_summary(
        "morning",
        cfg.underlyings.len() as i64,
        0,
        trades_placed as i64,
        vix_value,
    ).await.ok();

    console_log!("Morning scan complete. Placed {} trade(s) in {}ms", trades_placed, duration_ms);
    Ok(())
}

/// Minimal UUID v4 for Cloudflare Workers (no rand crate needed for WASM).
fn uuid_v4() -> String {
    use worker::js_sys::Math;
    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (Math::random() * 0xffff_ffff_u64 as f64) as u64,
        (Math::random() * 0xffff_u64 as f64) as u64,
        (Math::random() * 0x0fff_u64 as f64) as u64,
        ((Math::random() * 0x3fff_u64 as f64) as u64) | 0x8000,
        (Math::random() * 0x0000_ffff_ffff_u64 as f64) as u64
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
}
