use worker::{console_log, Env, Result};

use crate::broker::alpaca::{build_occ_symbol, AlpacaClient};
use crate::broker::types::SpreadOrder;
use crate::config::SpreadConfig;
use crate::db::d1::D1Client;
use crate::db::kv::KvClient;
use crate::notifications::discord::DiscordClient;
use crate::types::ExitReason;

/// Determines if an open position should be closed, and why.
///
/// Checks in priority order: gamma explosion (DTE) → profit target → stop loss.
pub fn check_exit_conditions(
    entry_credit: f64,
    current_debit: f64,
    dte: i64,
    cfg: &SpreadConfig,
) -> Option<ExitReason> {
    if dte <= cfg.gamma_exit_dte {
        return Some(ExitReason::Expiration);
    }
    if cfg.should_exit_profit(entry_credit, current_debit) {
        return Some(ExitReason::ProfitTarget);
    }
    if cfg.should_exit_stop_loss(entry_credit, current_debit) {
        return Some(ExitReason::StopLoss);
    }
    None
}

pub async fn run(env: &Env) -> Result<()> {
    use crate::analysis::greeks::days_to_expiry;

    console_log!("Position monitor starting...");

    let db = D1Client::new(env.d1("DB")?);
    let kv = KvClient::new(env.kv("KV")?);
    let discord = DiscordClient::new(
        env.var("DISCORD_BOT_TOKEN")?.to_string(),
        env.var("DISCORD_CHANNEL_ID")?.to_string(),
    );

    let cb_state = kv.get_circuit_breaker().await?;
    if cb_state.halted {
        return Ok(());
    }

    let alpaca = AlpacaClient::new(
        env.var("ALPACA_API_KEY")?.to_string(),
        env.var("ALPACA_SECRET_KEY")?.to_string(),
        env.var("ENVIRONMENT")?.to_string() == "paper",
    );

    if !alpaca.is_market_open().await? {
        return Ok(());
    }

    let open_trades = db.get_open_trades().await?;
    if open_trades.is_empty() {
        return Ok(());
    }

    let cfg = SpreadConfig::default();

    for trade in &open_trades {
        if matches!(trade.status, crate::types::TradeStatus::PendingFill) {
            if let Some(ref order_id) = trade.broker_order_id {
                match alpaca.get_order(order_id).await {
                    Ok(order) => {
                        if matches!(order.status, crate::broker::types::OrderStatus::Filled) {
                            let fill_price = order.filled_price.unwrap_or(trade.entry_credit);
                            db.mark_trade_filled(&trade.id, fill_price).await.ok();
                            console_log!("Trade {} filled at {:.2}", trade.id, fill_price);
                        }
                    }
                    Err(e) => console_log!("Could not check order {}: {:?}", order_id, e),
                }
            }
            continue;
        }

        let dte = days_to_expiry(&trade.expiration).unwrap_or(0);
        let current_debit = match get_spread_debit(&alpaca, trade).await {
            Some(d) => d,
            None => {
                console_log!("Could not get current price for trade {}", trade.id);
                continue;
            }
        };

        if let Some(exit_reason) = check_exit_conditions(trade.entry_credit, current_debit, dte, &cfg) {
            console_log!(
                "Exiting {} {}/{} via {:?} (entry: {:.2}, current: {:.2}, DTE: {})",
                trade.underlying,
                trade.short_strike,
                trade.long_strike,
                exit_reason,
                trade.entry_credit,
                current_debit,
                dte,
            );

            let option_type_str = match trade.spread_type {
                crate::types::SpreadType::BullPut => "P",
                crate::types::SpreadType::BearCall => "C",
            };
            let short_occ = build_occ_symbol(&trade.underlying, &trade.expiration, option_type_str, trade.short_strike);
            let long_occ = build_occ_symbol(&trade.underlying, &trade.expiration, option_type_str, trade.long_strike);

            let close_order = SpreadOrder {
                underlying: trade.underlying.clone(),
                short_occ_symbol: short_occ,
                long_occ_symbol: long_occ,
                contracts: trade.contracts,
                limit_price: current_debit,
            };

            let placed = match alpaca.place_closing_spread_order(&close_order).await {
                Ok(o) => o,
                Err(e) => {
                    console_log!("Failed to place closing order for trade {}: {:?}", trade.id, e);
                    discord.send_error(
                        &format!("Exit Order Failed: {}", trade.underlying),
                        &format!("{:?}", e),
                    ).await.ok();
                    continue;
                }
            };

            let net_pnl = (trade.entry_credit - current_debit) * 100.0 * trade.contracts as f64;

            match db.close_trade(&trade.id, current_debit, &exit_reason, net_pnl, None).await {
                Err(e) => {
                    console_log!(
                        "CRITICAL: DB close_trade failed for trade {} after close order {}.",
                        trade.id, placed.id
                    );
                    discord.send_error(
                        "DB WRITE FAILED AFTER CLOSE ORDER",
                        &format!("Trade {} / Order {}: {:?}. Position may be double-closed.", trade.id, placed.id, e),
                    ).await.ok();
                    continue;
                }
                Ok(false) => {
                    // Another concurrent invocation already closed this trade — skip notification.
                    console_log!("Trade {} already closed by concurrent invocation, skipping.", trade.id);
                    continue;
                }
                Ok(true) => {}
            }

            discord.send_position_exit(
                &trade.underlying,
                trade.short_strike,
                trade.long_strike,
                exit_reason.as_str(),
                net_pnl,
            ).await.ok();
        }
    }

    Ok(())
}

async fn get_spread_debit(alpaca: &AlpacaClient, trade: &crate::types::Trade) -> Option<f64> {
    let chain = alpaca.get_options_chain(&trade.underlying).await.ok()?;

    let option_type = match trade.spread_type {
        crate::types::SpreadType::BullPut => crate::broker::types::OptionType::Put,
        crate::types::SpreadType::BearCall => crate::broker::types::OptionType::Call,
    };

    let short = chain.contracts.iter().find(|c| {
        (c.strike - trade.short_strike).abs() < 0.01
            && c.expiration == trade.expiration
            && c.option_type == option_type
    })?;

    let long = chain.contracts.iter().find(|c| {
        (c.strike - trade.long_strike).abs() < 0.01
            && c.expiration == trade.expiration
            && c.option_type == option_type
    })?;

    Some(short.ask - long.bid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpreadConfig;

    #[test]
    fn identifies_profit_target_exit() {
        // 75% profit: entry 1.00, debit 0.25 (profit = 0.75 = 75%)
        let cfg = SpreadConfig::default();
        let result = check_exit_conditions(1.00, 0.25, 40, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::ProfitTarget)));
    }

    #[test]
    fn identifies_stop_loss_exit() {
        // 210% loss: entry 1.00, debit 3.10 (loss = 2.10 > 200% threshold)
        let cfg = SpreadConfig::default();
        let result = check_exit_conditions(1.00, 3.10, 40, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::StopLoss)));
    }

    #[test]
    fn identifies_gamma_explosion_exit() {
        let cfg = SpreadConfig::default();
        let result = check_exit_conditions(1.00, 0.90, 5, &cfg);
        assert!(matches!(result, Some(crate::types::ExitReason::Expiration)));
    }

    #[test]
    fn no_exit_when_within_normal_range() {
        let cfg = SpreadConfig::default();
        let result = check_exit_conditions(1.00, 0.90, 30, &cfg);
        assert!(result.is_none());
    }
}
