use worker::{console_log, Env, Result};

use crate::broker::alpaca::AlpacaClient;
use crate::db::d1::D1Client;
use crate::db::kv::KvClient;
use crate::notifications::discord::DiscordClient;

#[derive(Debug, Clone)]
pub struct ClosedTradeSummary {
    pub net_pnl: f64,
}

/// Calculate win rate from closed trades.
/// Returns 1.0 (100%) when there are no closed trades to avoid NaN.
pub fn calculate_win_rate(trades: &[ClosedTradeSummary]) -> f64 {
    if trades.is_empty() {
        return 1.0;
    }
    let wins = trades.iter().filter(|t| t.net_pnl >= 0.0).count();
    wins as f64 / trades.len() as f64
}

pub async fn run(env: &Env) -> Result<()> {
    use chrono::Utc;
    console_log!("EOD summary starting...");

    let db = D1Client::new(env.d1("DB")?);
    let kv = KvClient::new(env.kv("KV")?);
    let discord = DiscordClient::new(
        env.var("DISCORD_BOT_TOKEN")?.to_string(),
        env.var("DISCORD_CHANNEL_ID")?.to_string(),
    );

    let today = Utc::now().format("%Y-%m-%d").to_string();

    let trades_today = db.get_trades_closed_on(&today).await?;

    let closed_summaries: Vec<ClosedTradeSummary> = trades_today
        .iter()
        .filter_map(|t| t.net_pnl.map(|pnl| ClosedTradeSummary { net_pnl: pnl }))
        .collect();

    let realized_pnl: f64 = closed_summaries.iter().map(|t| t.net_pnl).sum();
    let win_rate = calculate_win_rate(&closed_summaries);

    let alpaca = AlpacaClient::new(
        env.var("ALPACA_API_KEY")?.to_string(),
        env.var("ALPACA_SECRET_KEY")?.to_string(),
        env.var("ENVIRONMENT")?.to_string() == "paper",
    );

    for symbol in &["SPY", "QQQ", "IWM"] {
        match alpaca.get_options_chain(symbol).await {
            Ok(chain) => {
                let atm_iv = chain.contracts.iter()
                    .filter(|c| (c.strike - chain.underlying_price).abs() < chain.underlying_price * 0.02)
                    .filter_map(|c| c.implied_volatility)
                    .fold((0.0f64, 0usize), |(sum, n), iv| (sum + iv, n + 1));

                if atm_iv.1 > 0 {
                    let iv = atm_iv.0 / atm_iv.1 as f64;
                    db.upsert_iv_history(symbol, &today, iv).await.ok();
                    console_log!("{}: IV snapshot saved ({:.2}%)", symbol, iv * 100.0);
                }
            }
            Err(e) => console_log!("Could not fetch chain for {} IV snapshot: {:?}", symbol, e),
        }
    }

    let mut daily_stats = kv.get_daily_stats().await?;
    daily_stats.realized_pnl = realized_pnl;
    daily_stats.trades_closed = closed_summaries.len() as i64;
    kv.set_daily_stats(&daily_stats).await.ok();

    discord.send_eod_summary(&today, realized_pnl, closed_summaries.len() as i64, win_rate).await.ok();

    console_log!(
        "EOD summary complete: {} closed, P&L ${:.0}, win rate {:.0}%",
        closed_summaries.len(),
        realized_pnl,
        win_rate * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn win_rate_is_1_for_no_trades() {
        let rate = calculate_win_rate(&[]);
        assert!((rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn win_rate_is_50_pct_for_one_win_one_loss() {
        let trades = vec![
            ClosedTradeSummary { net_pnl: 50.0 },
            ClosedTradeSummary { net_pnl: -30.0 },
        ];
        let rate = calculate_win_rate(&trades);
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn win_rate_is_1_for_all_winners() {
        let trades = vec![
            ClosedTradeSummary { net_pnl: 50.0 },
            ClosedTradeSummary { net_pnl: 25.0 },
        ];
        assert!((calculate_win_rate(&trades) - 1.0).abs() < 1e-9);
    }
}
