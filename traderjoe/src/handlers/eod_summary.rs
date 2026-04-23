use std::collections::HashMap;
use worker::{console_log, Env, Result};

use crate::broker::alpaca::AlpacaClient;
use crate::broker::types::OptionsChain;
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

pub fn build_chain_map(chains: Vec<OptionsChain>) -> HashMap<String, OptionsChain> {
    let mut map = HashMap::new();
    for c in chains {
        map.insert(c.underlying.clone(), c);
    }
    map
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

    let mut chains: Vec<OptionsChain> = Vec::new();
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
                chains.push(chain);
            }
            Err(e) => console_log!("Could not fetch chain for {} IV snapshot: {:?}", symbol, e),
        }
    }
    let chain_map = build_chain_map(chains);

    let open_trades = db.get_open_trades().await.unwrap_or_default();
    let pg = crate::measurement::portfolio_greeks::PortfolioGreeksAggregator::compute(
        &open_trades, &chain_map,
    );
    db.insert_portfolio_greeks_eod(&today, &pg).await.ok();

    let vixy = alpaca.get_vix().await.ok().flatten().map(|v| v.vix);

    let spy_bars_52w = alpaca.get_bars("SPY", 260).await.unwrap_or_default();
    let spy_bars_20d: Vec<_> = spy_bars_52w.iter().rev().take(20).rev().cloned().collect();
    let spy_stats = crate::measurement::market_context::compute_spy_stats(&spy_bars_20d, &spy_bars_52w);

    let spot_vix_reading = fetch_fred_spot_vix().await;
    let (spot_vix_val, src, src_date) = match spot_vix_reading {
        Some(r) => (Some(r.value), r.source, Some(r.source_date)),
        None => {
            discord.send_error(
                "VIX Data Unavailable",
                &format!("FRED VIXCLS fetch failed for {}. spot_vix stored as NULL — regime tagging for today will be absent.", today),
            ).await.ok();
            (None, "unavailable", None)
        }
    };
    db.insert_market_context_daily(
        &today, spot_vix_val, src, src_date.as_deref(),
        vixy,
        Some(spy_stats.realized_vol_20d),
        Some(spy_stats.return_20d),
        Some(spy_stats.drawdown_from_52w_high),
    ).await.ok();

    let mut daily_stats = kv.get_daily_stats().await?;
    daily_stats.realized_pnl = realized_pnl;
    daily_stats.trades_closed = closed_summaries.len() as i64;
    kv.set_daily_stats(&daily_stats).await.ok();

    discord.send_eod_summary(&today, realized_pnl, closed_summaries.len() as i64, win_rate).await.ok();

    let account_final = alpaca.get_account().await.ok();
    let (eq, cash) = account_final.as_ref().map(|a| (a.equity, a.cash)).unwrap_or((0.0, 0.0));
    let snap = crate::measurement::equity::build_equity_snapshot(
        crate::measurement::equity::EquityEvent::Eod,
        crate::measurement::equity::SnapshotInputs {
            timestamp: chrono::Utc::now().to_rfc3339(),
            equity: eq, cash,
            open_position_mtm: 0.0,
            realized_pnl_day: realized_pnl,
            unrealized_pnl_day: 0.0,
            open_position_count: open_trades.len() as i64,
            trade_id_ref: None,
        },
    );
    db.insert_equity_snapshot(&snap).await.ok();

    console_log!(
        "EOD summary complete: {} closed, P&L ${:.0}, win rate {:.0}%",
        closed_summaries.len(),
        realized_pnl,
        win_rate * 100.0
    );
    Ok(())
}

async fn fetch_fred_spot_vix() -> Option<crate::measurement::market_context::SpotVixReading> {
    use worker::{Fetch, Method, Request, RequestInit};
    let url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS";
    let mut init = RequestInit::new();
    init.with_method(Method::Get);
    let req = Request::new_with_init(url, &init).ok()?;
    let mut resp = Fetch::Request(req).send().await.ok()?;
    let body = resp.text().await.ok()?;
    crate::measurement::market_context::parse_fred_vix_csv(&body)
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

    #[test]
    fn collect_chains_for_greeks_returns_map_keyed_by_underlying() {
        use crate::broker::types::OptionsChain;
        let chain_spy = OptionsChain {
            underlying: "SPY".to_string(), underlying_price: 480.0,
            expirations: vec![], contracts: vec![],
        };
        let chain_qqq = OptionsChain {
            underlying: "QQQ".to_string(), underlying_price: 500.0,
            expirations: vec![], contracts: vec![],
        };
        let map = build_chain_map(vec![chain_spy, chain_qqq]);
        assert!(map.contains_key("SPY"));
        assert!(map.contains_key("QQQ"));
    }
}
