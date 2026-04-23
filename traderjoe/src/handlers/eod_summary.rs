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

/// Compute total unrealized P&L of open positions using current chain prices.
/// current_debit = short_ask - long_bid (cost to close). MTM = entry_credit - current_debit.
pub fn compute_open_mtm(trades: &[crate::types::Trade], chain_map: &HashMap<String, OptionsChain>) -> f64 {
    trades.iter().map(|t| {
        let chain = match chain_map.get(&t.underlying) {
            Some(c) => c,
            None => return 0.0,
        };
        let option_type = match t.spread_type {
            crate::types::SpreadType::BullPut => crate::broker::types::OptionType::Put,
            crate::types::SpreadType::BearCall => crate::broker::types::OptionType::Call,
        };
        let short = chain.contracts.iter().find(|c|
            (c.strike - t.short_strike).abs() < 0.01
            && c.option_type == option_type
            && c.expiration == t.expiration
        );
        let long = chain.contracts.iter().find(|c|
            (c.strike - t.long_strike).abs() < 0.01
            && c.option_type == option_type
            && c.expiration == t.expiration
        );
        match (short, long) {
            (Some(s), Some(l)) => (t.entry_credit - (s.ask - l.bid)) * 100.0 * t.contracts as f64,
            _ => 0.0,
        }
    }).sum()
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

    let account_final = alpaca.get_account().await.ok();
    let (eq, cash) = account_final.as_ref().map(|a| (a.equity, a.cash)).unwrap_or((0.0, 0.0));
    let open_mtm = compute_open_mtm(&open_trades, &chain_map);
    let spy_price = chain_map.get("SPY").map(|c| c.underlying_price);
    let max_loss_remaining: f64 = open_trades.iter().map(|t| t.max_loss * t.contracts as f64).sum();
    let dte_buckets = bucket_dtes(&open_trades, &today);

    discord.send_eod_summary(&crate::notifications::discord::EodEmbedInput {
        date: &today,
        realized_pnl,
        trades_closed: closed_summaries.len() as i64,
        win_rate,
        open_count: open_trades.len() as i64,
        open_mtm,
        max_loss_remaining,
        greeks: Some(&pg),
        spy_price,
        dte_buckets,
        spot_vix: spot_vix_val,
        vix_source: src,
        vixy,
        spy_stats: Some(&spy_stats),
        equity: account_final.as_ref().map(|_| eq),
        cash: account_final.as_ref().map(|_| cash),
    }).await.ok();

    let snap = crate::measurement::equity::build_equity_snapshot(
        crate::measurement::equity::EquityEvent::Eod,
        crate::measurement::equity::SnapshotInputs {
            timestamp: chrono::Utc::now().to_rfc3339(),
            equity: eq, cash,
            open_position_mtm: open_mtm,
            realized_pnl_day: realized_pnl,
            unrealized_pnl_day: open_mtm,
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

fn bucket_dtes(trades: &[crate::types::Trade], today: &str) -> (i64, i64, i64) {
    use chrono::NaiveDate;
    let today_d = NaiveDate::parse_from_str(today, "%Y-%m-%d").ok();
    let mut b = (0i64, 0i64, 0i64);
    for t in trades {
        let exp = NaiveDate::parse_from_str(&t.expiration, "%Y-%m-%d").ok();
        let dte = match (today_d, exp) {
            (Some(a), Some(e)) => (e - a).num_days(),
            _ => continue,
        };
        if dte <= 7 { b.0 += 1; }
        else if dte <= 21 { b.1 += 1; }
        else { b.2 += 1; }
    }
    b
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
    fn compute_open_mtm_sums_unrealized_pnl_from_chain() {
        use crate::broker::types::{OptionContract, OptionType, OptionsChain};
        use crate::types::{SpreadType, Trade, TradeStatus};

        let trade = Trade {
            id: "t1".to_string(),
            created_at: "2026-04-22T10:00:00Z".to_string(),
            underlying: "SPY".to_string(),
            spread_type: SpreadType::BullPut,
            short_strike: 460.0, long_strike: 455.0,
            expiration: "2026-05-15".to_string(),
            contracts: 2,
            entry_credit: 0.75, max_loss: 425.0,
            broker_order_id: None, status: TradeStatus::Open,
            fill_price: None, fill_time: None,
            exit_price: None, exit_time: None, exit_reason: None, net_pnl: None,
            iv_rank: None, short_delta: None, short_theta: None,
            entry_short_bid: None, entry_short_ask: None,
            entry_long_bid: None, entry_long_ask: None, entry_net_mid: None,
            exit_short_bid: None, exit_short_ask: None,
            exit_long_bid: None, exit_long_ask: None, exit_net_mid: None,
            entry_short_gamma: None, entry_short_vega: None,
            entry_long_delta: None, entry_long_gamma: None, entry_long_vega: None,
            nbbo_displayed_size_short: None, nbbo_displayed_size_long: None,
            nbbo_snapshot_time: None,
        };
        let mk = |strike: f64, bid: f64, ask: f64| OptionContract {
            symbol: format!("SPY{}", strike),
            underlying: "SPY".to_string(),
            expiration: "2026-05-15".to_string(),
            strike, option_type: OptionType::Put,
            bid, ask, last: (bid + ask) / 2.0,
            volume: 0, open_interest: 0,
            implied_volatility: None,
            delta: None, gamma: None, theta: None, vega: None,
            bid_size: None, ask_size: None,
        };
        // short_ask=0.50, long_bid=0.10 → current_debit=0.40 → mtm=(0.75-0.40)*100*2=70
        let chain = OptionsChain {
            underlying: "SPY".to_string(), underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string()],
            contracts: vec![mk(460.0, 0.49, 0.50), mk(455.0, 0.10, 0.12)],
        };
        let mut map = HashMap::new();
        map.insert("SPY".to_string(), chain);
        let mtm = compute_open_mtm(&[trade], &map);
        assert!((mtm - 70.0).abs() < 1e-9, "expected 70.0, got {}", mtm);
    }

    #[test]
    fn bucket_dtes_groups_trades_by_expiry_window() {
        use crate::types::{SpreadType, Trade, TradeStatus};
        let mk = |exp: &str| Trade {
            id: exp.to_string(), created_at: "2026-04-23T00:00:00Z".to_string(),
            underlying: "SPY".to_string(), spread_type: SpreadType::BullPut,
            short_strike: 460.0, long_strike: 455.0,
            expiration: exp.to_string(),
            contracts: 1, entry_credit: 0.5, max_loss: 450.0,
            broker_order_id: None, status: TradeStatus::Open,
            fill_price: None, fill_time: None,
            exit_price: None, exit_time: None, exit_reason: None, net_pnl: None,
            iv_rank: None, short_delta: None, short_theta: None,
            entry_short_bid: None, entry_short_ask: None,
            entry_long_bid: None, entry_long_ask: None, entry_net_mid: None,
            exit_short_bid: None, exit_short_ask: None,
            exit_long_bid: None, exit_long_ask: None, exit_net_mid: None,
            entry_short_gamma: None, entry_short_vega: None,
            entry_long_delta: None, entry_long_gamma: None, entry_long_vega: None,
            nbbo_displayed_size_short: None, nbbo_displayed_size_long: None,
            nbbo_snapshot_time: None,
        };
        // Today 2026-04-23; expiries: +5d, +14d, +30d
        let trades = vec![mk("2026-04-28"), mk("2026-05-07"), mk("2026-05-23")];
        let (short, mid, long) = bucket_dtes(&trades, "2026-04-23");
        assert_eq!((short, mid, long), (1, 1, 1));
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
