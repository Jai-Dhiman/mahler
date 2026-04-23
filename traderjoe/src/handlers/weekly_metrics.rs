use worker::{console_log, Env, Result};

use crate::db::d1::{D1Client, MetricsWeeklyRow};
use crate::measurement::metrics::{compute_metrics, MetricsInput, SampleSizeTag};
use crate::notifications::discord::DiscordClient;

pub async fn run(env: &Env) -> Result<()> {
    use chrono::{Duration, Utc};
    console_log!("Weekly metrics starting...");

    let db = D1Client::new(env.d1("DB")?);
    let discord = DiscordClient::new(
        env.var("DISCORD_BOT_TOKEN")?.to_string(),
        env.var("DISCORD_CHANNEL_ID")?.to_string(),
    );

    let end = Utc::now().date_naive();
    let start = end - Duration::days(30);
    let start_s = start.format("%Y-%m-%d").to_string();
    let end_s = end.format("%Y-%m-%d").to_string();

    let trades = db.get_closed_trades_in_window(&start_s, &end_s).await?;
    let equity = db.get_equity_history_in_window(&start_s, &end_s).await?;
    let greeks = db.get_portfolio_greeks_history_in_window(&start_s, &end_s).await?;
    let ctx = db.get_market_context_in_window(&start_s, &end_s).await?;

    let bundle = compute_metrics(MetricsInput {
        window_start: start_s.clone(),
        window_end: end_s.clone(),
        trades,
        equity_eod: equity,
        greeks_eod: greeks,
        market_context: ctx,
    });

    let tag_str = match bundle.sample_size_tag {
        SampleSizeTag::Ok => "OK",
        SampleSizeTag::Weak => "WEAK",
        SampleSizeTag::Insufficient => "INSUFFICIENT",
    };

    let regime_json = serde_json::to_string(&bundle.regime_buckets)
        .unwrap_or_else(|_| "{}".to_string());
    let greek_json = serde_json::to_string(&bundle.greek_ranges)
        .unwrap_or_else(|_| "{}".to_string());

    db.insert_metrics_weekly(MetricsWeeklyRow {
        generated_at: Utc::now().to_rfc3339(),
        window_start: start_s,
        window_end: end_s,
        trade_count: bundle.trade_count as i64,
        sharpe: Some(bundle.sharpe),
        sortino: Some(bundle.sortino),
        profit_factor: Some(bundle.profit_factor),
        win_rate: Some(bundle.win_rate),
        pnl_skew: Some(bundle.pnl_skew),
        max_drawdown_pct: Some(bundle.max_drawdown_pct),
        mean_slippage_vs_mid: Some(bundle.mean_slippage_vs_mid),
        max_slippage_vs_mid: Some(bundle.max_slippage_vs_mid),
        slippage_vs_orats_ratio: Some(bundle.slippage_vs_orats_ratio),
        fill_size_violation_count: bundle.fill_size_violation_count as i64,
        fill_size_violation_pct: Some(bundle.fill_size_violation_pct),
        regime_buckets_json: regime_json,
        greek_ranges_json: greek_json,
        sample_size_tag: tag_str.to_string(),
    })
    .await?;

    discord.send_metrics_report(&bundle).await.ok();
    console_log!("Weekly metrics complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_signature_compiles_wasm_constrained() {
        // Never executed: WASM prevents async handler calls in tests.
        fn _check_sig<'a>(env: &'a worker::Env) -> impl std::future::Future<Output = worker::Result<()>> + 'a {
            run(env)
        }
    }
}
