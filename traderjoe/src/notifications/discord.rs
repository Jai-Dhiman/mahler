use serde_json::{json, Value};
use worker::{Fetch, Headers, Method, Request, RequestInit, Result};

use crate::measurement::market_context::SpyStats;
use crate::measurement::portfolio_greeks::PortfolioGreeks;

const COLOR_GREEN: u32 = 0x57F287;
const COLOR_RED: u32 = 0xED4245;
const COLOR_BLUE: u32 = 0x5865F2;

/// Build a Discord embed for a placed trade.
pub fn build_trade_embed(
    underlying: &str,
    short_strike: f64,
    long_strike: f64,
    contracts: i64,
    credit: f64,
    order_id: &str,
) -> Value {
    json!({
        "title": format!("Trade Placed: {} Bull Put Spread", underlying),
        "color": COLOR_GREEN,
        "description": format!(
            "Placed {} contract(s) of ${}/{} spread",
            contracts, short_strike, long_strike
        ),
        "fields": [
            { "name": "Credit", "value": format!("${:.2}/share", credit), "inline": true },
            { "name": "Max Loss", "value": format!("${:.0}/contract", (short_strike - long_strike - credit) * 100.0), "inline": true },
            { "name": "Order ID", "value": format!("`{}`", order_id), "inline": false }
        ]
    })
}

/// Build a Discord embed for a position exit.
pub fn build_exit_embed(
    underlying: &str,
    short_strike: f64,
    long_strike: f64,
    exit_reason: &str,
    net_pnl: f64,
) -> Value {
    let color = if net_pnl >= 0.0 { COLOR_GREEN } else { COLOR_RED };
    json!({
        "title": format!("Position Closed: {}", underlying),
        "color": color,
        "description": format!(
            "${}/{} spread closed via {}. P&L: ${:.0}",
            short_strike, long_strike, exit_reason, net_pnl
        )
    })
}

/// Build a Discord embed for scan summary.
pub fn build_scan_summary_embed(
    scan_type: &str,
    underlyings_scanned: i64,
    opportunities_found: i64,
    trades_placed: i64,
    vix: Option<f64>,
) -> Value {
    let vix_str = vix.map(|v| format!("{:.1}", v)).unwrap_or_else(|| "N/A".to_string());
    json!({
        "title": format!("{} Scan Complete", scan_type.replace('_', " ").to_uppercase()),
        "color": COLOR_BLUE,
        "description": format!(
            "Scanned {} underlyings, found {} opportunities, placed {} trade(s). VIX: {}",
            underlyings_scanned, opportunities_found, trades_placed, vix_str
        )
    })
}

/// Build a Discord embed for an error.
pub fn build_error_embed(title: &str, error: &str) -> Value {
    json!({
        "title": title,
        "color": COLOR_RED,
        "description": format!("```\n{}\n```", &error[..error.len().min(1500)])
    })
}

pub struct EodEmbedInput<'a> {
    pub date: &'a str,
    pub realized_pnl: f64,
    pub trades_closed: i64,
    pub win_rate: f64,
    pub open_count: i64,
    pub open_mtm: f64,
    pub max_loss_remaining: f64,
    pub greeks: Option<&'a PortfolioGreeks>,
    pub spy_price: Option<f64>,
    pub dte_buckets: (i64, i64, i64),
    pub spot_vix: Option<f64>,
    pub vix_source: &'a str,
    pub vixy: Option<f64>,
    pub spy_stats: Option<&'a SpyStats>,
    pub equity: Option<f64>,
    pub cash: Option<f64>,
}

pub fn regime_tag(spot_vix: Option<f64>) -> &'static str {
    match spot_vix {
        Some(v) if v < 15.0 => "calm",
        Some(v) if v < 20.0 => "normal",
        Some(v) if v < 30.0 => "elevated",
        Some(_) => "stress",
        None => "unknown",
    }
}

fn fmt_dollar(v: Option<f64>) -> String {
    v.map(|x| format!("${:.0}", x)).unwrap_or_else(|| "—".into())
}

pub fn build_eod_embed(i: &EodEmbedInput) -> Value {
    let color = if i.realized_pnl >= 0.0 { COLOR_GREEN } else { COLOR_RED };

    let (dollar_delta, theta_day, vega) = match (i.greeks, i.spy_price) {
        (Some(g), Some(p)) => (
            format!("${:+.0}", g.beta_weighted_delta * p),
            format!("${:+.0}", g.total_theta),
            format!("${:+.0}", g.total_vega),
        ),
        _ => ("—".into(), "—".into(), "—".into()),
    };

    let (rv, r20, ddn) = match i.spy_stats {
        Some(s) => (
            format!("{:.1}%", s.realized_vol_20d * 100.0),
            format!("{:+.1}%", s.return_20d * 100.0),
            format!("-{:.1}%", s.drawdown_from_52w_high * 100.0),
        ),
        None => ("—".into(), "—".into(), "—".into()),
    };

    let (d7, d21, d_long) = i.dte_buckets;
    let dte_str = format!("{} ≤7d · {} 8-21d · {} >21d", d7, d21, d_long);

    let vix_str = match i.spot_vix {
        Some(v) => format!("{:.2} ({})", v, i.vix_source),
        None => format!("— ({})", i.vix_source),
    };

    json!({
        "title": format!("EOD Summary — {}", i.date),
        "color": color,
        "description": format!("Regime: **{}**", regime_tag(i.spot_vix)),
        "fields": [
            { "name": "Realized P&L",   "value": format!("${:.0}", i.realized_pnl), "inline": true },
            { "name": "Trades Closed",  "value": i.trades_closed.to_string(),       "inline": true },
            { "name": "Win Rate",       "value": format!("{:.0}%", i.win_rate * 100.0), "inline": true },

            { "name": "Open Positions",    "value": format!("{} · MTM ${:+.0}", i.open_count, i.open_mtm), "inline": true },
            { "name": "Max Loss at Risk",  "value": format!("${:.0}", i.max_loss_remaining), "inline": true },
            { "name": "DTE Mix",           "value": dte_str, "inline": true },

            { "name": "$Δ (per $1 SPY)", "value": dollar_delta, "inline": true },
            { "name": "Θ / day",         "value": theta_day,    "inline": true },
            { "name": "Vega / 1% IV",    "value": vega,         "inline": true },

            { "name": "Spot VIX",            "value": vix_str, "inline": true },
            { "name": "VIXY",                "value": i.vixy.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "—".into()), "inline": true },
            { "name": "SPY 20d RV/Ret/DD",   "value": format!("{} · {} · {}", rv, r20, ddn), "inline": true },

            { "name": "Equity", "value": fmt_dollar(i.equity), "inline": true },
            { "name": "Cash",   "value": fmt_dollar(i.cash),   "inline": true },
            { "name": "\u{200b}", "value": "\u{200b}", "inline": true }
        ]
    })
}

/// Build a Discord embed for weekly metrics report.
pub fn build_metrics_embed(b: &crate::measurement::metrics::MetricBundle) -> Value {
    use crate::measurement::metrics::SampleSizeTag;
    let (color, tag_str) = match b.sample_size_tag {
        SampleSizeTag::Ok => (COLOR_BLUE, "OK"),
        SampleSizeTag::Weak => (0xF0B232u32, "WEAK"),
        SampleSizeTag::Insufficient => (COLOR_RED, "INSUFFICIENT"),
    };
    json!({
        "title": format!("Weekly Metrics — {} to {}", b.window_start, b.window_end),
        "color": color,
        "description": format!(
            "Window: {} → {}\nSample tag: **{}** (n={})",
            b.window_start, b.window_end, tag_str, b.trade_count
        ),
        "fields": [
            { "name": "Sharpe", "value": format!("{:.2}", b.sharpe), "inline": true },
            { "name": "Sortino", "value": format!("{:.2}", b.sortino), "inline": true },
            { "name": "Profit Factor", "value": format!("{:.2}", b.profit_factor), "inline": true },
            { "name": "Win Rate", "value": format!("{:.0}%", b.win_rate * 100.0), "inline": true },
            { "name": "Max DD", "value": format!("{:.1}%", b.max_drawdown_pct * 100.0), "inline": true },
            { "name": "P&L Skew", "value": format!("{:.2}", b.pnl_skew), "inline": true },
            { "name": "Slippage Ratio vs ORATS", "value": format!("{:.2}", b.slippage_vs_orats_ratio), "inline": true },
            { "name": "Fill Size Violations", "value": format!("{} ({:.1}%)", b.fill_size_violation_count, b.fill_size_violation_pct * 100.0), "inline": true },
        ]
    })
}

/// Client for sending Discord notifications via bot API.
pub struct DiscordClient {
    bot_token: String,
    channel_id: String,
}

impl DiscordClient {
    pub fn new(bot_token: impl Into<String>, channel_id: impl Into<String>) -> Self {
        DiscordClient {
            bot_token: bot_token.into(),
            channel_id: channel_id.into(),
        }
    }

    pub async fn send(&self, content: &str, embeds: Vec<Value>) -> Result<()> {
        let url = format!(
            "https://discord.com/api/v10/channels/{}/messages",
            self.channel_id
        );
        let body = serde_json::to_string(&json!({
            "content": content,
            "embeds": embeds
        }))
        .map_err(|e| worker::Error::RustError(e.to_string()))?;

        let mut headers = Headers::new();
        headers.set("Content-Type", "application/json")?;
        headers.set("Authorization", &format!("Bot {}", self.bot_token))?;

        let mut init = RequestInit::new();
        init.with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(body.into()));

        let req = Request::new_with_init(&url, &init)?;
        Fetch::Request(req).send().await?;
        Ok(())
    }

    pub async fn send_trade_placed(
        &self,
        underlying: &str,
        short_strike: f64,
        long_strike: f64,
        contracts: i64,
        credit: f64,
        order_id: &str,
    ) -> Result<()> {
        let embed = build_trade_embed(underlying, short_strike, long_strike, contracts, credit, order_id);
        self.send("**Trade Placed**", vec![embed]).await
    }

    pub async fn send_scan_summary(
        &self,
        scan_type: &str,
        underlyings_scanned: i64,
        opportunities_found: i64,
        trades_placed: i64,
        vix: Option<f64>,
    ) -> Result<()> {
        let embed = build_scan_summary_embed(scan_type, underlyings_scanned, opportunities_found, trades_placed, vix);
        self.send("", vec![embed]).await
    }

    pub async fn send_error(&self, title: &str, error: &str) -> Result<()> {
        let embed = build_error_embed(title, error);
        self.send("**Error**", vec![embed]).await
    }

    pub async fn send_eod_summary(&self, input: &EodEmbedInput<'_>) -> Result<()> {
        let embed = build_eod_embed(input);
        self.send("", vec![embed]).await
    }

    pub async fn send_position_exit(
        &self,
        underlying: &str,
        short_strike: f64,
        long_strike: f64,
        exit_reason: &str,
        net_pnl: f64,
    ) -> Result<()> {
        let embed = build_exit_embed(underlying, short_strike, long_strike, exit_reason, net_pnl);
        self.send("", vec![embed]).await
    }

    pub async fn send_metrics_report(&self, b: &crate::measurement::metrics::MetricBundle) -> Result<()> {
        let embed = build_metrics_embed(b);
        self.send("**Weekly Metrics**", vec![embed]).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trade_embed_has_green_color_and_correct_title() {
        let embed = build_trade_embed("SPY", 460.0, 455.0, 2, 0.75, "order-abc");
        assert_eq!(embed["color"], COLOR_GREEN);
        let title = embed["title"].as_str().expect("title field");
        assert!(title.contains("SPY"), "title should mention underlying");
    }

    #[test]
    fn error_embed_has_red_color() {
        let embed = build_error_embed("Scan failed", "NullPointerError: at line 42");
        assert_eq!(embed["color"], COLOR_RED);
    }

    #[test]
    fn scan_summary_embed_reports_correct_trade_count() {
        let embed = build_scan_summary_embed("morning", 3, 5, 2, Some(22.5));
        let description = embed["description"].as_str().expect("description");
        assert!(description.contains("2"), "should mention 2 trades placed");
    }

    use crate::measurement::metrics::{MetricBundle, SampleSizeTag};
    use std::collections::HashMap;

    fn mk_bundle(tag: SampleSizeTag) -> MetricBundle {
        MetricBundle {
            window_start: "2026-03-22".to_string(), window_end: "2026-04-22".to_string(),
            trade_count: 42, sharpe: 1.2, sortino: 1.8, profit_factor: 1.5,
            win_rate: 0.72, pnl_skew: -1.1, max_drawdown_pct: 0.06,
            mean_slippage_vs_mid: 0.012, max_slippage_vs_mid: 0.04,
            slippage_vs_orats_ratio: 1.1,
            fill_size_violation_count: 0, fill_size_violation_pct: 0.0,
            regime_buckets: HashMap::new(), greek_ranges: HashMap::new(),
            sample_size_tag: tag,
        }
    }

    #[test]
    fn metrics_embed_includes_window_and_tag() {
        let embed = build_metrics_embed(&mk_bundle(SampleSizeTag::Weak));
        let desc = embed["description"].as_str().unwrap_or("");
        assert!(desc.contains("2026-03-22"));
        assert!(desc.contains("WEAK"));
    }

    fn mk_greeks() -> PortfolioGreeks {
        use std::collections::HashMap;
        PortfolioGreeks {
            beta_weighted_delta: -42.0,
            total_gamma: -0.08,
            total_vega: -35.0,
            total_theta: 18.0,
            delta_by_underlying: HashMap::new(),
            max_gamma_single_position: 0.05,
            max_vega_single_position: 20.0,
            open_position_count: 3,
        }
    }

    fn mk_stats() -> SpyStats {
        SpyStats { realized_vol_20d: 0.123, return_20d: 0.012, drawdown_from_52w_high: 0.034 }
    }

    #[test]
    fn eod_embed_zero_trades_still_shows_open_book_and_regime() {
        let g = mk_greeks();
        let s = mk_stats();
        let embed = build_eod_embed(&EodEmbedInput {
            date: "2026-04-23",
            realized_pnl: 0.0, trades_closed: 0, win_rate: 1.0,
            open_count: 3, open_mtm: 45.0, max_loss_remaining: 1275.0,
            greeks: Some(&g), spy_price: Some(480.0),
            dte_buckets: (0, 2, 1),
            spot_vix: Some(17.45), vix_source: "fred",
            vixy: Some(21.30),
            spy_stats: Some(&s),
            equity: Some(102_430.0), cash: Some(98_100.0),
        });
        let desc = embed["description"].as_str().unwrap_or("");
        assert!(desc.contains("normal"), "regime should be normal at VIX=17.45");
        let fields = embed["fields"].as_array().expect("fields");
        let joined: String = fields.iter()
            .filter_map(|f| f["value"].as_str())
            .collect::<Vec<_>>()
            .join("|");
        assert!(joined.contains("MTM"), "should show open MTM even on zero-close day");
        assert!(joined.contains("≤7d"), "should show DTE mix");
        // $Δ = -42 * 480 = -20160
        assert!(joined.contains("-20160") || joined.contains("-20,160"),
                "should show $delta scaled by spy price; got: {}", joined);
    }

    #[test]
    fn regime_tag_boundaries() {
        assert_eq!(regime_tag(Some(14.9)), "calm");
        assert_eq!(regime_tag(Some(15.0)), "normal");
        assert_eq!(regime_tag(Some(19.9)), "normal");
        assert_eq!(regime_tag(Some(20.0)), "elevated");
        assert_eq!(regime_tag(Some(29.9)), "elevated");
        assert_eq!(regime_tag(Some(30.0)), "stress");
        assert_eq!(regime_tag(None), "unknown");
    }

    #[test]
    fn eod_embed_shows_em_dash_when_greeks_unavailable() {
        let embed = build_eod_embed(&EodEmbedInput {
            date: "2026-04-23",
            realized_pnl: 0.0, trades_closed: 0, win_rate: 1.0,
            open_count: 0, open_mtm: 0.0, max_loss_remaining: 0.0,
            greeks: None, spy_price: None,
            dte_buckets: (0, 0, 0),
            spot_vix: None, vix_source: "unavailable",
            vixy: None, spy_stats: None,
            equity: None, cash: None,
        });
        let fields = embed["fields"].as_array().expect("fields");
        let greek_field = fields.iter().find(|f| f["name"] == "$Δ (per $1 SPY)").unwrap();
        assert_eq!(greek_field["value"], "—");
    }
}
