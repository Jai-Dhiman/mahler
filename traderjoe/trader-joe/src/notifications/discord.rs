use serde_json::{json, Value};
use worker::{Fetch, Headers, Method, Request, RequestInit, Result};

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

/// Build a Discord embed for EOD summary.
pub fn build_eod_embed(
    date: &str,
    realized_pnl: f64,
    trades_closed: i64,
    win_rate: f64,
) -> Value {
    let color = if realized_pnl >= 0.0 { COLOR_GREEN } else { COLOR_RED };
    json!({
        "title": format!("EOD Summary — {}", date),
        "color": color,
        "fields": [
            { "name": "Realized P&L", "value": format!("${:.0}", realized_pnl), "inline": true },
            { "name": "Trades Closed", "value": trades_closed.to_string(), "inline": true },
            { "name": "Win Rate", "value": format!("{:.0}%", win_rate * 100.0), "inline": true }
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

    pub async fn send_eod_summary(&self, date: &str, pnl: f64, closed: i64, win_rate: f64) -> Result<()> {
        let embed = build_eod_embed(date, pnl, closed, win_rate);
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
}
