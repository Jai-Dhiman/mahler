use serde_json::Value;
use worker::{Fetch, Headers, Method, Request, RequestInit, Result};

use crate::broker::types::{
    Account, Bar, OptionContract, OptionType, OptionsChain, Order, OrderStatus,
    SpreadOrder, VixData,
};

/// Build an OCC-format option symbol.
///
/// Format: UNDERLYINGYYMMDDCBBBBB (C=Call, P=Put, BBBBB=strike * 1000 padded to 8 digits)
/// Example: SPY260515P00460000
pub fn build_occ_symbol(underlying: &str, expiration: &str, option_type: &str, strike: f64) -> String {
    let parts: Vec<&str> = expiration.split('-').collect();
    let yy = &parts[0][2..]; // last 2 digits of year
    let mm = parts[1];
    let dd = parts[2];
    let strike_int = (strike * 1000.0).round() as u64;
    format!("{}{}{}{}{}{:08}", underlying, yy, mm, dd, option_type, strike_int)
}

pub struct AlpacaClient {
    api_key: String,
    secret_key: String,
    base_url: String,
    data_url: String,
}

impl AlpacaClient {
    pub fn new(api_key: impl Into<String>, secret_key: impl Into<String>, paper: bool) -> Self {
        AlpacaClient {
            api_key: api_key.into(),
            secret_key: secret_key.into(),
            base_url: if paper {
                "https://paper-api.alpaca.markets".to_string()
            } else {
                "https://api.alpaca.markets".to_string()
            },
            data_url: "https://data.alpaca.markets".to_string(),
        }
    }

    fn auth_headers(&self) -> Result<Headers> {
        let mut h = Headers::new();
        h.set("APCA-API-KEY-ID", &self.api_key)?;
        h.set("APCA-API-SECRET-KEY", &self.secret_key)?;
        h.set("Accept", "application/json")?;
        Ok(h)
    }

    async fn get(&self, url: &str) -> Result<Value> {
        let headers = self.auth_headers()?;
        let mut init = RequestInit::new();
        init.with_method(Method::Get).with_headers(headers);
        let req = Request::new_with_init(url, &init)?;
        let mut resp = Fetch::Request(req).send().await?;
        resp.json().await
    }

    pub async fn is_market_open(&self) -> Result<bool> {
        let url = format!("{}/v2/clock", self.base_url);
        let json: Value = self.get(&url).await?;
        Ok(json["is_open"].as_bool().unwrap_or(false))
    }

    pub async fn get_account(&self) -> Result<Account> {
        let url = format!("{}/v2/account", self.base_url);
        let json: Value = self.get(&url).await?;
        Ok(Account {
            equity: json["equity"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            buying_power: json["buying_power"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
            cash: json["cash"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0),
        })
    }

    pub async fn get_vix(&self) -> Result<Option<VixData>> {
        // VIXY is a short-term VIX futures ETF, not spot VIX. It is affected by
        // futures roll costs and contango/backwardation, and can diverge from spot
        // VIX by 20-40% during high-volatility events. The circuit breaker thresholds
        // (vix_halt=50, vix_caution=30) and position sizer thresholds (40/50) were
        // calibrated against spot VIX. VIXY typically trades at a discount to spot
        // VIX, so these thresholds err on the side of caution (under-react to rising
        // vol). Alpaca does not expose a spot VIX endpoint; migrate to
        // /v2/indices/snapshots?symbols=VIX if Alpaca adds index data access.
        let url = format!(
            "{}/v2/stocks/snapshots?symbols=VIXY,UVXY&feed=iex",
            self.data_url
        );
        match self.get(&url).await {
            Ok(json) => {
                // latestTrade.p is only available during market hours (IEX feed).
                // After close, fall back to dailyBar.c which persists all day.
                let vix = json["VIXY"]["latestTrade"]["p"].as_f64()
                    .or_else(|| json["VIXY"]["dailyBar"]["c"].as_f64());
                Ok(vix.map(|v| VixData { vix: v, vix3m: None }))
            }
            Err(_) => Ok(None),
        }
    }

    pub async fn get_options_chain(&self, symbol: &str) -> Result<OptionsChain> {
        use chrono::{Duration, Utc};

        // Fetch only contracts in the 30-45 DTE window using date filters.
        // Without filters, limit=1000 returns near-term contracts alphabetically,
        // none of which fall in the 30-45 DTE window the screener needs.
        let today = Utc::now().date_naive();
        let exp_min = (today + Duration::days(30)).format("%Y-%m-%d").to_string();
        let exp_max = (today + Duration::days(45)).format("%Y-%m-%d").to_string();

        // The options snapshots endpoint does not include an underlying price field.
        // Fetch stock snapshot separately for the current price.
        let stock_url = format!(
            "{}/v2/stocks/snapshots?symbols={}&feed=iex",
            self.data_url, symbol
        );
        let stock_json: Value = self.get(&stock_url).await.unwrap_or_default();
        let underlying_price = stock_json[symbol]["latestTrade"]["p"]
            .as_f64()
            .or_else(|| stock_json[symbol]["latestQuote"]["ap"].as_f64())
            .unwrap_or(0.0);

        let mut contracts: Vec<OptionContract> = Vec::new();
        let mut expirations: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut page_token: Option<String> = None;

        loop {
            let url = if let Some(ref token) = page_token {
                format!(
                    "{}/v1beta1/options/snapshots/{}?feed=indicative&limit=1000&expiration_date_gte={}&expiration_date_lte={}&next_page_token={}",
                    self.data_url, symbol, exp_min, exp_max, token
                )
            } else {
                format!(
                    "{}/v1beta1/options/snapshots/{}?feed=indicative&limit=1000&expiration_date_gte={}&expiration_date_lte={}",
                    self.data_url, symbol, exp_min, exp_max
                )
            };

            let json: Value = self.get(&url).await?;

            if let Some(snapshots) = json["snapshots"].as_object() {
                for (occ_symbol, data) in snapshots {
                let greeks = &data["greeks"];
                let quote = &data["latestQuote"];

                if occ_symbol.len() < symbol.len() + 9 {
                    continue;
                }
                let rest = &occ_symbol[symbol.len()..];
                if rest.len() < 9 {
                    continue;
                }
                let date_part = &rest[..6];
                let option_char = &rest[6..7];
                let strike_part = &rest[7..];

                let strike = strike_part.parse::<f64>().unwrap_or(0.0) / 1000.0;
                let expiration = format!(
                    "20{}-{}-{}",
                    &date_part[..2],
                    &date_part[2..4],
                    &date_part[4..6]
                );
                let option_type = if option_char == "C" { OptionType::Call } else { OptionType::Put };

                expirations.insert(expiration.clone());

                contracts.push(OptionContract {
                    symbol: occ_symbol.clone(),
                    underlying: symbol.to_string(),
                    expiration,
                    strike,
                    option_type,
                    bid: quote["bp"].as_f64().unwrap_or(0.0),
                    ask: quote["ap"].as_f64().unwrap_or(0.0),
                    last: quote["ap"].as_f64().unwrap_or(0.0),
                    volume: data["dailyBar"]["v"].as_i64().unwrap_or(0),
                    open_interest: data["openInterest"].as_i64().unwrap_or(0),
                    implied_volatility: data["impliedVolatility"].as_f64(),
                    delta: greeks["delta"].as_f64(),
                    gamma: greeks["gamma"].as_f64(),
                    theta: greeks["theta"].as_f64(),
                    vega: greeks["vega"].as_f64(),
                    bid_size: quote["bs"].as_i64(),
                    ask_size: quote["as"].as_i64(),
                });
            }
        }

            // Paginate until all contracts in the DTE window are fetched.
            match json["next_page_token"].as_str() {
                Some(token) if !token.is_empty() => page_token = Some(token.to_string()),
                _ => break,
            }
        }

        let mut expirations_vec: Vec<String> = expirations.into_iter().collect();
        expirations_vec.sort();

        Ok(OptionsChain {
            underlying: symbol.to_string(),
            underlying_price,
            expirations: expirations_vec,
            contracts,
        })
    }

    pub async fn get_bars(&self, symbol: &str, limit: i64) -> Result<Vec<Bar>> {
        let url = format!(
            "{}/v2/stocks/{}/bars?timeframe=1Day&limit={}&adjustment=raw&feed=iex",
            self.data_url, symbol, limit
        );
        let json: Value = self.get(&url).await?;
        let bars = json["bars"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|b| Bar {
                timestamp: b["t"].as_str().unwrap_or("").to_string(),
                open: b["o"].as_f64().unwrap_or(0.0),
                high: b["h"].as_f64().unwrap_or(0.0),
                low: b["l"].as_f64().unwrap_or(0.0),
                close: b["c"].as_f64().unwrap_or(0.0),
                volume: b["v"].as_i64().unwrap_or(0),
            })
            .collect();
        Ok(bars)
    }

    pub async fn place_spread_order(&self, order: &SpreadOrder) -> Result<Order> {
        let url = format!("{}/v2/orders", self.base_url);
        let body = serde_json::to_string(&serde_json::json!({
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "qty": order.contracts.to_string(),
            "limit_price": format!("{:.2}", order.limit_price),
            "legs": [
                {
                    "symbol": order.short_occ_symbol,
                    "ratio_qty": "1",
                    "side": "sell",
                    "position_effect": "open"
                },
                {
                    "symbol": order.long_occ_symbol,
                    "ratio_qty": "1",
                    "side": "buy",
                    "position_effect": "open"
                }
            ]
        }))
        .map_err(|e| worker::Error::RustError(e.to_string()))?;

        let mut headers = self.auth_headers()?;
        headers.set("Content-Type", "application/json")?;

        let mut init = RequestInit::new();
        init.with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(body.into()));

        let req = Request::new_with_init(&url, &init)?;
        let mut resp = Fetch::Request(req).send().await?;
        let status = resp.status_code();
        let json: Value = resp.json().await?;

        if json["id"].is_null() || json["id"].as_str().map(|s| s.is_empty()).unwrap_or(true) {
            return Err(worker::Error::RustError(format!(
                "Order rejected (HTTP {}): {}",
                status, json
            )));
        }

        Ok(Order {
            id: json["id"].as_str().unwrap_or("").to_string(),
            status: OrderStatus::Pending,
            filled_price: None,
        })
    }

    pub async fn place_closing_spread_order(&self, order: &SpreadOrder) -> Result<Order> {
        let url = format!("{}/v2/orders", self.base_url);
        let body = serde_json::to_string(&serde_json::json!({
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "qty": order.contracts.to_string(),
            "limit_price": format!("{:.2}", order.limit_price),
            "legs": [
                {
                    "symbol": order.short_occ_symbol,
                    "ratio_qty": "1",
                    "side": "buy",
                    "position_effect": "close"
                },
                {
                    "symbol": order.long_occ_symbol,
                    "ratio_qty": "1",
                    "side": "sell",
                    "position_effect": "close"
                }
            ]
        }))
        .map_err(|e| worker::Error::RustError(e.to_string()))?;

        let mut headers = self.auth_headers()?;
        headers.set("Content-Type", "application/json")?;

        let mut init = RequestInit::new();
        init.with_method(Method::Post)
            .with_headers(headers)
            .with_body(Some(body.into()));

        let req = Request::new_with_init(&url, &init)?;
        let mut resp = Fetch::Request(req).send().await?;
        let status = resp.status_code();
        let json: Value = resp.json().await?;

        if json["id"].is_null() || json["id"].as_str().map(|s| s.is_empty()).unwrap_or(true) {
            return Err(worker::Error::RustError(format!(
                "Closing order rejected (HTTP {}): {}",
                status, json
            )));
        }

        Ok(Order {
            id: json["id"].as_str().unwrap_or("").to_string(),
            status: OrderStatus::Pending,
            filled_price: None,
        })
    }

    pub async fn get_order(&self, order_id: &str) -> Result<Order> {
        let url = format!("{}/v2/orders/{}", self.base_url, order_id);
        let json: Value = self.get(&url).await?;
        let status = match json["status"].as_str().unwrap_or("") {
            "filled" => OrderStatus::Filled,
            "canceled" | "cancelled" => OrderStatus::Cancelled,
            "rejected" => OrderStatus::Rejected,
            _ => OrderStatus::Pending,
        };
        let filled_price = json["filled_avg_price"]
            .as_str()
            .and_then(|s| s.parse::<f64>().ok());
        Ok(Order { id: order_id.to_string(), status, filled_price })
    }

    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let url = format!("{}/v2/orders/{}", self.base_url, order_id);
        let headers = self.auth_headers()?;
        let mut init = RequestInit::new();
        init.with_method(Method::Delete).with_headers(headers);
        let req = Request::new_with_init(&url, &init)?;
        Fetch::Request(req).send().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occ_symbol_format_matches_alpaca_convention() {
        let symbol = build_occ_symbol("SPY", "2026-05-15", "P", 460.0);
        assert_eq!(symbol, "SPY260515P00460000");
    }

    #[test]
    fn occ_symbol_with_fractional_strike() {
        let symbol = build_occ_symbol("SPY", "2026-05-15", "P", 462.5);
        assert_eq!(symbol, "SPY260515P00462500");
    }

    #[test]
    fn occ_symbol_for_call() {
        let symbol = build_occ_symbol("QQQ", "2026-06-20", "C", 500.0);
        assert_eq!(symbol, "QQQ260620C00500000");
    }
}
