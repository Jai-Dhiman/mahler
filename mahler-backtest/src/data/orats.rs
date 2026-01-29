//! ORATS API client for downloading historical options data.
//!
//! API Constraints:
//! - Row limit: 5,000 rows per request
//! - Rate limit: 1,000 requests/minute
//! - Monthly limit: 20,000 requests/month
//! - Date range: 2007-01-01 to present

use std::collections::HashMap;
use std::time::{Duration, Instant};

use chrono::NaiveDate;
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::{Greeks, OptionQuote, OptionType, OptionsChain, OptionsSnapshot};

/// ORATS API base URL.
const BASE_URL: &str = "https://api.orats.io/datav2";

/// Maximum rows per API request.
pub const MAX_ROWS_PER_REQUEST: usize = 5000;

/// Minimum interval between requests (100ms = max 600 req/min, well under limit).
const MIN_REQUEST_INTERVAL: Duration = Duration::from_millis(100);

/// ORATS API errors.
#[derive(Error, Debug)]
pub enum ORATSError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Invalid response format: {0}")]
    InvalidResponse(String),

    #[error("No data available for {ticker} on {date}")]
    NoData { ticker: String, date: NaiveDate },
}

/// API response wrapper - ORATS wraps all responses in {"data": [...]}
#[derive(Debug, Clone, Deserialize)]
pub struct ApiResponse<T> {
    pub data: T,
}

/// Ticker availability information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerInfo {
    pub ticker: String,
    pub min: String, // Date as string from API
    pub max: String,
}

impl TickerInfo {
    pub fn min_date(&self) -> Option<NaiveDate> {
        NaiveDate::parse_from_str(&self.min, "%Y-%m-%d").ok()
    }

    pub fn max_date(&self) -> Option<NaiveDate> {
        NaiveDate::parse_from_str(&self.max, "%Y-%m-%d").ok()
    }
}

/// Raw strike record from ORATS API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawStrikeRecord {
    pub ticker: String,
    pub trade_date: String,
    pub expir_date: String,
    pub dte: i32,
    pub strike: f64,
    pub stock_price: f64,

    // Call data
    #[serde(default)]
    pub call_volume: Option<i64>,
    #[serde(default)]
    pub call_open_interest: Option<i64>,
    #[serde(default)]
    pub call_bid_price: Option<f64>,
    #[serde(default)]
    pub call_ask_price: Option<f64>,
    #[serde(default)]
    pub call_value: Option<f64>,
    #[serde(default)]
    pub call_bid_iv: Option<f64>,
    #[serde(default)]
    pub call_mid_iv: Option<f64>,
    #[serde(default)]
    pub call_ask_iv: Option<f64>,

    // Put data
    #[serde(default)]
    pub put_volume: Option<i64>,
    #[serde(default)]
    pub put_open_interest: Option<i64>,
    #[serde(default)]
    pub put_bid_price: Option<f64>,
    #[serde(default)]
    pub put_ask_price: Option<f64>,
    #[serde(default)]
    pub put_value: Option<f64>,
    #[serde(default)]
    pub put_bid_iv: Option<f64>,
    #[serde(default)]
    pub put_mid_iv: Option<f64>,
    #[serde(default)]
    pub put_ask_iv: Option<f64>,

    // Common fields
    #[serde(default)]
    pub smv_vol: Option<f64>,
    #[serde(default)]
    pub delta: Option<f64>,
    #[serde(default)]
    pub gamma: Option<f64>,
    #[serde(default)]
    pub theta: Option<f64>,
    #[serde(default)]
    pub vega: Option<f64>,
    #[serde(default)]
    pub rho: Option<f64>,
    #[serde(default)]
    pub residual_rate: Option<f64>,
}

impl RawStrikeRecord {
    /// Convert to call OptionQuote.
    pub fn to_call_quote(&self) -> Option<OptionQuote> {
        let trade_date = NaiveDate::parse_from_str(&self.trade_date, "%Y-%m-%d").ok()?;
        let expiration = NaiveDate::parse_from_str(&self.expir_date, "%Y-%m-%d").ok()?;
        let bid = Decimal::try_from(self.call_bid_price?).ok()?;
        let ask = Decimal::try_from(self.call_ask_price?).ok()?;

        Some(OptionQuote {
            ticker: self.ticker.clone(),
            trade_date,
            expiration,
            dte: self.dte,
            strike: Decimal::try_from(self.strike).ok()?,
            option_type: OptionType::Call,
            stock_price: Decimal::try_from(self.stock_price).ok()?,
            bid,
            ask,
            mid: (bid + ask) / Decimal::from(2),
            theoretical_value: Decimal::try_from(self.call_value.unwrap_or(0.0)).ok()?,
            volume: self.call_volume.unwrap_or(0),
            open_interest: self.call_open_interest.unwrap_or(0),
            bid_iv: self.call_bid_iv.unwrap_or(0.0),
            mid_iv: self.call_mid_iv.unwrap_or(0.0),
            ask_iv: self.call_ask_iv.unwrap_or(0.0),
            smv_vol: self.smv_vol.unwrap_or(0.0),
            greeks: Greeks {
                delta: self.delta.unwrap_or(0.0),
                gamma: self.gamma.unwrap_or(0.0),
                theta: self.theta.unwrap_or(0.0),
                vega: self.vega.unwrap_or(0.0),
                rho: self.rho.unwrap_or(0.0),
            },
            residual_rate: self.residual_rate.unwrap_or(0.0),
        })
    }

    /// Convert to put OptionQuote.
    pub fn to_put_quote(&self) -> Option<OptionQuote> {
        let trade_date = NaiveDate::parse_from_str(&self.trade_date, "%Y-%m-%d").ok()?;
        let expiration = NaiveDate::parse_from_str(&self.expir_date, "%Y-%m-%d").ok()?;
        let bid = Decimal::try_from(self.put_bid_price?).ok()?;
        let ask = Decimal::try_from(self.put_ask_price?).ok()?;

        // For puts, delta is typically negative but ORATS returns the absolute value
        // We store as negative for puts
        let delta = -self.delta.unwrap_or(0.0).abs();

        Some(OptionQuote {
            ticker: self.ticker.clone(),
            trade_date,
            expiration,
            dte: self.dte,
            strike: Decimal::try_from(self.strike).ok()?,
            option_type: OptionType::Put,
            stock_price: Decimal::try_from(self.stock_price).ok()?,
            bid,
            ask,
            mid: (bid + ask) / Decimal::from(2),
            theoretical_value: Decimal::try_from(self.put_value.unwrap_or(0.0)).ok()?,
            volume: self.put_volume.unwrap_or(0),
            open_interest: self.put_open_interest.unwrap_or(0),
            bid_iv: self.put_bid_iv.unwrap_or(0.0),
            mid_iv: self.put_mid_iv.unwrap_or(0.0),
            ask_iv: self.put_ask_iv.unwrap_or(0.0),
            smv_vol: self.smv_vol.unwrap_or(0.0),
            greeks: Greeks {
                delta,
                gamma: self.gamma.unwrap_or(0.0),
                theta: self.theta.unwrap_or(0.0),
                vega: self.vega.unwrap_or(0.0),
                rho: self.rho.unwrap_or(0.0),
            },
            residual_rate: self.residual_rate.unwrap_or(0.0),
        })
    }
}

/// ORATS API client.
pub struct ORATSClient {
    client: Client,
    token: String,
    last_request: Instant,
    request_count: u64,
}

impl ORATSClient {
    /// Create a new ORATS client.
    pub fn new(token: String) -> Self {
        Self {
            client: Client::new(),
            token,
            last_request: Instant::now() - MIN_REQUEST_INTERVAL,
            request_count: 0,
        }
    }

    /// Rate-limited request helper.
    async fn request<T: for<'de> Deserialize<'de>>(
        &mut self,
        endpoint: &str,
        params: &[(&str, &str)],
    ) -> Result<T, ORATSError> {
        // Rate limiting
        let elapsed = self.last_request.elapsed();
        if elapsed < MIN_REQUEST_INTERVAL {
            tokio::time::sleep(MIN_REQUEST_INTERVAL - elapsed).await;
        }

        let url = format!("{}/{}", BASE_URL, endpoint);
        let mut all_params: Vec<(&str, &str)> = params.to_vec();
        all_params.push(("token", &self.token));

        let response = self.client.get(&url).query(&all_params).send().await?;

        self.last_request = Instant::now();
        self.request_count += 1;

        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ORATSError::RateLimitExceeded);
        }

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ORATSError::ApiError(format!("{}: {}", status, text)));
        }

        response.json().await.map_err(|e| {
            ORATSError::InvalidResponse(format!("Failed to parse response: {}", e))
        })
    }

    /// Get request count for monitoring.
    pub fn request_count(&self) -> u64 {
        self.request_count
    }

    /// Get available tickers with date ranges.
    pub async fn get_tickers(&mut self, ticker: Option<&str>) -> Result<Vec<TickerInfo>, ORATSError> {
        let params: Vec<(&str, &str)> = match ticker {
            Some(t) => vec![("ticker", t)],
            None => vec![],
        };
        let response: ApiResponse<Vec<TickerInfo>> = self.request("tickers", &params).await?;
        Ok(response.data)
    }

    /// Get historical strikes for a single date.
    pub async fn get_strikes_history(
        &mut self,
        ticker: &str,
        trade_date: NaiveDate,
    ) -> Result<Vec<RawStrikeRecord>, ORATSError> {
        let date_str = trade_date.format("%Y-%m-%d").to_string();
        let params = vec![("ticker", ticker), ("tradeDate", &date_str)];

        let response: ApiResponse<Vec<RawStrikeRecord>> = self.request("hist/strikes", &params).await?;
        Ok(response.data)
    }

    /// Get historical strikes for a date range.
    ///
    /// Note: Large date ranges may exceed row limits. Use with caution.
    pub async fn get_strikes_history_range(
        &mut self,
        ticker: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<Vec<RawStrikeRecord>, ORATSError> {
        let date_range = format!(
            "{},{}",
            start_date.format("%Y-%m-%d"),
            end_date.format("%Y-%m-%d")
        );
        let params = vec![("ticker", ticker), ("tradeDate", &date_range)];

        let response: ApiResponse<Vec<RawStrikeRecord>> = self.request("hist/strikes", &params).await?;
        Ok(response.data)
    }

    /// Get IV rank history.
    pub async fn get_ivrank_history(
        &mut self,
        ticker: &str,
        start_date: NaiveDate,
        end_date: NaiveDate,
    ) -> Result<serde_json::Value, ORATSError> {
        let date_range = format!(
            "{},{}",
            start_date.format("%Y-%m-%d"),
            end_date.format("%Y-%m-%d")
        );
        let params = vec![("ticker", ticker), ("tradeDate", &date_range)];

        self.request("hist/ivrank", &params).await
    }
}

/// Convert raw strike records to an OptionsSnapshot.
pub fn records_to_snapshot(
    ticker: &str,
    trade_date: NaiveDate,
    records: Vec<RawStrikeRecord>,
) -> OptionsSnapshot {
    // Group by expiration
    let mut chains_map: HashMap<NaiveDate, OptionsChain> = HashMap::new();
    let mut underlying_price = Decimal::ZERO;

    for record in records {
        // Get underlying price from first record
        if underlying_price.is_zero() {
            if let Ok(price) = Decimal::try_from(record.stock_price) {
                underlying_price = price;
            }
        }

        let expiration = match NaiveDate::parse_from_str(&record.expir_date, "%Y-%m-%d") {
            Ok(d) => d,
            Err(_) => continue,
        };

        let chain = chains_map
            .entry(expiration)
            .or_insert_with(|| OptionsChain::new(expiration, record.dte));

        // Add call quote
        if let Some(call) = record.to_call_quote() {
            chain.add_quote(call);
        }

        // Add put quote
        if let Some(put) = record.to_put_quote() {
            chain.add_quote(put);
        }
    }

    // Convert to sorted vector
    let mut chains: Vec<OptionsChain> = chains_map.into_values().collect();
    chains.sort_by_key(|c| c.expiration);

    OptionsSnapshot {
        date: trade_date,
        ticker: ticker.to_string(),
        underlying_price,
        chains,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticker_info_date_parsing() {
        let info = TickerInfo {
            ticker: "SPY".to_string(),
            min: "2007-01-03".to_string(),
            max: "2024-01-15".to_string(),
        };

        assert_eq!(info.min_date(), Some(NaiveDate::from_ymd_opt(2007, 1, 3).unwrap()));
        assert_eq!(info.max_date(), Some(NaiveDate::from_ymd_opt(2024, 1, 15).unwrap()));
    }
}
