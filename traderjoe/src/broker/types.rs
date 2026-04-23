use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionContract {
    pub symbol: String,
    pub underlying: String,
    pub expiration: String,
    pub strike: f64,
    pub option_type: OptionType,
    pub bid: f64,
    pub ask: f64,
    pub last: f64,
    pub volume: i64,
    pub open_interest: i64,
    pub implied_volatility: Option<f64>,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub theta: Option<f64>,
    pub vega: Option<f64>,
    pub bid_size: Option<i64>,
    pub ask_size: Option<i64>,
}

impl OptionContract {
    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    pub fn bid_ask_spread_pct(&self) -> f64 {
        let mid = self.mid_price();
        if mid <= 0.0 {
            return 1.0;
        }
        (self.ask - self.bid) / mid
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsChain {
    pub underlying: String,
    pub underlying_price: f64,
    pub expirations: Vec<String>,
    pub contracts: Vec<OptionContract>,
}

impl OptionsChain {
    pub fn get_puts(&self, expiration: &str) -> Vec<OptionContract> {
        self.contracts
            .iter()
            .filter(|c| c.expiration == expiration && c.option_type == OptionType::Put)
            .cloned()
            .collect()
    }

    pub fn get_calls(&self, expiration: &str) -> Vec<OptionContract> {
        self.contracts
            .iter()
            .filter(|c| c.expiration == expiration && c.option_type == OptionType::Call)
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub equity: f64,
    pub buying_power: f64,
    pub cash: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub status: OrderStatus,
    pub filled_price: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SpreadOrder {
    pub underlying: String,
    pub short_occ_symbol: String,
    pub long_occ_symbol: String,
    pub contracts: i64,
    pub limit_price: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VixData {
    pub vix: f64,
    pub vix3m: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub timestamp: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_contract(strike: f64, option_type: OptionType, expiration: &str) -> OptionContract {
        OptionContract {
            symbol: format!("TEST{}", strike),
            underlying: "TEST".to_string(),
            expiration: expiration.to_string(),
            strike,
            option_type,
            bid: 1.00,
            ask: 1.05,
            last: 1.02,
            volume: 100,
            open_interest: 500,
            implied_volatility: Some(0.22),
            delta: Some(-0.10),
            gamma: None,
            theta: None,
            vega: None,
            bid_size: Some(10),
            ask_size: Some(10),
        }
    }

    #[test]
    fn get_puts_filters_to_puts_for_given_expiration() {
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string(), "2026-06-20".to_string()],
            contracts: vec![
                make_contract(460.0, OptionType::Put, "2026-05-15"),
                make_contract(480.0, OptionType::Call, "2026-05-15"),
                make_contract(450.0, OptionType::Put, "2026-06-20"),
            ],
        };
        let puts = chain.get_puts("2026-05-15");
        assert_eq!(puts.len(), 1);
        assert_eq!(puts[0].strike, 460.0);
        assert!(matches!(puts[0].option_type, OptionType::Put));
    }

    #[test]
    fn get_calls_filters_to_calls_for_given_expiration() {
        let chain = OptionsChain {
            underlying: "SPY".to_string(),
            underlying_price: 480.0,
            expirations: vec!["2026-05-15".to_string()],
            contracts: vec![
                make_contract(460.0, OptionType::Put, "2026-05-15"),
                make_contract(500.0, OptionType::Call, "2026-05-15"),
            ],
        };
        let calls = chain.get_calls("2026-05-15");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].strike, 500.0);
    }

    #[test]
    fn option_contract_round_trips_bid_and_ask_size() {
        let contract = OptionContract {
            symbol: "SPY260515P00460000".to_string(),
            underlying: "SPY".to_string(),
            expiration: "2026-05-15".to_string(),
            strike: 460.0,
            option_type: OptionType::Put,
            bid: 1.00, ask: 1.05, last: 1.02,
            volume: 100, open_interest: 500,
            implied_volatility: Some(0.22),
            delta: Some(-0.10), gamma: Some(0.01), theta: Some(-0.05), vega: Some(0.10),
            bid_size: Some(25),
            ask_size: Some(40),
        };
        let json = serde_json::to_string(&contract).expect("serialize");
        let back: OptionContract = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.bid_size, Some(25));
        assert_eq!(back.ask_size, Some(40));
    }
}
