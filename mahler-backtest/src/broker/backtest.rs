use chrono::{NaiveDateTime, NaiveTime};
use rust_decimal::Decimal;

use crate::backtest::{CommissionModel, SlippageModel};
use crate::engine_core::events::{FillEvent, LegIntent, OrderIntent, OrderSide};
use crate::data::OptionsSnapshot;

use super::{Broker, BrokerError};

pub struct SimulatedBroker {
    slippage: SlippageModel,
    commission: CommissionModel,
}

impl SimulatedBroker {
    pub fn new(slippage: SlippageModel, commission: CommissionModel) -> Self {
        Self { slippage, commission }
    }

    fn find_quote_prices(
        &self,
        leg: &LegIntent,
        snapshot: &OptionsSnapshot,
    ) -> Result<(Decimal, Decimal), BrokerError> {
        for chain in &snapshot.chains {
            if chain.expiration != leg.expiration {
                continue;
            }
            let quotes = match leg.option_type {
                crate::data::OptionType::Call => &chain.calls,
                crate::data::OptionType::Put => &chain.puts,
            };
            if let Some(quote) = quotes.iter().find(|q| q.strike == leg.strike) {
                return Ok((quote.bid, quote.ask));
            }
        }
        Err(BrokerError::NoMatchingQuote {
            strike: leg.strike,
            option_type: leg.option_type,
            expiration: leg.expiration,
        })
    }
}

impl Broker for SimulatedBroker {
    fn submit_order(
        &mut self,
        order: &OrderIntent,
        snapshot: &OptionsSnapshot,
    ) -> Result<FillEvent, BrokerError> {
        let num_legs = order.legs.len();
        let slippage = self.slippage.for_legs(num_legs);

        let mut fill_prices = Vec::with_capacity(num_legs);
        let mut max_contracts: i32 = 0;

        for leg in &order.legs {
            let (bid, ask) = self.find_quote_prices(leg, snapshot)?;
            let fill = match leg.side {
                OrderSide::Sell => slippage.sell_fill(bid, ask),
                OrderSide::Buy => slippage.buy_fill(bid, ask),
            };
            fill_prices.push(fill);
            let abs = leg.contracts.abs();
            if abs > max_contracts {
                max_contracts = abs;
            }
        }

        // Commission is per spread unit (max contracts per leg) * legs
        let commission = self.commission.calculate(max_contracts, num_legs).total;

        let time = NaiveTime::from_hms_opt(15, 46, 0).unwrap();
        let timestamp = NaiveDateTime::new(snapshot.date, time);

        Ok(FillEvent {
            order: order.clone(),
            fill_prices,
            commission,
            timestamp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::backtest::{SpreadType, CommissionModel, SlippageModel};
    use crate::engine_core::events::{OrderIntent, LegIntent, OrderSide};
    use crate::data::{OptionType, OptionQuote, OptionsChain, OptionsSnapshot, Greeks};

    fn make_snapshot_with_quotes() -> OptionsSnapshot {
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let exp = NaiveDate::from_ymd_opt(2024, 2, 16).unwrap();
        let mut snapshot = OptionsSnapshot::new(date, "SPY".to_string(), dec!(480));
        let mut chain = OptionsChain::new(exp, 32);

        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(), trade_date: date, expiration: exp, dte: 32,
            strike: dec!(470), option_type: OptionType::Put, stock_price: dec!(480),
            bid: dec!(2.40), ask: dec!(2.60), mid: dec!(2.50),
            theoretical_value: dec!(2.50), volume: 1000, open_interest: 5000,
            bid_iv: 0.18, mid_iv: 0.19, ask_iv: 0.20, smv_vol: 0.19,
            greeks: Greeks { delta: -0.15, gamma: 0.02, theta: -0.05, vega: 0.10, rho: 0.01 },
            residual_rate: 0.05,
        });

        chain.add_quote(OptionQuote {
            ticker: "SPY".to_string(), trade_date: date, expiration: exp, dte: 32,
            strike: dec!(465), option_type: OptionType::Put, stock_price: dec!(480),
            bid: dec!(1.40), ask: dec!(1.60), mid: dec!(1.50),
            theoretical_value: dec!(1.50), volume: 800, open_interest: 4000,
            bid_iv: 0.19, mid_iv: 0.20, ask_iv: 0.21, smv_vol: 0.20,
            greeks: Greeks { delta: -0.10, gamma: 0.015, theta: -0.03, vega: 0.08, rho: 0.008 },
            residual_rate: 0.05,
        });

        snapshot.chains.push(chain);
        snapshot
    }

    #[test]
    fn test_simulated_broker_fills_spread() {
        let mut broker = SimulatedBroker::new(
            SlippageModel::default(),
            CommissionModel::default(),
        );
        let snapshot = make_snapshot_with_quotes();

        let order = OrderIntent::enter_spread(
            "SPY".to_string(),
            SpreadType::PutCreditSpread,
            vec![
                LegIntent {
                    option_type: OptionType::Put, strike: dec!(470),
                    expiration: NaiveDate::from_ymd_opt(2024, 2, 16).unwrap(),
                    contracts: -1, side: OrderSide::Sell,
                },
                LegIntent {
                    option_type: OptionType::Put, strike: dec!(465),
                    expiration: NaiveDate::from_ymd_opt(2024, 2, 16).unwrap(),
                    contracts: 1, side: OrderSide::Buy,
                },
            ],
        );

        let fill = broker.submit_order(&order, &snapshot).unwrap();
        assert_eq!(fill.fill_prices.len(), 2);

        let sell_fill = fill.fill_prices[0];
        assert!(sell_fill >= dec!(2.40) && sell_fill <= dec!(2.60),
            "Sell fill {} should be between bid 2.40 and ask 2.60", sell_fill);

        let buy_fill = fill.fill_prices[1];
        assert!(buy_fill >= dec!(1.40) && buy_fill <= dec!(1.60),
            "Buy fill {} should be between bid 1.40 and ask 1.60", buy_fill);

        assert_eq!(fill.commission, dec!(2));
    }

    #[test]
    fn test_simulated_broker_missing_quote_returns_error() {
        let mut broker = SimulatedBroker::new(
            SlippageModel::default(),
            CommissionModel::default(),
        );
        let snapshot = make_snapshot_with_quotes();

        let order = OrderIntent::enter_spread(
            "SPY".to_string(),
            SpreadType::PutCreditSpread,
            vec![
                LegIntent {
                    option_type: OptionType::Put, strike: dec!(999),
                    expiration: NaiveDate::from_ymd_opt(2024, 2, 16).unwrap(),
                    contracts: -1, side: OrderSide::Sell,
                },
            ],
        );

        let result = broker.submit_order(&order, &snapshot);
        assert!(result.is_err());
    }
}
