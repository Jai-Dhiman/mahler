use rust_decimal::Decimal;
use tracing::info;

use crate::backtest::{BacktestConfig, BacktestResult, CommissionModel};
use crate::broker::Broker;
use crate::data::DataSource;
use crate::portfolio::PortfolioTracker;
use crate::risk::gate::RiskGate;
use crate::strategy::Strategy;

use super::events::OrderAction;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Data error: {0}")]
    Data(String),

    #[error("Broker error: {0}")]
    Broker(#[from] crate::broker::BrokerError),
}

pub struct Engine;

impl Engine {
    pub fn run<D, S, B, R>(
        mut data_source: D,
        mut strategy: S,
        mut broker: B,
        mut risk_gate: R,
        initial_equity: Decimal,
        commission: CommissionModel,
    ) -> Result<BacktestResult, EngineError>
    where
        D: DataSource,
        S: Strategy,
        B: Broker,
        R: RiskGate,
    {
        let mut portfolio = PortfolioTracker::new(initial_equity);
        let mut start_date = None;
        let mut end_date = None;
        let mut ticker = String::new();

        while let Some(event) = data_source.next_event() {
            let snapshot = &event.snapshot;

            if start_date.is_none() {
                start_date = Some(snapshot.date);
                ticker = snapshot.ticker.clone();
            }
            end_date = Some(snapshot.date);

            // 1. Update MTM for open positions
            portfolio.update_mtm(snapshot);

            // 2. Update risk gate with current state
            let view = portfolio.view();
            risk_gate.update(snapshot, &view);

            // 3. Get orders from strategy
            let orders = strategy.on_snapshot(snapshot, &view);

            // 4. Process each order through risk gate and broker
            for order in orders {
                let current_view = portfolio.view();

                match &order.action {
                    OrderAction::Close { position_id, reason } => {
                        let position_id = *position_id;
                        let reason = *reason;
                        if let Some(pos) = current_view.open_positions.iter().find(|p| p.id == position_id) {
                            let exit_commission = commission.calculate(
                                pos.total_contracts(),
                                pos.num_legs(),
                            ).total;
                            portfolio.close_position(
                                position_id,
                                snapshot.date,
                                reason,
                                pos.current_value,
                                exit_commission,
                            );
                        }
                    }
                    OrderAction::Open => {
                        let decision = risk_gate.check(&order, &current_view);
                        if !decision.is_allowed() {
                            continue;
                        }

                        match broker.submit_order(&order, snapshot) {
                            Ok(fill) => {
                                let position = build_position_from_fill(&order, &fill, snapshot);
                                portfolio.add_position(position);
                                strategy.on_fill(&fill);
                            }
                            Err(e) => {
                                info!("Broker error: {}", e);
                            }
                        }
                    }
                }
            }

            // 5. Cleanup and record
            portfolio.cleanup_closed();
            portfolio.record_equity(snapshot.date);
        }

        // Close remaining positions at end of backtest with correct per-position commission
        if let Some(end) = end_date {
            let open_positions: Vec<(u64, i32, usize, Decimal)> = portfolio.view().open_positions.iter()
                .map(|p| (p.id, p.total_contracts(), p.num_legs(), p.current_value))
                .collect();
            for (id, contracts, legs, current_value) in open_positions {
                let exit_commission = commission.calculate(contracts, legs).total;
                portfolio.close_position(
                    id,
                    end,
                    crate::backtest::ExitReason::EndOfPeriod,
                    current_value,
                    exit_commission,
                );
            }
        }

        let sd = start_date.unwrap_or_default();
        let ed = end_date.unwrap_or_default();

        Ok(build_result(&portfolio, vec![ticker], sd, ed, initial_equity))
    }
}

fn build_position_from_fill(
    order: &super::events::OrderIntent,
    fill: &super::events::FillEvent,
    snapshot: &crate::data::OptionsSnapshot,
) -> crate::backtest::Position {
    use crate::backtest::CreditSpreadBuilder;
    use super::events::OrderSide;

    let short_idx = order.legs.iter().position(|l| l.side == OrderSide::Sell).unwrap_or(0);
    let long_idx = order.legs.iter().position(|l| l.side == OrderSide::Buy).unwrap_or(1);

    let short_leg = &order.legs[short_idx];
    let long_leg = &order.legs[long_idx];

    let short_fill = fill.fill_prices[short_idx];
    let long_fill = fill.fill_prices[long_idx];

    let dte = snapshot.chains.iter()
        .find(|c| c.expiration == short_leg.expiration)
        .map(|c| c.dte)
        .unwrap_or(0);

    let short_delta = snapshot.chains.iter()
        .flat_map(|c| c.puts.iter().chain(c.calls.iter()))
        .find(|q| q.strike == short_leg.strike && q.expiration == short_leg.expiration)
        .map(|q| q.greeks.delta)
        .unwrap_or(0.0);

    let long_delta = snapshot.chains.iter()
        .flat_map(|c| c.puts.iter().chain(c.calls.iter()))
        .find(|q| q.strike == long_leg.strike && q.expiration == long_leg.expiration)
        .map(|q| q.greeks.delta)
        .unwrap_or(0.0);

    // TODO: support additional spread types when second strategy is added (Phase 2)
    CreditSpreadBuilder::put_credit_spread(&order.ticker, snapshot.date)
        .stock_price(snapshot.underlying_price)
        .short_leg(short_leg.strike, short_fill, short_delta)
        .long_leg(long_leg.strike, long_fill, long_delta)
        .expiration(short_leg.expiration, dte)
        .contracts(short_leg.contracts.abs())
        .commission(fill.commission)
        .build()
}

fn build_result(
    portfolio: &PortfolioTracker,
    tickers: Vec<String>,
    start_date: chrono::NaiveDate,
    end_date: chrono::NaiveDate,
    initial_equity: Decimal,
) -> BacktestResult {
    let equity = portfolio.equity();
    let initial_f64: f64 = initial_equity.try_into().unwrap_or(1.0);
    let final_f64: f64 = equity.try_into().unwrap_or(1.0);
    let total_return_pct = (final_f64 - initial_f64) / initial_f64 * 100.0;

    let peak: f64 = portfolio.peak_equity().try_into().unwrap_or(1.0);
    let max_drawdown = portfolio.peak_equity() - equity;
    let max_dd_f64: f64 = max_drawdown.try_into().unwrap_or(0.0);
    let max_drawdown_pct = if peak > 0.0 { max_dd_f64 / peak * 100.0 } else { 0.0 };

    let trades = portfolio.closed_trades();
    let winning_trades = trades.iter().filter(|t| t.is_winner()).count();
    let losing_trades = trades.len() - winning_trades;
    let total_pnl: Decimal = trades.iter().map(|t| t.pnl()).sum();
    let gross_profit: Decimal = trades.iter().filter(|t| t.is_winner()).map(|t| t.pnl()).sum();
    let gross_loss: Decimal = trades.iter().filter(|t| !t.is_winner()).map(|t| t.pnl()).sum();

    BacktestResult {
        config: BacktestConfig::default(),
        tickers,
        start_date,
        end_date,
        trades: trades.to_vec(),
        equity_curve: portfolio.equity_curve().to_vec(),
        final_equity: equity,
        total_return_pct,
        trading_days: portfolio.equity_curve().len(),
        peak_equity: portfolio.peak_equity(),
        max_drawdown,
        max_drawdown_pct,
        total_trades: trades.len(),
        winning_trades,
        losing_trades,
        total_pnl,
        gross_profit,
        gross_loss,
        total_commission: portfolio.total_commission(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;
    use crate::backtest::{CommissionModel, SlippageModel};
    use crate::broker::backtest::SimulatedBroker;
    use crate::data::HistoricalDataSource;
    use crate::risk::circuit_breakers::CircuitBreakerConfig;
    use crate::risk::gate::DefaultRiskGate;
    use crate::strategy::NoOpStrategy;
    use crate::strategy::put_spread::{PutCreditSpreadStrategy, PutSpreadConfig};

    #[test]
    fn test_engine_runs_with_noop_strategy() {
        let source = HistoricalDataSource::new(
            "data/orats",
            "SPY",
            NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 31).unwrap(),
        );

        match source {
            Ok(data_source) => {
                let strategy = NoOpStrategy;
                let broker = SimulatedBroker::new(SlippageModel::default(), CommissionModel::default());
                let risk_gate = DefaultRiskGate::new(CircuitBreakerConfig::default(), dec!(100_000));

                let result = Engine::run(
                    data_source,
                    strategy,
                    broker,
                    risk_gate,
                    dec!(100_000),
                    CommissionModel::default(),
                );

                assert!(result.is_ok());
                let result = result.unwrap();
                assert_eq!(result.total_trades, 0);
                assert_eq!(result.final_equity, dec!(100_000));
                assert!(result.trading_days > 0);
            }
            Err(_) => eprintln!("Skipping: data not available"),
        }
    }

    #[test]
    fn test_engine_runs_with_put_spread_strategy() {
        let source = HistoricalDataSource::new(
            "data/orats",
            "SPY",
            NaiveDate::from_ymd_opt(2020, 6, 1).unwrap(),
            NaiveDate::from_ymd_opt(2020, 12, 31).unwrap(),
        );

        match source {
            Ok(data_source) => {
                let config = PutSpreadConfig::default();
                let strategy = PutCreditSpreadStrategy::new(config);
                let broker = SimulatedBroker::new(SlippageModel::default(), CommissionModel::default());
                let risk_gate = DefaultRiskGate::new(CircuitBreakerConfig::default(), dec!(100_000));

                let result = Engine::run(
                    data_source,
                    strategy,
                    broker,
                    risk_gate,
                    dec!(100_000),
                    CommissionModel::default(),
                );

                assert!(result.is_ok());
                let result = result.unwrap();
                assert!(result.total_trades > 0, "Expected trades, got 0");
                assert!(result.trading_days > 100);
            }
            Err(_) => eprintln!("Skipping: data not available"),
        }
    }
}
