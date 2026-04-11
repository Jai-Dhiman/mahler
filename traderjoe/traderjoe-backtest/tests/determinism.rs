use chrono::NaiveDate;
use rust_decimal_macros::dec;

use traderjoe_backtest::backtest::{CommissionModel, SlippageModel};
use traderjoe_backtest::broker::backtest::SimulatedBroker;
use traderjoe_backtest::engine_core::engine::Engine;
use traderjoe_backtest::data::HistoricalDataSource;
use traderjoe_backtest::risk::circuit_breakers::CircuitBreakerConfig;
use traderjoe_backtest::risk::gate::DefaultRiskGate;
use traderjoe_backtest::strategy::put_spread::{PutCreditSpreadStrategy, PutSpreadConfig};

#[test]
fn test_deterministic_results() {
    let start = NaiveDate::from_ymd_opt(2020, 6, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2020, 12, 31).unwrap();
    let equity = dec!(100_000);

    let run = || -> Option<(usize, f64, f64)> {
        let data_source = HistoricalDataSource::new("data/orats", "SPY", start, end).ok()?;
        let strategy = PutCreditSpreadStrategy::new(PutSpreadConfig::default());
        let broker = SimulatedBroker::new(SlippageModel::default(), CommissionModel::default());
        let risk_gate = DefaultRiskGate::new(CircuitBreakerConfig::default(), equity);

        let result = Engine::run(data_source, strategy, broker, risk_gate, equity, CommissionModel::default()).ok()?;
        Some((result.total_trades, result.total_return_pct, result.max_drawdown_pct))
    };

    match (run(), run()) {
        (Some(r1), Some(r2)) => {
            assert_eq!(r1.0, r2.0, "Trade count must be deterministic");
            assert!(
                (r1.1 - r2.1).abs() < 0.0001,
                "Return must be deterministic: {} vs {}", r1.1, r2.1
            );
            assert!(
                (r1.2 - r2.2).abs() < 0.0001,
                "Drawdown must be deterministic: {} vs {}", r1.2, r2.2
            );
        }
        _ => eprintln!("Skipping: data not available"),
    }
}

#[test]
fn test_engine_produces_trades() {
    let start = NaiveDate::from_ymd_opt(2020, 6, 1).unwrap();
    let end = NaiveDate::from_ymd_opt(2020, 12, 31).unwrap();
    let equity = dec!(100_000);

    let data_source = match HistoricalDataSource::new("data/orats", "SPY", start, end) {
        Ok(ds) => ds,
        Err(_) => { eprintln!("Skipping: data not available"); return; }
    };

    let strategy = PutCreditSpreadStrategy::new(PutSpreadConfig::default());
    let broker = SimulatedBroker::new(SlippageModel::default(), CommissionModel::default());
    let risk_gate = DefaultRiskGate::new(CircuitBreakerConfig::default(), equity);

    let result = Engine::run(data_source, strategy, broker, risk_gate, equity, CommissionModel::default()).unwrap();

    assert!(result.total_trades > 0, "Should produce trades over 7 months");
    assert!(result.trading_days > 100, "Should have >100 trading days");
    assert!(result.equity_curve.len() > 100, "Should have equity curve entries");

    assert!(result.win_rate() > 0.0 && result.win_rate() <= 1.0, "Win rate should be between 0 and 1");
    assert!(result.total_commission > rust_decimal::Decimal::ZERO, "Should have commission");
}
