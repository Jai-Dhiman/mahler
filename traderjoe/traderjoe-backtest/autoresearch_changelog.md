# Autoresearch Changelog — SPY Put Credit Spread Optimization

Goal: Maximize avg out-of-sample (test) Sharpe ratio, walk-forward 2007-2023
Metric: "Out-of-Sample Test Sharpe" from optimize command
Guard: test_sharpe > -1.0, trades >= 20, win_rate >= 30%, sharpe_degradation < 80%

## Iteration 0 — BASELINE
Config: profit_target=50%, stop_loss=125%, dte=30-60, delta=0.20-0.30, iv=off (0.0)
