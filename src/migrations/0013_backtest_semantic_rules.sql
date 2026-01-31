-- Backtest Semantic Rules Migration
-- Validated rules from walk-forward analysis (2007-2025, QQQ)
-- Source: mahler-backtest/analysis/walkforward_findings_2026-01-30.log

-- Rule 1: IV Filter Removal (Highest Impact: +59% CAGR)
INSERT OR IGNORE INTO semantic_rules (id, rule_text, rule_type, source, applies_to_agent, supporting_trades, p_value, effect_size, is_active)
VALUES ('bt_rule_001', 'Trade in all IV environments, not just elevated IV (>50th percentile). Removing IV filter increases CAGR by 59% while improving Sharpe ratio.', 'entry', 'initial', 'all', 2215, 0.001, 0.59, 1);

-- Rule 2: Delta Selection (Higher Win Rate)
INSERT OR IGNORE INTO semantic_rules (id, rule_text, rule_type, source, applies_to_agent, supporting_trades, p_value, effect_size, is_active)
VALUES ('bt_rule_002', 'Prefer delta 0.05-0.15 over 0.20-0.30 for higher win rate (69.9% vs 67.9%) and better profit factor (6.10 vs 3.16). More OTM deltas are also more robust to slippage.', 'entry', 'initial', 'all', 1626, 0.01, 0.10, 1);

-- Rule 3: Slippage Awareness (Risk Factor)
INSERT OR IGNORE INTO semantic_rules (id, rule_text, rule_type, source, applies_to_agent, supporting_trades, p_value, effect_size, is_active)
VALUES ('bt_rule_003', 'Strategy edge degrades rapidly above 80% slippage. Monitor fill quality - at 75% slippage CAGR drops to 4.97%, at 100% strategy loses money.', 'sizing', 'initial', 'all', 2215, 0.001, -0.52, 1);

-- Rule 4: Stress Period Behavior (Regime Insight)
INSERT OR IGNORE INTO semantic_rules (id, rule_text, rule_type, source, applies_to_agent, supporting_trades, p_value, effect_size, is_active)
VALUES ('bt_rule_004', 'Strategy thrives in high-IV bear markets. 2022 bear market: +22.97% return with 89.7% win rate and profit factor of 52.05.', 'regime', 'initial', 'all', 114, 0.001, 0.23, 1);
