# Mahler

**AI-Assisted Options Trading System**

---

## Product Overview

Mahler is a human-in-the-loop options trading system that uses AI for market analysis and trade recommendations while requiring explicit human approval for all executions. The system targets swing trading with 30-45 DTE credit spreads, optimizing for a solo operator with minimal infrastructure costs.

---

## Problem Statement

Retail options traders face information asymmetry against institutional players who deploy sophisticated analysis tools. Manual analysis is time-consuming, emotionally biased, and inconsistent. Fully autonomous trading systems conflict with regulatory guidance and introduce unacceptable risk for individual accounts. There is a gap for intelligent analysis tools that augment human decision-making without removing accountability.

---

## Target User

Solo developer/trader seeking systematic options income through defined-risk strategies. Comfortable with technology but not necessarily trading expertise. Willing to invest time in system development and learning. Account size: $10,000-$50,000. Time commitment: 15-30 minutes daily for trade review and approval.

---

## Core Objectives

1. **Analysis Automation**: Scan markets, calculate Greeks, identify high-probability credit spread opportunities without manual screening.

2. **Human-in-the-Loop Execution**: Surface actionable recommendations via Discord notifications; require explicit approval before any order placement.

3. **Risk Enforcement**: Implement hard-coded position sizing, drawdown limits, and circuit breakers that cannot be overridden by the AI or user in the moment.

4. **Self-Improvement**: Learn from trade outcomes through reflection patterns and evolving playbook, improving recommendation quality over time.

5. **Paper Trading Validation**: Achieve 55% win rate with >1.5 profit factor across 100+ paper trades before live deployment.

---

## Strategy Specification

### Primary Strategy: Credit Spreads

| Parameter | Specification |
|-----------|---------------|
| Instruments | SPY, QQQ, IWM (high-liquidity ETFs) |
| Structure | Bull put spreads (bullish/neutral), Bear call spreads (bearish/neutral) |
| Days to Expiration | 30-45 DTE (sweet spot for theta decay) |
| Short Strike Delta | 0.20-0.30 (70-80% probability OTM) |
| Long Strike Delta | 0.10-0.15 (defines max loss) |
| Entry Trigger | IV Rank >= 50 (preferably >= 70) |
| Profit Target | 50% of maximum credit received |
| Stop Loss | 200% of credit received |
| Time Exit | Close at 21 DTE regardless of P/L |
| Trading Cadence | 1-3 trades per week |

---

## Functional Requirements

### Market Analysis Engine

- Fetch real-time and delayed options chain data for target underlyings
- Calculate IV Rank using 52-week IV history
- Compute Greeks (delta, gamma, theta, vega) for all candidate strikes
- Screen for spreads matching entry criteria
- Rank opportunities by risk-adjusted expected value

### AI Analysis Layer

- Generate natural language trade thesis for each recommendation
- Assess macro context (earnings, Fed events, VIX regime)
- Provide confidence score (low/medium/high) with reasoning
- Flag potential risks and contrary indicators

### Notification and Approval System

- Send trade recommendations via Discord with approve/reject buttons
- Include full trade details: strikes, premium, max profit/loss, Greeks
- Set approval expiration (15 minutes during market hours)
- Log all recommendations and decisions for analysis

### Order Execution

- Place limit orders at mid-price with configurable offset
- Monitor fill status with timeout and price adjustment logic
- Confirm execution via notification
- Support paper trading mode for validation

### Position Management

- Track all open positions with real-time P/L
- Monitor exit conditions (profit target, stop loss, time exit)
- Generate exit recommendations with approval flow
- Calculate portfolio Greeks and total exposure

### Learning and Reflection

- Generate post-trade reflection after each closed position
- Identify what worked, what didn't, and lessons learned
- Store reflections for retrieval during future analysis
- Update strategy playbook based on accumulated learnings

---

## Risk Management Requirements

| Rule | Specification |
|------|---------------|
| Position Sizing | Maximum 2% account risk per trade |
| Single Position | Maximum 5% of account in one position |
| Portfolio Heat | Maximum 10% total open risk |
| Daily Loss Limit | Halt new trades after 2% daily drawdown |
| Weekly Loss Limit | Require manual review after 5% weekly drawdown |
| Maximum Drawdown | Close all positions and disable trading at 15% drawdown |
| Stale Data | Halt on no quote update >10 seconds |
| API Errors | Halt on 5+ errors per minute |
| Rapid Loss | Halt on 1% loss in <5 minutes |
| High Volatility | Reduce position sizes 75% when VIX > 40 |
| Extreme Volatility | Halt new trades when VIX > 50 |

---

## Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Reliability | 99.5% uptime during market hours (9:30 AM - 4:00 PM ET) |
| Latency | <5 second analysis cycle; <1 second order submission |
| Infrastructure Cost | <$25/month for compute and data |
| Auditability | Complete trade log with timestamps retained 3+ years |
| Security | API keys encrypted, no sensitive data in logs |

---

## Out of Scope

- Web or mobile frontend (Discord notifications only)
- Multi-user support
- Day trading or intraday strategies
- Complex multi-leg strategies (iron condors, butterflies) in v1
- Cryptocurrency or futures trading
- Fully autonomous execution without approval

---

## Success Criteria

### Paper Trading Phase

- 100+ completed trades across 3-6 months
- Win rate >= 55%
- Profit factor >= 1.5
- Maximum drawdown < 15%
- Tested through at least one VIX > 30 event

### Live Trading Phase

- Consistent monthly returns (positive 8+ months per year)
- Sharpe ratio > 1.0
- Zero unintended trades or execution errors
- Operator time commitment < 30 minutes daily

---

## Brokerage Selection

| Phase | Broker | Rationale |
|-------|--------|-----------|
| Paper Trading | Alpaca | Free, paper trading default, good API |
| Live Trading | Tastytrade | Options-first API, $1/contract open, $0 close |
