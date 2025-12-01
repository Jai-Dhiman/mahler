# Trader Jim

**AI-Assisted Options Trading System**

---

## Product Overview

Trader Jim is a human-in-the-loop options trading system that uses AI for market analysis and trade recommendations while requiring explicit human approval for all executions. The system targets swing trading with 30-45 DTE credit spreads, optimizing for a solo operator with minimal infrastructure costs.

---

## Problem Statement

Retail options traders face information asymmetry against institutional players who deploy sophisticated analysis tools. Manual analysis is time-consuming, emotionally biased, and inconsistent. Fully autonomous trading systems conflict with regulatory guidance and introduce unacceptable risk for individual accounts. There is a gap for intelligent analysis tools that augment human decision-making without removing accountability.

---

## Target User

Solo developer/trader seeking systematic options income through defined-risk strategies. Comfortable with technology but not necessarily trading expertise. Willing to invest time in system development and learning. Account size: $10,000–$50,000. Time commitment: 15–30 minutes daily for trade review and approval.

---

## Core Objectives

1. **Analysis Automation**: Scan markets, calculate Greeks, identify high-probability credit spread opportunities without manual screening.

2. **Human-in-the-Loop Execution**: Surface actionable recommendations via notifications; require explicit approval before any order placement.

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
| Days to Expiration | 30–45 DTE (sweet spot for theta decay) |
| Short Strike Delta | 0.20–0.30 (70–80% probability OTM) |
| Long Strike Delta | 0.10–0.15 (defines max loss) |
| Entry Trigger | IV Rank ≥ 50 (preferably ≥ 70) |
| Profit Target | 50% of maximum credit received |
| Stop Loss | 200% of credit received |
| Time Exit | Close at 21 DTE regardless of P/L |
| Trading Cadence | 1–3 trades per week, swing trading |

---

## Functional Requirements

### Market Analysis Engine

1. Fetch real-time and delayed options chain data for target underlyings
2. Calculate IV Rank using 52-week IV history
3. Compute Greeks (delta, gamma, theta, vega) for all candidate strikes
4. Screen for spreads matching entry criteria
5. Rank opportunities by risk-adjusted expected value

### AI Analysis Layer

1. Generate natural language trade thesis for each recommendation
2. Assess macro context (earnings, Fed events, VIX regime)
3. Provide confidence score (low/medium/high) with reasoning
4. Flag potential risks and contrary indicators

### Notification & Approval System

1. Send trade recommendations via Slack with one-click approve/reject
2. Include full trade details: strikes, premium, max profit/loss, Greeks
3. Set approval expiration (e.g., 15 minutes during market hours)
4. Log all recommendations and decisions for analysis

### Order Execution

1. Place limit orders at mid-price with configurable offset
2. Monitor fill status with timeout and price adjustment logic
3. Confirm execution via notification
4. Support paper trading mode for validation

### Position Management

1. Track all open positions with real-time P/L
2. Monitor exit conditions (profit target, stop loss, time exit)
3. Generate exit recommendations with approval flow
4. Calculate portfolio Greeks and total exposure

### Learning & Reflection

1. Generate post-trade reflection after each closed position
2. Identify what worked, what didn't, and lessons learned
3. Store reflections in episodic memory for retrieval
4. Update strategy playbook based on accumulated learnings
5. Generate weekly/monthly performance summaries

---

## Risk Management Requirements

1. **Position Sizing**: Maximum 2% account risk per trade, 5% maximum single position, 10% maximum portfolio heat

2. **Daily Loss Limit**: Halt new trades after 2% daily drawdown

3. **Weekly Loss Limit**: Require manual review after 5% weekly drawdown

4. **Maximum Drawdown**: Close all positions and disable trading at 15% drawdown

5. **Kill Switches**: Automatic halt on stale data (>10s), API errors (5+ per minute), rapid loss (1% in <5 min)

6. **Volatility Adjustment**: Reduce position sizes 75% when VIX > 40

---

## Non-Functional Requirements

1. **Reliability**: 99.5% uptime during market hours (9:30 AM – 4:00 PM ET)

2. **Latency**: <5 second analysis cycle; <1 second order submission

3. **Infrastructure Cost**: <$20/month for compute and data

4. **Auditability**: Complete trade log with timestamps retained 3+ years

5. **Security**: API keys encrypted at rest, no sensitive data in logs

---

## Out of Scope

- Web or mobile frontend (notifications only)
- Multi-user support
- Day trading or intraday strategies
- Complex multi-leg strategies (iron condors, butterflies) in v1
- Cryptocurrency or futures trading
- Fully autonomous execution without approval

---

## Success Criteria

### Paper Trading Phase

- 100+ completed trades across 3–6 months
- Win rate ≥ 55%
- Profit factor ≥ 1.5
- Maximum drawdown < 15%
- Tested through at least one VIX > 30 event

### Live Trading Phase

- Consistent monthly returns (positive 8+ months per year)
- Sharpe ratio > 1.0
- Zero unintended trades or execution errors
- Operator time commitment < 30 minutes daily

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1–4)

1. Set up Rust project with core dependencies
2. Implement Alpaca API client for paper trading
3. Build basic credit spread screening logic
4. Deploy to Hetzner VPS with systemd timer

### Phase 2: Risk Infrastructure (Months 2–3)

1. Implement position sizing and portfolio heat tracking
2. Add circuit breakers and kill switches
3. Build trade logging and audit trail
4. Set up Slack notifications with approval flow

### Phase 3: AI Integration (Months 3–6)

1. Integrate Claude API for trade analysis
2. Implement reflection system for post-trade learning
3. Build episodic memory and playbook evolution
4. Paper trade 100+ positions; validate success criteria

### Phase 4: Live Deployment (Month 6+)

1. Transition to Tastytrade for production trading
2. Deploy with 25% of intended capital
3. Scale capital as performance demonstrates consistency
4. Continuous improvement based on live trading data

---

## Brokerage Selection

| Phase | Broker | Rationale |
|-------|--------|-----------|
| Paper Trading | Alpaca | $0 commission, paper trading default, excellent docs |
| Live Trading | Tastytrade | Options-first API, $1/contract open, $0 close |
