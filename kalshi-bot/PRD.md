# Product Requirements Document: Kalshi Informed Market Making System

**Project Codename:** Mahler-PM (Prediction Markets)  
**Author:** Jai  
**Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Draft

---

## Executive Summary

This document defines the requirements for an automated trading system targeting steady, index-fund-comparable returns (10-20% annually) on Kalshi prediction markets. The system combines market making with ML-based probability estimation to generate edge in niche, lower-volume markets where high-frequency trading competition is structurally limited.

### Key Strategic Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Platform | Kalshi only | Legal for US citizens; ~1.75% fees reduce HFT competition |
| Primary Strategy | Informed market making | Combines reliable spread capture with ML probability edge |
| Core Technology | Rust execution + Python ML | Reliability and efficiency for trading; ecosystem maturity for ML |
| Target Markets | Computable outcomes (economic, weather, sports stats) | Probability models can achieve genuine edge |
| Capital Range | $5,000 - $25,000 | Minimum viable for infrastructure costs; max before complexity increases |

---

## Problem Statement

### The Opportunity

Prediction markets are less efficient than traditional financial markets, particularly in niche segments. Research indicates:

- 86% of Polymarket traders lose money, suggesting inefficiencies exist but are captured by sophisticated participants
- Market making on prediction markets generated $200-800/day for early automated participants
- Computable markets (economic indicators, weather, sports statistics) have measurable true probabilities that often diverge from market prices

### The Challenge

Naive approaches fail because:

1. **Speed-based strategies are competed away** — HFT bots with dedicated infrastructure capture arbitrage opportunities in milliseconds
2. **Simple market making is capital-intensive** — Requires $10K+ to overcome transaction costs and inventory risk
3. **Pure ML prediction is insufficient** — Markets already aggregate information; beating collective wisdom requires domain expertise
4. **Regulatory complexity for US citizens** — Polymarket is legally inaccessible; Kalshi has state-level challenges

### The Hypothesis

A hybrid approach combining:

- **Market making** for consistent spread capture and liquidity rewards
- **ML probability estimation** to tilt quotes toward true value
- **Niche market focus** where competition is lower

...can achieve 10-20% annual returns with acceptable risk, while providing a meaningful Rust learning project.

---

## Goals and Success Metrics

### Primary Goal

Achieve consistent, positive risk-adjusted returns comparable to or exceeding passive index fund investing, while building expertise in Rust systems programming and quantitative trading.

### Success Metrics

| Metric | Target | Measurement Period | Rationale |
|--------|--------|-------------------|-----------|
| Net Annual Return | 12-18% | Rolling 12 months | Exceeds savings (5%) and approaches index (10%) |
| Sharpe Ratio | > 1.0 | Rolling 3 months | Risk-adjusted return quality |
| Maximum Drawdown | < 20% | Any period | Capital preservation |
| Win Rate | > 55% | Per trade | Edge validation |
| System Uptime | > 99% | Monthly | Reliability for continuous operation |
| Model Calibration Error | < 5% | Monthly | Probability estimates are accurate |

### Non-Goals

- **Competing on latency** — We accept 50-200ms execution; sub-millisecond is out of scope
- **High-frequency trading** — We target 10-100 trades/day, not thousands
- **Polymarket access** — Legal risk is unacceptable; revisit if US platform launches
- **Full automation without oversight** — Human review of risk metrics remains required
- **Maximizing absolute returns** — Risk management takes priority over return optimization

---

## User Personas

### Primary User: Solo Operator (Jai)

**Background:**

- ML specialist with founding engineer experience
- Comfortable with PyTorch, JAX/Flax, recommendation systems
- Learning Rust; wants production system as learning vehicle
- Based in US (San Francisco); regulatory compliance required

**Goals:**

- Generate steady returns (10-20% annually) on $5-25K capital
- Learn Rust through building real production system
- Minimize time commitment after initial build (target: 2-5 hrs/week maintenance)
- Understand quantitative trading mechanics

**Constraints:**

- Cannot use Polymarket (US citizen)
- Limited to evenings/weekends for development
- No dedicated trading infrastructure budget (< $100/mo)
- Risk tolerance: can lose 50% of capital without financial hardship

**Pain Points:**

- Skeptical of "get rich quick" Twitter claims
- Wants honest assessment of realistic returns
- Values learning over pure profit
- Needs system that runs reliably without constant attention

---

## Product Requirements

### Functional Requirements

#### FR-1: Market Data Ingestion

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-1.1 | System shall maintain real-time orderbook state for all active markets | P0 | WebSocket-based updates |
| FR-1.2 | System shall persist tick-level data to time-series database | P1 | Required for backtesting and model training |
| FR-1.3 | System shall ingest external data sources (BLS, TSA, weather APIs) | P1 | Features for probability models |
| FR-1.4 | System shall handle API rate limits gracefully with backoff | P0 | Kalshi rate limits vary by tier |
| FR-1.5 | System shall recover automatically from connection failures | P0 | 24/7 operation requirement |

#### FR-2: Probability Estimation

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-2.1 | System shall generate calibrated probability estimates for target markets | P0 | Core edge source |
| FR-2.2 | System shall provide confidence intervals for all estimates | P0 | Used for spread calculation |
| FR-2.3 | System shall update estimates at configurable intervals (default: 5 min) | P1 | Balance freshness vs. compute cost |
| FR-2.4 | System shall support multiple model types per market category | P2 | Different models for TSA vs. weather |
| FR-2.5 | System shall track model performance and calibration over time | P1 | Detect model degradation |

#### FR-3: Quote Generation

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-3.1 | System shall generate two-sided quotes (bid and ask) for each market | P0 | Market making requirement |
| FR-3.2 | Quotes shall incorporate model probability estimates | P0 | Edge source |
| FR-3.3 | Quotes shall adjust spread based on model confidence | P0 | Wider spread = less confidence |
| FR-3.4 | Quotes shall incorporate inventory skew to manage position risk | P0 | Risk management |
| FR-3.5 | Quotes shall account for Kalshi fee structure | P0 | ~1.75% at midprice |
| FR-3.6 | System shall implement hysteresis to avoid excessive order churn | P1 | Reduce API calls, improve fills |

#### FR-4: Order Management

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-4.1 | System shall submit orders to Kalshi API with proper authentication | P0 | RSA-PSS signing |
| FR-4.2 | System shall track all open orders and their states | P0 | Required for reconciliation |
| FR-4.3 | System shall cancel and replace orders when quotes change significantly | P0 | Core trading loop |
| FR-4.4 | System shall handle partial fills correctly | P0 | Common occurrence |
| FR-4.5 | System shall implement retry logic for transient failures | P0 | Network reliability |
| FR-4.6 | System shall maintain audit log of all order actions | P1 | Debugging and compliance |

#### FR-5: Risk Management

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-5.1 | System shall enforce per-market position limits | P0 | Kalshi caps at $25K anyway |
| FR-5.2 | System shall enforce total portfolio exposure limits | P0 | Capital preservation |
| FR-5.3 | System shall enforce daily loss limits with automatic halt | P0 | Circuit breaker |
| FR-5.4 | System shall track and limit inventory imbalance per market | P0 | Prevent one-sided exposure |
| FR-5.5 | System shall identify and limit correlated market exposure | P1 | Related markets risk |
| FR-5.6 | System shall support manual kill switch | P0 | Emergency intervention |

#### FR-6: Monitoring and Alerting

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-6.1 | System shall expose real-time P&L metrics | P0 | Core monitoring |
| FR-6.2 | System shall alert on risk limit breaches | P0 | Discord/mobile notification |
| FR-6.3 | System shall alert on system health issues (disconnects, errors) | P0 | Operational awareness |
| FR-6.4 | System shall provide daily summary reports | P1 | Performance tracking |
| FR-6.5 | System shall track and display model performance metrics | P1 | Edge validation |
| FR-6.6 | System shall support Grafana dashboard integration | P2 | Visualization |

### Non-Functional Requirements

#### NFR-1: Performance

| ID | Requirement | Target | Notes |
|----|-------------|--------|-------|
| NFR-1.1 | Order submission latency | < 500ms p99 | Not HFT; reliability over speed |
| NFR-1.2 | Quote recalculation time | < 100ms | Responsive to market changes |
| NFR-1.3 | Memory usage (trading engine) | < 500MB | Efficient VPS utilization |
| NFR-1.4 | CPU usage (idle) | < 10% | Cost efficiency |
| NFR-1.5 | Markets monitored simultaneously | 50+ | Scale requirement |

#### NFR-2: Reliability

| ID | Requirement | Target | Notes |
|----|-------------|--------|-------|
| NFR-2.1 | System uptime | 99.5% | ~1.8 hrs downtime/month acceptable |
| NFR-2.2 | Automatic recovery from crashes | < 60 seconds | Supervisor/systemd |
| NFR-2.3 | Data durability | No tick data loss | Persistent storage |
| NFR-2.4 | Order state consistency | 100% | No orphaned orders |

#### NFR-3: Security

| ID | Requirement | Notes |
|----|-------------|-------|
| NFR-3.1 | API keys stored encrypted at rest | Environment variables or secrets manager |
| NFR-3.2 | Private keys never logged | Audit logging excludes sensitive data |
| NFR-3.3 | VPS access via SSH key only | No password authentication |
| NFR-3.4 | Regular security updates | Automated or scheduled |

#### NFR-4: Operability

| ID | Requirement | Notes |
|----|-------------|-------|
| NFR-4.1 | Configuration via files, not code changes | YAML/TOML config |
| NFR-4.2 | Log levels configurable at runtime | Debug capability without restart |
| NFR-4.3 | Graceful shutdown preserving state | No order state corruption |
| NFR-4.4 | Deployment via single command | Docker or simple script |

---

## Market Selection Criteria

### Target Market Characteristics

The system focuses on markets with "computable" outcomes where probability models can achieve genuine edge:

| Market Type | Examples | Why Tractable | Data Sources |
|-------------|----------|---------------|--------------|
| Economic Indicators | CPI, jobs report, Fed rate | Historical patterns, leading indicators | BLS, Fed, economic calendars |
| Transportation Metrics | TSA passenger volume | Seasonal patterns, holiday effects | TSA daily data, fuel prices |
| Weather Events | Temperature thresholds, precipitation | Ensemble weather models | NOAA, Weather.gov APIs |
| Sports Statistics | Over/under totals, player props | Historical stats, recent form | Sports reference sites, APIs |
| Scheduled Events | Earnings surprises, launch dates | Analyst estimates, historical hit rates | SEC filings, company data |

### Market Selection Algorithm

Markets are ranked by a composite score:

```
Score = (Model_Edge × Confidence) / (Spread + Fee_Drag) × Liquidity_Factor
```

Where:

- **Model_Edge:** Absolute difference between model probability and market price
- **Confidence:** Model's calibrated confidence (0-1)
- **Spread:** Current bid-ask spread in the market
- **Fee_Drag:** Expected Kalshi fees for round-trip
- **Liquidity_Factor:** Discount for thin markets (harder to fill)

### Markets to Avoid

| Market Type | Reason |
|-------------|--------|
| Breaking news dependent | Cannot model; information arrives unpredictably |
| Political sentiment | Prices driven by emotion, not computable probability |
| Long-dated (> 3 months) | Capital locked; hard to model |
| Very low volume (< $1K/day) | Cannot execute meaningful size |
| Resolution ambiguity | Dispute risk |

---

## Constraints and Assumptions

### Constraints

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| US citizenship | Cannot use Polymarket legally | Kalshi-only focus |
| Kalshi state restrictions | Some states may restrict access | Monitor regulatory developments; diversify market types |
| $25K position limits | Cannot scale single-market positions indefinitely | Diversify across markets |
| API rate limits | Cannot poll infinitely fast | Efficient WebSocket usage; respect tiers |
| Capital: $5-25K | Returns capped by capital size | Focus on percentage returns, not absolute |
| Time: 5-10 hrs/week ongoing | Cannot actively manage positions | Automation is essential |

### Assumptions

| Assumption | If Wrong... |
|------------|-------------|
| Kalshi remains operational and legal | Project becomes non-viable; pivot to paper trading/research |
| ML models can achieve > 5% edge on some markets | Pure market making returns may be insufficient |
| Niche markets remain less competitive | Returns compress; may need to find new niches |
| Infrastructure costs stay < $100/mo | ROI calculation changes; may need more capital |
| Rust learning curve is manageable | Timeline extends; consider Python fallback for trading engine |

---

## Risk Register

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model overfitting | High | Medium | Out-of-sample testing, regular recalibration |
| API breaking changes | Medium | High | Version pinning, monitor Kalshi announcements |
| Infrastructure failure | Low | High | Health checks, auto-restart, alerts |
| Data feed gaps | Medium | Medium | Fallback to REST polling, gap detection |
| Order state desync | Low | High | Periodic reconciliation, conservative position tracking |

### Market Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Adverse selection | High | Medium | Wider spreads, inventory limits |
| Liquidity disappears | Medium | High | Position limits, diversification |
| Correlated losses | Medium | High | Correlation tracking, exposure limits |
| Black swan event | Low | Critical | Daily loss limits, automatic halt |
| Fee structure changes | Low | Medium | Monitor announcements, adjust spreads |

### Regulatory Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Kalshi loses CFTC status | Low | Critical | No mitigation; accept as existential risk |
| State-level restrictions expand | Medium | Medium | Focus on federally-clear market types |
| Tax treatment changes | Low | Low | Maintain detailed records |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Burnout/time constraints | Medium | Medium | Minimize ongoing maintenance requirement |
| Key person dependency | High | High | Document everything; modular design |
| Credential compromise | Low | Critical | Secure storage, minimal permissions |

---

## Phased Rollout Plan

### Phase 0: Research and Validation (Complete)

- [x] Platform analysis (Polymarket vs. Kalshi)
- [x] Strategy evaluation
- [x] Realistic return expectations
- [x] Architecture design

### Phase 1: Foundation (Weeks 1-4)

**Goal:** Working data pipeline + paper trading

**Deliverables:**

- [ ] Kalshi demo API integration
- [ ] Rust data ingestion service
- [ ] TimescaleDB tick storage
- [ ] Basic quote generator (spread-based, no ML)
- [ ] Paper trading harness

**Exit Criteria:**

- System logs theoretical trades for 5+ markets
- No crashes over 48-hour test period
- Theoretical P&L tracking functional

### Phase 2: ML Integration (Weeks 5-8)

**Goal:** Add probability edge

**Deliverables:**

- [ ] Feature engineering pipeline (Python)
- [ ] Trained models for 2-3 market types
- [ ] Calibration validation
- [ ] Model output integration with quote generator
- [ ] A/B testing framework (ML vs. naive quotes)

**Exit Criteria:**

- ML-informed quotes show improvement in paper P&L
- Model calibration error < 10%
- End-to-end pipeline runs without intervention

### Phase 3: Live Trading (Weeks 9-12)

**Goal:** Validate with real capital

**Deliverables:**

- [ ] Production deployment on VPS
- [ ] Full monitoring and alerting
- [ ] Risk management enforcement
- [ ] $500-1,000 live capital deployment
- [ ] 3-5 markets active

**Exit Criteria:**

- Sharpe ratio > 0.5 over 4 weeks (lower bar for small sample)
- No risk limit breaches
- System uptime > 99%
- Positive P&L (even if small)

### Phase 4: Scale (Months 4-6)

**Goal:** Reach target returns

**Deliverables:**

- [ ] Expand to 10-20 markets
- [ ] Increase capital to $5-10K
- [ ] Add more market types
- [ ] Correlation-aware risk management
- [ ] Performance attribution reporting

**Exit Criteria:**

- Sharpe ratio > 1.0 over 3 months
- Net returns > 10% annualized
- Maximum drawdown < 20%
- Maintenance time < 5 hrs/week

---

## Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| Dec 2024 | Kalshi only (no Polymarket) | Legal risk unacceptable for US citizen | VPN access, offshore entity |
| Dec 2024 | Rust for trading engine | Learning goal + reliability + efficiency | Python, Go |
| Dec 2024 | Python for ML components | Ecosystem maturity, existing expertise | Rust ML (immature), Julia |
| Dec 2024 | Informed market making strategy | Combines spread capture with ML edge | Pure MM, pure prediction, arbitrage |
| Dec 2024 | Niche market focus | Lower competition, computable probabilities | High-volume markets |
| Dec 2024 | $5-25K capital range | Minimum viable for infrastructure ROI | Smaller (insufficient), larger (complexity) |

---

## Open Questions

| Question | Owner | Due Date | Status |
|----------|-------|----------|--------|
| Which specific markets to target first? | Jai | Phase 1 | Open |
| Optimal model retraining frequency? | Jai | Phase 2 | Open |
| VPS provider selection (Hetzner vs. others)? | Jai | Phase 1 | Open |
| Kalshi tier upgrade criteria? | Jai | Phase 3 | Open |
| Tax reporting approach? | Jai | Phase 3 | Open |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| CLOB | Central Limit Order Book; the matching engine for orders |
| Calibration | How well predicted probabilities match actual outcome frequencies |
| Edge | The difference between estimated true probability and market price |
| Inventory | Net position (YES contracts minus NO contracts) in a market |
| Market Making | Providing liquidity by quoting both bid and ask prices |
| Sharpe Ratio | Risk-adjusted return metric: (Return - Risk-free rate) / Volatility |
| Spread | Difference between best bid and best ask prices |
| Tick | A single price/quantity update in the market |

## Appendix B: Reference Links

- Kalshi API Documentation: <https://docs.kalshi.com/>
- Kalshi Fee Schedule: <https://kalshi.com/docs/kalshi-fee-schedule.pdf>
- defiance_cr Market Making Interview: <https://news.polymarket.com/p/automated-market-making-on-polymarket>
- poly-maker Open Source Bot: <https://github.com/warproxxx/poly-maker>
- kalshi-rust Crate: <https://lib.rs/crates/kalshi>
- TSA Trading Bot Series: <https://ferraijv.github.io/kalshi_tsa_trading_bot_overview/>
