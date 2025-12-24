# System Architecture: Kalshi Informed Market Making System

**Project Codename:** Mahler-PM  
**Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Design Complete

---

## Overview

This document describes the technical architecture for an automated prediction market trading system. The system combines real-time market data ingestion, ML-based probability estimation, and automated order management to execute an informed market making strategy on Kalshi.

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Reliability over speed** | We're not competing on latency; 99.5% uptime matters more than microseconds |
| **Separation of concerns** | Trading engine (Rust) and ML pipeline (Python) are independent services |
| **Fail-safe defaults** | On uncertainty, reduce position size and widen spreads |
| **Observable by default** | Every component emits metrics; debugging production issues is expected |
| **Configuration over code** | Strategy parameters change without redeployment |
| **Idempotent operations** | Network failures and retries must not corrupt state |

### Technology Selection

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Trading Engine | Rust | Memory safety, no GC pauses, efficient resource usage, learning goal |
| ML Pipeline | Python | PyTorch/sklearn ecosystem, rapid iteration, existing expertise |
| Time-Series DB | TimescaleDB | PostgreSQL compatibility, efficient tick storage, mature tooling |
| Cache/State Store | Redis | Sub-millisecond reads, pub/sub for inter-service communication |
| Message Queue | Redis Streams | Simple, sufficient throughput, no additional infrastructure |
| Monitoring | Prometheus + Grafana | Industry standard, excellent Rust support, free tier available |
| Alerting | Discord Webhooks | Already used in Mahler, mobile notifications, free |
| Deployment | Docker + systemd | Simple, reliable, no Kubernetes complexity needed |

---

## System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SYSTEMS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   Kalshi API    │  │  External Data  │  │      Notification           │ │
│  │                 │  │    Sources      │  │        Services             │ │
│  │  • REST API     │  │                 │  │                             │ │
│  │  • WebSocket    │  │  • BLS (jobs)   │  │  • Discord                  │ │
│  │  • Auth (RSA)   │  │  • TSA data     │  │  • (future: SMS/email)      │ │
│  │                 │  │  • Weather APIs │  │                             │ │
│  │                 │  │  • News APIs    │  │                             │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘ │
│           │                    │                          │                 │
└───────────┼────────────────────┼──────────────────────────┼─────────────────┘
            │                    │                          │
            ▼                    ▼                          ▲
┌───────────────────────────────────────────────────────────┼─────────────────┐
│                                                           │                 │
│                    MAHLER-PM SYSTEM                       │                 │
│                                                           │                 │
│  ┌─────────────────────────────────────────────────────┐ │                 │
│  │              Data Ingestion Layer                    │ │                 │
│  │                   (Rust)                             │ │                 │
│  └─────────────────────────┬───────────────────────────┘ │                 │
│                            │                              │                 │
│                            ▼                              │                 │
│  ┌─────────────────────────────────────────────────────┐ │                 │
│  │              Probability Engine                      │ │                 │
│  │                  (Python)                            │ │                 │
│  └─────────────────────────┬───────────────────────────┘ │                 │
│                            │                              │                 │
│                            ▼                              │                 │
│  ┌─────────────────────────────────────────────────────┐ │                 │
│  │              Trading Engine                          │─┼─────────────────┘
│  │                   (Rust)                             │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│                      ┌───────────┐                        │
│                      │  Operator │                        │
│                      │   (Jai)   │                        │
│                      └───────────┘                        │
│                                                           │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Layer 1: Data Ingestion (Rust)

**Purpose:** Maintain real-time market state and persist historical data.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │  Kalshi WebSocket    │    │  External Data       │                      │
│  │  Handler             │    │  Fetchers            │                      │
│  │                      │    │                      │                      │
│  │  • Orderbook deltas  │    │  • BLS client        │                      │
│  │  • Trade stream      │    │  • TSA scraper       │                      │
│  │  • Fill notifications│    │  • Weather API       │                      │
│  │  • Market metadata   │    │  • News aggregator   │                      │
│  │                      │    │                      │                      │
│  └──────────┬───────────┘    └──────────┬───────────┘                      │
│             │                           │                                   │
│             └─────────────┬─────────────┘                                   │
│                           │                                                 │
│                           ▼                                                 │
│             ┌─────────────────────────────┐                                │
│             │     Event Processor         │                                │
│             │                             │                                │
│             │  • Normalize data formats   │                                │
│             │  • Validate/sanitize        │                                │
│             │  • Timestamp alignment      │                                │
│             │  • Deduplication            │                                │
│             └──────────────┬──────────────┘                                │
│                            │                                                │
│              ┌─────────────┴─────────────┐                                 │
│              │                           │                                 │
│              ▼                           ▼                                 │
│  ┌───────────────────────┐   ┌───────────────────────┐                    │
│  │    TimescaleDB        │   │       Redis           │                    │
│  │    (Persistence)      │   │    (Hot State)        │                    │
│  │                       │   │                       │                    │
│  │  • Tick data          │   │  • Current orderbooks │                    │
│  │  • Trade history      │   │  • Latest prices      │                    │
│  │  • External events    │   │  • Market metadata    │                    │
│  │  • Model predictions  │   │  • Model estimates    │                    │
│  └───────────────────────┘   └───────────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | Responsibility | Crate Dependencies |
|-----------|---------------|-------------------|
| WebSocket Handler | Maintain persistent connection to Kalshi, parse messages | tokio-tungstenite, serde_json |
| External Fetchers | Poll external APIs on schedule | reqwest, tokio-cron |
| Event Processor | Normalize, validate, route events | custom |
| DB Writer | Batch inserts to TimescaleDB | sqlx, deadpool-postgres |
| Cache Writer | Update Redis hot state | redis-rs |

**Data Flow:**

1. WebSocket handler receives orderbook delta from Kalshi
2. Event processor normalizes to internal `OrderBookUpdate` struct
3. Parallel writes: (a) batch to TimescaleDB, (b) immediate to Redis
4. Trading engine reads from Redis for low-latency access

**Reliability Patterns:**

- **Automatic reconnection** with exponential backoff (1s, 2s, 4s, max 60s)
- **Heartbeat monitoring** — disconnect if no message for 30s
- **Sequence number tracking** — detect gaps, request snapshot on mismatch
- **Graceful degradation** — fall back to REST polling if WebSocket fails

---

### Layer 2: Probability Engine (Python)

**Purpose:** Generate calibrated probability estimates that inform quote generation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROBABILITY ENGINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Scheduler (APScheduler)                         │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │   │  Every 5m   │  │  Every 1h   │  │  Daily      │                 │   │
│  │   │  Inference  │  │  Retrain    │  │  Report     │                 │   │
│  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │   │
│  │          │                │                │                         │   │
│  └──────────┼────────────────┼────────────────┼─────────────────────────┘   │
│             │                │                │                             │
│             ▼                ▼                ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Model Manager                                   │   │
│  │                                                                      │   │
│  │   ┌───────────────────┐  ┌───────────────────┐                      │   │
│  │   │   Model Registry  │  │  Feature Store    │                      │   │
│  │   │                   │  │                   │                      │   │
│  │   │  • TSA_v1.pkl     │  │  • Computed       │                      │   │
│  │   │  • Weather_v1.pkl │  │    features       │                      │   │
│  │   │  • SPX_v1.pkl     │  │  • Raw inputs     │                      │   │
│  │   │  • (per market)   │  │  • Cached values  │                      │   │
│  │   └───────────────────┘  └───────────────────┘                      │   │
│  │                                                                      │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Inference Pipeline                              │   │
│  │                                                                      │   │
│  │   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐ │   │
│  │   │  Feature   │   │   Model    │   │ Calibration│   │  Publish   │ │   │
│  │   │  Engineer  │──▶│  Predict   │──▶│   Layer    │──▶│  to Redis  │ │   │
│  │   │            │   │            │   │            │   │            │ │   │
│  │   └────────────┘   └────────────┘   └────────────┘   └────────────┘ │   │
│  │                                                                      │   │
│  │   Outputs per market:                                                │   │
│  │   • probability: float (0.0 - 1.0)                                   │   │
│  │   • confidence: float (0.0 - 1.0)                                    │   │
│  │   • model_version: string                                            │   │
│  │   • timestamp: datetime                                              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Model Architecture per Market Type:**

| Market Type | Model Approach | Key Features |
|-------------|---------------|--------------|
| TSA Volume | Gradient Boosting + Isotonic Calibration | Day of week, holidays, fuel price, lag values |
| Weather | Ensemble of weather model outputs | Multiple forecast sources, historical accuracy |
| S&P Ranges | Black-Scholes-inspired + regime detection | Current price, volatility, time to expiry |
| Economic (CPI) | Survey aggregation + historical deviation | Analyst estimates, recent prints, seasonal |

**Calibration Approach:**

All models output probabilities through a calibration layer:

1. **Platt Scaling** — Logistic regression on held-out validation set
2. **Isotonic Regression** — Non-parametric monotonic calibration
3. **Temperature Scaling** — Simple divisor for neural network outputs

Calibration is validated using:

- **Reliability diagrams** — Visual check of predicted vs. actual frequencies
- **Expected Calibration Error (ECE)** — Quantitative metric, target < 5%
- **Brier Score** — Overall probability quality

**Output Contract:**

Models publish to Redis with key pattern `model:prob:{market_ticker}`:

```json
{
  "market_id": "TSA-25DEC31-T2400000",
  "probability": 0.62,
  "confidence": 0.78,
  "model_version": "tsa_v1.2.3",
  "features_used": ["dow", "holiday_proximity", "lag_1w", "fuel_price"],
  "timestamp": "2024-12-23T14:30:00Z",
  "ttl_seconds": 300
}
```

---

### Layer 3: Trading Engine (Rust)

**Purpose:** Generate quotes, manage orders, enforce risk limits.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING ENGINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Main Event Loop                               │   │
│  │                                                                      │   │
│  │   ┌─────────────┐                                                   │   │
│  │   │  Tick       │  Every 1 second:                                  │   │
│  │   │  Timer      │  1. Read current state from Redis                 │   │
│  │   │             │  2. Generate target quotes per market             │   │
│  │   │             │  3. Compare to open orders                        │   │
│  │   │             │  4. Submit cancel/replace as needed               │   │
│  │   └─────────────┘  5. Update position tracking                      │   │
│  │                                                                      │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Quote Generator                                │   │
│  │                                                                      │   │
│  │   Inputs:                          Outputs:                         │   │
│  │   • Model probability              • Bid price                      │   │
│  │   • Model confidence               • Bid size                       │   │
│  │   • Current orderbook              • Ask price                      │   │
│  │   • Current inventory              • Ask size                       │   │
│  │   • Config (spreads, limits)       • Skip flag (if no edge)         │   │
│  │                                                                      │   │
│  │   Logic:                                                            │   │
│  │   1. Calculate fair value from model                                │   │
│  │   2. Determine spread (base + confidence adjustment)                │   │
│  │   3. Apply inventory skew                                           │   │
│  │   4. Apply fee cushion                                              │   │
│  │   5. Size via Kelly criterion (capped)                              │   │
│  │                                                                      │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Risk Manager                                   │   │
│  │                                                                      │   │
│  │   Pre-Trade Checks:                Post-Trade Updates:              │   │
│  │   • Position limit per market      • Update positions               │   │
│  │   • Total exposure limit           • Update P&L                     │   │
│  │   • Daily loss limit               • Check halt conditions          │   │
│  │   • Inventory imbalance            • Log for audit                  │   │
│  │   • Correlation exposure                                            │   │
│  │                                                                      │   │
│  │   Circuit Breakers:                                                 │   │
│  │   • Daily loss > $X → HALT                                          │   │
│  │   • API errors > Y/min → PAUSE                                      │   │
│  │   • Position desync detected → RECONCILE                            │   │
│  │                                                                      │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Order Manager                                  │   │
│  │                                                                      │   │
│  │   Order State Machine:                                              │   │
│  │                                                                      │   │
│  │      ┌─────────┐    submit    ┌─────────┐    fill    ┌─────────┐   │   │
│  │      │ PENDING │─────────────▶│  OPEN   │───────────▶│ FILLED  │   │   │
│  │      └─────────┘              └─────────┘            └─────────┘   │   │
│  │           │                        │                               │   │
│  │           │ reject                 │ cancel                        │   │
│  │           ▼                        ▼                               │   │
│  │      ┌─────────┐              ┌─────────┐                          │   │
│  │      │REJECTED │              │CANCELLED│                          │   │
│  │      └─────────┘              └─────────┘                          │   │
│  │                                                                      │   │
│  │   Hysteresis: Don't cancel/replace unless price moved > threshold   │   │
│  │                                                                      │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Kalshi Client                                  │   │
│  │                                                                      │   │
│  │   • REST client for order submission                                │   │
│  │   • WebSocket client for fills and order updates                    │   │
│  │   • RSA-PSS request signing                                         │   │
│  │   • Rate limit tracking and backoff                                 │   │
│  │   • Request/response logging                                        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Quote Generation Algorithm:**

```
FUNCTION generate_quote(market_id):
    
    // Gather inputs
    model_estimate = redis.get("model:prob:{market_id}")
    orderbook = redis.get("orderbook:{market_id}")
    inventory = redis.get("inventory:{market_id}")
    config = load_config(market_id)
    
    // Check if we should quote this market
    IF model_estimate.confidence < config.min_confidence:
        RETURN skip_market
    
    IF orderbook.spread_bps > config.max_spread_to_enter:
        RETURN skip_market
    
    // Calculate fair value
    fair_value = model_estimate.probability
    
    // Calculate spread
    base_spread = config.base_spread_bps / 10000
    confidence_adjustment = (1 - model_estimate.confidence) * config.confidence_scaling
    fee_cushion = config.kalshi_fee_bps * 2 / 10000  // Round trip
    
    spread = (base_spread + fee_cushion) * (1 + confidence_adjustment)
    
    // Apply inventory skew
    net_inventory = inventory.yes_contracts - inventory.no_contracts
    skew_per_contract = config.skew_bps_per_contract / 10000
    inventory_skew = -1 * net_inventory * skew_per_contract
    
    // Generate prices
    bid_price = round(fair_value - spread/2 + inventory_skew, 2)
    ask_price = round(fair_value + spread/2 + inventory_skew, 2)
    
    // Clamp to valid range
    bid_price = clamp(bid_price, 0.01, 0.99)
    ask_price = clamp(ask_price, 0.01, 0.99)
    
    // Size using Kelly criterion (simplified)
    edge = abs(model_estimate.probability - orderbook.mid_price)
    kelly_fraction = edge / (1 - edge)
    max_size_usd = config.max_position_usd * kelly_fraction
    size_contracts = floor(max_size_usd / fair_value)
    size_contracts = min(size_contracts, config.max_order_size)
    
    RETURN QuoteResult {
        bid_price,
        bid_size: size_contracts,
        ask_price,
        ask_size: size_contracts,
        model_prob: model_estimate.probability,
        confidence: model_estimate.confidence
    }
```

**Risk Manager Rules:**

| Rule | Limit | Action on Breach |
|------|-------|------------------|
| Per-market position | $500 (configurable) | Reject new orders that increase position |
| Total exposure | $5,000 (configurable) | Reject all new orders |
| Daily loss | $200 (configurable) | HALT all trading, alert operator |
| Inventory imbalance | 70% one side | Only allow orders that reduce imbalance |
| Correlation group exposure | $1,000 per group | Reject orders in correlated markets |
| API error rate | 10/minute | PAUSE 5 minutes, alert operator |
| Order fill latency | > 5 seconds | Log warning, continue |

---

### Layer 4: Monitoring and Alerting

**Purpose:** Operational visibility, performance tracking, incident response.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MONITORING AND ALERTING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Metrics Collection                             │   │
│  │                                                                      │   │
│  │   Trading Engine (Rust) ──────────────────────▶ Prometheus          │   │
│  │   • orders_submitted_total                                          │   │
│  │   • orders_filled_total                                             │   │
│  │   • order_latency_seconds                                           │   │
│  │   • position_value_usd{market, side}                                │   │
│  │   • daily_pnl_usd                                                   │   │
│  │   • inventory_imbalance{market}                                     │   │
│  │                                                                      │   │
│  │   Probability Engine (Python) ────────────────▶ Prometheus          │   │
│  │   • model_inference_duration_seconds                                │   │
│  │   • model_probability{market}                                       │   │
│  │   • model_confidence{market}                                        │   │
│  │   • calibration_error                                               │   │
│  │                                                                      │   │
│  │   Data Ingestion (Rust) ──────────────────────▶ Prometheus          │   │
│  │   • websocket_messages_total                                        │   │
│  │   • websocket_reconnects_total                                      │   │
│  │   • tick_lag_seconds                                                │   │
│  │   • db_write_duration_seconds                                       │   │
│  │                                                                      │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Grafana Dashboards                             │   │
│  │                                                                      │   │
│  │   Dashboard: Trading Overview                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  Daily P&L    │  Open Positions  │  Win Rate (7d)           │   │   │
│  │   │   +$42.50     │     $1,234       │    58.2%                 │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │              Cumulative P&L (30 days)                       │   │   │
│  │   │  $800 ─┐                                           ╱        │   │   │
│  │   │        │                                     ╱───╱          │   │   │
│  │   │  $400 ─┼─────────────────────────╱──────────╱               │   │   │
│  │   │        │                   ╱────╱                           │   │   │
│  │   │    $0 ─┴──────────────────╱─────────────────────────────────│   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   Dashboard: System Health                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  WS Status   │  API Latency (p99)  │  Model Freshness       │   │   │
│  │   │   🟢 CONN    │      187ms          │    2m 34s ago          │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └──────────────────────────────────┬──────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Alerting Rules                                 │   │
│  │                                                                      │   │
│  │   CRITICAL (immediate Discord + mobile push):                       │   │
│  │   • daily_pnl_usd < -$200 (daily loss limit)                        │   │
│  │   • trading_engine_up == 0 (system down)                            │   │
│  │   • websocket_connected == 0 for > 5 minutes                        │   │
│  │                                                                      │   │
│  │   WARNING (Discord only):                                           │   │
│  │   • inventory_imbalance > 0.6 for any market                        │   │
│  │   • model_confidence < 0.5 for active market                        │   │
│  │   • order_fill_rate < 0.3 over 1 hour                               │   │
│  │   • api_error_rate > 5/minute                                       │   │
│  │                                                                      │   │
│  │   INFO (daily digest):                                              │   │
│  │   • Daily P&L summary                                               │   │
│  │   • Best/worst performing markets                                   │   │
│  │   • Model calibration report                                        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Architecture

### Database Schema (TimescaleDB)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TIMESCALEDB SCHEMA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TABLE: orderbook_ticks (hypertable, partitioned by time)                  │
│  ─────────────────────────────────────────────────────────                 │
│  • timestamp         TIMESTAMPTZ    PRIMARY KEY (part of composite)        │
│  • market_id         TEXT           PRIMARY KEY (part of composite)        │
│  • bid_price         DECIMAL(10,4)                                         │
│  • bid_size          INTEGER                                                │
│  • ask_price         DECIMAL(10,4)                                         │
│  • ask_size          INTEGER                                                │
│  • mid_price         DECIMAL(10,4)                                         │
│  • spread_bps        INTEGER                                                │
│                                                                             │
│  Retention: 90 days                                                        │
│  Compression: After 7 days                                                 │
│  Indexes: (market_id, timestamp DESC)                                      │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  TABLE: trades (hypertable)                                                │
│  ──────────────────────────                                                │
│  • timestamp         TIMESTAMPTZ    PRIMARY KEY                            │
│  • market_id         TEXT                                                  │
│  • price             DECIMAL(10,4)                                         │
│  • size              INTEGER                                                │
│  • side              TEXT           (buy/sell)                             │
│  • taker_order_id    TEXT                                                  │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  TABLE: model_predictions                                                  │
│  ────────────────────────────                                              │
│  • timestamp         TIMESTAMPTZ    PRIMARY KEY                            │
│  • market_id         TEXT                                                  │
│  • probability       DECIMAL(5,4)                                          │
│  • confidence        DECIMAL(5,4)                                          │
│  • model_version     TEXT                                                  │
│  • features_json     JSONB                                                 │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  TABLE: orders (regular table)                                             │
│  ─────────────────────────────                                             │
│  • order_id          TEXT           PRIMARY KEY                            │
│  • market_id         TEXT                                                  │
│  • side              TEXT                                                  │
│  • price             DECIMAL(10,4)                                         │
│  • size              INTEGER                                                │
│  • filled_size       INTEGER                                                │
│  • status            TEXT                                                  │
│  • created_at        TIMESTAMPTZ                                           │
│  • updated_at        TIMESTAMPTZ                                           │
│  • model_prob        DECIMAL(5,4)   (probability at time of order)         │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  TABLE: fills (hypertable)                                                 │
│  ─────────────────────────                                                 │
│  • timestamp         TIMESTAMPTZ    PRIMARY KEY                            │
│  • order_id          TEXT                                                  │
│  • market_id         TEXT                                                  │
│  • side              TEXT                                                  │
│  • price             DECIMAL(10,4)                                         │
│  • size              INTEGER                                                │
│  • fee_usd           DECIMAL(10,4)                                         │
│  • realized_pnl      DECIMAL(10,4)                                         │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  TABLE: external_events                                                    │
│  ──────────────────────────                                                │
│  • timestamp         TIMESTAMPTZ    PRIMARY KEY                            │
│  • event_type        TEXT           (tsa_report, cpi_release, etc.)        │
│  • event_data        JSONB                                                 │
│  • source            TEXT                                                  │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  TABLE: daily_summary (regular table, one row per day)                     │
│  ─────────────────────────────────────────────────────                     │
│  • date              DATE           PRIMARY KEY                            │
│  • total_pnl         DECIMAL(10,2)                                         │
│  • gross_profit      DECIMAL(10,2)                                         │
│  • gross_loss        DECIMAL(10,2)                                         │
│  • trades_count      INTEGER                                                │
│  • win_count         INTEGER                                                │
│  • fees_paid         DECIMAL(10,2)                                         │
│  • max_drawdown      DECIMAL(10,2)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Redis Key Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            REDIS KEY SCHEMA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HOT MARKET DATA (updated every tick)                                      │
│  ────────────────────────────────────                                      │
│  orderbook:{market_id}          HASH    Current orderbook snapshot         │
│    • bid_price                                                             │
│    • bid_size                                                              │
│    • ask_price                                                             │
│    • ask_size                                                              │
│    • mid_price                                                             │
│    • spread_bps                                                            │
│    • updated_at                                                            │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  MODEL OUTPUTS (updated every 5 minutes)                                   │
│  ───────────────────────────────────────                                   │
│  model:prob:{market_id}         STRING  JSON blob with estimate            │
│    TTL: 300 seconds                                                        │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  TRADING STATE (updated on every order/fill)                               │
│  ───────────────────────────────────────────                               │
│  inventory:{market_id}          HASH    Current position                   │
│    • yes_contracts                                                         │
│    • no_contracts                                                          │
│    • avg_yes_price                                                         │
│    • avg_no_price                                                          │
│    • unrealized_pnl                                                        │
│                                                                             │
│  orders:open:{market_id}        SET     Set of open order IDs              │
│                                                                             │
│  order:{order_id}               HASH    Order details                      │
│    • market_id                                                             │
│    • side                                                                  │
│    • price                                                                 │
│    • size                                                                  │
│    • filled_size                                                           │
│    • status                                                                │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  RISK STATE (updated on fills and periodically)                            │
│  ──────────────────────────────────────────────                            │
│  risk:daily_pnl                 STRING  Current day's P&L                  │
│  risk:total_exposure            STRING  Sum of all position values         │
│  risk:is_halted                 STRING  "true" or "false"                  │
│  risk:halt_reason               STRING  Reason if halted                   │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│  CONFIGURATION (updated manually or on deploy)                             │
│  ─────────────────────────────────────────────                             │
│  config:markets                 SET     Active market IDs                  │
│  config:market:{market_id}      HASH    Per-market settings                │
│    • base_spread_bps                                                       │
│    • max_position_usd                                                      │
│    • min_confidence                                                        │
│    • enabled                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

### Infrastructure Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              INTERNET                                       │
│                                  │                                          │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                     │
│              │                   │                   │                     │
│              ▼                   ▼                   ▼                     │
│     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │
│     │   Kalshi API    │ │  External APIs  │ │    Discord      │           │
│     │  (trading-api.  │ │  (BLS, TSA,     │ │   (webhooks)    │           │
│     │   kalshi.com)   │ │   weather)      │ │                 │           │
│     └────────┬────────┘ └────────┬────────┘ └────────▲────────┘           │
│              │                   │                   │                     │
│              └───────────────────┼───────────────────┘                     │
│                                  │                                          │
│ ┌────────────────────────────────┼────────────────────────────────────────┐│
│ │                                │                                        ││
│ │                    VPS (Hetzner CPX21)                                  ││
│ │                    3 vCPU, 4GB RAM, 80GB SSD                            ││
│ │                    Location: US-East (Ashburn)                          ││
│ │                                │                                        ││
│ │    ┌───────────────────────────┴────────────────────────────────┐      ││
│ │    │                                                            │      ││
│ │    │                     Docker Compose                         │      ││
│ │    │                                                            │      ││
│ │    │  ┌─────────────────┐  ┌─────────────────┐                 │      ││
│ │    │  │  trading-engine │  │  data-ingest    │                 │      ││
│ │    │  │  (Rust binary)  │  │  (Rust binary)  │                 │      ││
│ │    │  │                 │  │                 │                 │      ││
│ │    │  │  Port: internal │  │  Port: internal │                 │      ││
│ │    │  │  Memory: 100MB  │  │  Memory: 100MB  │                 │      ││
│ │    │  └─────────────────┘  └─────────────────┘                 │      ││
│ │    │                                                            │      ││
│ │    │  ┌─────────────────┐  ┌─────────────────┐                 │      ││
│ │    │  │  probability-   │  │  prometheus     │                 │      ││
│ │    │  │  engine         │  │                 │                 │      ││
│ │    │  │  (Python)       │  │  Port: 9090     │                 │      ││
│ │    │  │                 │  │  Memory: 200MB  │                 │      ││
│ │    │  │  Memory: 500MB  │  │                 │                 │      ││
│ │    │  └─────────────────┘  └─────────────────┘                 │      ││
│ │    │                                                            │      ││
│ │    │  ┌─────────────────┐  ┌─────────────────┐                 │      ││
│ │    │  │  timescaledb    │  │  redis          │                 │      ││
│ │    │  │                 │  │                 │                 │      ││
│ │    │  │  Port: 5432     │  │  Port: 6379     │                 │      ││
│ │    │  │  Memory: 1GB    │  │  Memory: 256MB  │                 │      ││
│ │    │  │  Volume: /data  │  │                 │                 │      ││
│ │    │  └─────────────────┘  └─────────────────┘                 │      ││
│ │    │                                                            │      ││
│ │    └────────────────────────────────────────────────────────────┘      ││
│ │                                                                        ││
│ │    Volumes:                                                            ││
│ │    • /data/timescale - 40GB (persistent, backed up daily)              ││
│ │    • /data/redis - 1GB (RDB snapshots)                                 ││
│ │    • /data/logs - 5GB (rotated weekly)                                 ││
│ │                                                                        ││
│ └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │                                                                        │ │
│ │                    External Services                                   │ │
│ │                                                                        │ │
│ │    ┌─────────────────┐  ┌─────────────────┐                           │ │
│ │    │  Grafana Cloud  │  │  Backblaze B2   │                           │ │
│ │    │  (Free tier)    │  │  (Backups)      │                           │ │
│ │    │                 │  │                 │                           │ │
│ │    │  Dashboards     │  │  Daily DB dumps │                           │ │
│ │    │  via Prometheus │  │  $0.005/GB/mo   │                           │ │
│ │    └─────────────────┘  └─────────────────┘                           │ │
│ │                                                                        │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Resource Requirements

| Component | CPU | Memory | Disk | Network |
|-----------|-----|--------|------|---------|
| Trading Engine | 0.5 vCPU | 100MB | Minimal | Low (API calls) |
| Data Ingestion | 0.5 vCPU | 100MB | Minimal | Medium (WebSocket) |
| Probability Engine | 1 vCPU | 500MB | 1GB (models) | Low |
| TimescaleDB | 0.5 vCPU | 1GB | 40GB | Low |
| Redis | 0.2 vCPU | 256MB | 1GB | Low |
| Prometheus | 0.2 vCPU | 200MB | 5GB | Low |
| **Total** | **2.9 vCPU** | **2.2GB** | **47GB** | - |

**VPS Selection:** Hetzner CPX21 (3 vCPU, 4GB RAM, 80GB SSD) at ~$10/month provides adequate headroom.

### Deployment Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT WORKFLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LOCAL DEVELOPMENT                                                        │
│   ──────────────────                                                       │
│   1. Edit code                                                             │
│   2. Run tests locally                                                     │
│   3. Commit to Git                                                         │
│                                                                             │
│         │                                                                  │
│         ▼                                                                  │
│                                                                             │
│   GITHUB ACTIONS (CI)                                                      │
│   ───────────────────                                                      │
│   1. cargo test (Rust)                                                     │
│   2. pytest (Python)                                                       │
│   3. cargo build --release                                                 │
│   4. Build Docker images                                                   │
│   5. Push to GitHub Container Registry                                     │
│                                                                             │
│         │                                                                  │
│         ▼                                                                  │
│                                                                             │
│   MANUAL DEPLOY (SSH to VPS)                                               │
│   ──────────────────────────                                               │
│   1. ssh mahler-pm                                                         │
│   2. cd /opt/mahler-pm                                                     │
│   3. git pull                                                              │
│   4. docker compose pull                                                   │
│   5. docker compose up -d                                                  │
│   6. docker compose logs -f (verify startup)                               │
│                                                                             │
│         │                                                                  │
│         ▼                                                                  │
│                                                                             │
│   POST-DEPLOY VERIFICATION                                                 │
│   ────────────────────────                                                 │
│   1. Check Grafana dashboard for healthy metrics                           │
│   2. Verify WebSocket connected                                            │
│   3. Confirm model predictions updating                                    │
│   4. Check no error alerts in Discord                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration Management

All configuration via YAML files, no code changes needed:

**config/markets.yaml:**

```yaml
markets:
  - ticker: "TSA-*"
    enabled: true
    base_spread_bps: 300
    max_position_usd: 500
    min_confidence: 0.6
    model: "tsa_v1"
    
  - ticker: "INXD-*"
    enabled: true
    base_spread_bps: 250
    max_position_usd: 300
    min_confidence: 0.65
    model: "spx_range_v1"
```

**config/risk.yaml:**

```yaml
risk:
  max_daily_loss_usd: 200
  max_total_exposure_usd: 5000
  max_inventory_imbalance: 0.7
  correlation_groups:
    politics:
      - "PRES-*"
      - "SEN-*"
      - "HOUSE-*"
    macro:
      - "CPI-*"
      - "FOMC-*"
      - "NFP-*"
```

---

## Security Architecture

### Authentication and Authorization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SECRETS MANAGEMENT                                                        │
│  ───────────────────                                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Environment Variables                             │   │
│  │                    (Docker secrets or .env)                          │   │
│  │                                                                      │   │
│  │  KALSHI_API_KEY_ID=xxxxx                                            │   │
│  │  KALSHI_PRIVATE_KEY_PATH=/run/secrets/kalshi_key.pem                │   │
│  │  DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...           │   │
│  │  DATABASE_URL=postgres://user:pass@localhost/mahler                 │   │
│  │  REDIS_URL=redis://localhost:6379                                   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  NETWORK SECURITY                                                          │
│  ────────────────                                                          │
│                                                                             │
│  • All services bind to localhost only (no external exposure)              │
│  • SSH access via key authentication only                                  │
│  • UFW firewall: allow 22 (SSH), deny all other inbound                    │
│  • Outbound: allow HTTPS (443) to Kalshi, external APIs                    │
│                                                                             │
│  API AUTHENTICATION                                                        │
│  ──────────────────                                                        │
│                                                                             │
│  Kalshi uses RSA-PSS signatures:                                           │
│  1. Generate RSA key pair (done once during setup)                         │
│  2. Register public key with Kalshi dashboard                              │
│  3. Sign each request with private key                                     │
│  4. Include signature in Authorization header                              │
│                                                                             │
│  AUDIT LOGGING                                                             │
│  ─────────────                                                             │
│                                                                             │
│  All order actions logged with:                                            │
│  • Timestamp                                                               │
│  • Action type (submit, cancel, fill)                                      │
│  • Order details (no API keys or signatures)                               │
│  • Risk check results                                                      │
│  • Model probability at time of action                                     │
│                                                                             │
│  Logs retained for 90 days, then archived to cold storage.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Failure Modes and Recovery

### Failure Scenarios

| Scenario | Detection | Automatic Response | Manual Response |
|----------|-----------|-------------------|-----------------|
| WebSocket disconnect | No message for 30s | Reconnect with backoff; fall back to REST polling | Check Kalshi status page |
| Kalshi API errors | HTTP 5xx or timeout | Exponential backoff; pause trading after 10 failures | Check Kalshi status; wait for recovery |
| Database unavailable | Connection error | Retry writes; queue in Redis | Restart TimescaleDB container |
| Redis unavailable | Connection error | Trading engine enters read-only mode | Restart Redis; reconcile state |
| Model inference fails | Exception or timeout | Use last known estimate (if < 1 hour old); else skip market | Check Python logs; restart container |
| Order state desync | Fill notification for unknown order | Fetch all open orders from Kalshi API; reconcile | Review logs; manual position check |
| Daily loss limit hit | daily_pnl < -$200 | HALT all trading; cancel all open orders | Review trades; decide whether to reset |
| VPS crash | No heartbeat to monitoring | systemd auto-restart; Docker Compose restart policy | SSH to investigate; check disk space |

### Recovery Procedures

**Procedure: Full System Restart**

```
1. SSH to VPS
2. docker compose down
3. docker compose up -d
4. Verify: docker compose logs -f (watch for errors)
5. Verify: Check Grafana dashboard for healthy metrics
6. Verify: Confirm WebSocket connected in logs
7. Verify: Check Discord for any alerts
```

**Procedure: Position Reconciliation**

```
1. Trading engine: Set is_halted = true in Redis
2. Fetch all open orders from Kalshi API
3. Fetch all positions from Kalshi API
4. Compare to local Redis state
5. Update Redis to match Kalshi (source of truth)
6. Review discrepancies in logs
7. If clean: Set is_halted = false
8. If discrepancies: Manual review before resuming
```

**Procedure: Database Recovery**

```
1. docker compose stop timescaledb
2. Restore from latest backup: pg_restore -d mahler /backup/latest.dump
3. docker compose start timescaledb
4. Verify: psql -c "SELECT COUNT(*) FROM orderbook_ticks WHERE timestamp > NOW() - INTERVAL '1 day'"
5. Note: Some recent ticks may be lost; acceptable since Redis has hot state
```

---

## Development and Testing

### Local Development Setup

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LOCAL DEVELOPMENT ENVIRONMENT                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PREREQUISITES                                                             │
│  ─────────────                                                             │
│  • Rust (rustup, stable channel)                                           │
│  • Python 3.11+                                                            │
│  • Docker and Docker Compose                                               │
│  • Kalshi demo API credentials                                             │
│                                                                             │
│  SETUP                                                                     │
│  ─────                                                                     │
│                                                                             │
│  # Clone repository                                                        │
│  git clone git@github.com:jai/mahler-pm.git                                │
│  cd mahler-pm                                                              │
│                                                                             │
│  # Start infrastructure                                                    │
│  docker compose -f docker-compose.dev.yml up -d                            │
│                                                                             │
│  # Setup Rust components                                                   │
│  cd trading-engine && cargo build                                          │
│  cd ../data-ingest && cargo build                                          │
│                                                                             │
│  # Setup Python components                                                 │
│  cd ../probability-engine                                                  │
│  python -m venv venv                                                       │
│  source venv/bin/activate                                                  │
│  pip install -r requirements.txt                                           │
│                                                                             │
│  # Configure                                                               │
│  cp .env.example .env                                                      │
│  # Edit .env with Kalshi demo credentials                                  │
│                                                                             │
│  RUNNING LOCALLY                                                           │
│  ───────────────                                                           │
│                                                                             │
│  # Terminal 1: Data ingestion                                              │
│  cd data-ingest && RUST_LOG=info cargo run                                 │
│                                                                             │
│  # Terminal 2: Probability engine                                          │
│  cd probability-engine && python main.py                                   │
│                                                                             │
│  # Terminal 3: Trading engine (paper mode)                                 │
│  cd trading-engine && PAPER_TRADING=true cargo run                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Testing Strategy

| Test Type | Scope | Tools | Frequency |
|-----------|-------|-------|-----------|
| Unit Tests | Individual functions, pure logic | cargo test, pytest | Every commit |
| Integration Tests | Service interactions | Docker Compose test env | Every PR |
| Backtest | Strategy on historical data | Custom Python harness | Weekly or on model changes |
| Paper Trading | Full system on live markets, no real orders | Kalshi demo API | Continuous (Phase 1-2) |
| Live Validation | Real orders with minimal capital | Kalshi production | Phase 3+ |

### Backtesting Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKTESTING ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                                                                     │
│  ─────                                                                     │
│  • Historical orderbook ticks from TimescaleDB                             │
│  • Historical model predictions                                            │
│  • Configuration (same as production)                                      │
│                                                                             │
│  SIMULATION                                                                │
│  ──────────                                                                │
│  1. Replay ticks in chronological order                                    │
│  2. Quote generator produces target quotes at each tick                    │
│  3. Simulate fills based on orderbook depth                                │
│  4. Track positions, P&L, risk metrics                                     │
│                                                                             │
│  ASSUMPTIONS                                                               │
│  ───────────                                                               │
│  • Our orders don't impact market (reasonable at small size)               │
│  • Fills occur at quoted price if liquidity available                      │
│  • Latency simulated at 200ms                                              │
│  • Fees calculated per Kalshi schedule                                     │
│                                                                             │
│  OUTPUT                                                                    │
│  ──────                                                                    │
│  • Cumulative P&L curve                                                    │
│  • Sharpe ratio, Sortino ratio                                             │
│  • Max drawdown                                                            │
│  • Win rate, average win/loss                                              │
│  • Per-market breakdown                                                    │
│  • Model calibration metrics                                               │
│                                                                             │
│  VALIDATION                                                                │
│  ──────────                                                                │
│  • Compare backtest to paper trading results                               │
│  • Compare paper trading to live trading results                           │
│  • Discrepancies indicate simulation bugs or market impact                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Rust Crate Dependencies

### Trading Engine (trading-engine/Cargo.toml)

| Crate | Version | Purpose |
|-------|---------|---------|
| tokio | 1.x | Async runtime |
| reqwest | 0.11 | HTTP client for REST API |
| tokio-tungstenite | 0.20 | WebSocket client |
| serde / serde_json | 1.x | Serialization |
| rust_decimal | 1.x | Precise decimal arithmetic |
| redis | 0.23 | Redis client |
| sqlx | 0.7 | PostgreSQL client |
| tracing / tracing-subscriber | 0.1 | Structured logging |
| prometheus | 0.13 | Metrics export |
| rsa | 0.9 | RSA-PSS signing for Kalshi auth |
| sha2 | 0.10 | Hashing for signatures |
| config | 0.13 | Configuration loading |
| anyhow / thiserror | 1.x | Error handling |

### Data Ingestion (data-ingest/Cargo.toml)

Same as trading engine, plus:

| Crate | Version | Purpose |
|-------|---------|---------|
| tokio-cron-scheduler | 0.9 | Scheduled external data fetches |
| scraper | 0.17 | HTML parsing for TSA data |

## Appendix B: Python Dependencies

### Probability Engine (probability-engine/requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.24+ | Numerical operations |
| pandas | 2.0+ | Data manipulation |
| scikit-learn | 1.3+ | ML models, calibration |
| xgboost | 2.0+ | Gradient boosting models |
| lightgbm | 4.0+ | Alternative boosting |
| redis | 5.0+ | Redis client |
| psycopg2-binary | 2.9+ | PostgreSQL client |
| apscheduler | 3.10+ | Task scheduling |
| requests | 2.31+ | HTTP client |
| prometheus-client | 0.17+ | Metrics export |
| pydantic | 2.0+ | Data validation |

## Appendix C: API Reference

### Kalshi API Endpoints Used

| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| /trade-api/v2/login | POST | Authenticate, get session token | N/A |
| /trade-api/v2/markets | GET | List available markets | 10/sec |
| /trade-api/v2/markets/{ticker} | GET | Get market details | 10/sec |
| /trade-api/v2/markets/{ticker}/orderbook | GET | Get orderbook snapshot | 10/sec |
| /trade-api/v2/portfolio/orders | POST | Submit new order | 10/sec |
| /trade-api/v2/portfolio/orders/{order_id} | DELETE | Cancel order | 10/sec |
| /trade-api/v2/portfolio/positions | GET | Get current positions | 10/sec |
| /trade-api/ws/v2 | WebSocket | Real-time orderbook, fills | N/A |

### Internal Redis Pub/Sub Channels

| Channel | Publisher | Subscriber | Message Type |
|---------|-----------|------------|--------------|
| orderbook_updates | data-ingest | trading-engine | OrderBookSnapshot |
| model_updates | probability-engine | trading-engine | ModelEstimate |
| fill_notifications | trading-engine | monitoring | FillEvent |
| alerts | all services | monitoring | AlertMessage |

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| Backoff | Progressively increasing delay between retry attempts |
| Calibration | Process of adjusting model outputs so predicted probabilities match actual frequencies |
| Circuit Breaker | Automatic mechanism to halt trading when risk limits are breached |
| CLOB | Central Limit Order Book; order matching system |
| Hysteresis | Threshold buffer to prevent oscillating behavior (e.g., don't cancel order for 1 cent move) |
| Hypertable | TimescaleDB's term for a time-partitioned table |
| Inventory | Net position in a market (YES contracts minus NO contracts) |
| Kelly Criterion | Formula for optimal bet sizing based on edge and odds |
| Market Making | Providing liquidity by quoting both buy and sell prices |
| P&L | Profit and Loss |
| RSA-PSS | RSA Probabilistic Signature Scheme; used by Kalshi for API authentication |
| Sharpe Ratio | (Return - Risk-free rate) / Standard deviation; measures risk-adjusted return |
| Skew | Adjustment to bid/ask prices to manage inventory risk |
| Spread | Difference between best bid and best ask prices |
| Tick | Single update to market data (price, size, etc.) |
| TTL | Time To Live; expiration time for cached data |
