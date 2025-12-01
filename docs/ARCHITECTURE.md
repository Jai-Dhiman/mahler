# Trader Jim Architecture

A human-in-the-loop options trading system built in Rust, using Claude for market analysis and trade recommendations with explicit approval gates before execution.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADER JIM SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   Scheduler  │───▶│   Analysis   │───▶│  Recommender │───▶│  Notifier │  │
│  │   (systemd)  │    │   Engine     │    │   (Claude)   │    │  (Slack)  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────┬─────┘  │
│                                                                      │      │
│                                                                      ▼      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │    Memory    │◀───│  Reflection  │◀───│   Executor   │◀───│  Approval │  │
│  │    Store     │    │    System    │    │              │    │   Gate    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘  │
│         │                                        │                          │
│         │            ┌──────────────┐            │                          │
│         └───────────▶│     Risk     │◀───────────┘                          │
│                      │   Manager    │                                       │
│                      └──────────────┘                                       │
│                             │                                               │
│                             ▼                                               │
│                      ┌──────────────┐                                       │
│                      │   Broker API │                                       │
│                      │ (Alpaca/TT)  │                                       │
│                      └──────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Scheduler

**Purpose**: Orchestrates the daily trading workflow on a fixed schedule.

**Implementation**: systemd timer on Hetzner VPS (not cron—systemd provides better logging, dependency management, and failure handling).

**Schedule**:

| Time (ET) | Task |
|-----------|------|
| 9:00 AM | Pre-market analysis: fetch overnight data, calculate IV Rank |
| 9:35 AM | Opening scan: identify opportunities after first 5 minutes |
| 12:00 PM | Midday check: monitor open positions, scan for new setups |
| 3:30 PM | Closing scan: final opportunity check, position reviews |
| 4:15 PM | End-of-day: close audit, generate daily summary |

**Key Design Decisions**:

- Single-threaded async Tokio runtime (no need for multi-core for this workload)
- Each scheduled task is idempotent—safe to retry on failure
- All times converted to ET internally regardless of server timezone

---

### 2. Analysis Engine

**Purpose**: Fetch market data, calculate Greeks, screen for opportunities.

**Data Sources**:

- **Options chains**: Broker API (Alpaca for paper, Tastytrade for live)
- **IV Rank**: Calculated from 52-week IV history (stored locally in SQLite)
- **Greeks**: Computed via Black-Scholes using `optionstratlib` crate

**Screening Pipeline**:

```
Fetch Options Chain
        │
        ▼
Filter by DTE (30-45 days)
        │
        ▼
Filter by Liquidity (bid-ask spread < 10% of mid)
        │
        ▼
Calculate IV Rank (require ≥ 50)
        │
        ▼
Identify Valid Spreads:
  - Short strike delta: 0.20-0.30
  - Long strike delta: 0.10-0.15
  - Width: $1-$5 depending on underlying
        │
        ▼
Rank by Expected Value:
  EV = (POP × MaxProfit) - ((1-POP) × MaxLoss)
        │
        ▼
Top 3-5 Candidates → Recommender
```

**Data Structures**:

```rust
struct SpreadCandidate {
    underlying: Symbol,
    spread_type: SpreadType,  // BullPut or BearCall
    short_strike: Strike,
    long_strike: Strike,
    expiration: NaiveDate,
    bid: Decimal,
    ask: Decimal,
    mid: Decimal,
    greeks: SpreadGreeks,
    iv_rank: Decimal,
    pop: Decimal,             // Probability of profit
    expected_value: Decimal,
}

struct SpreadGreeks {
    delta: Decimal,
    gamma: Decimal,
    theta: Decimal,
    vega: Decimal,
}
```

---

### 3. Recommender (Claude Integration)

**Purpose**: Generate natural language trade thesis, assess macro context, provide confidence scoring.

**Integration Pattern**: Synchronous API call per analysis cycle (not streaming—we need complete analysis before notification).

**Prompt Structure**:

```
<system>
You are a trading analyst assistant. Analyze options trade opportunities
and provide clear, actionable recommendations. Be direct about risks.
Never recommend trades—only analyze candidates the system has identified.
</system>

<context>
Current market conditions: {vix_level}, {spy_trend}, {upcoming_events}
Portfolio state: {open_positions}, {daily_pnl}, {available_capital}
Historical performance: {win_rate}, {avg_winner}, {avg_loser}
Relevant past trades: {retrieved_memories}
</context>

<candidates>
{spread_candidates_json}
</candidates>

<task>
For each candidate, provide:
1. Trade thesis (2-3 sentences)
2. Key risks
3. Confidence level (low/medium/high) with reasoning
4. Suggested position size (as % of available capital, respecting 2% max risk)
</task>
```

**Response Parsing**: Structured JSON output with fallback to regex extraction if JSON fails.

**Rate Limiting**: Maximum 10 API calls per hour to control costs (~$0.50/day at current Claude pricing).

---

### 4. Notifier (Slack Integration)

**Purpose**: Deliver trade recommendations and receive approval/rejection.

**Message Format**:

```
🎯 *Trade Recommendation: SPY Bull Put Spread*

*Setup*
• Short: 580 Put @ $2.45
• Long: 575 Put @ $1.20
• Net Credit: $1.25 ($125 per contract)
• Max Loss: $3.75 ($375 per contract)
• DTE: 38 days
• IV Rank: 72%

*Analysis*
SPY holding above 20-day MA with VIX elevated. High IV rank 
creates favorable premium environment. Risk defined at $375 
with 33% return potential in 38 days.

*Greeks*
Delta: -0.24 | Theta: +$3.42/day | Vega: -$8.21

*Confidence*: HIGH
*Suggested Size*: 2 contracts ($750 risk = 1.5% of account)

*Risk Flags*: None

[✅ Approve]  [❌ Reject]  [⏸️ Skip]
```

**Approval Flow**:

1. User clicks button → Slack sends webhook to Trader Jim
2. Trader Jim validates: Is recommendation still valid? Has market moved significantly?
3. If valid → forward to Executor
4. If stale (>15 min) or market moved (>1% underlying move) → notify user, require fresh analysis

**Expiration**: Recommendations expire after 15 minutes during market hours.

---

### 5. Approval Gate

**Purpose**: Enforce human-in-the-loop requirement before any order execution.

**Implementation**: HTTP endpoint receiving Slack interactive message payloads.

**Validation Checks**:

- Recommendation not expired
- Underlying price within 1% of analysis price
- IV Rank still above threshold
- No circuit breakers triggered
- Daily/weekly loss limits not exceeded

**State Machine**:

```
PENDING → APPROVED → EXECUTING → FILLED
    │         │           │
    │         │           └──→ PARTIALLY_FILLED → FILLED
    │         │                      │
    │         └──→ VALIDATION_FAILED └──→ CANCELLED
    │
    └──→ REJECTED
    │
    └──→ EXPIRED
```

---

### 6. Executor

**Purpose**: Place orders with the broker, handle fills, manage order lifecycle.

**Order Placement Strategy**:

1. Calculate limit price: mid-price - $0.02 (slightly below mid for better fill)
2. Submit order with 60-second timeout
3. If not filled, adjust price by $0.01 toward natural (up to 3 adjustments)
4. If still not filled, notify user with option to market order or cancel

**Broker Abstraction**:

```rust
#[async_trait]
trait BrokerClient {
    async fn get_account(&self) -> Result<Account>;
    async fn get_options_chain(&self, symbol: &str, expiration: NaiveDate) -> Result<OptionsChain>;
    async fn place_spread_order(&self, order: SpreadOrder) -> Result<OrderId>;
    async fn get_order_status(&self, order_id: OrderId) -> Result<OrderStatus>;
    async fn cancel_order(&self, order_id: OrderId) -> Result<()>;
    async fn get_positions(&self) -> Result<Vec<Position>>;
}

struct AlpacaClient { /* paper trading */ }
struct TastytradeClient { /* live trading */ }
```

**Paper Trading Adjustments**:

- Add random 1-5% slippage to fills
- Simulate 80% fill rate on limit orders
- Add realistic latency (100-500ms)

---

### 7. Risk Manager

**Purpose**: Enforce position sizing rules and circuit breakers. This component has veto power over all trades.

**Pre-Trade Checks**:

```rust
struct RiskCheck {
    fn check_position_size(&self, trade: &Trade) -> Result<()>;      // Max 2% risk per trade
    fn check_portfolio_heat(&self, trade: &Trade) -> Result<()>;     // Max 10% total open risk
    fn check_correlation(&self, trade: &Trade) -> Result<()>;        // Max 15% correlated exposure
    fn check_daily_loss(&self) -> Result<()>;                        // 2% daily limit
    fn check_weekly_loss(&self) -> Result<()>;                       // 5% weekly limit
    fn check_max_drawdown(&self) -> Result<()>;                      // 15% max drawdown
}
```

**Circuit Breakers** (trigger immediate halt):

| Condition | Action |
|-----------|--------|
| 1% loss in < 5 minutes | Halt trading, alert user |
| 3 consecutive losses | 15-minute pause |
| No quote update > 10 seconds | Halt, alert: stale data |
| 5+ API errors in 1 minute | Halt, alert: system issue |
| VIX > 40 | Reduce position sizes 75% |
| VIX > 50 | Halt new trades |

**Kill Switch**: Manual override via Slack command `/traderjim kill` closes all positions and disables trading until manual re-enable.

---

### 8. Reflection System

**Purpose**: Learn from trade outcomes to improve future recommendations.

**Trigger**: After each position closes (profit, loss, or time exit).

**Reflection Prompt**:

```
<context>
Trade: {trade_details}
Entry thesis: {original_thesis}
Outcome: {profit_loss}, {holding_period}
Market during trade: {price_action}, {vix_movement}
</context>

<task>
Analyze this completed trade:
1. Was the original thesis correct? Why or why not?
2. What market signals did we miss or correctly identify?
3. Would different entry/exit timing have improved outcome?
4. What should we do differently for similar setups?

Provide a concise lesson (1-2 sentences) to remember.
</task>
```

**Memory Storage**:

```rust
struct TradeReflection {
    trade_id: Uuid,
    entry_date: DateTime<Utc>,
    exit_date: DateTime<Utc>,
    underlying: Symbol,
    strategy: Strategy,
    outcome: Outcome,
    profit_loss: Decimal,
    original_thesis: String,
    reflection: String,
    lesson: String,
    tags: Vec<String>,  // e.g., ["high_iv", "earnings_play", "vix_spike"]
}
```

---

### 9. Memory Store

**Purpose**: Persist trade history, reflections, and evolving playbook for retrieval during analysis.

**Storage**: SQLite with full-text search (using FTS5).

**Schema**:

```sql
-- Core trade log
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    underlying TEXT,
    strategy TEXT,
    entry_price DECIMAL,
    exit_price DECIMAL,
    quantity INTEGER,
    profit_loss DECIMAL,
    outcome TEXT,  -- 'win', 'loss', 'scratch'
    thesis TEXT,
    reflection TEXT,
    lesson TEXT
);

-- Full-text search for lessons
CREATE VIRTUAL TABLE lessons_fts USING fts5(
    trade_id,
    lesson,
    tags
);

-- Playbook rules (evolving)
CREATE TABLE playbook (
    id TEXT PRIMARY KEY,
    rule TEXT,
    source TEXT,  -- 'initial', 'learned'
    confidence DECIMAL,
    supporting_trades TEXT,  -- JSON array of trade_ids
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Daily/weekly aggregates for performance tracking
CREATE TABLE performance (
    period TEXT,  -- '2024-01-15' or '2024-W03'
    period_type TEXT,  -- 'daily', 'weekly', 'monthly'
    trades INTEGER,
    wins INTEGER,
    losses INTEGER,
    total_pnl DECIMAL,
    max_drawdown DECIMAL,
    sharpe_ratio DECIMAL
);
```

**Retrieval for Analysis**:

```rust
// Before generating recommendations, retrieve relevant context
async fn get_relevant_memories(&self, candidate: &SpreadCandidate) -> Vec<TradeReflection> {
    // 1. Similar setups (same underlying, similar IV rank, similar delta)
    // 2. Recent lessons (last 30 days)
    // 3. Playbook rules tagged with relevant conditions
    
    let query = format!(
        "underlying:{} AND iv_rank:{}-{} AND outcome:*",
        candidate.underlying,
        candidate.iv_rank - 10,
        candidate.iv_rank + 10
    );
    
    self.db.search_reflections(&query, 5).await
}
```

---

## Data Flow

### Trade Lifecycle

```
1. SCAN
   Scheduler triggers → Analysis Engine fetches data → Candidates identified

2. ANALYZE  
   Candidates + Context → Claude API → Recommendations with thesis

3. NOTIFY
   Recommendations → Slack message with approve/reject buttons

4. APPROVE
   User clicks Approve → Validation checks → Risk Manager approval

5. EXECUTE
   Order placed → Monitor for fill → Confirm via Slack

6. MONITOR
   Track P/L → Check exit conditions → Generate exit recommendation when triggered

7. EXIT
   Exit recommendation → Approval flow → Close position

8. REFLECT
   Closed trade → Reflection prompt → Store lesson → Update playbook
```

---

## Technology Stack

### Core Dependencies

```toml
[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# HTTP client
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Financial math - CRITICAL: never use f64 for money
rust_decimal = { version = "1.39", features = ["serde", "maths"] }
rust_decimal_macros = "1.39"

# Dates and times
chrono = { version = "0.4", features = ["serde"] }
chrono-tz = "0.9"

# Options pricing and Greeks
optionstratlib = "0.10"

# Database
rusqlite = { version = "0.32", features = ["bundled", "serde_json"] }

# Error handling
thiserror = "2.0"
anyhow = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }

# Configuration
config = "0.14"
dotenvy = "0.15"

# UUID generation
uuid = { version = "1", features = ["v4", "serde"] }
```

### Project Structure

```
trader-jim/
├── Cargo.toml
├── config/
│   ├── default.toml        # Default configuration
│   ├── paper.toml          # Paper trading overrides
│   └── live.toml           # Live trading overrides
├── src/
│   ├── main.rs             # Entry point, scheduler setup
│   ├── lib.rs              # Library root
│   ├── config.rs           # Configuration loading
│   ├── broker/
│   │   ├── mod.rs          # Broker trait definition
│   │   ├── alpaca.rs       # Alpaca implementation
│   │   └── tastytrade.rs   # Tastytrade implementation
│   ├── analysis/
│   │   ├── mod.rs
│   │   ├── screener.rs     # Options screening logic
│   │   ├── greeks.rs       # Greeks calculation
│   │   └── iv_rank.rs      # IV Rank calculation
│   ├── ai/
│   │   ├── mod.rs
│   │   ├── claude.rs       # Claude API client
│   │   ├── prompts.rs      # Prompt templates
│   │   └── parser.rs       # Response parsing
│   ├── risk/
│   │   ├── mod.rs
│   │   ├── position_sizer.rs
│   │   ├── circuit_breaker.rs
│   │   └── validators.rs
│   ├── execution/
│   │   ├── mod.rs
│   │   ├── order_manager.rs
│   │   └── fill_monitor.rs
│   ├── notification/
│   │   ├── mod.rs
│   │   ├── slack.rs        # Slack integration
│   │   └── approval.rs     # Approval webhook handler
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── store.rs        # SQLite operations
│   │   ├── reflection.rs   # Reflection generation
│   │   └── playbook.rs     # Playbook management
│   └── types/
│       ├── mod.rs
│       ├── trade.rs        # Trade-related types
│       ├── position.rs     # Position types
│       └── market.rs       # Market data types
├── migrations/
│   └── 001_initial.sql     # Database schema
└── tests/
    ├── integration/
    └── fixtures/
```

---

## Deployment

### Infrastructure

**Primary**: Hetzner Cloud CX22 (2 vCPU, 4GB RAM) — €4.85/month

**Why Hetzner over alternatives**:

- Full Rust support (no WASM restrictions like Cloudflare Workers)
- Reliable cron via systemd (unlike GitHub Actions with 15-60min delays)
- Persistent storage for SQLite
- European servers = lower latency to NYSE/NASDAQ

### Deployment Process

```bash
# Build release binary locally (cross-compile for Linux)
cargo build --release --target x86_64-unknown-linux-gnu

# Deploy to server
scp target/x86_64-unknown-linux-gnu/release/trader-jim user@server:/opt/trader-jim/
scp config/*.toml user@server:/opt/trader-jim/config/

# On server: set up systemd service
sudo cp trader-jim.service /etc/systemd/system/
sudo systemctl enable trader-jim
sudo systemctl start trader-jim
```

### Systemd Service

```ini
# /etc/systemd/system/trader-jim.service
[Unit]
Description=Trader Jim Trading System
After=network.target

[Service]
Type=simple
User=traderjim
WorkingDirectory=/opt/trader-jim
Environment=RUST_LOG=info
Environment=TRADER_JIM_ENV=paper
ExecStart=/opt/trader-jim/trader-jim
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Secrets Management

```bash
# Store secrets in systemd drop-in
sudo mkdir -p /etc/systemd/system/trader-jim.service.d/
sudo cat > /etc/systemd/system/trader-jim.service.d/secrets.conf << EOF
[Service]
Environment=ALPACA_API_KEY=xxx
Environment=ALPACA_SECRET_KEY=xxx
Environment=ANTHROPIC_API_KEY=xxx
Environment=SLACK_BOT_TOKEN=xxx
Environment=SLACK_SIGNING_SECRET=xxx
EOF
sudo chmod 600 /etc/systemd/system/trader-jim.service.d/secrets.conf
```

---

## Monitoring & Observability

### Logging

Structured JSON logs via `tracing`:

```rust
tracing::info!(
    trade_id = %trade.id,
    underlying = %trade.underlying,
    action = "order_placed",
    price = %order.limit_price,
    "Placed spread order"
);
```

### Health Checks

**Healthchecks.io** (free tier):

- Ping on each successful analysis cycle
- Alert if no ping received within expected window

### Alerts

| Event | Channel | Priority |
|-------|---------|----------|
| Trade filled | Slack | Normal |
| Circuit breaker triggered | Slack + SMS | High |
| API errors | Slack | Normal |
| Daily P/L summary | Slack | Low |
| System down | SMS (via Healthchecks.io) | Critical |

---

## Security Considerations

1. **API Keys**: Never in code or config files—environment variables only
2. **Slack Verification**: Validate signing secret on all incoming webhooks  
3. **Rate Limiting**: Enforce on approval endpoint to prevent abuse
4. **Audit Log**: Immutable record of all orders and decisions
5. **Network**: Only outbound connections required; no inbound ports except Slack webhook
6. **Principle of Least Privilege**: Broker API keys scoped to trading only (no withdrawal capability)

---

## Testing Strategy

### Unit Tests

- Greeks calculations against known values
- Position sizing logic
- Circuit breaker state machine

### Integration Tests

- Broker API client against paper trading endpoint
- Full analysis → recommendation → approval flow with mocked Slack

### Backtesting

- Historical options data from CBOE DataShop or Polygon.io
- Replay through system with simulated fills
- Validate against known profitable periods

### Paper Trading Validation

- 100+ trades minimum before live deployment
- Must include at least one VIX > 30 event
- Track all metrics: win rate, profit factor, max drawdown, Sharpe

---

## Future Enhancements (Post-MVP)

1. **Iron Condors**: Add neutral strategy for range-bound markets
2. **Earnings Plays**: Specialized analysis for earnings volatility
3. **Position Adjustments**: Roll spreads that are being tested
4. **Multi-Account**: Support for multiple brokerage accounts
5. **Web Dashboard**: Read-only view of performance and open positions
6. **Reinforcement Learning**: PPO-based position sizing optimization
