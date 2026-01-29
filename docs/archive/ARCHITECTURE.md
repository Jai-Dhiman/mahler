# Mahler Architecture

A serverless, human-in-the-loop options trading system built on Cloudflare Workers with Python.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Mahler                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SCHEDULED WORKERS (Cron Triggers)                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Morning     │  │   Midday     │  │  Afternoon   │  │   EOD        │     │ 
│  │  Scan        │  │   Check      │  │  Scan        │  │   Summary    │     │
│  │  9:35 AM ET  │  │  12:00 PM ET │  │  3:30 PM ET  │  │  4:15 PM ET  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                                             │
│  ┌──────────────┐                                                           │
│  │  Position    │  every 5 min during market hours                          │
│  │  Monitor     │                                                           │
│  └──────────────┘                                                           │
│         │                                                                   │
│         └─────────────────────────┬──────────────────────────────────────── │
│                                   │                                         │
│  HTTP WORKERS                     │                                         │
│  ┌──────────────┐  ┌──────────────┤                                         │
│  │   Discord    │  │   Health     │                                         │
│  │   Webhook    │  │   Check      │                                         │
│  └──────────────┘  └──────────────┘                                         │
│         │                         │                                         │
│         ▼                         ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      CLOUDFLARE BINDINGS                            │    │
│  │                                                                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │     D1      │  │     KV      │  │     R2      │                  │    │
│  │  │  (SQLite)   │  │  (State)    │  │  (Archive)  │                  │    │
│  │  │             │  │             │  │             │                  │    │
│  │  │ trades      │  │ circuit     │  │ daily       │                  │    │
│  │  │ recs        │  │ breaker     │  │ snapshots   │                  │    │
│  │  │ positions   │  │ daily       │  │ options     │                  │    │
│  │  │ playbook    │  │ limits      │  │ chains      │                  │    │
│  │  │ performance │  │ rate limits │  │ backups     │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌───────────┐  ┌───────────┐  ┌───────────┐
             │  Alpaca   │  │  Claude   │  │  Discord  │
             │  (Broker) │  │  (AI)     │  │  (Notify) │
             └───────────┘  └───────────┘  └───────────┘
```

---

## Components

### Scheduled Workers

Cron-triggered workers handle the analysis and monitoring workload.

| Worker | Cron (UTC) | ET Time | Purpose |
|--------|------------|---------|---------|
| morning_scan | `35 13 * * 1-5` | 9:35 AM | Post-open opportunity scan |
| midday_check | `0 16 * * 1-5` | 12:00 PM | Position monitoring, new setups |
| afternoon_scan | `30 19 * * 1-5` | 3:30 PM | Final scan before close |
| eod_summary | `15 20 * * 1-5` | 4:15 PM | Daily P/L, reflection generation |
| position_monitor | `*/5 13-20 * * 1-5` | Every 5 min | Exit condition monitoring |

### HTTP Workers

| Worker | Route | Purpose |
|--------|-------|---------|
| discord_webhook | `POST /discord/interactions` | Handle approve/reject button clicks |
| health | `GET /health` | Health check endpoint |

### Storage Bindings

**D1 (SQLite)** - Primary relational storage

- Trade history and audit log
- Recommendations and their status
- Position snapshots
- Playbook rules and reflections
- Performance metrics

**KV** - Fast key-value state

- Circuit breaker status
- Daily trading limits
- Rate limiting counters
- Session state

**R2** - Object storage

- Historical options chain data
- Daily market snapshots
- Database backups
- Large analysis artifacts

---

## Data Flow

### Trade Recommendation Flow

```
Morning Scan Worker
       │
       ├──► Fetch options chains (Alpaca API)
       ├──► Screen candidates (IV rank, delta, DTE)
       ├──► Retrieve relevant memories (D1)
       ├──► Analyze with Claude API
       ├──► Save recommendation (D1)
       └──► Send to Discord with buttons
```

### Trade Approval Flow

```
User clicks [Approve] in Discord
       │
       ▼
Discord Webhook Worker
       │
       ├──► Verify Discord signature
       ├──► Load recommendation (D1)
       ├──► Check not expired
       ├──► Validate price hasn't moved >1%
       ├──► Check circuit breakers (KV)
       ├──► Place order (Alpaca API)
       ├──► Update recommendation status (D1)
       ├──► Create trade record (D1)
       └──► Update Discord message with confirmation
```

### Position Monitoring Flow

```
Position Monitor Worker (every 5 min)
       │
       ├──► Get open positions (Alpaca API)
       ├──► For each position:
       │    ├──► Calculate current P/L
       │    ├──► Check profit target (50%)
       │    ├──► Check stop loss (200%)
       │    └──► Check time exit (21 DTE)
       ├──► Update positions table (D1)
       └──► Send exit alerts to Discord if triggered
```

---

## Project Structure

```
mahler/
├── src/
│   ├── workers/
│   │   ├── router.py            # Main entry point
│   │   ├── morning_scan.py
│   │   ├── midday_check.py
│   │   ├── afternoon_scan.py
│   │   ├── eod_summary.py
│   │   ├── position_monitor.py
│   │   ├── discord_webhook.py
│   │   └── health.py
│   │
│   ├── core/
│   │   ├── broker/
│   │   │   ├── alpaca.py
│   │   │   └── types.py
│   │   ├── analysis/
│   │   │   ├── screener.py
│   │   │   ├── greeks.py
│   │   │   └── iv_rank.py
│   │   ├── ai/
│   │   │   ├── claude.py
│   │   │   └── prompts.py
│   │   ├── notifications/
│   │   │   └── discord.py
│   │   ├── risk/
│   │   │   ├── position_sizer.py
│   │   │   ├── circuit_breaker.py
│   │   │   └── validators.py
│   │   ├── db/
│   │   │   ├── d1.py
│   │   │   ├── kv.py
│   │   │   └── r2.py
│   │   └── types.py
│   │
│   └── migrations/
│       └── 0001_initial.sql
│
├── tests/
├── wrangler.toml
├── pyproject.toml
└── requirements.txt
```

---

## Database Schema

### D1 Tables

```sql
-- Recommendations awaiting approval
CREATE TABLE recommendations (
    id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, approved, rejected, expired, executed

    underlying TEXT NOT NULL,
    spread_type TEXT NOT NULL,      -- bull_put, bear_call
    short_strike REAL NOT NULL,
    long_strike REAL NOT NULL,
    expiration TEXT NOT NULL,

    credit REAL NOT NULL,
    max_loss REAL NOT NULL,

    iv_rank REAL,
    delta REAL,
    theta REAL,

    thesis TEXT,
    confidence TEXT,                -- low, medium, high
    suggested_contracts INTEGER,

    analysis_price REAL,
    discord_message_id TEXT
);

-- Executed trades
CREATE TABLE trades (
    id TEXT PRIMARY KEY,
    recommendation_id TEXT REFERENCES recommendations(id),

    opened_at TEXT,
    closed_at TEXT,
    status TEXT,                    -- open, closed

    entry_credit REAL,
    exit_debit REAL,
    profit_loss REAL,

    contracts INTEGER,
    broker_order_id TEXT,

    reflection TEXT,
    lesson TEXT
);

-- Position snapshots
CREATE TABLE positions (
    id TEXT PRIMARY KEY,
    trade_id TEXT REFERENCES trades(id),

    underlying TEXT,
    short_strike REAL,
    long_strike REAL,
    expiration TEXT,
    contracts INTEGER,

    current_value REAL,
    unrealized_pnl REAL,

    updated_at TEXT DEFAULT (datetime('now'))
);

-- Daily performance
CREATE TABLE daily_performance (
    date TEXT PRIMARY KEY,
    starting_balance REAL,
    ending_balance REAL,
    realized_pnl REAL,
    trades_opened INTEGER DEFAULT 0,
    trades_closed INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0
);

-- Playbook rules
CREATE TABLE playbook (
    id TEXT PRIMARY KEY,
    rule TEXT NOT NULL,
    source TEXT,                    -- initial, learned
    supporting_trade_ids TEXT,      -- JSON array
    created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX idx_recommendations_status ON recommendations(status);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_positions_underlying ON positions(underlying);
```

### KV Keys

| Key Pattern | Value | TTL |
|-------------|-------|-----|
| `circuit_breaker` | `{halted, reason, updated_at}` | None |
| `daily:{YYYY-MM-DD}` | `{trades, realized_pnl, losses}` | 7 days |
| `rate_limit:claude` | Request count | 1 hour |

---

## Configuration

### wrangler.toml

```toml
name = "mahler"
main = "src/workers/router.py"
compatibility_date = "2024-12-01"
compatibility_flags = ["python_workers"]

[triggers]
crons = [
    "35 13 * * 1-5",      # morning scan
    "0 16 * * 1-5",       # midday check
    "30 19 * * 1-5",      # afternoon scan
    "15 20 * * 1-5",      # eod summary
    "*/5 13-20 * * 1-5"   # position monitor
]

[[d1_databases]]
binding = "DB"
database_name = "mahler"
database_id = "<your-database-id>"

[[kv_namespaces]]
binding = "STATE"
id = "<your-kv-id>"

[[r2_buckets]]
binding = "MAHLER_BUCKET"
bucket_name = "mahler-bucket"

[vars]
ENVIRONMENT = "paper"
```

### Secrets

Set via `wrangler secret put`:

```
ALPACA_API_KEY
ALPACA_SECRET_KEY
ANTHROPIC_API_KEY
DISCORD_BOT_TOKEN
DISCORD_PUBLIC_KEY
```

---

## Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python | Works in Workers via Pyodide, familiar |
| Runtime | Cloudflare Workers | Serverless, cron triggers, native bindings |
| Database | D1 (SQLite) | Included with Workers, SQL support |
| State | KV | Fast reads for circuit breakers |
| Storage | R2 | Cheap object storage for archives |
| Broker | Alpaca | Free paper trading, good API |
| AI | Claude API | Strong reasoning for trade analysis |
| Notifications | Discord | Interactive buttons, rich embeds |

---

## Monitoring

### Logging

Workers logs available via:

- Cloudflare dashboard (real-time)
- `wrangler tail` (CLI streaming)

### Health Checks

External monitoring (e.g., Healthchecks.io) pings `/health` endpoint.

### Alerts

| Event | Channel |
|-------|---------|
| Trade recommendation | Discord |
| Trade executed | Discord |
| Exit condition triggered | Discord |
| Circuit breaker activated | Discord |
| Daily summary | Discord |
| Worker errors | Cloudflare notifications |

---

## Security

- API keys stored as Cloudflare secrets (encrypted at rest)
- Discord interaction signature verification on all incoming requests
- Broker API keys scoped to trading only (no withdrawal)
- No sensitive data in logs
- D1/KV/R2 access restricted to worker bindings

---

## Cost Estimate

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| Cloudflare Workers | Existing plan | $0 incremental |
| D1 | <1GB | $0 (included) |
| KV | Minimal | $0 (included) |
| R2 | <1GB | $0 (free tier) |
| Claude API | ~10 calls/day | $15-20 |
| Alpaca | Paper trading | $0 |
| Discord | Free | $0 |
| **Total** | | **~$15-20/mo** |

---

## Runtime Capabilities (Pro Plan)

Mahler runs on **Cloudflare Workers Pro Plan** with Python via Pyodide, which provides significant computational resources.

### CPU Time Limits

| Trigger Type | Interval | CPU Time Limit |
|--------------|----------|----------------|
| Cron (>=1hr intervals) | morning_scan, midday_check, afternoon_scan, eod_summary | **15 minutes** |
| Cron (<1hr intervals) | position_monitor (every 5 min) | **30 seconds** |
| HTTP requests | discord_webhook, health | **30 seconds** |

CPU time is configurable via `cpu_ms` in wrangler.toml (default 30s, max 15min for hourly+ crons).

### Scientific Python Stack

Pyodide supports the **full scientific Python ecosystem** via pyproject.toml dependencies. Cloudflare uses memory snapshots at deploy time for fast cold starts.

**Available packages:**
- **NumPy** - Array operations, linear algebra, correlation matrices
- **Pandas** - Time series analysis, DataFrames, rolling windows
- **SciPy** - Optimization (scipy.optimize), statistics (scipy.stats), signal processing
- **scikit-learn** - GaussianMixture for regime detection, clustering, ML models
- **statsmodels** - Statistical tests, time series models (ARIMA, etc.)
- **matplotlib** - Charting (if needed for reports)

### Memory

| Resource | Limit |
|----------|-------|
| Worker memory | 128MB |
| D1 database | 10GB |

---

## Limitations

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| 128MB memory | Large datasets must be chunked | Process in batches; use generators |
| Cron precision | +/- few seconds | Acceptable for swing trading |
| D1 row limits | 10GB max | Archive old data to R2 |
| position_monitor CPU | 30s limit (5-min interval) | Keep monitoring logic lightweight |
