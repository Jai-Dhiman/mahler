# Mahler V2: Autonomous Multi-Agent Options Trading System

## Architecture Design Document

**Version**: 2.0
**Status**: Design Phase
**Last Updated**: 2026-01-27

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Principles](#2-design-principles)
   - 2.1 [Source-Backed Design Decisions](#21-source-backed-design-decisions)
   - 2.2 [Options-Specific Requirements](#22-options-specific-requirements)
   - 2.3 [Risk Philosophy](#23-risk-philosophy)
   - 2.4 [Key Architectural Decisions](#24-key-architectural-decisions)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Layer 1: Data & Backtesting Infrastructure](#4-layer-1-data--backtesting-infrastructure)
5. [Layer 2: Multi-Agent Analysis System](#5-layer-2-multi-agent-analysis-system)
6. [Layer 3: Memory & Reflection System](#6-layer-3-memory--reflection-system)
7. [Layer 4: Decision & Risk Management](#7-layer-4-decision--risk-management)
8. [Layer 5: Execution & Monitoring](#8-layer-5-execution--monitoring)
9. [Options-Specific Components](#9-options-specific-components)
10. [Data Synthesis & Continuous Learning](#10-data-synthesis--continuous-learning)
11. [Technology Stack](#11-technology-stack)
    - 11.5 [LLM Migration Path](#115-llm-migration-path)
12. [Performance Targets](#12-performance-targets)
13. [References](#13-references)
14. [Appendix A: Migration Path from V1](#appendix-a-migration-path-from-v1)
15. [Appendix B: Key Decisions Summary](#appendix-b-key-decisions-summary)

---

## 1. Executive Summary

Mahler V2 is a fully autonomous options trading system designed for credit spread strategies on high-liquidity ETFs. The system is **Cloudflare-native**, leveraging the existing V1 infrastructure while adding sophisticated multi-agent capabilities.

The system combines:

- **Multi-agent debate architecture** inspired by [TradingAgents](https://arxiv.org/abs/2412.20138) and [FINCON](https://arxiv.org/abs/2407.06567)
- **Layered memory with self-reflection** from [FinMem](https://arxiv.org/abs/2311.13743) and [TradingGroup](https://arxiv.org/abs/2508.17565)
- **Rigorous backtesting** using [ORATS](https://orats.com/) historical options data
- **Options-specific analytics** via [vollib](https://vollib.org/) and custom IV analysis
- **Hard-coded circuit breakers** that cannot be overridden by any agent
- **Cloudflare infrastructure** (Workers, D1, Vectorize, KV, R2) for zero-ops serverless deployment

### Key Differences from V1

| Aspect | V1 (Current) | V2 (Proposed) |
|--------|--------------|---------------|
| Human Approval | Required (Discord) | Fully Autonomous |
| Agent Architecture | Single Claude call | Multi-agent debate |
| Memory | Last 5 playbook rules | Layered (working/episodic/semantic) |
| Learning | Text-based reflections | Quantitative feedback + fine-tuning |
| Backtesting | None | Walk-forward validation |
| Strategy Validation | Heuristics | Statistically validated parameters |
| Infrastructure | Cloudflare (D1/KV/R2) | Cloudflare (D1/KV/R2 + **Vectorize**) |
| LLM | Claude Sonnet | Claude Sonnet now, **fine-tuned Qwen later** |

---

## 2. Design Principles

### 2.1 Source-Backed Design Decisions

Every architectural choice cites academic research or production-tested frameworks:

1. **Multi-agent debate reduces hallucination** - TradingAgents achieved Sharpe ratios of 5.60-8.21 vs baselines of 2.07-4.14 using bull/bear researcher debate [[1]](#ref-1)

2. **Hierarchical communication prevents information loss** - FINCON's manager-analyst structure outperformed peer-to-peer by avoiding the "telephone effect" [[2]](#ref-2)

3. **Self-reflection improves decisions** - TradingGroup's reflection mechanism achieved 40.46% cumulative return vs 13.27% baseline on AMZN [[3]](#ref-3)

4. **Layered memory mirrors human cognition** - FinMem's architecture enables retention of critical information beyond human limits [[4]](#ref-4)

5. **Autonomous outperforms human-in-the-loop** - Stanford research shows AI beat 93% of fund managers over 30 years [[5]](#ref-5)

### 2.2 Options-Specific Requirements

Credit spread trading differs fundamentally from directional stock trading:

- **Non-linear payoffs**: Max profit/loss defined at entry
- **Time decay (theta)**: Primary profit source, must be modeled
- **Volatility sensitivity (vega)**: IV crush/expansion critical
- **Correlation risk**: SPY/QQQ/IWM are 86-92% correlated
- **Discrete events**: Earnings, Fed, economic data cause IV spikes

### 2.3 Risk Philosophy

From [credit spread research](https://optionalpha.com/blog/spy-put-credit-spread-backtest):

> "Introducing profit targets, stop-losses, and contract rolling had a substantial impact on overall strategy performance. These additions dramatically lowered volatility, shrinking the max loss from $12,000 to $2,000, with a notably better Sharpe ratio."

**Implication**: Hard-coded risk rules outperform AI-controlled risk decisions.

### 2.4 Key Architectural Decisions

These decisions were made after evaluating alternatives and are documented here with rationale.

#### Decision 1: Claude API Now, Fine-Tuned Qwen Later

| Option | Pros | Cons |
|--------|------|------|
| Claude API | Best reasoning, zero infrastructure, immediate start | Per-call cost, can't fine-tune |
| Self-hosted Qwen3-8B | No per-call cost, can fine-tune, lower latency | GPU costs ($500-2000/mo), need training data first |

**Decision**: Start with Claude Sonnet API, migrate to fine-tuned Qwen3-8B after accumulating 500+ labeled trades.

**Rationale**:

- [TradingGroup](https://arxiv.org/abs/2508.17565) needed 1,080 labeled trajectories to fine-tune effectively
- Claude's reasoning is stronger than base Qwen for debate/analysis tasks
- Cost at 10-20 calls/day is ~$15-30/month, not worth infrastructure complexity initially
- Clear migration path: accumulate data -> fine-tune Qwen3-8B with LoRA -> deploy to Cloudflare Workers AI or external GPU service

#### Decision 2: Cloudflare Vectorize + D1 (Not PostgreSQL + pgvector)

| Option | Pros | Cons |
|--------|------|------|
| PostgreSQL + pgvector | 99%+ accuracy, full SQL, mature | $15-50/mo, ops overhead, separate infra |
| Cloudflare Vectorize + D1 | Zero ops, free tier sufficient, existing infra | ~80% accuracy (approximate), 10M vector limit |

**Decision**: Stay on Cloudflare stack (Vectorize + D1 + KV + R2).

**Rationale**:

- Expected corpus size: <5,000 vectors (trades + rules). At this scale, approximate search has near-perfect recall
- [Cloudflare Vectorize](https://developers.cloudflare.com/vectorize/) supports `returnValues: true` for high-precision mode when needed
- V1 already uses D1/KV/R2 - adding Vectorize is one line in wrangler.toml
- Operational simplicity matters for a solo developer
- Free tier: 30M queried dimensions/month, 5M stored dimensions (sufficient)

#### Decision 3: Simple RAG (No Hybrid Retrieval, No Reranking)

| Technique | When Helpful | Our Situation |
|-----------|--------------|---------------|
| Hybrid (BM25 + Vector) | Large corpus, keyword-heavy queries | Small corpus, structured queries |
| Reranking (Cross-Encoder) | Many false positives, need top 5 from 50+ | Retrieving 5-10 candidates max |
| Query Classification | Mixed query types, free-form input | Predictable system-generated queries |

**Decision**: Start with simple vector similarity retrieval. Add complexity only if retrieval quality is poor.

**Rationale**:

- [RAG best practices research](https://arxiv.org/html/2407.01219v1) shows hybrid + reranking helps for large, diverse corpora
- Our queries are structured and predictable (not free-form user input)
- Claude's large context window (200K) can handle 10+ full trade memories
- Keyword queries (e.g., "SPY trades in March") can use D1 SQL directly
- Premature optimization: add hybrid/reranking in Phase 2 if needed

**Future Enhancement Path**:

```
Phase 1: Vectorize similarity (topK=5-10)
Phase 2 (if needed): Add cross-encoder reranking (ms-marco-MiniLM-L-12-v2)
Phase 3 (if needed): Add hybrid retrieval (D1 full-text + Vectorize)
```

---

## 3. System Architecture Overview

```
+===========================================================================+
|                     MAHLER V2: AUTONOMOUS OPTIONS TRADING                  |
+===========================================================================+
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  | LAYER 5: HARD CIRCUIT BREAKERS (Non-AI, Rule-Based)                 |  |
|  |   - Max daily loss: 2% -> HALT                                      |  |
|  |   - Max drawdown: 15% -> HALT                                       |  |
|  |   - VIX > 50 -> HALT                                                |  |
|  |   CANNOT BE OVERRIDDEN BY ANY AGENT                                 |  |
|  +---------------------------------------------------------------------+  |
|                                    |                                      |
|  +---------------------------------------------------------------------+  |
|  | LAYER 4: DECISION & RISK MANAGEMENT                                 |  |
|  |   +------------------+    +------------------+    +---------------+  |  |
|  |   | Trading Decision |<---| Risk Manager     |<---| Position      |  |  |
|  |   | Agent            |    | (3 perspectives) |    | Sizer         |  |  |
|  |   +------------------+    +------------------+    +---------------+  |  |
|  +---------------------------------------------------------------------+  |
|                                    ^                                      |
|  +---------------------------------------------------------------------+  |
|  | LAYER 3: MEMORY & REFLECTION                                        |  |
|  |   +----------------+  +----------------+  +-------------------+      |  |
|  |   | Working Memory |  | Episodic Memory|  | Semantic Memory   |      |  |
|  |   | (current state)|  | (recent trades)|  | (validated rules) |      |  |
|  |   +----------------+  +----------------+  +-------------------+      |  |
|  |                              |                                       |  |
|  |   +--------------------------------------------------------+        |  |
|  |   | Self-Reflection Engine: Compare predicted vs actual     |        |  |
|  |   | Update beliefs, propagate to relevant agents            |        |  |
|  |   +--------------------------------------------------------+        |  |
|  +---------------------------------------------------------------------+  |
|                                    ^                                      |
|  +---------------------------------------------------------------------+  |
|  | LAYER 2: MULTI-AGENT ANALYSIS                                       |  |
|  |                                                                     |  |
|  |   ANALYST AGENTS (parallel)                                         |  |
|  |   +------------+ +------------+ +------------+ +------------+       |  |
|  |   | IV/Term    | | Technical  | | Macro/     | | Greeks/    |       |  |
|  |   | Structure  | | Analyst    | | Events     | | Risk       |       |  |
|  |   +------------+ +------------+ +------------+ +------------+       |  |
|  |          |             |              |              |              |  |
|  |          +-------------+--------------+--------------+              |  |
|  |                              |                                      |  |
|  |   DEBATE LAYER (sequential rounds)                                  |  |
|  |   +----------------------------+  +----------------------------+    |  |
|  |   | Bull Researcher            |  | Bear Researcher            |    |  |
|  |   | "IV at 75%, contango,      |  | "VIX rising, earnings in   |    |  |
|  |   |  favorable for selling"    |  |  5 days, avoid entry"      |    |  |
|  |   +----------------------------+  +----------------------------+    |  |
|  |                              |                                      |  |
|  |   +--------------------------------------------------------+        |  |
|  |   | Debate Facilitator: Synthesize, record outcome          |        |  |
|  |   +--------------------------------------------------------+        |  |
|  +---------------------------------------------------------------------+  |
|                                    ^                                      |
|  +---------------------------------------------------------------------+  |
|  | LAYER 1: DATA & BACKTESTING                                         |  |
|  |   +----------------+  +----------------+  +-------------------+      |  |
|  |   | ORATS API      |  | Alpaca Data    |  | Options Chain     |      |  |
|  |   | (historical)   |  | (real-time)    |  | Calculator        |      |  |
|  |   +----------------+  +----------------+  +-------------------+      |  |
|  |          |                   |                    |                 |  |
|  |   +--------------------------------------------------------+        |  |
|  |   | Backtesting Engine (walk-forward validation)            |        |  |
|  |   | - Train: 6 months | Validate: 1 month | Roll forward    |        |  |
|  |   +--------------------------------------------------------+        |  |
|  +---------------------------------------------------------------------+  |
|                                                                           |
+===========================================================================+
```

---

## 4. Layer 1: Data & Backtesting Infrastructure

### 4.1 Data Sources

#### 4.1.1 Historical Options Data (ORATS)

**Source**: [ORATS API Documentation](https://docs.orats.io/)

ORATS provides:

- Historical data from 2007 to present
- Greeks, IV, and theoretical values
- 5,000+ symbols
- Near-EOD snapshots (14 minutes before close)

**API Endpoints**:

```
POST https://api.orats.io/backtest/submit   # Submit backtest
POST https://api.orats.io/backtest/status   # Check status
GET  https://api.orats.io/backtest/results  # Get results
```

**Data Schema**:

```python
@dataclass
class OptionsSnapshot:
    underlying_symbol: str
    underlying_price: float
    quote_date: date
    expiration_date: date
    strike: float
    option_type: Literal["call", "put"]
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
```

#### 4.1.2 Real-Time Data (Alpaca)

For live trading, continue using Alpaca's options API:

- Real-time quotes during market hours
- Order placement and management
- Position tracking

### 4.2 Backtesting Engine

**Design inspired by**: [ORATS Backtester Methodology](https://docs.orats.io/backtest-api-guide/backtester-methodology.html)

#### 4.2.1 Walk-Forward Validation

From [alpha decay research](https://www.mavensecurities.com/alpha-decay-what-does-it-look-like-and-what-does-it-mean-for-systematic-traders/):

> "Studies have shown that alpha on new trades decays in about 12 months on average."

**Implementation**:

```
Training Window:   6 months
Validation Window: 1 month
Test Window:       1 month (held out)
Roll Forward:      Monthly

Example:
  Train:    Jan 2023 - Jun 2023
  Validate: Jul 2023
  Test:     Aug 2023

  Then roll:
  Train:    Feb 2023 - Jul 2023
  Validate: Aug 2023
  Test:     Sep 2023
```

#### 4.2.2 Backtesting Parameters

Based on [Option Alpha backtesting research](https://optionalpha.com/blog/spy-put-credit-spread-backtest):

```python
@dataclass
class BacktestConfig:
    # Entry criteria
    dte_min: int = 30
    dte_max: int = 45
    short_delta_min: float = 0.10  # Research: further OTM is more consistent
    short_delta_max: float = 0.15
    iv_percentile_min: float = 50

    # Exit criteria
    profit_target_pct: float = 50   # 50% of max credit
    stop_loss_pct: float = 125      # Research: 125% better than 200%
    dte_exit: int = 21              # Time-based exit

    # Position sizing
    max_risk_per_trade_pct: float = 2.0
    max_portfolio_risk_pct: float = 10.0

    # Slippage (from ORATS methodology)
    slippage_pct: float = 0.10  # 10% of bid-ask spread
    commission_per_contract: float = 1.00
```

#### 4.2.3 Performance Metrics

```python
@dataclass
class BacktestResults:
    total_trades: int
    win_rate: float                 # Target: >= 70% (further OTM)
    profit_factor: float            # Target: >= 1.5
    sharpe_ratio: float             # Target: >= 1.0
    max_drawdown_pct: float         # Target: <= 15%
    avg_days_in_trade: float
    cumulative_return_pct: float
    annualized_return_pct: float

    # By regime
    performance_by_regime: dict[MarketRegime, RegimeMetrics]
```

### 4.3 Options Analytics (vollib)

**Source**: [vollib Documentation](https://vollib.org/)

```python
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho

class OptionsCalculator:
    """
    Greeks calculation using vollib's Black-Scholes implementation.

    vollib uses Peter Jackel's "Let's Be Rational" algorithm for
    implied volatility, achieving machine precision in 2 iterations.
    """

    def calculate_greeks(
        self,
        flag: Literal["c", "p"],
        S: float,      # Underlying price
        K: float,      # Strike
        t: float,      # Time to expiration (years)
        r: float,      # Risk-free rate
        sigma: float   # Implied volatility
    ) -> Greeks:
        return Greeks(
            delta=delta(flag, S, K, t, r, sigma),
            gamma=gamma(flag, S, K, t, r, sigma),
            theta=theta(flag, S, K, t, r, sigma),
            vega=vega(flag, S, K, t, r, sigma),
            rho=rho(flag, S, K, t, r, sigma)
        )

    def implied_vol_from_price(
        self,
        price: float,
        flag: Literal["c", "p"],
        S: float,
        K: float,
        t: float,
        r: float
    ) -> float:
        return implied_volatility(price, S, K, t, r, flag)
```

---

## 5. Layer 2: Multi-Agent Analysis System

### 5.1 Architecture Overview

**Inspired by**:

- [TradingAgents](https://arxiv.org/abs/2412.20138): Multi-agent debate with bull/bear researchers
- [FINCON](https://arxiv.org/abs/2407.06567): Manager-analyst hierarchy

```
                    +------------------+
                    |  Debate          |
                    |  Facilitator     |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                                 |
    +-------v-------+                 +-------v-------+
    | Bull          |    DEBATE       | Bear          |
    | Researcher    |<--------------->| Researcher    |
    +-------^-------+   (N rounds)    +-------^-------+
            |                                 |
            +----------------+----------------+
                             |
    +------------------------+------------------------+
    |           |            |            |           |
+---v---+  +---v---+   +----v----+  +----v----+      |
| IV/   |  | Tech  |   | Macro/  |  | Greeks/ |      |
| Term  |  | Analy |   | Events  |  | Risk    |      |
| Struct|  | st    |   | Agent   |  | Agent   |      |
+-------+  +-------+   +---------+  +---------+      |
    |           |            |            |           |
    +------------------------+------------------------+
                             |
                    +--------v---------+
                    |  Market Data     |
                    |  (Options Chain) |
                    +------------------+
```

### 5.2 Analyst Agents

Each analyst agent processes specific data and outputs structured insights.

#### 5.2.1 IV/Term Structure Analyst

**Purpose**: Assess volatility environment for premium selling

```python
@dataclass
class IVAnalystOutput:
    iv_percentile: float          # 0-100, primary filter
    iv_rank: float                # 0-100, secondary context
    term_structure: Literal["contango", "flat", "backwardation"]
    term_slope: float             # Positive = contango (favorable)
    mean_reversion_signal: Literal["sell_vol", "buy_vol", "neutral"]
    mean_reversion_zscore: float
    recommendation: str           # Natural language summary
    confidence: float             # 0-1
```

**System Prompt**:

```
You are an options volatility analyst specializing in term structure and
mean reversion. Your job is to assess whether the current IV environment
is favorable for selling credit spreads.

Key considerations:
- IV Percentile > 50% indicates elevated premiums worth selling
- Contango (short-term IV < long-term IV) favors selling near-term premium
- Z-score > 2.0 from long-term IV mean suggests reversion opportunity

You receive: IV percentile, IV rank, term structure data, historical IV
You output: Structured JSON with your assessment
```

#### 5.2.2 Technical Analyst

**Purpose**: Identify support/resistance levels and trend context

```python
@dataclass
class TechnicalAnalystOutput:
    trend: Literal["bullish", "bearish", "neutral"]
    trend_strength: float         # 0-1
    key_support_levels: list[float]
    key_resistance_levels: list[float]
    rsi_14: float
    macd_signal: Literal["bullish", "bearish", "neutral"]
    bollinger_position: float     # 0-1 (0=lower band, 1=upper)
    recommendation: str
    confidence: float
```

**System Prompt**:

```
You are a technical analyst specializing in options strike selection.
Your job is to identify optimal strike placement relative to support/resistance.

For bull put spreads: Sell puts BELOW strong support zones
For bear call spreads: Sell calls ABOVE strong resistance zones

You receive: OHLCV data, technical indicators (RSI, MACD, Bollinger, ATR)
You output: Structured JSON with support/resistance levels and trend assessment
```

#### 5.2.3 Macro/Events Analyst

**Purpose**: Flag upcoming events that could impact positions

```python
@dataclass
class MacroAnalystOutput:
    upcoming_events: list[MarketEvent]
    event_risk_score: float       # 0-1 (1 = high risk)
    vix_level: float
    vix_regime: Literal["low", "normal", "elevated", "high", "extreme"]
    fed_calendar_risk: bool
    earnings_within_dte: bool
    recommendation: str
    confidence: float

@dataclass
class MarketEvent:
    date: date
    event_type: str               # earnings, fed, economic_data
    symbol_affected: str | None
    expected_iv_impact: Literal["high", "medium", "low"]
```

**System Prompt**:

```
You are a macro analyst monitoring events that impact options positions.
Credit spreads are vulnerable to:
- Earnings announcements (IV crush or gap risk)
- FOMC meetings (market-wide volatility)
- Economic data releases (employment, CPI, GDP)

For 30-45 DTE positions, events within the holding period are critical.

You receive: Economic calendar, earnings dates, VIX level, news headlines
You output: Structured JSON with event assessment and risk score
```

#### 5.2.4 Greeks/Risk Analyst

**Purpose**: Portfolio-level risk assessment

```python
@dataclass
class GreeksAnalystOutput:
    portfolio_delta: float        # Target: <= 0.30 absolute
    portfolio_gamma: float        # Target: <= 0.20
    portfolio_theta: float        # Daily decay (positive = good)
    portfolio_vega: float         # IV sensitivity
    correlation_risk: float       # 0-1 (1 = highly correlated positions)
    concentration_risk: str       # Which underlyings dominate
    max_loss_scenario: float      # Worst case portfolio loss
    recommendation: str
    confidence: float
```

### 5.3 Debate Layer

**Source**: TradingAgents "n rounds of natural language dialogue" between opposing researchers [[1]](#ref-1)

#### 5.3.1 Bull Researcher

**System Prompt**:

```
You are a bullish researcher advocating for the trade opportunity.
Your job is to highlight positive signals and reasons to enter.

You receive: All analyst outputs
You argue: Why this trade has favorable risk/reward

Be specific and cite data:
- "IV percentile at 72% provides adequate premium cushion"
- "Strong support at $445 is 3% below short strike"
- "No earnings for SPY, low event risk"

Acknowledge risks briefly, but emphasize why they are manageable.
```

#### 5.3.2 Bear Researcher

**System Prompt**:

```
You are a bearish researcher highlighting risks and reasons to avoid.
Your job is to identify potential problems with the trade.

You receive: All analyst outputs
You argue: Why this trade should be avoided or sized down

Be specific and cite data:
- "VIX trending higher suggests increasing uncertainty"
- "Term structure in backwardation - smart money buying protection"
- "Portfolio delta already at 0.25, adding more increases directional risk"

Acknowledge positive signals, but emphasize why risks outweigh them.
```

#### 5.3.3 Debate Facilitator

**Inspired by**: TradingAgents' "designated facilitator agent guides debates, reviews discussion history, selects the prevailing perspective"

```python
@dataclass
class DebateConfig:
    max_rounds: int = 3
    consensus_threshold: float = 0.7  # Agreement level to stop early

@dataclass
class DebateOutcome:
    rounds_conducted: int
    winning_perspective: Literal["bull", "bear", "neutral"]
    confidence: float
    key_bull_arguments: list[str]
    key_bear_arguments: list[str]
    synthesis: str                # Natural language summary
    trade_recommendation: Literal["enter", "skip", "reduce_size"]
```

**System Prompt**:

```
You are the debate facilitator synthesizing bull and bear perspectives.

After each round:
1. Identify the strongest arguments from each side
2. Note any arguments that were not adequately countered
3. Determine if consensus is emerging

After final round:
1. Declare which perspective prevailed and why
2. Record the key unchallenged arguments
3. Provide a clear trade recommendation

Your output determines whether the system enters the trade.
```

### 5.4 Agent Communication Protocol

**Inspired by**: FINCON's hybrid structured documents + natural language dialogue [[2]](#ref-2)

```python
@dataclass
class AgentMessage:
    agent_id: str
    timestamp: datetime
    message_type: Literal["analysis", "argument", "synthesis"]
    content: str                  # Natural language
    structured_data: dict | None  # JSON-serializable metrics
    confidence: float

class CommunicationBus:
    """
    Manages agent communication following FINCON's hierarchical pattern.

    Analysts -> Researchers -> Facilitator -> Decision Agent

    This avoids the "telephone effect" where information degrades
    through extended peer-to-peer conversations.
    """

    def broadcast_to_researchers(
        self,
        analyst_outputs: list[AgentMessage]
    ) -> None:
        """Analysts send structured outputs to both researchers."""
        pass

    def conduct_debate_round(
        self,
        bull_argument: AgentMessage,
        bear_argument: AgentMessage
    ) -> DebateRoundResult:
        """Facilitator processes one round of debate."""
        pass
```

---

## 6. Layer 3: Memory & Reflection System

### 6.1 Layered Memory Architecture

**Source**: FinMem's cognitive structure that "aligns closely with human traders" [[4]](#ref-4)

```
+------------------------------------------------------------------+
|                     MEMORY SYSTEM                                 |
+------------------------------------------------------------------+
|                                                                   |
|  WORKING MEMORY (Immediate Context)                               |
|  +-------------------------------------------------------------+  |
|  | - Current market state (prices, IV, VIX)                    |  |
|  | - Open positions with Greeks                                |  |
|  | - Pending orders                                            |  |
|  | - Today's P&L                                               |  |
|  | - Active debate context                                     |  |
|  | TTL: Current session only                                   |  |
|  +-------------------------------------------------------------+  |
|                              |                                    |
|  EPISODIC MEMORY (Recent Trades)                                  |
|  +-------------------------------------------------------------+  |
|  | - Last 30 days of trades with full context                  |  |
|  | - Entry thesis, debate outcome, analyst outputs             |  |
|  | - Actual outcome vs predicted                               |  |
|  | - Reflection notes                                          |  |
|  | TTL: 30 days rolling                                        |  |
|  +-------------------------------------------------------------+  |
|                              |                                    |
|  SEMANTIC MEMORY (Validated Rules)                                |
|  +-------------------------------------------------------------+  |
|  | - Statistically validated trading rules                     |  |
|  | - Rules require p < 0.05 and minimum 10 supporting trades   |  |
|  | - Examples:                                                  |  |
|  |   "Skip trades when VIX > 30" (p=0.02, n=15)               |  |
|  |   "Reduce size when term structure inverted" (p=0.04, n=12)|  |
|  | TTL: Until invalidated by new data                          |  |
|  +-------------------------------------------------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

### 6.2 Memory Schema

```python
@dataclass
class WorkingMemory:
    """Volatile, session-scoped memory."""
    market_state: MarketState
    open_positions: list[Position]
    pending_orders: list[Order]
    daily_pnl: float
    current_debate: DebateContext | None

@dataclass
class EpisodicMemory:
    """Recent trade history with full context."""
    trade_id: str
    entry_date: date
    exit_date: date | None

    # Entry context
    entry_thesis: str
    debate_outcome: DebateOutcome
    analyst_outputs: dict[str, AgentMessage]
    predicted_outcome: PredictedOutcome

    # Actual results
    actual_outcome: ActualOutcome | None
    pnl: float | None
    pnl_pct: float | None

    # Reflection (populated after close)
    reflection: TradeReflection | None

@dataclass
class SemanticMemory:
    """Long-term validated rules."""
    rule_id: str
    rule_text: str
    source: Literal["initial", "learned"]

    # Statistical validation
    supporting_trades: list[str]  # trade_ids
    opposing_trades: list[str]
    effect_size: float            # Avg P&L improvement
    p_value: float
    last_validated: date

    # Propagation (from FINCON)
    applies_to_agents: list[str]  # Which agents receive this rule
```

### 6.3 Self-Reflection Engine

**Source**: TradingGroup's self-reflection that "extracts recent successful and failed cases" and "summarizes patterns and root causes" [[3]](#ref-3)

```python
class SelfReflectionEngine:
    """
    Inspired by TradingGroup's self-reflection mechanism:

    "The self-reflection mechanism can more accurately obtain historical
    successful and failed cases and conduct efficient analysis compared
    to traditional RAG approaches."

    Key difference from V1: Quantitative comparison, not just text generation.
    """

    def generate_reflection(
        self,
        trade: EpisodicMemory
    ) -> TradeReflection:
        """
        Compare predicted vs actual outcome.

        Returns structured analysis, not just narrative.
        """
        return TradeReflection(
            # What we predicted
            predicted_direction=trade.predicted_outcome.direction,
            predicted_confidence=trade.predicted_outcome.confidence,
            predicted_max_drawdown=trade.predicted_outcome.max_drawdown,

            # What actually happened
            actual_direction=self._calculate_direction(trade.pnl),
            actual_max_drawdown=trade.actual_outcome.max_drawdown,

            # Analysis
            prediction_accurate=self._was_accurate(trade),
            key_factors_correct=self._identify_correct_factors(trade),
            key_factors_wrong=self._identify_wrong_factors(trade),

            # Actionable insight
            lesson=self._extract_lesson(trade),
            applies_to_regime=trade.actual_outcome.market_regime,
            confidence_in_lesson=self._calculate_lesson_confidence(trade)
        )

    def update_semantic_memory(
        self,
        recent_reflections: list[TradeReflection],
        current_rules: list[SemanticMemory]
    ) -> list[RuleUpdate]:
        """
        Inspired by FINCON's conceptual verbal reinforcement:

        "Selectively propagates insights back to relevant agents
        rather than broadcasting system-wide."

        Only create rules with statistical validation.
        """
        candidate_rules = self._identify_patterns(recent_reflections)
        validated_rules = []

        for rule in candidate_rules:
            p_value = self._mann_whitney_test(rule)
            if p_value < 0.05 and rule.supporting_trades >= 10:
                validated_rules.append(RuleUpdate(
                    action="add",
                    rule=rule,
                    propagate_to=self._identify_relevant_agents(rule)
                ))

        return validated_rules
```

### 6.4 Memory Retrieval

**Inspired by**: FinMem's "adjustable cognitive span" and TradingGroup's similarity-based retrieval

**Implementation**: Simple vector similarity via Cloudflare Vectorize (no hybrid retrieval or reranking initially - see Section 2.4 Decision 3).

```python
class MemoryRetriever:
    """
    Retrieves relevant memories using Cloudflare Vectorize.

    Simple approach (Phase 1):
    - Vector similarity for episodic memory
    - D1 SQL queries for semantic rules
    - No hybrid retrieval or reranking (add if needed)

    Uses three scoring dimensions (from LLM agent research):
    - Recency: More recent memories weighted higher
    - Relevancy: Cosine similarity of market conditions
    - Importance: Based on P&L magnitude and lesson confidence
    """

    def __init__(self, vectorize_index, d1_client, embedding_model):
        self.vectorize = vectorize_index      # Cloudflare Vectorize
        self.d1 = d1_client                   # Cloudflare D1
        self.embedder = embedding_model       # e.g., all-MiniLM-L6-v2

    async def retrieve_relevant_episodes(
        self,
        current_context: WorkingMemory,
        max_results: int = 5
    ) -> list[EpisodicMemory]:
        """
        Find similar past trades using Vectorize.

        Simple vector similarity - no hybrid retrieval for now.
        If retrieval quality is poor, add reranking in Phase 2.
        """
        # Embed current market conditions
        context_embedding = self.embedder.encode(
            self._context_to_text(current_context)
        )

        # Query Vectorize for similar trades
        # returnValues=True for high-precision scoring
        results = await self.vectorize.query(
            vector=context_embedding,
            topK=max_results * 2,  # Over-fetch for post-filtering
            returnValues=True,      # High precision mode
            returnMetadata="all"
        )

        # Post-process with recency and importance weighting
        scored = []
        for match in results.matches:
            episode = await self._load_episode(match.id)
            score = (
                match.score * 0.5 +                           # Relevancy (from Vectorize)
                self._recency_score(episode) * 0.3 +          # Recency
                self._importance_score(episode) * 0.2         # P&L magnitude
            )
            scored.append((episode, score))

        return [ep for ep, _ in sorted(scored, key=lambda x: -x[1])[:max_results]]

    async def get_applicable_rules(
        self,
        current_context: WorkingMemory,
        agent_id: str
    ) -> list[SemanticMemory]:
        """
        Get validated rules from D1 (structured query, not vector search).

        Rules are stored in D1 with agent_id filtering.
        No vector search needed - rules are small corpus with exact matching.
        """
        rules = await self.d1.execute(
            """
            SELECT * FROM semantic_rules
            WHERE applies_to_agent = ? OR applies_to_agent = 'all'
            AND p_value < 0.05
            ORDER BY effect_size DESC
            """,
            [agent_id]
        )
        return [
            SemanticMemory(**row) for row in rules
            if self._rule_applies_to_context(row, current_context)
        ]
```

**Vectorize Index Schema**:

```python
# wrangler.toml
[[vectorize]]
binding = "EPISODIC_MEMORY"
index_name = "episodic-trades"
dimensions = 384  # all-MiniLM-L6-v2
metric = "cosine"

# Vector metadata stored with each trade embedding
metadata = {
    "trade_id": str,
    "entry_date": str,
    "underlying": str,
    "regime": str,
    "pnl_pct": float,
    "win": bool
}
```

---

## 7. Layer 4: Decision & Risk Management

### 7.1 Trading Decision Agent

**Inspired by**: FINCON's manager agent as "sole decision-maker" [[2]](#ref-2)

```python
class TradingDecisionAgent:
    """
    The manager agent that makes final trade decisions.

    Receives:
    - Debate outcome from researchers
    - Relevant episodic memories
    - Applicable semantic rules
    - Risk manager assessment

    Outputs:
    - Trade decision (enter/skip)
    - Position size
    - Entry parameters
    """

    def make_decision(
        self,
        debate_outcome: DebateOutcome,
        relevant_memories: list[EpisodicMemory],
        applicable_rules: list[SemanticMemory],
        risk_assessment: RiskAssessment,
        spread_candidates: list[CreditSpread]
    ) -> TradeDecision:

        # Check hard rules first
        for rule in applicable_rules:
            if rule.blocks_trade(self.working_memory):
                return TradeDecision(
                    action="skip",
                    reason=f"Blocked by rule: {rule.rule_text}"
                )

        # Process debate outcome
        if debate_outcome.winning_perspective == "bear":
            if debate_outcome.confidence > 0.8:
                return TradeDecision(action="skip", reason="Bear won debate decisively")
            else:
                # Reduce size if bear had good arguments
                size_multiplier = 0.5
        else:
            size_multiplier = 1.0

        # Get position size from risk manager
        base_size = risk_assessment.recommended_contracts
        final_size = int(base_size * size_multiplier)

        if final_size == 0:
            return TradeDecision(action="skip", reason="Size reduced to zero")

        # Select best spread
        best_spread = self._select_best_spread(spread_candidates, applicable_rules)

        return TradeDecision(
            action="enter",
            spread=best_spread,
            contracts=final_size,
            entry_thesis=debate_outcome.synthesis,
            confidence=debate_outcome.confidence
        )
```

### 7.2 Risk Manager (Three Perspectives)

**Inspired by**: TradingAgents' three-perspective risk deliberation [[1]](#ref-1)

```python
class RiskManager:
    """
    Three-perspective risk assessment inspired by TradingAgents:
    "Three risk perspectives (aggressive, neutral, conservative)
    monitor portfolio exposure and adjust strategies."
    """

    def assess_risk(
        self,
        proposed_trade: CreditSpread,
        current_portfolio: Portfolio,
        market_state: MarketState
    ) -> RiskAssessment:

        # Get three perspectives
        aggressive = self._aggressive_assessment(proposed_trade, current_portfolio)
        neutral = self._neutral_assessment(proposed_trade, current_portfolio)
        conservative = self._conservative_assessment(proposed_trade, current_portfolio)

        # Deliberate
        perspectives = [aggressive, neutral, conservative]

        # Final recommendation uses weighted average
        # In high VIX: weight conservative more heavily
        if market_state.vix > 30:
            weights = [0.1, 0.3, 0.6]  # Conservative-weighted
        elif market_state.vix > 20:
            weights = [0.2, 0.5, 0.3]  # Neutral-weighted
        else:
            weights = [0.3, 0.5, 0.2]  # Slightly aggressive

        recommended_contracts = int(sum(
            p.recommended_contracts * w
            for p, w in zip(perspectives, weights)
        ))

        return RiskAssessment(
            recommended_contracts=recommended_contracts,
            aggressive_view=aggressive,
            neutral_view=neutral,
            conservative_view=conservative,
            deliberation_summary=self._summarize_deliberation(perspectives)
        )
```

### 7.3 Position Sizer

```python
class PositionSizer:
    """
    Correlation-aware position sizing.

    Key insight: SPY/QQQ/IWM are 86-92% correlated.
    Treat as single concentrated bet, not diversification.
    """

    def __init__(self):
        self.max_risk_per_trade = 0.02      # 2% of account
        self.max_single_position = 0.05     # 5% of account
        self.max_portfolio_risk = 0.10      # 10% total exposure
        self.max_equity_exposure = 0.50     # 50% in correlated equities

        # Asset correlations (updated daily)
        self.correlation_groups = {
            "equity_etf": ["SPY", "QQQ", "IWM"],  # Highly correlated
            "treasury": ["TLT"],                   # Negative correlation
            "commodity": ["GLD"]                   # Low correlation
        }

    def calculate_size(
        self,
        spread: CreditSpread,
        account: Account,
        current_positions: list[Position]
    ) -> PositionSizeResult:

        # Base size from max risk per trade
        max_loss_per_contract = spread.max_loss
        risk_budget = account.equity * self.max_risk_per_trade
        base_contracts = int(risk_budget / max_loss_per_contract)

        # Adjust for correlation
        existing_equity_exposure = self._get_equity_exposure(current_positions)
        if spread.underlying in self.correlation_groups["equity_etf"]:
            remaining_equity_budget = (
                account.equity * self.max_equity_exposure - existing_equity_exposure
            )
            max_from_correlation = int(remaining_equity_budget / max_loss_per_contract)
            base_contracts = min(base_contracts, max_from_correlation)

        # Adjust for portfolio heat
        current_risk = self._calculate_portfolio_risk(current_positions)
        remaining_risk_budget = (
            account.equity * self.max_portfolio_risk - current_risk
        )
        max_from_portfolio = int(remaining_risk_budget / max_loss_per_contract)

        final_contracts = max(0, min(base_contracts, max_from_portfolio))

        return PositionSizeResult(
            contracts=final_contracts,
            risk_amount=final_contracts * max_loss_per_contract,
            risk_pct=final_contracts * max_loss_per_contract / account.equity,
            limiting_factor=self._identify_limiting_factor(...)
        )
```

### 7.4 Dynamic Exit Management

**Inspired by**: TradingGroup's dynamic risk management with volatility-adjusted thresholds [[3]](#ref-3)

```python
class ExitManager:
    """
    Dynamic exit thresholds based on TradingGroup's formula:

    T_SL = m_s^sl * sigma_d,10
    T_TP = m_s^tp * sigma_d,10

    Where sigma_d,10 is 10-day historical volatility and
    multipliers adjust based on trading style.
    """

    def __init__(self):
        # Base thresholds (from Option Alpha research)
        self.base_profit_target = 0.50      # 50% of credit
        self.base_stop_loss = 1.25          # 125% of credit
        self.dte_exit = 21                  # Mandatory time exit

        # Style multipliers
        self.style_multipliers = {
            "aggressive": {"tp": 0.6, "sl": 1.5},
            "neutral": {"tp": 0.5, "sl": 1.25},
            "conservative": {"tp": 0.4, "sl": 1.0}
        }

    def check_exit_conditions(
        self,
        position: Position,
        current_price: float,
        current_vol: float,
        style: str = "neutral"
    ) -> ExitSignal | None:

        # Calculate dynamic thresholds
        vol_10d = self._calculate_10d_vol(position.underlying)
        mult = self.style_multipliers[style]

        adjusted_tp = self.base_profit_target * (1 + 0.1 * (vol_10d - 0.15))
        adjusted_sl = self.base_stop_loss * mult["sl"]

        # Check conditions
        current_pnl_pct = (position.entry_credit - current_price) / position.entry_credit

        if current_pnl_pct >= adjusted_tp:
            return ExitSignal(reason="profit_target", pnl_pct=current_pnl_pct)

        if current_pnl_pct <= -adjusted_sl:
            return ExitSignal(reason="stop_loss", pnl_pct=current_pnl_pct)

        if position.dte <= self.dte_exit:
            return ExitSignal(reason="time_exit", pnl_pct=current_pnl_pct)

        return None
```

---

## 8. Layer 5: Execution & Monitoring

### 8.1 Hard Circuit Breakers

**Critical Design Decision**: These rules CANNOT be overridden by any agent.

```python
class CircuitBreaker:
    """
    Non-AI circuit breakers that halt trading.

    From research: "Autonomous systems perform best within the boundaries
    of their training data. When encountering scenarios outside this training,
    they can falter."

    These rules protect against scenarios outside training distribution.
    """

    # IMMUTABLE THRESHOLDS
    DAILY_LOSS_HALT = 0.02          # 2% daily loss -> full halt
    WEEKLY_LOSS_HALT = 0.05         # 5% weekly loss -> full halt
    MAX_DRAWDOWN_HALT = 0.15        # 15% drawdown -> full halt
    VIX_HALT = 50                   # VIX > 50 -> halt new trades
    STALE_DATA_SECONDS = 30         # No quote update -> halt
    API_ERRORS_PER_MINUTE = 5       # API issues -> halt

    def check(self, state: SystemState) -> CircuitBreakerStatus:
        """
        Check all circuit breakers.
        Returns HALT if any threshold breached.
        """

        if state.daily_pnl_pct <= -self.DAILY_LOSS_HALT:
            return CircuitBreakerStatus(
                halted=True,
                reason=f"Daily loss {state.daily_pnl_pct:.1%} exceeds {self.DAILY_LOSS_HALT:.1%}",
                action="close_all_positions"
            )

        if state.weekly_pnl_pct <= -self.WEEKLY_LOSS_HALT:
            return CircuitBreakerStatus(
                halted=True,
                reason=f"Weekly loss {state.weekly_pnl_pct:.1%} exceeds {self.WEEKLY_LOSS_HALT:.1%}",
                action="close_all_positions"
            )

        if state.drawdown_pct >= self.MAX_DRAWDOWN_HALT:
            return CircuitBreakerStatus(
                halted=True,
                reason=f"Drawdown {state.drawdown_pct:.1%} exceeds {self.MAX_DRAWDOWN_HALT:.1%}",
                action="close_all_positions_disable_trading"
            )

        if state.vix >= self.VIX_HALT:
            return CircuitBreakerStatus(
                halted=True,
                reason=f"VIX {state.vix} exceeds {self.VIX_HALT}",
                action="halt_new_trades"
            )

        # ... additional checks ...

        return CircuitBreakerStatus(halted=False)
```

### 8.2 Order Execution

```python
class OrderExecutor:
    """
    Handles order placement and monitoring.
    """

    def __init__(self, broker: AlpacaClient):
        self.broker = broker
        self.slippage_assumption = 0.10  # 10% of spread (from ORATS)

    async def execute_spread(
        self,
        spread: CreditSpread,
        contracts: int
    ) -> ExecutionResult:

        # Calculate limit price with slippage buffer
        mid_price = (spread.bid + spread.ask) / 2
        bid_ask_spread = spread.ask - spread.bid
        limit_price = mid_price - (bid_ask_spread * self.slippage_assumption)

        # Place multi-leg order
        order = SpreadOrder(
            legs=[
                OrderLeg(
                    symbol=spread.short_leg.symbol,
                    side="sell_to_open",
                    quantity=contracts
                ),
                OrderLeg(
                    symbol=spread.long_leg.symbol,
                    side="buy_to_open",
                    quantity=contracts
                )
            ],
            order_type="limit",
            limit_price=limit_price,
            time_in_force="day"
        )

        result = await self.broker.place_order(order)

        # Monitor for fill
        return await self._monitor_fill(result.order_id, timeout_seconds=300)
```

### 8.3 Monitoring & Alerting

```python
class MonitoringService:
    """
    Continuous monitoring with alerts.

    Unlike V1, there's no Discord approval flow.
    Notifications are informational only.
    """

    def __init__(self, notifier: DiscordNotifier):
        self.notifier = notifier

    async def run_monitoring_loop(self):
        """Run every 5 minutes during market hours."""

        while True:
            # Check circuit breakers first
            cb_status = self.circuit_breaker.check(self.state)
            if cb_status.halted:
                await self.notifier.send_alert(
                    level="critical",
                    message=f"CIRCUIT BREAKER: {cb_status.reason}"
                )
                await self._execute_halt_action(cb_status.action)

            # Check exit conditions for open positions
            for position in self.positions:
                exit_signal = self.exit_manager.check_exit_conditions(position)
                if exit_signal:
                    await self._execute_exit(position, exit_signal)
                    await self.notifier.send_alert(
                        level="info",
                        message=f"Closed {position.symbol}: {exit_signal.reason}"
                    )

            await asyncio.sleep(300)  # 5 minutes
```

---

## 9. Options-Specific Components

### 9.1 Credit Spread Screener

```python
class CreditSpreadScreener:
    """
    Options-specific screening for credit spreads.

    Parameters based on research:
    - Delta 0.10-0.15 (further OTM for consistency)
    - DTE 30-45 (theta decay sweet spot)
    - IV Percentile >= 50 (elevated premium)
    - Credit >= 25% of spread width (adequate premium)
    """

    def __init__(self):
        self.config = ScreenerConfig(
            dte_range=(30, 45),
            short_delta_range=(0.10, 0.15),  # Research: further OTM
            iv_percentile_min=50,
            spread_width_range=(2, 10),
            min_credit_pct=0.25,             # 25% of width
            max_bid_ask_spread_pct=0.08,     # 8% max slippage
            min_open_interest=100,
            min_volume=10
        )

    def screen(
        self,
        underlying: str,
        options_chain: OptionsChain,
        iv_metrics: IVMetrics
    ) -> list[CreditSpread]:

        # Filter by IV first
        if iv_metrics.iv_percentile < self.config.iv_percentile_min:
            return []

        candidates = []

        # Screen bull put spreads
        for expiration in self._get_valid_expirations(options_chain):
            puts = options_chain.get_puts(expiration)

            for short_put in puts:
                if not self._valid_short_strike(short_put):
                    continue

                long_put = self._find_long_strike(short_put, puts)
                if long_put is None:
                    continue

                spread = self._construct_spread(short_put, long_put, "bull_put")
                if self._passes_filters(spread):
                    candidates.append(spread)

        # Screen bear call spreads similarly...

        # Score and rank
        return self._score_and_rank(candidates)

    def _score_and_rank(self, spreads: list[CreditSpread]) -> list[CreditSpread]:
        """
        Regime-conditional scoring from V1, with research-backed weights.
        """
        for spread in spreads:
            # Expected value calculation
            prob_otm = 1 - abs(spread.short_delta)
            ev = (spread.credit * prob_otm) - (spread.max_loss * (1 - prob_otm))

            # Composite score (weights from backtesting)
            spread.score = (
                0.35 * self._normalize(ev) +
                0.25 * self._normalize(spread.iv_percentile) +
                0.25 * self._normalize(1 - abs(spread.short_delta)) +
                0.15 * self._normalize(spread.credit / spread.width)
            )

        return sorted(spreads, key=lambda s: -s.score)
```

### 9.2 IV Term Structure Analysis

```python
class IVTermStructureAnalyzer:
    """
    Analyze IV term structure for credit spread entry.

    Contango (short-term IV < long-term IV):
    - Favorable for selling near-term premium
    - Market expects calm near-term, uncertainty later

    Backwardation (short-term IV > long-term IV):
    - Unfavorable for selling premium
    - Market pricing near-term uncertainty/events
    """

    def analyze(
        self,
        iv_surface: dict[int, float]  # DTE -> IV
    ) -> TermStructureAnalysis:

        # Get key tenors
        iv_30d = self._interpolate_iv(iv_surface, 30)
        iv_60d = self._interpolate_iv(iv_surface, 60)
        iv_90d = self._interpolate_iv(iv_surface, 90)

        # Calculate ratio (TradingAgents style)
        ratio_30_90 = iv_30d / iv_90d

        if ratio_30_90 < 0.95:
            structure = "contango"
            signal = "favorable"
        elif ratio_30_90 > 1.05:
            structure = "backwardation"
            signal = "unfavorable"
        else:
            structure = "flat"
            signal = "neutral"

        # Calculate slope via regression
        dtes = list(iv_surface.keys())
        ivs = list(iv_surface.values())
        slope = self._calculate_slope(dtes, ivs)

        return TermStructureAnalysis(
            structure=structure,
            ratio_30_90=ratio_30_90,
            slope=slope,
            signal=signal,
            iv_30d=iv_30d,
            iv_60d=iv_60d,
            iv_90d=iv_90d
        )
```

### 9.3 Greeks Portfolio Aggregation

```python
class PortfolioGreeksCalculator:
    """
    Aggregate Greeks across all positions.

    For credit spreads:
    - Delta: Net directional exposure
    - Gamma: Acceleration of delta
    - Theta: Daily time decay (positive = profit)
    - Vega: IV sensitivity (typically negative for credit spreads)
    """

    def calculate(
        self,
        positions: list[Position],
        underlying_prices: dict[str, float]
    ) -> PortfolioGreeks:

        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for position in positions:
            # Get current Greeks for each leg
            short_greeks = self.calculator.calculate_greeks(
                position.short_leg,
                underlying_prices[position.underlying]
            )
            long_greeks = self.calculator.calculate_greeks(
                position.long_leg,
                underlying_prices[position.underlying]
            )

            # Net Greeks (short - long for credit spread)
            contracts = position.contracts * 100  # Convert to shares

            total_delta += (short_greeks.delta - long_greeks.delta) * contracts
            total_gamma += (short_greeks.gamma - long_greeks.gamma) * contracts
            total_theta += (short_greeks.theta - long_greeks.theta) * contracts
            total_vega += (short_greeks.vega - long_greeks.vega) * contracts

        return PortfolioGreeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            delta_dollars=total_delta * sum(underlying_prices.values()) / len(underlying_prices)
        )
```

---

## 10. Data Synthesis & Continuous Learning

### 10.1 Automated Data Synthesis Pipeline

**Source**: TradingGroup's "automated data-synthesis and annotation pipeline" [[3]](#ref-3)

```python
class DataSynthesisPipeline:
    """
    Inspired by TradingGroup:

    "The pipeline automatically logs every agent's inputs, outputs,
    full CoT, and daily evaluation metrics, resulting in labeled
    trajectories that serve as the distillation dataset."

    This enables continuous improvement through:
    1. Collecting agent traces
    2. Auto-labeling outcomes
    3. Fine-tuning on successful patterns
    """

    def __init__(self):
        self.trajectory_store = TrajectoryStore()

    def log_trade_trajectory(
        self,
        trade_id: str,
        analyst_outputs: dict[str, AgentMessage],
        debate_transcript: list[DebateRound],
        decision: TradeDecision,
        risk_assessment: RiskAssessment
    ) -> None:
        """Log full agent trajectory for later labeling."""

        self.trajectory_store.save(TradeTrajectory(
            trade_id=trade_id,
            timestamp=datetime.now(),

            # Agent outputs
            analyst_outputs=analyst_outputs,
            debate_transcript=debate_transcript,
            decision=decision,
            risk_assessment=risk_assessment,

            # Chain-of-thought
            bull_reasoning=self._extract_cot(debate_transcript, "bull"),
            bear_reasoning=self._extract_cot(debate_transcript, "bear"),

            # To be filled after trade closes
            outcome=None,
            label=None
        ))

    def label_trajectory(
        self,
        trade_id: str,
        actual_pnl: float,
        actual_pnl_pct: float,
        benchmark_return: float
    ) -> None:
        """
        Auto-label trajectory based on outcome.

        From TradingGroup:
        "reward_a = simulated_return - benchmark_return - transaction_costs"
        """
        trajectory = self.trajectory_store.get(trade_id)

        # Calculate reward
        transaction_costs = 0.01  # Approximate
        reward = actual_pnl_pct - benchmark_return - transaction_costs

        # Label
        if reward > 0.02:  # Outperformed by 2%+
            label = "strong_positive"
        elif reward > 0:
            label = "positive"
        elif reward > -0.02:
            label = "negative"
        else:
            label = "strong_negative"

        trajectory.outcome = TradeOutcome(
            pnl=actual_pnl,
            pnl_pct=actual_pnl_pct,
            reward=reward
        )
        trajectory.label = label

        self.trajectory_store.update(trajectory)

    def generate_training_data(
        self,
        min_trades: int = 100
    ) -> list[TrainingExample]:
        """
        Generate training examples from labeled trajectories.

        Filter for high-quality examples (strong positive/negative).
        """
        trajectories = self.trajectory_store.get_labeled()

        if len(trajectories) < min_trades:
            return []

        examples = []
        for traj in trajectories:
            if traj.label in ["strong_positive", "strong_negative"]:
                examples.append(TrainingExample(
                    input=self._format_input(traj),
                    output=self._format_output(traj),
                    label=traj.label
                ))

        return examples
```

### 10.2 Model Fine-Tuning Schedule

**From TradingGroup**: "Fine-tuned Qwen3-8B using LoRA with int8 quantization"

```python
class ModelFineTuner:
    """
    Periodic fine-tuning on accumulated trading data.

    Schedule: Monthly (requires 100+ new labeled trades)
    Method: LoRA fine-tuning for efficiency
    """

    def should_fine_tune(self) -> bool:
        labeled_since_last = self.trajectory_store.get_labeled_since(
            self.last_fine_tune_date
        )
        return len(labeled_since_last) >= 100

    def fine_tune(
        self,
        base_model: str,
        training_data: list[TrainingExample]
    ) -> str:
        """
        Fine-tune using LoRA (Parameter-Efficient Fine-Tuning).

        Returns: Path to fine-tuned model
        """
        # Implementation would use transformers + peft libraries
        pass
```

---

## 11. Technology Stack

### 11.1 Core Components

**Decision**: Stay on Cloudflare-native stack for zero-ops serverless deployment (see Section 2.4).

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Runtime | Cloudflare Workers (Python via Pyodide) | Existing V1 infra, serverless |
| LLM (Phase 1) | Claude Sonnet API | Best reasoning, start immediately |
| LLM (Phase 2) | Fine-tuned Qwen3-8B | After 500+ labeled trades |
| Options Pricing | [vollib](https://vollib.org/) | Fast Greeks, Jackel's algorithm |
| Backtesting Data | [ORATS](https://orats.com/) | Best options historical data |
| Live Data/Trading | Alpaca | Free, good API |
| Relational DB | Cloudflare D1 (SQLite) | Trades, rules, performance |
| Vector Store | Cloudflare Vectorize | Episodic memory similarity |
| Cache | Cloudflare KV | Working memory, circuit breaker state |
| Object Storage | Cloudflare R2 | Historical snapshots, model artifacts |
| Scheduling | Cloudflare Cron Triggers | Morning scan, position monitor, EOD |

### 11.2 Cloudflare Limits & Pricing

**Vectorize** ([docs](https://developers.cloudflare.com/vectorize/platform/limits/)):

- Vectors per index: 10,000,000 (sufficient for years of trades)
- Dimensions: up to 1536 (using 384 for all-MiniLM-L6-v2)
- Free tier: 30M queried dimensions/month, 5M stored dimensions
- Accuracy: ~80% approximate, higher with `returnValues: true`

**D1** ([docs](https://developers.cloudflare.com/d1/)):

- 10GB storage per database
- 25M row reads/month free
- Full SQLite compatibility

**Workers** ([docs](https://developers.cloudflare.com/workers/)):

- 10ms CPU time (free), 30s (paid)
- 100,000 requests/day free

### 11.3 Dependencies

```toml
# wrangler.toml
name = "mahler-v2"
main = "src/main.py"
compatibility_date = "2024-01-01"
compatibility_flags = ["python_workers"]

[vars]
ENVIRONMENT = "production"

# D1 Database
[[d1_databases]]
binding = "DB"
database_name = "mahler-db"
database_id = "xxx"

# Vectorize Index
[[vectorize]]
binding = "EPISODIC_MEMORY"
index_name = "episodic-trades"

# KV Namespaces
[[kv_namespaces]]
binding = "CACHE"
id = "xxx"

# R2 Bucket
[[r2_buckets]]
binding = "STORAGE"
bucket_name = "mahler-storage"

# Cron Triggers
[triggers]
crons = [
    "35 14 * * 1-5",   # Morning scan (10:35 AM ET)
    "0 17 * * 1-5",    # Midday check (12:00 PM ET)
    "30 20 * * 1-5",   # Afternoon scan (3:30 PM ET)
    "15 21 * * 1-5",   # EOD summary (4:15 PM ET)
]

# Requirements (Pyodide-compatible)
[build]
command = ""

# Python requirements loaded via Pyodide
# numpy, scipy, scikit-learn available by default
# anthropic, alpaca-py loaded as pure Python
```

```python
# requirements.txt (for local development/backtesting)
anthropic>=0.40.0
alpaca-py>=0.30.0
py-vollib>=1.0.3
numpy>=1.26.0
scipy>=1.12.0
pandas>=2.2.0
sentence-transformers>=3.0.0  # For generating embeddings
scikit-learn>=1.4.0
httpx>=0.27.0
pydantic>=2.6.0
```

### 11.4 Architecture Diagram (Cloudflare-Native)

```
+===========================================================================+
|                     CLOUDFLARE WORKERS ENVIRONMENT                         |
+===========================================================================+
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  | CRON TRIGGERS                                                       |  |
|  |   10:35 AM ET: Morning Scan Worker                                  |  |
|  |   12:00 PM ET: Midday Check Worker                                  |  |
|  |   3:30 PM ET:  Afternoon Scan Worker                                |  |
|  |   4:15 PM ET:  EOD Summary Worker                                   |  |
|  |   Every 5min:  Position Monitor Worker                              |  |
|  +---------------------------------------------------------------------+  |
|                              |                                            |
|                     +--------v---------+                                  |
|                     |  Agent           |                                  |
|                     |  Orchestrator    |                                  |
|                     +--------+---------+                                  |
|                              |                                            |
|      +---------------+-------+-------+---------------+                    |
|      |               |               |               |                    |
|      v               v               v               v                    |
|  +---+---+     +-----+-----+   +-----+-----+   +-----+-----+              |
|  | IV    |     | Technical |   | Macro     |   | Greeks    |              |
|  | Agent |     | Agent     |   | Agent     |   | Agent     |              |
|  +---+---+     +-----+-----+   +-----+-----+   +-----+-----+              |
|      |               |               |               |                    |
|      +---------------+-------+-------+---------------+                    |
|                              |                                            |
|                     +--------v---------+                                  |
|                     | Debate Layer     |                                  |
|                     | (Bull vs Bear)   |                                  |
|                     +--------+---------+                                  |
|                              |                                            |
+===========================================================================+
                               |
       +-----------------------+-----------------------+
       |                       |                       |
       v                       v                       v
+------+------+        +-------+-------+       +-------+-------+
| Claude API  |        | ORATS API     |       | Alpaca API    |
| (LLM)       |        | (Backtest)    |       | (Live Data)   |
+-------------+        +---------------+       +---------------+

+===========================================================================+
|                     CLOUDFLARE STORAGE                                     |
+===========================================================================+
|                                                                           |
|  +----------------+  +----------------+  +----------------+  +----------+ |
|  | D1 (SQLite)    |  | Vectorize      |  | KV             |  | R2       | |
|  |                |  |                |  |                |  |          | |
|  | - trades       |  | - episodic     |  | - working      |  | - daily  | |
|  | - positions    |  |   memory       |  |   memory       |  |   snaps  | |
|  | - rules        |  |   embeddings   |  | - circuit      |  | - models | |
|  | - performance  |  |                |  |   breaker      |  | - chains | |
|  | - trajectories |  |                |  | - rate limits  |  |          | |
|  +----------------+  +----------------+  +----------------+  +----------+ |
|                                                                           |
+===========================================================================+
```

### 11.5 LLM Migration Path

```
PHASE 1: Claude API (Now)
+------------------+
| Claude Sonnet    |  - Best reasoning for debate/analysis
| via API          |  - ~$15-30/month at expected volume
+------------------+  - Zero infrastructure

        |
        | After 500+ labeled trades (~6-12 months)
        v

PHASE 2: Evaluate Fine-Tuning
+------------------+
| Qwen3-8B         |  - Fine-tune with LoRA on trade trajectories
| Fine-tuned       |  - Test on validation set
+------------------+  - Compare to Claude baseline

        |
        | If performance matches or exceeds Claude
        v

PHASE 3: Deploy Fine-Tuned Model
+------------------+
| Option A:        |  - Cloudflare Workers AI (if Qwen supported)
| Workers AI       |  - Zero infrastructure change
+------------------+

+------------------+
| Option B:        |  - Together.ai, Fireworks, or Modal
| External GPU     |  - ~$50-100/month for dedicated inference
+------------------+

+------------------+
| Option C:        |  - Keep Claude for complex reasoning
| Hybrid           |  - Use fine-tuned model for simple tasks
+------------------+
```

---

## 12. Performance Targets

### 12.1 Trading Performance

Based on research benchmarks:

| Metric | Target | Source |
|--------|--------|--------|
| Win Rate | >= 70% | [Option Alpha](https://optionalpha.com/blog/spy-put-credit-spread-backtest): 93% at delta 0.10 |
| Profit Factor | >= 1.5 | PRD requirement |
| Sharpe Ratio | >= 1.0 | [TradingAgents](https://arxiv.org/abs/2412.20138): achieved 5.60-8.21 |
| Max Drawdown | <= 15% | PRD requirement |
| Monthly Win Rate | >= 8/12 | Consistent profitability |

### 12.2 System Performance

| Metric | Target |
|--------|--------|
| Analysis Cycle | < 60 seconds |
| Debate Rounds | 2-3 per opportunity |
| Memory Retrieval | < 100ms |
| Order Placement | < 1 second |
| Position Monitor | Every 5 minutes |
| Uptime | 99.5% during market hours |

### 12.3 Validation Requirements

Before live trading:

1. **Backtest Validation** (ORATS data, 2020-2025)
   - 500+ simulated trades
   - Walk-forward validation across 12+ periods
   - Tested through VIX > 30 events (March 2020, 2022)

2. **Paper Trading** (Alpaca, 3 months)
   - 100+ actual paper trades
   - Verify backtested parameters hold
   - Validate agent behavior in real-time

3. **Shadow Mode** (1 month)
   - Run alongside manual trading
   - Compare recommendations to human decisions
   - Final validation before autonomous

---

## 13. References

<a name="ref-1"></a>
**[1]** Xiao, Y., Sun, E., Luo, D., & Wang, W. (2025). *TradingAgents: Multi-Agents LLM Financial Trading Framework*. arXiv:2412.20138. <https://arxiv.org/abs/2412.20138>

<a name="ref-2"></a>
**[2]** Yu, Y., et al. (2024). *FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making*. NeurIPS 2024. <https://arxiv.org/abs/2407.06567>

<a name="ref-3"></a>
**[3]** TradingGroup Authors. (2025). *TradingGroup: A Multi-Agent Trading System with Self-Reflection and Data-Synthesis*. arXiv:2508.17565. <https://arxiv.org/abs/2508.17565>

<a name="ref-4"></a>
**[4]** Yu, Y., et al. (2023). *FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design*. arXiv:2311.13743. <https://arxiv.org/abs/2311.13743>

<a name="ref-5"></a>
**[5]** Stanford/Boston College Research on AI Trading Performance. <https://www.godofprompt.ai/blog/ai-trading-bots-outperforming-human-investors>

<a name="ref-6"></a>
**[6]** Option Alpha. *SPY Put Credit Spread Backtest Results*. <https://optionalpha.com/blog/spy-put-credit-spread-backtest>

<a name="ref-7"></a>
**[7]** Maven Securities. *Alpha Decay: What does it look like?*. <https://www.mavensecurities.com/alpha-decay-what-does-it-look-like-and-what-does-it-mean-for-systematic-traders/>

<a name="ref-8"></a>
**[8]** ORATS. *Backtest API Documentation*. <https://docs.orats.io/backtest-api-guide/>

<a name="ref-9"></a>
**[9]** vollib. *Options Pricing Library*. <https://vollib.org/>

<a name="ref-10"></a>
**[10]** FinRL. *Deep Reinforcement Learning Library*. <https://github.com/AI4Finance-Foundation/FinRL>

<a name="ref-11"></a>
**[11]** Hull, J., et al. (2021). *Deep Hedging of Derivatives Using Reinforcement Learning*. <https://arxiv.org/abs/2103.16409>

<a name="ref-12"></a>
**[12]** MIT Sloan. *Retail investors lose big in options markets*. <https://mitsloan.mit.edu/ideas-made-to-matter/retail-investors-lose-big-options-markets-research-shows>

<a name="ref-13"></a>
**[13]** Cloudflare. *Vectorize Documentation*. <https://developers.cloudflare.com/vectorize/>

<a name="ref-14"></a>
**[14]** Cloudflare. *Building Vectorize: A Distributed Vector Database*. <https://blog.cloudflare.com/building-vectorize-a-distributed-vector-database-on-cloudflare-developer-platform/>

<a name="ref-15"></a>
**[15]** Wang, X., et al. (2024). *Searching for Best Practices in Retrieval-Augmented Generation*. arXiv:2407.01219. <https://arxiv.org/abs/2407.01219>

<a name="ref-16"></a>
**[16]** Stack Overflow. *Practical Tips for Retrieval-Augmented Generation (RAG)*. <https://stackoverflow.blog/2024/08/15/practical-tips-for-retrieval-augmented-generation-rag/>

<a name="ref-17"></a>
**[17]** Iguazio. *Commercial vs. Self-Hosted LLMs: Cost Analysis*. <https://www.iguazio.com/blog/commercial-vs-self-hosted-llms/>

---

## Appendix A: Migration Path from V1

### Phase 1: Add Backtesting (2-3 weeks)

- Integrate ORATS data API
- Build backtesting framework (local Python, not Workers)
- Validate current V1 parameters vs research-backed parameters
- Establish performance baseline with walk-forward validation

### Phase 2: Add Vectorize Memory (1-2 weeks)

- Create Vectorize index for episodic memory
- Implement embedding generation (all-MiniLM-L6-v2)
- Build simple retrieval (no hybrid/reranking)
- Migrate V1 playbook rules to D1 semantic_rules table

### Phase 3: Implement Multi-Agent (3-4 weeks)

- Design analyst agent prompts
- Implement debate mechanism (bull vs bear)
- Build agent orchestrator
- Test against V1 recommendations (shadow mode)

### Phase 4: Remove Human-in-Loop (1-2 weeks)

- Implement hard circuit breakers (non-overridable)
- Remove Discord approval flow
- Update execution for full autonomy
- Add comprehensive Discord alerting (info only)

### Phase 5: Add Reflection System (2-3 weeks)

- Implement self-reflection engine
- Build trajectory logging for data synthesis
- Create rule validation pipeline (Mann-Whitney)
- Deploy continuous learning loop

### Phase 6: Evaluate Fine-Tuning (After 500+ trades)

- Export labeled trajectories
- Fine-tune Qwen3-8B with LoRA
- Compare to Claude baseline
- Deploy if performance matches

---

## Appendix B: Key Decisions Summary

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Infrastructure | Cloudflare (D1/Vectorize/KV/R2) | PostgreSQL + pgvector | Zero ops, existing infra, sufficient accuracy |
| LLM (Phase 1) | Claude Sonnet API | Self-hosted Qwen | Best reasoning, no training data yet |
| LLM (Phase 2) | Fine-tuned Qwen3-8B | Keep Claude | Lower cost at scale, domain-specific |
| RAG Strategy | Simple vector similarity | Hybrid + reranking | Small corpus, structured queries |
| Memory Store | Vectorize (episodic) + D1 (semantic) | All in pgvector | Separation of concerns, SQL for rules |

---

*Document generated: 2026-01-27*
*Last updated: 2026-01-27 (Added architectural decisions)*
*Next review: After backtesting validation*
