# Algorithmic Options Trading Research

**Date:** 2026-04-06
**Purpose:** Capture findings from research phase to inform strategy development and system design.

---

## 1. Rust Ecosystem for Options Trading

### Frameworks

| Library | Stars | Options Support | Status |
|---------|-------|-----------------|--------|
| barter-rs | ~2.1k | None (crypto/futures) | Active, not production-ready |
| NautilusTrader | Large | Partial (instrument defs) | Production-grade, Rust core + Python API |
| rust-ibapi | Active | Yes (via IBKR) | Community IBKR client |
| RustQuant | ~1.7k | Yes (pricing math) | Active, not production software |
| RustyQLib | Small | Derivative pricing | Active |
| QuantRS | Small | Quant finance | Newer |

**Assessment:** No complete Rust options trading framework exists. Build custom engine using barter-rs patterns for architecture, rust-ibapi for IBKR connectivity, RustQuant for pricing math.

### Python Frameworks Worth Studying

- **QuantConnect LEAN** (C#/Python): Most complete open-source options system. Options chains are first-class. Same code runs backtest and live.
- **NautilusTrader**: Rust core with nanosecond resolution, 5M rows/sec throughput. Best architecture reference.
- **Backtrader**: Cerebro engine shows clean composition of feeds/strategies/analyzers.
- **VectorBT**: Fast for screening but introduces look-ahead bias -- unsuitable for options.

**Consensus lesson:** Event-driven with strict time ordering is required for correct options backtesting.

---

## 2. Spintwig Backtest Findings

### SPX Iron Condor 45-DTE (18 backtests, 51,600+ trades, 2007-2024)

- **The call side has generally negative expected value**
- The put side has generally positive expected value
- Iron condors underperform short puts alone because the call side is a drag
- Win rate at 16-delta: ~77.6%
- Moving from 15-delta to 30-delta drops win rate from ~70% to ~34%
- Spintwig recommends discounting CAGR by 20% for realistic expectations

### SPX/SPY Credit Spreads (52,400+ vertical put trades)

- 5-delta short puts, 2018-2020: ~96% win rate (330 trades, 11 losers)
- Avg winner: $13.29, avg loser: $163.45
- 50% profit target captured ~50% of premium; hold-to-expiration captured ~55.43%
- 5-delta at 1x leverage: Sharpe ~2.12 (highest risk-adjusted)
- 10-delta at 5x leverage: CAGR ~10.8%

### Wheel Strategy (SPY, 10 backtests, 2,200+ trades)

- **94-99% of total return came from the long underlying position**, not premium
- Not a single wheel variant outperformed buy-and-hold SPY
- The "triple income" claim does not hold up in backtesting

### 0DTE SPX Iron Condors (CBOE, Henry Schwartz)

- 0DTE SPX options = 62% of total S&P 500 options volume (2024)
- ~50% from retail traders
- Example structure: 10-point spreads, ~$1 premium per side, ~$2 total credit
- Risk:reward per spread: 9:1
- Optimal: enter mid-day after volatility has occurred (post-CPI, post-economic data)
- **No published systematic backtest data** -- illustrative trades only

---

## 3. Broker API Comparison

| Broker | API Quality | Options | Rust Support | Cost/Contract | Paper Trading |
|--------|-------------|---------|--------------|---------------|---------------|
| IBKR | Excellent (complex) | Comprehensive | Multiple libs | $0.15-0.65 | Full, mirrors live |
| Alpaca | Good | Level 3 (iron condors) | Build REST client | Varies | Full, real-time data |
| Tastytrade | Good (simple) | Full | Build REST client | $0.75 open, $0 close | 60 req/min limit |
| Tradier | Excellent (clean) | Full | Build REST client | Low | 15-min delayed data only |
| Schwab | Good | Full | Sparse | Varies | Via thinkorswim |

**Paper trading recommendation:** Alpaca (real-time data, Level 3 options, no setup friction). IBKR for future production.

---

## 4. Market Data Sources

| Source | Type | Options Coverage | Historical | Cost |
|--------|------|-----------------|------------|------|
| Polygon.io | Real-time + historical | All US options | Back to 2014 | Tiered |
| Databento | Professional-grade | OPRA, ~2M tickers | Same API for live/historical | Tiered |
| ORATS | Research-grade | All US options | Back to 2007 | Higher |
| MarketData.app | Retail-friendly | Full OPRA | EOD back to 2010 | Affordable |
| Tradier | Broker-bundled | Full OPRA | With funded account | Free with account |

**Current data:** ORATS strike-level parquet files: SPY (2007-2026), QQQ (2008-2026), IWM (2007-2026). 4.1 GB total.

---

## 5. Options-Specific Considerations

### Why Options Differ from Equities

- Multi-dimensional instrument space (thousands of contracts per underlying)
- Greeks: delta, gamma, theta, vega, rho all matter
- Volatility surface (not a single number)
- Bid-ask spreads are wide ($0.05-$0.50+)
- Every position expires -- full lifecycle management required
- Assignment risk (American-style only; European-style SPX/NDX eliminates this)

### Pricing Models

- **Black-Scholes**: Foundational, assumes constant vol (wrong but universal baseline)
- **Heston**: Stochastic vol, mean-reverting vol-of-vol, better smile capture
- **SABR**: Analytic smile interpolation

For retail algo: BS Greeks from data provider is sufficient. No need for Heston calibration.

### Operational Risks

- **Bid-ask spread**: Major transaction cost, backtests often underestimate
- **Liquidity**: Filter for OI > threshold, tight spreads
- **Assignment**: Use European-style (SPX) to eliminate
- **Pin risk**: Close before expiration day (21 DTE exit or 50% profit target)
- **Correlation**: In crashes, all correlations go to 1

---

## 6. Proven Strategies for Algo Execution

### Recommended Starting Strategy: 45-DTE Put Credit Spreads on SPY

- Entry: IVR > 30%, 20-30 delta short strike, 30-45 DTE
- Exit: 50% profit target OR 21 DTE time stop OR 125% stop loss
- Position sizing: Max loss < 5% of portfolio per trade
- Spintwig-validated win rate: ~77% at 16-delta, ~96% at 5-delta

### Why Not Iron Condors First

Spintwig found the call side has negative expected value. Put spreads alone outperform the full iron condor. Start with puts, add calls later if data supports it.

### Other Strategies (future validation)

- Short strangles: Higher return, undefined risk
- 0DTE iron condors: High operational complexity, no systematic backtest data
- The Wheel: Basically buy-and-hold with extra costs

---

## 7. Architecture Patterns

### Event-Driven (Required for Options)

```
MarketDataEvent -> Strategy -> Signal -> RiskManager -> OrderEvent -> Execution -> FillEvent -> Portfolio
```

### Backtest/Live Parity

Strategy layer has zero knowledge of whether it's backtesting or live. Only DataSource and Broker implementations change.

### NautilusTrader Reference Architecture

- Rust core + Python API hybrid
- Message bus connects all components
- Same domain objects in backtest and live
- Nanosecond-precision timestamps for deterministic ordering
- Three-tier fill model for realistic slippage
- Configurable latency model

### Fill Simulation

- Fill at mid is optimistic
- Fill at 1/4 spread inside worst side is realistic
- Use historical bid/ask data
- ORATS methodology: 66% of spread for 2-leg, 53% for 4-leg

---

## 8. Common Pitfalls

1. **Look-ahead bias**: #1 sin in options backtesting. ORATS data is EOD snapshots, which naturally avoids intraday look-ahead.
2. **Overfitting**: Test across 2008, 2020 crash, 2022 rate hikes. Walk-forward validation is essential.
3. **Ignoring bid-ask spread**: Options spreads are wide. Always model pessimistic fills.
4. **Ignoring commissions**: $0.65/contract x 4 legs x many trades adds up fast.
5. **Kelly Criterion**: Dangerous for negative-skew strategies. Use fixed-fractional sizing.
6. **Win rate illusion**: 70% win rate means nothing if each loss is 4x each win. Evaluate on expected value and Sharpe.
7. **Correlation risk**: "Diversified" short-vol across SPY/QQQ/IWM all move together in crashes.

---

## 9. Key Resources

### Must-Read Before Building

- **Spintwig** (spintwig.com): Rigorous backtests of dozens of strategies with real data
- **"Option Volatility and Pricing" by Natenberg**: Foundational text
- **"Options, Futures, and Other Derivatives" by Hull**: Academic standard

### Communities

- r/algotrading (highest signal for systematic trading)
- r/thetagang (premium-selling focus)
- QuantConnect forums (systematic options discussions)
- NautilusTrader Discord

### Architecture References

- NautilusTrader docs (nautilustrader.io/docs)
- LEAN source (github.com/QuantConnect/Lean)
- barter-rs source (github.com/barter-rs/barter-rs)
