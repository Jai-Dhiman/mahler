"""Morning scan worker - runs at 10:00 AM ET.

Scans options chains for credit spread opportunities and sends
recommendations to Discord for approval.

Timing: Moved from 9:35 AM to 10:00 AM to avoid the first 30 minutes
when spreads are wide and quotes are stale.
"""

from datetime import datetime, timedelta

from core import http
from core.ai.claude import ClaudeClient, ClaudeRateLimitError
from core.analysis.greeks import (
    calculate_second_order_greeks,
    calculate_spread_second_order_greeks,
    years_to_expiry,
)
from core.analysis.iv_rank import (
    IVMeanReversion,
    IVTermStructure,
    MeanReversionSignal,
    TermStructurePoint,
    TermStructureRegime,
    calculate_iv_metrics,
)
from core.analysis.regime import InsufficientDataError, create_regime_detector
from core.analysis.screener import (
    MarketRegime,
    OptionsScreener,
    RegimeConditionalScorer,
    ScreenerConfig,
    ScoringWeights,
)
from core.broker.alpaca import AlpacaClient
from core.db.d1 import D1Client
from core.db.kv import KVClient
from core.notifications.discord import DiscordClient
from core.risk.circuit_breaker import CircuitBreaker, RiskLevel
from core.risk.position_sizer import PositionSizer
from core.types import Confidence, RecommendationStatus, SpreadType, TradeStatus

# V2 Multi-Agent System imports
from core.agents import (
    AgentOrchestrator,
    DebateConfig,
    IVAnalyst,
    TechnicalAnalyst,
    MacroAnalyst,
    GreeksAnalyst,
    BullResearcher,
    BearResearcher,
    DebateFacilitator,
    build_agent_context,
    PipelineResult,
)
from core.memory.vectorize import EpisodicMemoryStore
from core.memory.retriever import MemoryRetriever

# V2 Three-Perspective Risk and Trajectory Store
from core.risk.three_perspective import ThreePerspectiveRiskManager
from core.learning import TrajectoryStore, TradeTrajectory

# Underlyings to scan
# SPY/QQQ/IWM are equity ETFs (86-92% correlated)
# TLT is treasury ETF (negatively correlated with equities)
# GLD is gold ETF (low/variable correlation)
UNDERLYINGS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

# Maximum recommendations per scan
MAX_RECOMMENDATIONS = 3


def _create_v2_orchestrator(claude, debate_rounds: int = 2) -> AgentOrchestrator:
    """Create a configured V2 agent orchestrator with all analysts and debate agents.

    Args:
        claude: ClaudeClient instance
        debate_rounds: Number of debate rounds (default 2)

    Returns:
        Configured AgentOrchestrator ready for pipeline execution
    """
    orchestrator = AgentOrchestrator(
        claude=claude,
        debate_config=DebateConfig(
            max_rounds=debate_rounds,
            min_rounds=1,
            consensus_threshold=0.7,
        ),
    )

    # Register analyst agents
    orchestrator.register_analyst(IVAnalyst(claude))
    orchestrator.register_analyst(TechnicalAnalyst(claude))
    orchestrator.register_analyst(MacroAnalyst(claude))
    orchestrator.register_analyst(GreeksAnalyst(claude))

    # Register debate agents
    orchestrator.register_debater(BullResearcher(claude), "bull")
    orchestrator.register_debater(BearResearcher(claude), "bear")

    # Set facilitator for synthesis
    orchestrator.set_facilitator(DebateFacilitator(claude))

    return orchestrator


async def _run_v2_analysis(
    orchestrator: AgentOrchestrator,
    spread,
    underlying_price: float,
    iv_metrics,
    term_structure,
    mean_reversion,
    regime_result: dict | None,
    current_vix: float | None,
    vix_3m: float | None,
    price_bars: list[dict] | None,
    positions: list,
    portfolio_greeks,
    account,
    daily_pnl: float,
    weekly_pnl: float,
    playbook_rules: list,
    episodic_store: EpisodicMemoryStore | None,
    memory_retriever: MemoryRetriever | None,
):
    """Run V2 multi-agent analysis pipeline for a spread.

    Args:
        orchestrator: Configured AgentOrchestrator
        spread: CreditSpread to analyze
        ... (market data, portfolio context)
        episodic_store: EpisodicMemoryStore for similar trade retrieval and storage
        memory_retriever: MemoryRetriever for rule and context retrieval

    Returns:
        Tuple of (PipelineResult, similar_trades list)
    """
    # Retrieve similar trades from episodic memory (if available)
    similar_trades = []
    if memory_retriever:
        retrieved_context = await memory_retriever.retrieve_context(
            underlying=spread.underlying,
            spread_type=spread.spread_type.value,
            market_regime=regime_result.get("regime") if regime_result else None,
            iv_rank=iv_metrics.iv_rank if iv_metrics else None,
            vix=current_vix,
        )
        similar_trades = retrieved_context.similar_trades

    # Build agent context
    context = build_agent_context(
        spread=spread,
        underlying_price=underlying_price,
        iv_metrics=iv_metrics,
        term_structure=term_structure,
        mean_reversion=mean_reversion,
        regime=regime_result.get("regime") if regime_result else None,
        regime_probability=regime_result.get("probability") if regime_result else None,
        current_vix=current_vix,
        vix_3m=vix_3m,
        price_bars=price_bars,
        positions=positions,
        portfolio_greeks=portfolio_greeks,
        account_equity=account.equity,
        buying_power=account.buying_power,
        daily_pnl=daily_pnl,
        weekly_pnl=weekly_pnl,
        playbook_rules=playbook_rules,
        similar_trades=similar_trades,
        scan_type="morning",
    )

    # Run the multi-agent pipeline
    result = await orchestrator.run_pipeline(context)

    return result, similar_trades


def _map_v2_result_to_analysis(result: PipelineResult):
    """Map V2 PipelineResult to TradeAnalysis-compatible format.

    Args:
        result: PipelineResult from orchestrator

    Returns:
        Object with thesis and confidence attributes matching TradeAnalysis
    """
    from dataclasses import dataclass

    @dataclass
    class V2Analysis:
        thesis: str
        confidence: Confidence

    # Map V2 confidence (0.0-1.0) to Confidence enum
    if result.confidence >= 0.7:
        confidence = Confidence.HIGH
    elif result.confidence >= 0.4:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    return V2Analysis(
        thesis=result.thesis,
        confidence=confidence,
    )


async def _store_episodic_memory(
    episodic_store: EpisodicMemoryStore,
    trade_id: str | None,
    spread,
    result: PipelineResult,
    regime_result: dict | None,
    iv_metrics,
    current_vix: float | None,
) -> str | None:
    """Store episodic memory record for a trade.

    Args:
        episodic_store: EpisodicMemoryStore instance
        trade_id: Trade ID if trade was executed
        spread: CreditSpread
        result: PipelineResult from pipeline
        regime_result: Market regime dict
        iv_metrics: IV metrics
        current_vix: VIX at entry

    Returns:
        Memory ID if stored, None otherwise
    """
    # Build predicted outcome from synthesis
    predicted_outcome = None
    if result.synthesis_message and result.synthesis_message.structured_data:
        data = result.synthesis_message.structured_data
        predicted_outcome = {
            "recommendation": result.recommendation,
            "confidence": result.confidence,
            "expected_profit_probability": data.get("position_size_multiplier", 0.5),
            "key_bull_points": data.get("key_bull_points", []),
            "key_bear_points": data.get("key_bear_points", []),
            "thesis": result.thesis,
        }

    memory_id = await episodic_store.store_memory(
        trade_id=trade_id,
        underlying=spread.underlying,
        spread_type=spread.spread_type.value,
        short_strike=spread.short_strike,
        long_strike=spread.long_strike,
        expiration=spread.expiration,
        analyst_messages=result.analyst_messages,
        debate_messages=result.debate_messages,
        synthesis_message=result.synthesis_message,
        market_regime=regime_result.get("regime") if regime_result else None,
        iv_rank=iv_metrics.iv_rank if iv_metrics else None,
        vix_at_entry=current_vix,
        predicted_outcome=predicted_outcome,
    )

    return memory_id


async def _refresh_dynamic_betas(
    alpaca,
    db,
    kv,
) -> None:
    """Refresh dynamic betas for all traded symbols.

    Calculates EWMA and rolling betas relative to SPY.
    Caches results in KV and stores in D1 for history.
    """
    from core.risk.dynamic_beta import DynamicBetaCalculator

    symbols = ["QQQ", "IWM", "TLT", "GLD"]  # SPY is the benchmark
    calculator = DynamicBetaCalculator()

    # Fetch SPY bars (benchmark)
    try:
        spy_bars = await alpaca.get_historical_bars("SPY", timeframe="1Day", limit=80)
        if len(spy_bars) < 60:
            print(f"Insufficient SPY data for beta calculation ({len(spy_bars)} bars)")
            return
    except Exception as e:
        print(f"Failed to fetch SPY bars for beta calculation: {e}")
        return

    for symbol in symbols:
        try:
            bars = await alpaca.get_historical_bars(symbol, timeframe="1Day", limit=80)
            result = calculator.calculate_for_symbol(symbol, bars, spy_bars)

            # Cache in KV
            await kv.cache_beta(symbol, result.to_dict())

            # Store in D1 for history (only if not fallback)
            if not result.is_fallback:
                await db.save_dynamic_beta(
                    symbol=result.symbol,
                    beta_ewma=result.beta_ewma,
                    beta_rolling_20=result.beta_rolling_20,
                    beta_rolling_60=result.beta_rolling_60,
                    beta_blended=result.beta_blended,
                    correlation_spy=result.correlation_spy,
                    data_days=result.data_days,
                )

            status = "fallback" if result.is_fallback else f"{result.beta_blended:.2f}"
            print(f"Updated beta for {symbol}: {status}")

        except Exception as e:
            print(f"Failed to update beta for {symbol}: {e}")


async def handle_morning_scan(env):
    """Run the morning options scan."""
    print("Starting morning scan...")

    # Signal start to heartbeat monitor
    heartbeat_url = getattr(env, "HEARTBEAT_URL", None)
    await http.ping_heartbeat_start(heartbeat_url, "morning_scan")

    job_success = False
    try:
        await _run_morning_scan(env)
        job_success = True
    finally:
        # Ping heartbeat with success/failure
        await http.ping_heartbeat(heartbeat_url, "morning_scan", success=job_success)


async def _run_morning_scan(env):
    """Internal morning scan logic."""

    # Initialize clients
    db = D1Client(env.MAHLER_DB)
    kv = KVClient(env.MAHLER_KV)
    circuit_breaker = CircuitBreaker(kv)

    # Check circuit breaker with daily auto-reset
    # Research: Circuit breakers need "cooling off" periods, so we auto-reset at start of new day
    circuit_breaker_auto_reset = False
    previous_halt_reason = None
    status = await circuit_breaker.get_status()
    if status.halted:
        today = datetime.now().strftime("%Y-%m-%d")
        triggered_date = status.triggered_at.strftime("%Y-%m-%d") if status.triggered_at else None

        if triggered_date and triggered_date < today:
            # Halt was from a previous day - auto-reset for new trading day
            print(f"Auto-resetting circuit breaker (triggered {triggered_date}, now {today})")
            previous_halt_reason = status.reason
            await circuit_breaker.reset()
            circuit_breaker_auto_reset = True
        else:
            # Halt is from today - respect the halt
            print(f"Trading halted today: {status.reason}")
            return

    # Initialize external clients
    alpaca = AlpacaClient(
        api_key=env.ALPACA_API_KEY,
        secret_key=env.ALPACA_SECRET_KEY,
        paper=(env.ENVIRONMENT == "paper"),
    )

    discord = DiscordClient(
        bot_token=env.DISCORD_BOT_TOKEN,
        public_key=env.DISCORD_PUBLIC_KEY,
        channel_id=env.DISCORD_CHANNEL_ID,
    )

    claude = ClaudeClient(api_key=env.ANTHROPIC_API_KEY)

    # V2 Multi-Agent System initialization
    debate_rounds = int(getattr(env, "MULTI_AGENT_DEBATE_ROUNDS", "2"))

    print(f"V2 Multi-Agent System (debate_rounds={debate_rounds})")
    orchestrator = _create_v2_orchestrator(claude, debate_rounds=debate_rounds)

    # Initialize episodic memory if bindings are available
    episodic_store = None
    memory_retriever = None
    if hasattr(env, "EPISODIC_MEMORY") and hasattr(env, "AI"):
        episodic_store = EpisodicMemoryStore(
            vectorize_binding=env.EPISODIC_MEMORY,
            ai_binding=env.AI,
            d1_binding=env.MAHLER_DB,
        )
        memory_retriever = MemoryRetriever(
            d1_binding=env.MAHLER_DB,
            episodic_store=episodic_store,
        )
        print("V2 Episodic memory initialized")
    else:
        print("V2 Running without episodic memory (bindings not configured)")

    # Initialize trajectory store for learning
    trajectory_store = TrajectoryStore(env.MAHLER_DB)
    print("V2 Trajectory store initialized")

    # Initialize three-perspective risk manager
    three_persp_manager = None

    # Check if market is open
    market_open = await alpaca.is_market_open()
    print(f"Market open: {market_open}")
    if not market_open:
        print("Market is closed, skipping scan")
        return

    # Send scan start notification
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    await discord.send_message(
        content="**Morning Scan Starting**",
        embeds=[{
            "title": "Morning Scan Initiated",
            "color": 0x5865F2,  # Blurple
            "description": f"Starting options screening for credit spread opportunities.\n\n**Time:** {scan_time}",
            "fields": [
                {"name": "Underlyings", "value": ", ".join(UNDERLYINGS), "inline": True},
            ],
        }]
    )

    # Notify if circuit breaker was auto-reset
    if circuit_breaker_auto_reset:
        await discord.send_message(
            content="**Circuit Breaker Auto-Reset**",
            embeds=[{
                "title": "Trading Resumed",
                "color": 0x57F287,  # Green
                "description": f"Circuit breaker automatically reset for new trading day.\n\n**Previous halt reason:** {previous_halt_reason}",
            }]
        )

    # Refresh dynamic betas (daily calculation)
    try:
        await _refresh_dynamic_betas(alpaca, db, kv)
    except Exception as e:
        print(f"Beta refresh failed: {e}, using cached/static betas")

    # Get account info for position sizing
    account = await alpaca.get_account()
    positions = await db.get_all_positions()
    open_trades = await db.get_open_trades()

    # Initialize weekly stats (will only set if not already initialized this week)
    await kv.initialize_weekly_stats(starting_equity=account.equity)
    weekly_stats = await kv.get_weekly_stats()
    print(f"Weekly starting equity: ${weekly_stats['starting_equity']:,.2f}")

    # Get current VIX for position sizing and circuit breaker
    current_vix = None
    try:
        vix_data = await alpaca.get_vix_snapshot()
        if vix_data:
            current_vix = vix_data.get("vix")
            vix3m = vix_data.get("vix3m")
            if current_vix:
                print(f"VIX: {current_vix:.2f}")
                # Check for backwardation (VIX > VIX3M indicates near-term fear)
                if vix3m and current_vix / vix3m > 1.0:
                    print(f"VIX in backwardation ({current_vix/vix3m:.2f}x), elevated caution")
    except Exception as e:
        print(f"Could not fetch VIX: {e}")

    # Detect market regime using SPY
    regime_result = None
    regime_multiplier = 1.0
    try:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour = now.hour

        # Check cache first
        cached = await kv.get_cached_regime("SPY", date_str, hour)

        if cached:
            regime_result = cached
            regime_multiplier = cached["position_multiplier"]
            print(f"Using cached regime: {cached['regime']} ({cached['probability']:.0%})")
        else:
            # Fetch historical bars for SPY (need ~80 days for 50-day SMA)
            bars = await alpaca.get_historical_bars("SPY", timeframe="1Day", limit=80)

            if len(bars) >= 60:
                # Get SPY ATM IV
                spy_chain = await alpaca.get_options_chain("SPY")
                atm_contracts = [
                    c for c in spy_chain.contracts
                    if abs(c.strike - spy_chain.underlying_price) < spy_chain.underlying_price * 0.02
                    and c.implied_volatility
                ]
                if not atm_contracts:
                    raise ValueError("No SPY ATM contracts with IV data for regime detection")
                spy_iv = sum(c.implied_volatility for c in atm_contracts) / len(atm_contracts)

                # Detect regime (uses pre-computed model in production, sklearn in dev)
                detector = await create_regime_detector(env, kv)
                regime_result_obj = detector.detect_regime(bars, spy_iv, current_vix)
                regime_multiplier = regime_result_obj.position_multiplier
                regime_result = regime_result_obj.to_dict()

                # Cache result
                await kv.cache_regime("SPY", date_str, hour, regime_result)

                # Store in D1 for history
                await db.save_market_regime(
                    symbol="SPY",
                    regime=regime_result["regime"],
                    probability=regime_result["probability"],
                    position_multiplier=regime_result["position_multiplier"],
                    features=regime_result["features"],
                    detected_at=regime_result["detected_at"],
                )

                regime_display = regime_result["regime"].replace("_", " ").title()
                print(f"Detected regime: {regime_display} ({regime_result['probability']:.0%}), multiplier: {regime_multiplier}")
            else:
                print(f"Insufficient historical data for regime detection ({len(bars)} bars)")

    except InsufficientDataError as e:
        print(f"Insufficient data for regime detection: {e}")
    except Exception as e:
        print(f"Regime detection failed: {e}, using VIX-only logic")

    # Full graduated risk evaluation
    daily_stats = await kv.get_daily_stats()
    daily_starting_equity = daily_stats.get("starting_equity", account.equity)

    risk_state = await circuit_breaker.evaluate_all(
        starting_daily_equity=daily_starting_equity,
        starting_weekly_equity=weekly_stats["starting_equity"] or account.equity,
        peak_equity=max(daily_starting_equity, weekly_stats["starting_equity"] or account.equity),
        current_equity=account.equity,
        current_vix=current_vix,
    )

    # Log and handle risk state
    if risk_state.level != RiskLevel.NORMAL:
        print(f"Risk level: {risk_state.level.value}, size multiplier: {risk_state.size_multiplier}")
        if risk_state.reason:
            print(f"Reason: {risk_state.reason}")

    if risk_state.level == RiskLevel.HALTED:
        print(f"Trading halted: {risk_state.reason}")
        return

    # Get playbook rules for AI context
    playbook_rules = await db.get_playbook_rules()

    # Initialize regime-conditional scorer with optimized weights if available
    scorer = RegimeConditionalScorer()
    try:
        cached_weights = await kv.get_cached_weights()
        if cached_weights:
            scorer.load_from_dict(cached_weights)
            print(f"Loaded optimized weights for {len(cached_weights)} regimes")
        else:
            print("Using default scoring weights (no optimization yet)")
    except Exception as e:
        print(f"Error loading optimized weights: {e}, using defaults")

    # Initialize screener with scorer
    screener = OptionsScreener(ScreenerConfig(), scorer=scorer)
    sizer = PositionSizer(kv_client=kv)

    # V2: Initialize three-perspective risk manager
    three_persp_manager = ThreePerspectiveRiskManager(sizer)
    print("V2 Three-perspective risk manager initialized")

    # Combined size multiplier (risk + regime)
    # Use the more conservative (lower) of the two multipliers
    combined_size_multiplier = min(risk_state.size_multiplier, regime_multiplier)
    if combined_size_multiplier < 1.0:
        print(f"Size multiplier: {combined_size_multiplier:.2f} (risk: {risk_state.size_multiplier:.2f}, regime: {regime_multiplier:.2f})")

    all_opportunities = []

    # Screening tracking for logging and notifications
    screening_stats = {
        "total_underlyings_scanned": 0,
        "opportunities_found": 0,
        "opportunities_passed_filters": 0,
        "opportunities_sent_to_agents": 0,
        "opportunities_approved": 0,
    }
    skip_reasons: dict[str, int] = {}
    underlying_results: dict[str, dict] = {}
    iv_percentiles: dict[str, float] = {}

    # Scan each underlying
    for symbol in UNDERLYINGS:
        screening_stats["total_underlyings_scanned"] += 1
        underlying_results[symbol] = {"found": 0, "passed": 0, "reason": ""}

        try:
            print(f"Scanning {symbol}...")

            # Get options chain
            try:
                chain = await alpaca.get_options_chain(symbol)
            except Exception as e:
                raise RuntimeError(f"Options chain fetch failed: {e}") from e

            if not chain.contracts:
                print(f"No options data for {symbol}")
                underlying_results[symbol]["reason"] = "No options data"
                continue

            print(f"{symbol}: Got {len(chain.contracts)} contracts, price=${chain.underlying_price:.2f}")

            # Calculate IV metrics (using ATM options as proxy)
            atm_contracts = [
                c
                for c in chain.contracts
                if abs(c.strike - chain.underlying_price) < chain.underlying_price * 0.02
                and c.implied_volatility is not None
            ]

            if not atm_contracts:
                print(f"{symbol}: No ATM contracts with IV data - skipping")
                underlying_results[symbol]["reason"] = "No IV data available"
                skip_reasons["no_iv_data"] = skip_reasons.get("no_iv_data", 0) + 1
                continue

            # Use average IV from ATM contracts
            current_iv = sum(c.implied_volatility for c in atm_contracts) / len(atm_contracts)

            # Load real IV history from database
            try:
                historical_ivs = await db.get_iv_history(symbol, lookback_days=252)
            except Exception as e:
                raise RuntimeError(f"IV history fetch failed: {e}") from e
            iv_history_count = len(historical_ivs)

            # Calculate IV metrics - use historical if available, otherwise use current IV as baseline
            try:
                if iv_history_count >= 30:
                    iv_metrics = calculate_iv_metrics(current_iv, historical_ivs)
                else:
                    # Not enough history - use current IV with neutral rank
                    # This allows trading while IV history builds up
                    from core.analysis.iv_rank import IVMetrics
                    iv_metrics = IVMetrics(
                        current_iv=current_iv,
                        iv_rank=50.0,  # Neutral - no historical context
                        iv_percentile=50.0,
                        iv_high=current_iv,
                        iv_low=current_iv,
                    )
                    print(f"{symbol}: Insufficient IV history ({iv_history_count} days) - using neutral rank")
            except Exception as e:
                raise RuntimeError(f"IV metrics calculation failed: {e}") from e
            print(f"{symbol}: IV={current_iv:.2%}, Rank={iv_metrics.iv_rank:.1f}%, Percentile={iv_metrics.iv_percentile:.1f}%")

            # Track IV percentile for logging
            iv_percentiles[symbol] = iv_metrics.iv_percentile

            # IV Term Structure Analysis
            # Build term structure from options chain expirations
            term_structure_result = None
            try:
                # Group contracts by expiration and get ATM IV for each
                expirations_iv: dict[str, list[float]] = {}
                for c in chain.contracts:
                    if c.implied_volatility:
                        # Only use near-ATM contracts
                        if abs(c.strike - chain.underlying_price) < chain.underlying_price * 0.05:
                            if c.expiration not in expirations_iv:
                                expirations_iv[c.expiration] = []
                            expirations_iv[c.expiration].append(c.implied_volatility)

                # Build term structure points
                if len(expirations_iv) >= 2:
                    ts_points = []
                    today = datetime.now()
                    for exp_str, ivs in expirations_iv.items():
                        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                        dte = max(1, (exp_date - today).days)
                        avg_iv = sum(ivs) / len(ivs)
                        ts_points.append(TermStructurePoint(dte=dte, iv=avg_iv))

                    term_structure = IVTermStructure(ts_points)
                    term_structure_result = term_structure.detect_regime()

                    # Log term structure regime
                    regime_display = term_structure_result.regime.value.replace("_", " ").title()
                    print(f"{symbol}: Term structure {regime_display} (30/90 ratio: {term_structure_result.ratio_30_90:.2f})")

                    # Skip if backwardation (elevated near-term fear)
                    if term_structure_result.regime == TermStructureRegime.BACKWARDATION:
                        print(f"{symbol}: Skipping due to backwardation (elevated short-term fear)")
                        underlying_results[symbol]["reason"] = "Backwardation - elevated short-term fear"
                        skip_reasons["backwardation"] = skip_reasons.get("backwardation", 0) + 1
                        continue

            except Exception as e:
                print(f"{symbol}: Term structure analysis failed: {e}")

            # IV Mean Reversion Analysis
            mean_reversion_result = None
            if iv_history_count >= 60:
                try:
                    mr_analyzer = IVMeanReversion(historical_ivs)
                    mean_reversion_result = mr_analyzer.generate_signal(current_iv)

                    # Log mean reversion signal
                    if mean_reversion_result.signal != MeanReversionSignal.HOLD:
                        print(f"{symbol}: Mean reversion signal: {mean_reversion_result.signal.value} (z={mean_reversion_result.z_score:.2f})")

                except Exception as e:
                    print(f"{symbol}: Mean reversion analysis failed: {e}")

            # Screen for opportunities with regime-conditional scoring
            try:
                current_regime = regime_result.get("regime") if regime_result else None
                opportunities = screener.screen_chain(chain, iv_metrics, regime=current_regime)
            except Exception as e:
                raise RuntimeError(f"Screening failed: {e}") from e

            # Track found opportunities
            underlying_results[symbol]["found"] = len(opportunities) if opportunities else 0
            screening_stats["opportunities_found"] += len(opportunities) if opportunities else 0

            if opportunities:
                print(f"Found {len(opportunities)} opportunities for {symbol}")
                underlying_results[symbol]["passed"] = min(len(opportunities), 2)  # Top 2 per symbol
                screening_stats["opportunities_passed_filters"] += min(len(opportunities), 2)
                for opp in opportunities[:2]:  # Top 2 per symbol
                    # Include term structure and mean reversion context with opportunity
                    iv_context = {
                        "term_structure": term_structure_result,
                        "mean_reversion": mean_reversion_result,
                    }
                    all_opportunities.append((opp, chain.underlying_price, iv_metrics, iv_context))
            else:
                underlying_results[symbol]["reason"] = "No opportunities passed screening filters"
                skip_reasons["no_opportunities"] = skip_reasons.get("no_opportunities", 0) + 1

        except Exception as e:
            import traceback
            error_type = type(e).__name__
            print(f"Error scanning {symbol}: {error_type}: {e}")
            print(traceback.format_exc())
            underlying_results[symbol]["reason"] = f"{error_type}: {str(e)[:200]}"
            skip_reasons["error"] = skip_reasons.get("error", 0) + 1
            await circuit_breaker.check_api_errors()

    # Sort all opportunities by score
    all_opportunities.sort(key=lambda x: x[0].score, reverse=True)
    print(f"[DEBUG] Total opportunities found across all symbols: {len(all_opportunities)}")

    # Process top opportunities
    recommendations_sent = 0

    # Track opportunities sent to agent pipeline
    screening_stats["opportunities_sent_to_agents"] = min(len(all_opportunities), MAX_RECOMMENDATIONS)

    for opp, underlying_price, iv_metrics, iv_context in all_opportunities[:MAX_RECOMMENDATIONS]:
        try:
            spread = opp.spread

            # Calculate second-order Greeks (vanna/volga) for position sizing adjustment
            # These measure sensitivity to spot-vol correlation (vanna) and vol-of-vol (volga)
            spread_vanna = None
            spread_volga = None
            try:
                time_to_exp = years_to_expiry(spread.expiration)
                short_iv = spread.short_contract.implied_volatility or 0.20
                long_iv = spread.long_contract.implied_volatility or 0.20

                # Calculate second-order Greeks for each leg
                option_type = "put" if spread.spread_type.value == "bull_put" else "call"
                short_second = calculate_second_order_greeks(
                    spot=underlying_price,
                    strike=spread.short_strike,
                    time_to_expiry=time_to_exp,
                    volatility=short_iv,
                    option_type=option_type,
                )
                long_second = calculate_second_order_greeks(
                    spot=underlying_price,
                    strike=spread.long_strike,
                    time_to_expiry=time_to_exp,
                    volatility=long_iv,
                    option_type=option_type,
                )

                # Calculate net spread Greeks
                spread_greeks = calculate_spread_second_order_greeks(short_second, long_second)
                spread_vanna = spread_greeks.vanna
                spread_volga = spread_greeks.volga
            except Exception as e:
                print(f"Error calculating second-order Greeks: {e}")

            # Calculate position size (with vanna/volga adjustment)
            size_result = sizer.calculate_size(
                spread=spread,
                account_equity=account.equity,
                current_positions=positions,
                current_vix=current_vix,
                spread_vanna=spread_vanna,
                spread_volga=spread_volga,
            )

            if size_result.contracts == 0:
                print(f"Position size is 0 for {spread.underlying}: {size_result.reason}")
                skip_reasons["position_size_zero"] = skip_reasons.get("position_size_zero", 0) + 1
                continue

            # Apply graduated risk multiplier
            adjusted_contracts = max(1, int(size_result.contracts * combined_size_multiplier))
            if adjusted_contracts < size_result.contracts:
                print(f"Adjusted contracts: {size_result.contracts} -> {adjusted_contracts} (multiplier: {combined_size_multiplier:.2f})")

            # Build IV analysis context for AI
            iv_analysis_context = []
            ts_result = iv_context.get("term_structure")
            mr_result = iv_context.get("mean_reversion")

            if ts_result:
                iv_analysis_context.append(
                    f"Term structure: {ts_result.regime.value} (30/90 ratio: {ts_result.ratio_30_90:.2f})"
                )
            if mr_result:
                iv_analysis_context.append(
                    f"IV mean reversion: {mr_result.signal.value} (z-score: {mr_result.z_score:.2f})"
                )

            # V2 Multi-Agent Analysis Pipeline
            print(f"Running multi-agent analysis for {spread.underlying}...")

            # Get daily and weekly P&L for portfolio context
            daily_stats = await kv.get_daily_stats()
            daily_pnl = daily_stats.get("realized_pnl", 0)
            weekly_pnl = weekly_stats.get("realized_pnl", 0) if weekly_stats else 0

            # Get price bars for technical analysis
            price_bars = None
            try:
                price_bars = await alpaca.get_historical_bars(spread.underlying, timeframe="1Day", limit=50)
            except Exception as e:
                print(f"Could not fetch price bars: {e}")

            # Get portfolio Greeks if available
            portfolio_greeks = None  # TODO: Calculate from positions if needed

            v2_result, similar_trades = await _run_v2_analysis(
                orchestrator=orchestrator,
                spread=spread,
                underlying_price=underlying_price,
                iv_metrics=iv_metrics,
                term_structure=ts_result,
                mean_reversion=mr_result,
                regime_result=regime_result,
                current_vix=current_vix,
                vix_3m=vix_3m if 'vix_3m' in dir() else None,
                price_bars=price_bars,
                positions=positions,
                portfolio_greeks=portfolio_greeks,
                account=account,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                playbook_rules=playbook_rules,
                episodic_store=episodic_store,
                memory_retriever=memory_retriever,
            )

            print(f"Pipeline complete: recommendation={v2_result.recommendation}, confidence={v2_result.confidence:.0%}")

            # Log agent pipeline decisions to Discord
            try:
                # Serialize pipeline messages to dicts for Discord
                def serialize_message(msg):
                    if msg is None:
                        return None
                    return {
                        "agent_id": msg.agent_id,
                        "content": msg.content,
                        "confidence": msg.confidence,
                        "structured_data": msg.structured_data,
                    }

                pipeline_dict = {
                    "analyst_messages": [serialize_message(m) for m in (v2_result.analyst_messages or [])],
                    "debate_messages": [serialize_message(m) for m in (v2_result.debate_messages or [])],
                    "fund_manager_message": serialize_message(v2_result.synthesis_message),
                }
                await discord.send_agent_pipeline_log(
                    underlying=spread.underlying,
                    spread_type=spread.spread_type.value,
                    pipeline_result=pipeline_dict,
                )
            except Exception as e:
                print(f"Error sending agent pipeline log: {e}")

            # Map V2 result to TradeAnalysis format
            analysis = _map_v2_result_to_analysis(v2_result)

            # Apply V2 position size adjustment if synthesis suggests it
            if v2_result.synthesis_message and v2_result.synthesis_message.structured_data:
                v2_size_mult = v2_result.synthesis_message.structured_data.get("position_size_multiplier", 1.0)
                if v2_size_mult < 1.0:
                    new_adjusted = max(1, int(adjusted_contracts * v2_size_mult))
                    if new_adjusted < adjusted_contracts:
                        print(f"Reducing position size: {adjusted_contracts} -> {new_adjusted} (V2 multiplier: {v2_size_mult:.2f})")
                        adjusted_contracts = new_adjusted

            # Skip low confidence trades
            if analysis.confidence == Confidence.LOW:
                print(f"Skipping low confidence trade: {spread.underlying}")
                skip_reasons["low_confidence"] = skip_reasons.get("low_confidence", 0) + 1
                # Notify about skipped trade
                await discord.send_trade_decision(
                    underlying=spread.underlying,
                    spread_type=spread.spread_type.value,
                    short_strike=spread.short_strike,
                    long_strike=spread.long_strike,
                    expiration=spread.expiration,
                    credit=spread.credit,
                    decision="skipped",
                    reason="Low confidence from AI analysis",
                    ai_summary=v2_result.thesis,
                    confidence=v2_result.confidence,
                    iv_rank=iv_metrics.iv_rank,
                )
                continue

            # Skip if recommendation is "skip"
            if v2_result.recommendation == "skip":
                print(f"Skipping trade per multi-agent recommendation: {spread.underlying}")
                skip_reasons["agent_rejected"] = skip_reasons.get("agent_rejected", 0) + 1
                # Notify about rejected trade with AI reasoning
                await discord.send_trade_decision(
                    underlying=spread.underlying,
                    spread_type=spread.spread_type.value,
                    short_strike=spread.short_strike,
                    long_strike=spread.long_strike,
                    expiration=spread.expiration,
                    credit=spread.credit,
                    decision="rejected",
                    reason="Multi-agent analysis recommended skip",
                    ai_summary=v2_result.thesis,
                    confidence=v2_result.confidence,
                    iv_rank=iv_metrics.iv_rank,
                )
                continue

            # Get delta/theta from the scored spread or Greeks if available
            short_delta = None
            short_theta = None
            if spread.short_contract.greeks:
                short_delta = spread.short_contract.greeks.delta
                short_theta = spread.short_contract.greeks.theta

            # Create recommendation
            rec_id = await db.create_recommendation(
                underlying=spread.underlying,
                spread_type=spread.spread_type,
                short_strike=spread.short_strike,
                long_strike=spread.long_strike,
                expiration=spread.expiration,
                credit=spread.credit,
                max_loss=spread.max_loss,
                expires_at=datetime.now() + timedelta(minutes=15),
                iv_rank=iv_metrics.iv_rank,
                delta=short_delta,
                theta=short_theta,
                thesis=analysis.thesis,
                confidence=analysis.confidence,
                suggested_contracts=adjusted_contracts,
                analysis_price=spread.credit,
            )

            # Get the full recommendation
            rec = await db.get_recommendation(rec_id)

            # V2: Place order directly (autonomous mode)
            try:
                # Build OCC symbols
                exp_parts = spread.expiration.split("-")
                exp_str = exp_parts[0][2:] + exp_parts[1] + exp_parts[2]
                option_type = "P" if spread.spread_type == SpreadType.BULL_PUT else "C"
                short_symbol = f"{spread.underlying}{exp_str}{option_type}{int(spread.short_strike * 1000):08d}"
                long_symbol = f"{spread.underlying}{exp_str}{option_type}{int(spread.long_strike * 1000):08d}"

                # Place order
                from core.broker.types import SpreadOrder
                spread_order = SpreadOrder(
                    underlying=spread.underlying,
                    short_symbol=short_symbol,
                    long_symbol=long_symbol,
                    contracts=adjusted_contracts,
                    limit_price=spread.credit,
                )
                order = await alpaca.place_spread_order(spread_order)

                # Update recommendation status
                await db.update_recommendation_status(rec_id, RecommendationStatus.APPROVED)

                # Create trade record with pending_fill status
                # The position monitor will verify the order filled and update to 'open'
                trade_id = await db.create_trade(
                    recommendation_id=rec_id,
                    underlying=spread.underlying,
                    spread_type=spread.spread_type,
                    short_strike=spread.short_strike,
                    long_strike=spread.long_strike,
                    expiration=spread.expiration,
                    entry_credit=spread.credit,
                    contracts=adjusted_contracts,
                    broker_order_id=order.id,
                    status=TradeStatus.PENDING_FILL,
                )

                # Send autonomous notification (info-only, no buttons)
                await discord.send_autonomous_notification(
                    rec=rec,
                    v2_confidence=v2_result.confidence,
                    v2_thesis=v2_result.thesis,
                    order_id=order.id,
                )

                recommendations_sent += 1
                screening_stats["opportunities_approved"] += 1
                print(f"Order placed (pending fill): {trade_id}, Order: {order.id}")

                # Store episodic memory with trade ID
                if episodic_store:
                    try:
                        memory_id = await _store_episodic_memory(
                            episodic_store=episodic_store,
                            trade_id=trade_id,
                            spread=spread,
                            result=v2_result,
                            regime_result=regime_result,
                            iv_metrics=iv_metrics,
                            current_vix=current_vix,
                        )
                        print(f"Stored episodic memory: {memory_id}")
                    except Exception as e:
                        print(f"Error storing episodic memory: {e}")

                # Store trajectory for learning
                try:
                    # Build three-perspective result if available
                    three_persp_result = None
                    if three_persp_manager and current_vix:
                        three_persp_result = three_persp_manager.assess(
                            spread=spread,
                            account_equity=account.equity,
                            current_positions=positions,
                            current_vix=current_vix,
                        )

                    trajectory = TradeTrajectory.from_pipeline_result(
                        underlying=spread.underlying,
                        spread_type=spread.spread_type.value,
                        short_strike=spread.short_strike,
                        long_strike=spread.long_strike,
                        expiration=spread.expiration,
                        entry_credit=spread.credit,
                        contracts=adjusted_contracts,
                        analyst_messages=v2_result.analyst_messages,
                        debate_messages=v2_result.debate_messages,
                        synthesis_message=v2_result.synthesis_message,
                        decision_output=v2_result.synthesis_message.structured_data if v2_result.synthesis_message else None,
                        three_perspective=three_persp_result,
                        market_regime=regime_result.get("regime") if regime_result else None,
                        iv_rank=iv_metrics.iv_rank if iv_metrics else None,
                        vix_at_entry=current_vix,
                        trade_id=trade_id,
                    )
                    trajectory_id = await trajectory_store.store_trajectory(trajectory)
                    print(f"Stored trajectory: {trajectory_id}")
                except Exception as e:
                    print(f"Error storing trajectory: {e}")

            except Exception as e:
                print(f"Error placing order: {e}")

        except ClaudeRateLimitError as e:
            print(f"Claude API rate limit error: {e}")
            await discord.send_api_token_alert("Claude", str(e))
            # Continue processing other opportunities
        except Exception as e:
            import traceback
            print(f"Error processing opportunity: {e}")
            print(traceback.format_exc())

    print(f"Morning scan complete. Sent {recommendations_sent} recommendations.")

    # Save screening results to database
    try:
        scan_timestamp = datetime.now().isoformat()
        scan_date = datetime.now().strftime("%Y-%m-%d")

        # Build market context for logging
        market_context = {
            "vix": current_vix,
            "iv_percentile": iv_percentiles,
            "regime": regime_result.get("regime") if regime_result else None,
            "regime_probability": regime_result.get("probability") if regime_result else None,
            "risk_multiplier": risk_state.size_multiplier,
            "regime_multiplier": regime_multiplier,
            "combined_multiplier": combined_size_multiplier,
        }

        await db.save_screening_result(
            scan_date=scan_date,
            scan_time="morning",
            scan_timestamp=scan_timestamp,
            total_underlyings_scanned=screening_stats["total_underlyings_scanned"],
            opportunities_found=screening_stats["opportunities_found"],
            opportunities_passed_filters=screening_stats["opportunities_passed_filters"],
            opportunities_sent_to_agents=screening_stats["opportunities_sent_to_agents"],
            opportunities_approved=screening_stats["opportunities_approved"],
            skip_reasons=skip_reasons,
            underlying_results=underlying_results,
            market_context=market_context,
            circuit_breaker_active=False,
            circuit_breaker_reason=None,
            risk_multiplier=risk_state.size_multiplier,
            regime_multiplier=regime_multiplier,
            combined_multiplier=combined_size_multiplier,
        )
        print(f"Saved screening results: {screening_stats}")
    except Exception as e:
        print(f"Error saving screening results: {e}")

    # Send "no trade" notification if no trades were approved
    if recommendations_sent == 0:
        try:
            market_context_for_notification = {
                "vix": current_vix,
                "iv_percentile": iv_percentiles,
                "regime": regime_result.get("regime") if regime_result else None,
                "combined_multiplier": combined_size_multiplier,
            }

            await discord.send_no_trade_notification(
                scan_time="morning",
                underlyings_scanned=screening_stats["total_underlyings_scanned"],
                opportunities_found=screening_stats["opportunities_found"],
                opportunities_filtered=screening_stats["opportunities_passed_filters"],
                skip_reasons=skip_reasons,
                market_context=market_context_for_notification,
                underlying_details=underlying_results,
            )
            print("Sent no-trade notification")
        except Exception as e:
            print(f"Error sending no-trade notification: {e}")
