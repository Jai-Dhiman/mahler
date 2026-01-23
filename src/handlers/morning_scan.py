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

# Underlyings to scan
# SPY/QQQ/IWM are equity ETFs (86-92% correlated)
# TLT is treasury ETF (negatively correlated with equities)
# GLD is gold ETF (low/variable correlation)
UNDERLYINGS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

# Maximum recommendations per scan
MAX_RECOMMENDATIONS = 3


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

    # Check if market is open
    market_open = await alpaca.is_market_open()
    print(f"Market open: {market_open}")
    if not market_open:
        print("Market is closed, skipping scan")
        return

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
                spy_iv = (
                    sum(c.implied_volatility for c in atm_contracts) / len(atm_contracts)
                    if atm_contracts else 0.20
                )

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

    # Combined size multiplier (risk + regime)
    # Use the more conservative (lower) of the two multipliers
    combined_size_multiplier = min(risk_state.size_multiplier, regime_multiplier)
    if combined_size_multiplier < 1.0:
        print(f"Size multiplier: {combined_size_multiplier:.2f} (risk: {risk_state.size_multiplier:.2f}, regime: {regime_multiplier:.2f})")

    all_opportunities = []

    # Scan each underlying
    for symbol in UNDERLYINGS:
        try:
            print(f"Scanning {symbol}...")

            # Get options chain
            chain = await alpaca.get_options_chain(symbol)

            if not chain.contracts:
                print(f"No options data for {symbol}")
                continue

            print(f"{symbol}: Got {len(chain.contracts)} contracts, price=${chain.underlying_price:.2f}")

            # Calculate IV metrics (using ATM options as proxy)
            atm_contracts = [
                c
                for c in chain.contracts
                if abs(c.strike - chain.underlying_price) < chain.underlying_price * 0.02
            ]

            if atm_contracts and atm_contracts[0].implied_volatility:
                current_iv = atm_contracts[0].implied_volatility
            else:
                current_iv = 0.20  # Default

            # Load real IV history from database
            historical_ivs = await db.get_iv_history(symbol, lookback_days=252)
            iv_history_count = len(historical_ivs)

            # Use historical data if available, otherwise use fallback for testing
            if iv_history_count >= 30:
                iv_metrics = calculate_iv_metrics(current_iv, historical_ivs)
            else:
                # Fallback: estimate IV rank from VIX level (temporary for testing)
                # VIX 20-30 suggests elevated IV, use 70% rank as estimate
                from core.analysis.iv_rank import IVMetrics
                estimated_rank = min(90.0, max(50.0, current_vix * 2.5)) if current_vix else 70.0
                iv_metrics = IVMetrics(
                    current_iv=current_iv,
                    iv_rank=estimated_rank,
                    iv_percentile=estimated_rank,
                    iv_high=current_iv * 1.2,
                    iv_low=current_iv * 0.7,
                )
                print(f"{symbol}: Using estimated IV rank {estimated_rank:.0f}% (only {iv_history_count} days history)")
            print(f"{symbol}: IV={current_iv:.2%}, Rank={iv_metrics.iv_rank:.1f}%, Percentile={iv_metrics.iv_percentile:.1f}%")

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
            current_regime = regime_result.get("regime") if regime_result else None
            opportunities = screener.screen_chain(chain, iv_metrics, regime=current_regime)

            if opportunities:
                print(f"Found {len(opportunities)} opportunities for {symbol}")
                for opp in opportunities[:2]:  # Top 2 per symbol
                    # Include term structure and mean reversion context with opportunity
                    iv_context = {
                        "term_structure": term_structure_result,
                        "mean_reversion": mean_reversion_result,
                    }
                    all_opportunities.append((opp, chain.underlying_price, iv_metrics, iv_context))

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            await circuit_breaker.check_api_errors()

    # Sort all opportunities by score
    all_opportunities.sort(key=lambda x: x[0].score, reverse=True)
    print(f"[DEBUG] Total opportunities found across all symbols: {len(all_opportunities)}")

    # Process top opportunities
    recommendations_sent = 0

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

            # Get AI analysis with IV context
            analysis = await claude.analyze_trade(
                spread=spread,
                underlying_price=underlying_price,
                iv_rank=iv_metrics.iv_rank,
                current_iv=iv_metrics.current_iv,
                playbook_rules=playbook_rules,
                additional_context=iv_analysis_context if iv_analysis_context else None,
            )

            # Skip low confidence trades
            if analysis.confidence == Confidence.LOW:
                print(f"Skipping low confidence trade: {spread.underlying}")
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

            # Send to Discord
            message_id = await discord.send_recommendation(rec)
            await db.set_recommendation_discord_message_id(rec_id, message_id)

            recommendations_sent += 1
            print(f"Sent recommendation for {spread.underlying}: {rec_id}")

            # Auto-approve if enabled
            auto_approve = getattr(env, "AUTO_APPROVE_TRADES", "false").lower() == "true"
            if auto_approve:
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

                    # Update Discord message to show order placed (pending fill)
                    await discord.update_message(
                        message_id=message_id,
                        content=f"**Order Placed: {spread.underlying}** (awaiting fill)",
                        embeds=[{
                            "title": f"Trade Order Placed: {spread.underlying}",
                            "description": "Order submitted - awaiting fill confirmation",
                            "color": 0xFEE75C,  # Yellow for pending
                            "fields": [
                                {"name": "Strategy", "value": spread.spread_type.value.replace("_", " ").title(), "inline": True},
                                {"name": "Expiration", "value": spread.expiration, "inline": True},
                                {"name": "Strikes", "value": f"${spread.short_strike:.2f}/${spread.long_strike:.2f}", "inline": True},
                                {"name": "Credit", "value": f"${spread.credit:.2f}", "inline": True},
                                {"name": "Contracts", "value": str(adjusted_contracts), "inline": True},
                                {"name": "Order ID", "value": order.id, "inline": True},
                            ],
                        }],
                        components=[],  # Remove buttons
                    )

                    # Don't update daily stats yet - wait for fill confirmation
                    print(f"Order placed (pending fill): {trade_id}, Order: {order.id}")

                except Exception as e:
                    print(f"Error auto-approving trade: {e}")

        except ClaudeRateLimitError as e:
            print(f"Claude API rate limit error: {e}")
            await discord.send_api_token_alert("Claude", str(e))
            # Continue processing other opportunities
        except Exception as e:
            import traceback
            print(f"Error processing opportunity: {e}")
            print(traceback.format_exc())

    print(f"Morning scan complete. Sent {recommendations_sent} recommendations.")
