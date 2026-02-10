"""Main entry point for Cloudflare Workers.

Routes incoming requests (HTTP and cron) to appropriate handlers.

NOTE: Handler imports are lazy to avoid exceeding Cloudflare's startup CPU limit.
Heavy imports (numpy, etc.) are only loaded when handlers are actually invoked.
"""

from datetime import datetime

from workers import Response


# Lazy import helper to defer heavy module loading until needed
def _import_handler(name: str):
    """Import a handler function lazily."""
    if name == "morning_scan":
        from handlers.morning_scan import handle_morning_scan
        return handle_morning_scan
    elif name == "midday_check":
        from handlers.midday_check import handle_midday_check
        return handle_midday_check
    elif name == "afternoon_scan":
        from handlers.afternoon_scan import handle_afternoon_scan
        return handle_afternoon_scan
    elif name == "eod_summary":
        from handlers.eod_summary import handle_eod_summary
        return handle_eod_summary
    elif name == "position_monitor":
        from handlers.position_monitor import handle_position_monitor
        return handle_position_monitor
    elif name == "health":
        from handlers.health import handle_health
        return handle_health
    else:
        raise ValueError(f"Unknown handler: {name}")


async def on_fetch(request, env):
    """Handle HTTP requests."""
    import json

    url = request.url
    method = request.method

    try:
        # Health check
        if "/health" in url:
            handler = _import_handler("health")
            return await handler(request, env)

        # Test endpoints (for development only)
        if "/test/alpaca" in url:
            from core.broker.alpaca import AlpacaClient

            alpaca = AlpacaClient(
                api_key=env.ALPACA_API_KEY,
                secret_key=env.ALPACA_SECRET_KEY,
                paper=(env.ENVIRONMENT == "paper"),
            )
            account = await alpaca.get_account()
            market_open = await alpaca.is_market_open()
            return Response(
                json.dumps(
                    {
                        "status": "ok",
                        "account": {
                            "equity": account.equity,
                            "cash": account.cash,
                            "buying_power": account.buying_power,
                        },
                        "market_open": market_open,
                    }
                ),
                headers={"Content-Type": "application/json"},
            )

        if "/test/scan" in url and "/test/debug-scan" not in url:
            handler = _import_handler("morning_scan")
            await handler(env)
            return Response(
                '{"status": "ok", "message": "Morning scan completed"}',
                headers={"Content-Type": "application/json"},
            )

        if "/test/position-monitor" in url:
            handler = _import_handler("position_monitor")
            await handler(env)
            return Response(
                '{"status": "ok", "message": "Position monitor completed"}',
                headers={"Content-Type": "application/json"},
            )

        if "/test/midday" in url:
            handler = _import_handler("midday_check")
            await handler(env)
            return Response(
                '{"status": "ok", "message": "Midday check completed"}',
                headers={"Content-Type": "application/json"},
            )

        if "/test/afternoon" in url:
            handler = _import_handler("afternoon_scan")
            await handler(env)
            return Response(
                '{"status": "ok", "message": "Afternoon scan completed"}',
                headers={"Content-Type": "application/json"},
            )

        if "/test/eod" in url:
            handler = _import_handler("eod_summary")
            await handler(env)
            return Response(
                '{"status": "ok", "message": "EOD summary completed"}',
                headers={"Content-Type": "application/json"},
            )

        if "/test/v2-pipeline" in url:
            # Test V2 multi-agent pipeline with a mock spread
            from core.ai.router import LLMRouter
            from core.broker.alpaca import AlpacaClient
            from core.db.d1 import D1Client
            from core.db.kv import KVClient
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
                TradingDecisionAgent,
                RiskState,
                build_agent_context,
            )
            from core.memory.retriever import MemoryRetriever
            from core.risk.three_perspective import ThreePerspectiveRiskManager
            from core.risk.position_sizer import PositionSizer
            from core.types import SpreadType

            output = {"status": "running", "stages": {}}

            try:
                # Initialize clients
                db = D1Client(env.MAHLER_DB)
                kv = KVClient(env.MAHLER_KV)
                alpaca = AlpacaClient(
                    api_key=env.ALPACA_API_KEY,
                    secret_key=env.ALPACA_SECRET_KEY,
                    paper=(env.ENVIRONMENT == "paper"),
                )
                router = LLMRouter(api_key=env.ANTHROPIC_API_KEY)

                # Get market data
                account = await alpaca.get_account()
                vix_data = await alpaca.get_vix_snapshot()
                current_vix = vix_data.get("vix", 20.0) if vix_data else 20.0
                output["stages"]["market_data"] = {
                    "status": "ok",
                    "equity": account.equity,
                    "vix": current_vix,
                }

                # Get a real options chain for SPY
                chain = await alpaca.get_options_chain("SPY")
                if not chain.contracts:
                    return Response(
                        json.dumps({"error": "No options data available"}),
                        status=500,
                        headers={"Content-Type": "application/json"},
                    )

                # Find a suitable spread for testing
                from core.analysis.screener import OptionsScreener, ScreenerConfig
                from core.analysis.iv_rank import IVMetrics

                # Estimate IV metrics
                iv_metrics = IVMetrics(
                    current_iv=0.20,
                    iv_rank=65.0,
                    iv_percentile=65.0,
                    iv_high=0.30,
                    iv_low=0.15,
                )
                screener = OptionsScreener(ScreenerConfig())
                opportunities = screener.screen_chain(chain, iv_metrics)

                if not opportunities:
                    return Response(
                        json.dumps({"error": "No opportunities found", "chain_size": len(chain.contracts)}),
                        status=200,
                        headers={"Content-Type": "application/json"},
                    )

                spread = opportunities[0].spread
                output["stages"]["spread_selection"] = {
                    "status": "ok",
                    "underlying": spread.underlying,
                    "spread_type": spread.spread_type.value,
                    "strikes": f"{spread.short_strike}/{spread.long_strike}",
                    "expiration": spread.expiration,
                    "credit": spread.credit,
                }

                # Initialize V2 components
                memory_retriever = MemoryRetriever(env.MAHLER_DB, episodic_store=None)

                # Create orchestrator
                debate_rounds = int(env.MULTI_AGENT_DEBATE_ROUNDS or "2")
                orchestrator = AgentOrchestrator(
                    claude=router.get_client("facilitator"),
                    debate_config=DebateConfig(
                        max_rounds=debate_rounds,
                        min_rounds=1,
                        consensus_threshold=0.7,
                    ),
                )
                orchestrator.register_analyst(IVAnalyst(router=router))
                orchestrator.register_analyst(TechnicalAnalyst(router=router))
                orchestrator.register_analyst(MacroAnalyst(router=router))
                orchestrator.register_analyst(GreeksAnalyst(router=router))
                orchestrator.register_debater(BullResearcher(router=router), "bull")
                orchestrator.register_debater(BearResearcher(router=router), "bear")
                orchestrator.set_facilitator(DebateFacilitator(router=router))
                from core.agents.fund_manager import FundManagerAgent
                orchestrator.set_fund_manager(FundManagerAgent(router=router))

                output["stages"]["orchestrator_init"] = {"status": "ok", "debate_rounds": debate_rounds}

                # Build agent context
                context = build_agent_context(
                    spread=spread,
                    underlying_price=chain.underlying_price,
                    iv_metrics=iv_metrics,
                    regime=None,
                    regime_probability=None,
                    current_vix=current_vix,
                    vix_3m=current_vix,
                    price_bars=None,
                    positions=[],
                    portfolio_greeks=None,
                    account_equity=account.equity,
                    buying_power=account.buying_power,
                    daily_pnl=0.0,
                    weekly_pnl=0.0,
                    playbook_rules=await db.get_playbook_rules(),
                    similar_trades=[],
                )
                output["stages"]["context_build"] = {"status": "ok"}

                # Run V2 pipeline
                result = await orchestrator.run_pipeline(context)
                output["stages"]["pipeline_result"] = {
                    "status": "ok",
                    "recommendation": result.recommendation,
                    "confidence": result.confidence,
                    "thesis": result.thesis[:200] if result.thesis else None,
                    "analyst_count": len(result.analyst_messages),
                    "debate_rounds": len(result.debate_messages) // 2 if result.debate_messages else 0,
                    "duration_ms": result.duration_ms,
                }

                # Test three-perspective risk
                sizer = PositionSizer()
                three_persp = ThreePerspectiveRiskManager(sizer)
                risk_result = three_persp.assess(
                    spread=spread,
                    account_equity=account.equity,
                    current_positions=[],
                    current_vix=current_vix,
                )
                output["stages"]["three_perspective"] = {
                    "status": "ok",
                    "aggressive_contracts": risk_result.aggressive.recommended_contracts,
                    "neutral_contracts": risk_result.neutral.recommended_contracts,
                    "conservative_contracts": risk_result.conservative.recommended_contracts,
                    "weighted_contracts": risk_result.weighted_contracts,
                    "consensus": risk_result.consensus_recommendation,
                }

                # Test decision agent
                decision_agent = TradingDecisionAgent(router=router)
                risk_state = RiskState(
                    risk_level="normal",
                    size_multiplier=1.0,
                    portfolio_heat=0.0,
                    daily_pnl=0.0,
                    weekly_pnl=0.0,
                    is_halted=False,
                )

                # Retrieve context for decision
                retrieved_context = await memory_retriever.retrieve_context(
                    underlying=spread.underlying,
                    spread_type=spread.spread_type.value,
                    market_regime=None,
                    iv_rank=iv_metrics.iv_rank,
                    vix=current_vix,
                )

                decision = await decision_agent.make_decision(
                    context=context,
                    risk_state=risk_state,
                    retrieved_context=retrieved_context,
                    three_persp_manager=three_persp,
                    current_vix=current_vix,
                )
                output["stages"]["decision_agent"] = {
                    "status": "ok",
                    "decision": decision.decision,
                    "position_size": decision.position_size,
                    "confidence": decision.confidence,
                    "thesis": decision.final_thesis[:200] if decision.final_thesis else None,
                    "key_factors": decision.key_factors[:3],
                }

                output["status"] = "complete"

            except Exception as e:
                import traceback
                output["status"] = "error"
                output["error"] = str(e)
                output["traceback"] = traceback.format_exc()

            return Response(
                json.dumps(output, indent=2),
                headers={"Content-Type": "application/json"},
            )

        if "/test/v2-trace" in url:
            # Full V2 pipeline with traced output - bypasses screener with
            # a manually-constructed spread from real chain data to ensure
            # the agents get a debatable opportunity.
            from core.ai.router import LLMRouter
            from core.broker.alpaca import AlpacaClient
            from core.db.d1 import D1Client
            from core.db.kv import KVClient
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
                TradingDecisionAgent,
                RiskState,
                build_agent_context,
            )
            from core.memory.retriever import MemoryRetriever
            from core.risk.three_perspective import ThreePerspectiveRiskManager
            from core.risk.position_sizer import PositionSizer
            from core.types import CreditSpread, OptionContract, Greeks, SpreadType
            from core.analysis.iv_rank import (
                IVMetrics,
                IVMeanReversion,
                IVTermStructure,
                TermStructurePoint,
                calculate_iv_metrics,
            )
            from core.analysis.greeks import days_to_expiry

            try:
                # Initialize clients
                db = D1Client(env.MAHLER_DB)
                kv = KVClient(env.MAHLER_KV)
                alpaca = AlpacaClient(
                    api_key=env.ALPACA_API_KEY,
                    secret_key=env.ALPACA_SECRET_KEY,
                    paper=(env.ENVIRONMENT == "paper"),
                )
                router = LLMRouter(api_key=env.ANTHROPIC_API_KEY)

                # Get live market data
                account = await alpaca.get_account()
                vix_data = await alpaca.get_vix_snapshot()
                current_vix = vix_data.get("vix", 20.0) if vix_data else 20.0

                # Get real options chain for SPY
                chain = await alpaca.get_options_chain("SPY")
                if not chain.contracts:
                    return Response(
                        json.dumps({"error": "No options data available"}),
                        status=500,
                        headers={"Content-Type": "application/json"},
                    )

                underlying_price = chain.underlying_price

                # Fetch price bars for the technical analyst (IEX feed for paper)
                try:
                    price_bars = await alpaca.get_historical_bars("SPY", timeframe="1Day", limit=60)
                except Exception:
                    price_bars = None

                # Helper: convert broker OptionContract to core.types OptionContract
                def _to_core_contract(bc):
                    greeks = None
                    if bc.delta is not None:
                        greeks = Greeks(
                            delta=bc.delta or 0.0,
                            gamma=bc.gamma or 0.0,
                            theta=bc.theta or 0.0,
                            vega=bc.vega or 0.0,
                        )
                    return OptionContract(
                        symbol=bc.symbol,
                        underlying=bc.underlying,
                        expiration=bc.expiration,
                        strike=bc.strike,
                        option_type=bc.option_type,
                        bid=bc.bid,
                        ask=bc.ask,
                        last=bc.last,
                        volume=bc.volume,
                        open_interest=bc.open_interest,
                        implied_volatility=bc.implied_volatility or 0.0,
                        greeks=greeks,
                    )

                # Strategy: Find a spread with genuinely strong economics.
                # Previous attempts failed because far-OTM spreads have poor
                # theta, making the bear's math argument unbeatable.
                # Here we go closer to the money (delta 0.25-0.40) for much
                # better theta and credit, and use wider spreads ($5+).
                # Try bear call first (aligns with elevated VIX), then bull put.
                valid_expirations = [
                    exp for exp in chain.expirations
                    if 35 <= days_to_expiry(exp) <= 75
                ]

                best_spread = None
                spread_debug = {
                    "strategy": "bear_call_strong",
                    "expirations_checked": len(valid_expirations),
                    "candidates": [],
                }

                for exp in valid_expirations[:3]:
                    calls = sorted(chain.get_calls(exp), key=lambda c: c.strike)
                    for short_call in calls:
                        if short_call.strike <= underlying_price:
                            continue
                        if short_call.bid <= 0 or short_call.ask <= 0:
                            continue
                        if short_call.open_interest < 100 or short_call.volume < 20:
                            continue
                        short_delta = abs(short_call.delta) if short_call.delta is not None else None
                        if short_delta is None:
                            continue
                        # Closer to money for better theta and credit
                        if not (0.20 <= short_delta <= 0.40):
                            continue

                        for width in [5.0, 4.0, 3.0]:
                            target_strike = short_call.strike + width
                            long_candidates = [
                                c for c in calls
                                if abs(c.strike - target_strike) < 0.5
                                and c.ask > 0
                            ]
                            if not long_candidates:
                                continue
                            long_call = long_candidates[0]

                            short_mid = (short_call.bid + short_call.ask) / 2
                            long_mid = (long_call.bid + long_call.ask) / 2
                            credit = short_mid - long_mid
                            actual_width = long_call.strike - short_call.strike
                            credit_pct = credit / actual_width if actual_width > 0 else 0

                            # Also check theta quality
                            short_theta = abs(short_call.theta) if short_call.theta else 0
                            long_theta = abs(long_call.theta) if long_call.theta else 0
                            net_theta = short_theta - long_theta
                            max_loss_per_contract = (actual_width - credit) * 100
                            theta_ratio = (net_theta * 100 / max_loss_per_contract * 100) if max_loss_per_contract > 0 else 0

                            spread_debug["candidates"].append({
                                "strikes": f"{short_call.strike}/{long_call.strike}",
                                "exp": exp,
                                "credit": round(credit, 3),
                                "credit_pct": round(credit_pct * 100, 1),
                                "short_delta": round(short_delta, 3),
                                "width": actual_width,
                                "net_theta": round(net_theta, 4),
                                "theta_ratio_pct": round(theta_ratio, 2),
                            })

                            # Accept: reasonable credit matching screener threshold
                            if credit > 0.20 and credit_pct >= 0.12:
                                best_spread = CreditSpread(
                                    underlying="SPY",
                                    spread_type=SpreadType.BEAR_CALL,
                                    short_strike=short_call.strike,
                                    long_strike=long_call.strike,
                                    expiration=exp,
                                    short_contract=_to_core_contract(short_call),
                                    long_contract=_to_core_contract(long_call),
                                )
                                break
                        if best_spread:
                            break
                    if best_spread:
                        break

                if not best_spread:
                    return Response(
                        json.dumps({
                            "error": "Could not construct a suitable bear call spread",
                            "debug": spread_debug,
                            "underlying_price": underlying_price,
                        }, indent=2),
                        status=200,
                        headers={"Content-Type": "application/json"},
                    )

                spread = best_spread

                # Compute real IV metrics from chain + DB history
                atm_contracts = [
                    c for c in chain.contracts
                    if abs(c.strike - underlying_price) < underlying_price * 0.02
                    and c.implied_volatility is not None
                ]
                current_iv = sum(c.implied_volatility or 0.0 for c in atm_contracts) / len(atm_contracts) if atm_contracts else 0.20

                historical_ivs = await db.get_iv_history("SPY", lookback_days=252)
                if len(historical_ivs) >= 30:
                    iv_metrics = calculate_iv_metrics(current_iv, historical_ivs)
                else:
                    iv_metrics = IVMetrics(
                        current_iv=current_iv,
                        iv_rank=50.0,
                        iv_percentile=50.0,
                        iv_high=current_iv,
                        iv_low=current_iv,
                    )

                # Compute real term structure from chain
                term_structure_result = None
                expirations_iv: dict[str, list[float]] = {}
                for c in chain.contracts:
                    if c.implied_volatility:
                        if abs(c.strike - underlying_price) < underlying_price * 0.05:
                            if c.expiration not in expirations_iv:
                                expirations_iv[c.expiration] = []
                            expirations_iv[c.expiration].append(c.implied_volatility)

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

                # Compute real mean reversion if enough IV history
                mean_reversion_result = None
                if len(historical_ivs) >= 60:
                    mr_analyzer = IVMeanReversion(historical_ivs)
                    mean_reversion_result = mr_analyzer.generate_signal(current_iv)

                # Get VIX3M for context
                vix_3m = vix_data.get("vix3m") if vix_data else None

                # Detect regime from cache/DB
                regime_str = None
                regime_prob = None
                latest_regime = await db.get_latest_regime("SPY")
                if latest_regime:
                    regime_str = latest_regime["regime"]
                    regime_prob = latest_regime["probability"]

                # Create orchestrator
                debate_rounds = int(env.MULTI_AGENT_DEBATE_ROUNDS or "2")
                orchestrator = AgentOrchestrator(
                    claude=router.get_client("facilitator"),
                    debate_config=DebateConfig(
                        max_rounds=debate_rounds,
                        min_rounds=1,
                        consensus_threshold=0.7,
                    ),
                )
                orchestrator.register_analyst(IVAnalyst(router=router))
                orchestrator.register_analyst(TechnicalAnalyst(router=router))
                orchestrator.register_analyst(MacroAnalyst(router=router))
                orchestrator.register_analyst(GreeksAnalyst(router=router))
                orchestrator.register_debater(BullResearcher(router=router), "bull")
                orchestrator.register_debater(BearResearcher(router=router), "bear")
                orchestrator.set_facilitator(DebateFacilitator(router=router))

                # Register Fund Manager for full pipeline
                from core.agents.fund_manager import FundManagerAgent
                orchestrator.set_fund_manager(FundManagerAgent(router=router))

                # Build context with real market data
                context = build_agent_context(
                    spread=spread,
                    underlying_price=underlying_price,
                    iv_metrics=iv_metrics,
                    term_structure=term_structure_result,
                    mean_reversion=mean_reversion_result,
                    regime=regime_str,
                    regime_probability=regime_prob,
                    current_vix=current_vix,
                    vix_3m=vix_3m,
                    price_bars=price_bars,
                    positions=[],
                    portfolio_greeks=None,
                    account_equity=account.equity,
                    buying_power=account.buying_power,
                    daily_pnl=0.0,
                    weekly_pnl=0.0,
                    playbook_rules=await db.get_playbook_rules(),
                    similar_trades=[],
                )

                # Run V2 pipeline
                result = await orchestrator.run_pipeline(context)

                # Run decision agent separately
                memory_retriever = MemoryRetriever(env.MAHLER_DB, episodic_store=None)
                sizer = PositionSizer()
                three_persp = ThreePerspectiveRiskManager(sizer)
                risk_result = three_persp.assess(
                    spread=spread,
                    account_equity=account.equity,
                    current_positions=[],
                    current_vix=current_vix,
                )

                decision_agent = TradingDecisionAgent(router=router)
                risk_state = RiskState(
                    risk_level="normal",
                    size_multiplier=1.0,
                    portfolio_heat=0.0,
                    daily_pnl=0.0,
                    weekly_pnl=0.0,
                    is_halted=False,
                )
                retrieved_context = await memory_retriever.retrieve_context(
                    underlying=spread.underlying,
                    spread_type=spread.spread_type.value,
                    market_regime=regime_str,
                    iv_rank=iv_metrics.iv_rank,
                    vix=current_vix,
                )
                decision = await decision_agent.make_decision(
                    context=context,
                    risk_state=risk_state,
                    retrieved_context=retrieved_context,
                    three_persp_manager=three_persp,
                    current_vix=current_vix,
                )

                # Build full trace output
                output = {
                    "status": "complete",
                    "spread": {
                        "underlying": spread.underlying,
                        "type": spread.spread_type.value,
                        "short_strike": spread.short_strike,
                        "long_strike": spread.long_strike,
                        "expiration": spread.expiration,
                        "width": spread.width,
                        "credit": round(spread.credit, 3),
                        "credit_pct": round(spread.credit / spread.width * 100, 1),
                        "max_profit": round(spread.max_profit, 2),
                        "max_loss": round(spread.max_loss, 2),
                    },
                    "market": {
                        "spy_price": underlying_price,
                        "vix": current_vix,
                        "vix_3m": vix_3m,
                        "iv_rank": iv_metrics.iv_rank,
                        "iv_percentile": iv_metrics.iv_percentile,
                        "current_iv": round(current_iv, 4),
                        "iv_history_days": len(historical_ivs),
                        "regime": regime_str,
                        "equity": account.equity,
                    },
                    "pipeline_trace": result.to_dict(),
                    "decision_agent": {
                        "decision": decision.decision,
                        "position_size": decision.position_size,
                        "confidence": decision.confidence,
                        "final_thesis": decision.final_thesis,
                        "key_factors": decision.key_factors,
                    },
                    "three_perspective": {
                        "aggressive": risk_result.aggressive.recommended_contracts,
                        "neutral": risk_result.neutral.recommended_contracts,
                        "conservative": risk_result.conservative.recommended_contracts,
                        "weighted": risk_result.weighted_contracts,
                        "consensus": risk_result.consensus_recommendation,
                    },
                    "spread_selection_debug": spread_debug,
                }

                return Response(
                    json.dumps(output, indent=2, default=str),
                    headers={"Content-Type": "application/json"},
                )

            except Exception as e:
                import traceback
                return Response(
                    json.dumps({
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }, indent=2),
                    status=500,
                    headers={"Content-Type": "application/json"},
                )

        if "/test/debug-scan" in url:
            # Debug scan - bypasses market hours check and provides verbose output
            from core.analysis.iv_rank import IVMetrics, calculate_iv_metrics
            from core.analysis.screener import OptionsScreener, ScreenerConfig
            from core.broker.alpaca import AlpacaClient
            from core.db.d1 import D1Client
            from core.db.kv import KVClient
            from core.risk.circuit_breaker import CircuitBreaker

            def _get_screening_debug(chain, valid_expirations, config, current_iv):
                """Get detailed screening diagnostics."""
                from core.analysis.greeks import years_to_expiry
                from core.analysis.greeks_vollib import calculate_greeks_vollib

                debug = {"expirations": {}}
                for exp in valid_expirations[:2]:  # First 2 expirations
                    puts = chain.get_puts(exp)
                    calls = chain.get_calls(exp)

                    # Filter for liquidity
                    liquid_puts = []
                    for c in puts:
                        if (c.open_interest >= config.min_open_interest and
                            c.volume >= config.min_volume and
                            c.bid > 0 and c.ask > 0):
                            mid = (c.bid + c.ask) / 2
                            spread_pct = (c.ask - c.bid) / mid if mid > 0 else 1.0
                            if spread_pct <= config.max_bid_ask_spread_pct:
                                liquid_puts.append(c)

                    # Check delta filter
                    tte = years_to_expiry(exp)
                    puts_in_delta_range = 0
                    sample_deltas = []
                    for p in liquid_puts[:20]:
                        try:
                            if p.delta is not None:
                                delta = p.delta
                            else:
                                iv = p.implied_volatility or current_iv
                                greeks = calculate_greeks_vollib(
                                    spot=chain.underlying_price,
                                    strike=p.strike,
                                    time_to_expiry=tte,
                                    volatility=iv,
                                    option_type="put",
                                )
                                delta = greeks.delta
                            if config.min_delta <= abs(delta) <= config.max_delta:
                                puts_in_delta_range += 1
                            sample_deltas.append({
                                "strike": p.strike,
                                "delta": round(delta, 4),
                                "in_range": config.min_delta <= abs(delta) <= config.max_delta,
                            })
                        except Exception:
                            pass

                    debug["expirations"][exp] = {
                        "total_puts": len(puts),
                        "liquid_puts": len(liquid_puts),
                        "puts_in_delta_range": puts_in_delta_range,
                        "total_calls": len(calls),
                        "sample_put_deltas": sample_deltas[:5],
                    }
                return debug

            debug_output = {"status": "ok", "checks": {}, "scan_results": {}}

            # Initialize clients
            db = D1Client(env.MAHLER_DB)
            kv = KVClient(env.MAHLER_KV)
            circuit_breaker = CircuitBreaker(kv)
            alpaca = AlpacaClient(
                api_key=env.ALPACA_API_KEY,
                secret_key=env.ALPACA_SECRET_KEY,
                paper=(env.ENVIRONMENT == "paper"),
            )

            # Check circuit breaker
            cb_status = await circuit_breaker.get_status()
            debug_output["checks"]["circuit_breaker"] = {
                "halted": cb_status.halted,
                "reason": cb_status.reason,
            }

            # Check market status (but don't exit)
            market_open = await alpaca.is_market_open()
            debug_output["checks"]["market_open"] = market_open

            # Get account info
            account = await alpaca.get_account()
            debug_output["checks"]["account"] = {
                "equity": account.equity,
                "buying_power": account.buying_power,
            }

            # Get VIX
            try:
                vix_data = await alpaca.get_vix_snapshot()
                debug_output["checks"]["vix"] = vix_data
            except Exception as e:
                debug_output["checks"]["vix"] = {"error": str(e)}

            # Check IV history counts
            iv_history_counts = {}
            underlyings = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
            for symbol in underlyings:
                count = await db.get_iv_history_count(symbol)
                iv_history_counts[symbol] = count
            debug_output["checks"]["iv_history_counts"] = iv_history_counts

            # Scan each underlying (bypass market hours)
            screener = OptionsScreener(ScreenerConfig())
            scan_results = {}

            for symbol in underlyings:
                try:
                    chain = await alpaca.get_options_chain(symbol)
                    if not chain.contracts:
                        scan_results[symbol] = {"error": "No options data"}
                        continue

                    # Get current IV from ATM options
                    atm_contracts = [
                        c for c in chain.contracts
                        if abs(c.strike - chain.underlying_price) < chain.underlying_price * 0.02
                        and c.implied_volatility is not None
                    ]

                    if not atm_contracts:
                        scan_results[symbol] = {"error": "No ATM contracts with IV data"}
                        continue

                    current_iv = sum(c.implied_volatility for c in atm_contracts) / len(atm_contracts)

                    # Get IV history
                    historical_ivs = await db.get_iv_history(symbol, lookback_days=252)

                    # Calculate IV metrics
                    if len(historical_ivs) >= 30:
                        iv_metrics = calculate_iv_metrics(current_iv, historical_ivs)
                    else:
                        # Insufficient history - use neutral rank (debug only)
                        iv_metrics = IVMetrics(
                            current_iv=current_iv,
                            iv_rank=50.0,  # Neutral rank
                            iv_percentile=50.0,
                            iv_high=current_iv,
                            iv_low=current_iv,
                        )

                    # Screen for opportunities
                    opportunities = screener.screen_chain(chain, iv_metrics)

                    # Diagnostic: Count contracts passing each filter
                    from core.analysis.greeks import days_to_expiry
                    config = screener.config

                    # Filter diagnostics
                    valid_expirations = [
                        exp for exp in chain.expirations
                        if config.min_dte <= days_to_expiry(exp) <= config.max_dte
                    ]

                    # Count liquidity-passing contracts
                    liquidity_passing = 0
                    has_iv = 0
                    for c in chain.contracts:
                        if c.implied_volatility:
                            has_iv += 1
                        if (c.open_interest >= config.min_open_interest and
                            c.volume >= config.min_volume and
                            c.bid > 0 and c.ask > 0):
                            mid = (c.bid + c.ask) / 2
                            spread_pct = (c.ask - c.bid) / mid if mid > 0 else 1.0
                            if spread_pct <= config.max_bid_ask_spread_pct:
                                liquidity_passing += 1

                    scan_results[symbol] = {
                        "underlying_price": chain.underlying_price,
                        "contracts_count": len(chain.contracts),
                        "current_iv": round(current_iv * 100, 2),
                        "iv_rank": round(iv_metrics.iv_rank, 2),
                        "iv_percentile": round(iv_metrics.iv_percentile, 2),
                        "iv_history_days": len(historical_ivs),
                        "opportunities_found": len(opportunities),
                        "top_opportunities": [
                            {
                                "spread_type": opp.spread.spread_type.value,
                                "strikes": f"{opp.spread.short_strike}/{opp.spread.long_strike}",
                                "expiration": opp.spread.expiration,
                                "credit": round(opp.spread.credit, 2),
                                "score": round(opp.score, 2),
                            }
                            for opp in opportunities[:3]
                        ],
                        "filter_passed": iv_metrics.iv_percentile >= 50.0,
                        "diagnostics": {
                            "valid_expirations": len(valid_expirations),
                            "expirations_list": valid_expirations[:5],
                            "contracts_with_iv": has_iv,
                            "contracts_passing_liquidity": liquidity_passing,
                            "sample_atm_contract": {
                                "strike": atm_contracts[0].strike if atm_contracts else None,
                                "bid": atm_contracts[0].bid if atm_contracts else None,
                                "ask": atm_contracts[0].ask if atm_contracts else None,
                                "volume": atm_contracts[0].volume if atm_contracts else None,
                                "open_interest": atm_contracts[0].open_interest if atm_contracts else None,
                                "iv": atm_contracts[0].implied_volatility if atm_contracts else None,
                            } if atm_contracts else None,
                            "screening_debug": _get_screening_debug(chain, valid_expirations, config, iv_metrics.current_iv),
                        },
                    }

                except Exception as e:
                    scan_results[symbol] = {"error": str(e)}

            debug_output["scan_results"] = scan_results

            return Response(
                json.dumps(debug_output, indent=2),
                headers={"Content-Type": "application/json"},
            )

        if "/test/db" in url:
            from core.db.d1 import D1Client

            db = D1Client(env.MAHLER_DB)
            rules = await db.get_playbook_rules()
            return Response(
                json.dumps(
                    {
                        "status": "ok",
                        "playbook_rules_count": len(rules),
                        "sample_rules": [r.rule for r in rules[:3]],
                    }
                ),
                headers={"Content-Type": "application/json"},
            )

        if "/test/discord" in url:
            from core.notifications.discord import DiscordClient

            discord = DiscordClient(
                bot_token=env.DISCORD_BOT_TOKEN,
                public_key=env.DISCORD_PUBLIC_KEY,
                channel_id=env.DISCORD_CHANNEL_ID,
            )
            message_id = await discord.send_message(
                content="Mahler test message - if you see this, Discord integration is working!",
                embeds=[
                    {
                        "title": "System Test",
                        "description": "All systems operational",
                        "color": 0x00FF00,
                        "fields": [
                            {"name": "Environment", "value": env.ENVIRONMENT, "inline": True},
                            {
                                "name": "Timestamp",
                                "value": datetime.now().isoformat(),
                                "inline": True,
                            },
                        ],
                    }
                ],
            )
            return Response(
                json.dumps(
                    {
                        "status": "ok",
                        "message_id": message_id,
                        "channel_id": env.DISCORD_CHANNEL_ID,
                    }
                ),
                headers={"Content-Type": "application/json"},
            )

        # Admin endpoints for kill switch
        if "/admin/halt" in url and method == "POST":
            from core.db.kv import KVClient
            from core.risk.circuit_breaker import CircuitBreaker

            kv = KVClient(env.MAHLER_KV)
            circuit_breaker = CircuitBreaker(kv)
            await circuit_breaker.trip("Kill switch activated via admin endpoint")
            return Response(
                json.dumps({"status": "halted", "message": "Trading halted via admin endpoint"}),
                headers={"Content-Type": "application/json"},
            )

        if "/admin/resume" in url and method == "POST":
            from core.db.kv import KVClient
            from core.risk.circuit_breaker import CircuitBreaker

            kv = KVClient(env.MAHLER_KV)
            circuit_breaker = CircuitBreaker(kv)
            await circuit_breaker.reset()
            return Response(
                json.dumps({"status": "resumed", "message": "Trading resumed via admin endpoint"}),
                headers={"Content-Type": "application/json"},
            )

        if "/admin/status" in url:
            from core.db.kv import KVClient
            from core.risk.circuit_breaker import CircuitBreaker

            kv = KVClient(env.MAHLER_KV)
            circuit_breaker = CircuitBreaker(kv)
            status = await circuit_breaker.get_status()
            return Response(
                json.dumps({
                    "halted": status.halted,
                    "reason": status.reason,
                    "triggered_at": status.triggered_at.isoformat() if status.triggered_at else None,
                }),
                headers={"Content-Type": "application/json"},
            )

        if "/admin/seed-iv" in url:
            import math

            from core.broker.alpaca import AlpacaClient
            from core.db.d1 import D1Client

            db = D1Client(env.MAHLER_DB)
            alpaca = AlpacaClient(
                api_key=env.ALPACA_API_KEY,
                secret_key=env.ALPACA_SECRET_KEY,
                paper=(env.ENVIRONMENT == "paper"),
            )

            underlyings = ["SPY", "QQQ", "IWM"]
            total_seeded = 0
            results = {}

            for symbol in underlyings:
                try:
                    bars = await alpaca.get_historical_bars(symbol, timeframe="1Day", limit=90)
                    if len(bars) < 25:
                        results[symbol] = {"error": f"Insufficient bars ({len(bars)})"}
                        continue

                    # Compute 20-day rolling realized vol (stdev of log returns, annualized)
                    prices = [b["close"] for b in bars]
                    prices.reverse()  # oldest first

                    log_returns = []
                    for i in range(1, len(prices)):
                        log_returns.append(math.log(prices[i] / prices[i - 1]))

                    records = []
                    window = 20
                    dates = [b["timestamp"] for b in bars]
                    dates.reverse()  # oldest first

                    for i in range(window, len(log_returns)):
                        window_returns = log_returns[i - window:i]
                        mean_ret = sum(window_returns) / len(window_returns)
                        variance = sum((r - mean_ret) ** 2 for r in window_returns) / (len(window_returns) - 1)
                        stdev = math.sqrt(variance)
                        realized_vol = stdev * math.sqrt(252)

                        # dates[i+1] corresponds to the price at index i+1
                        # (log_returns[i] uses prices[i] and prices[i+1])
                        date_str = dates[i + 1] if (i + 1) < len(dates) else dates[-1]
                        # Normalize date format
                        if "T" in date_str:
                            date_str = date_str.split("T")[0]

                        records.append({
                            "date": date_str,
                            "underlying": symbol,
                            "atm_iv": round(realized_vol, 6),
                            "underlying_price": prices[i + 1] if (i + 1) < len(prices) else prices[-1],
                        })

                    count = await db.save_daily_iv_batch(records)
                    total_seeded += count
                    results[symbol] = {
                        "bars_fetched": len(bars),
                        "records_seeded": count,
                        "date_range": f"{records[0]['date']} to {records[-1]['date']}" if records else "none",
                    }

                except Exception as e:
                    results[symbol] = {"error": str(e)}

            return Response(
                json.dumps({
                    "status": "ok",
                    "total_records_seeded": total_seeded,
                    "results": results,
                }, indent=2),
                headers={"Content-Type": "application/json"},
            )

        # Default response
        return Response(
            '{"status": "ok", "service": "mahler"}',
            headers={"Content-Type": "application/json"},
        )

    except Exception as e:
        import traceback

        print(f"Error handling request: {e}")
        print(traceback.format_exc())
        return Response(
            json.dumps({"error": str(e)}),
            status=500,
            headers={"Content-Type": "application/json"},
        )


async def on_scheduled(event, env, ctx):
    """Handle cron triggers."""
    cron = event.cron

    try:
        print(f"Cron triggered: {cron} at {datetime.now().isoformat()}")

        # Route based on cron pattern (using lazy imports to avoid startup CPU limit)
        if cron == "0 15 * * MON-FRI":  # 10:00 AM ET (moved from 9:35 AM to avoid stale quotes)
            handler = _import_handler("morning_scan")
            await handler(env)
        elif cron == "0 17 * * MON-FRI":
            handler = _import_handler("midday_check")
            await handler(env)
        elif cron == "30 20 * * MON-FRI":
            handler = _import_handler("afternoon_scan")
            await handler(env)
        elif cron == "15 21 * * MON-FRI":
            handler = _import_handler("eod_summary")
            await handler(env)
        elif "*/5" in cron:
            handler = _import_handler("position_monitor")
            await handler(env)
        else:
            print(f"Unknown cron pattern: {cron}")

    except Exception as e:
        print(f"Error in scheduled handler: {e}")
        raise


# Cloudflare Workers entry points
def fetch(request, env):
    """Fetch handler for HTTP requests."""
    return on_fetch(request, env)


def scheduled(event, env, ctx):
    """Scheduled handler for cron triggers."""
    ctx.wait_until(on_scheduled(event, env, ctx))
