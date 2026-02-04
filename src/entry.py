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
            from core.ai.claude import ClaudeClient
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
                claude = ClaudeClient(api_key=env.ANTHROPIC_API_KEY)

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
                    claude=claude,
                    debate_config=DebateConfig(
                        max_rounds=debate_rounds,
                        min_rounds=1,
                        consensus_threshold=0.7,
                    ),
                )
                orchestrator.register_analyst(IVAnalyst(claude))
                orchestrator.register_analyst(TechnicalAnalyst(claude))
                orchestrator.register_analyst(MacroAnalyst(claude))
                orchestrator.register_analyst(GreeksAnalyst(claude))
                orchestrator.register_debater(BullResearcher(claude), "bull")
                orchestrator.register_debater(BearResearcher(claude), "bear")
                orchestrator.set_facilitator(DebateFacilitator(claude))

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
                decision_agent = TradingDecisionAgent(claude)
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
