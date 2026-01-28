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

        if "/test/debug-scan" in url:
            # Debug scan - bypasses market hours check and provides verbose output
            from core.analysis.iv_rank import IVMetrics, calculate_iv_metrics
            from core.analysis.screener import OptionsScreener, ScreenerConfig
            from core.broker.alpaca import AlpacaClient
            from core.db.d1 import D1Client
            from core.db.kv import KVClient
            from core.risk.circuit_breaker import CircuitBreaker

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
                    ]
                    current_iv = atm_contracts[0].implied_volatility if atm_contracts and atm_contracts[0].implied_volatility else 0.20

                    # Get IV history
                    historical_ivs = await db.get_iv_history(symbol, lookback_days=252)

                    # Calculate IV metrics
                    if len(historical_ivs) >= 30:
                        iv_metrics = calculate_iv_metrics(current_iv, historical_ivs)
                    else:
                        # Fallback estimation
                        current_vix = debug_output["checks"].get("vix", {}).get("vix", 20)
                        estimated_rank = min(90.0, max(50.0, current_vix * 2.5)) if current_vix else 70.0
                        iv_metrics = IVMetrics(
                            current_iv=current_iv,
                            iv_rank=estimated_rank,
                            iv_percentile=estimated_rank,
                            iv_high=current_iv * 1.2,
                            iv_low=current_iv * 0.7,
                        )

                    # Screen for opportunities
                    opportunities = screener.screen_chain(chain, iv_metrics)

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
