"""Midday check worker - runs at 12:00 PM ET.

Lighter version of morning scan - checks positions and
looks for new setups if capacity available.
"""

from datetime import datetime, timedelta

from core import http
from core.ai.claude import ClaudeClient, ClaudeRateLimitError
from core.analysis.iv_rank import calculate_iv_metrics
from core.analysis.screener import OptionsScreener, ScreenerConfig
from core.broker.alpaca import AlpacaClient
from core.db.d1 import D1Client
from core.db.kv import KVClient
from core.notifications.discord import DiscordClient
from core.broker.types import SpreadOrder
from core.risk.circuit_breaker import CircuitBreaker
from core.risk.position_sizer import PositionSizer
from core.types import Confidence, RecommendationStatus, SpreadType, TradeStatus

UNDERLYINGS = ["SPY", "QQQ", "IWM"]
MAX_RECOMMENDATIONS = 2  # Fewer than morning scan


async def handle_midday_check(env):
    """Run midday position check and opportunity scan."""
    print("Starting midday check...")

    # Signal start to heartbeat monitor
    heartbeat_url = getattr(env, "HEARTBEAT_URL", None)
    await http.ping_heartbeat_start(heartbeat_url, "midday_check")

    job_success = False
    try:
        await _run_midday_check(env)
        job_success = True
    finally:
        await http.ping_heartbeat(heartbeat_url, "midday_check", success=job_success)


async def _run_midday_check(env):
    """Internal midday check logic."""

    # Initialize clients
    db = D1Client(env.MAHLER_DB)
    kv = KVClient(env.MAHLER_KV)
    circuit_breaker = CircuitBreaker(kv)

    # Check circuit breaker
    if not await circuit_breaker.is_trading_allowed():
        status = await circuit_breaker.check_status()
        print(f"Trading halted: {status.reason}")
        return

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
    if not await alpaca.is_market_open():
        print("Market is closed, skipping midday check")
        return

    # Get account and position info
    account = await alpaca.get_account()
    positions = await db.get_all_positions()
    open_trades = await db.get_open_trades()

    sizer = PositionSizer()
    heat = sizer.calculate_portfolio_heat(positions, account.equity)

    # If at heat limit, skip scanning for new opportunities
    if heat["at_limit"]:
        print(f"Portfolio heat at limit ({heat['heat_percent']:.1%}), skipping scan")
        return

    # Check for pending recommendations that haven't been acted on
    pending = await db.get_pending_recommendations()
    if pending:
        print(f"Found {len(pending)} pending recommendations, skipping new scan")
        return

    # Run lighter scan (only if capacity available)
    playbook_rules = await db.get_playbook_rules()
    screener = OptionsScreener(ScreenerConfig())

    all_opportunities = []

    for symbol in UNDERLYINGS:
        try:
            chain = await alpaca.get_options_chain(symbol)
            if not chain.contracts:
                continue

            # Quick IV estimate
            atm = [
                c
                for c in chain.contracts
                if abs(c.strike - chain.underlying_price) < chain.underlying_price * 0.02
            ]
            current_iv = atm[0].implied_volatility if atm and atm[0].implied_volatility else 0.20
            iv_metrics = calculate_iv_metrics(current_iv, [current_iv * 0.8, current_iv * 1.2])

            # Only proceed if IV is elevated
            if iv_metrics.iv_rank < 50:
                continue

            opportunities = screener.screen_chain(chain, iv_metrics)
            for opp in opportunities[:1]:  # Just top 1 per symbol at midday
                all_opportunities.append((opp, chain.underlying_price, iv_metrics))

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    # Process best opportunity only
    if all_opportunities:
        all_opportunities.sort(key=lambda x: x[0].score, reverse=True)
        opp, underlying_price, iv_metrics = all_opportunities[0]
        spread = opp.spread

        size_result = sizer.calculate_size(
            spread=spread,
            account_equity=account.equity,
            current_positions=positions,
        )

        if size_result.contracts > 0:
            try:
                analysis = await claude.analyze_trade(
                    spread=spread,
                    underlying_price=underlying_price,
                    iv_rank=iv_metrics.iv_rank,
                    current_iv=iv_metrics.current_iv,
                    playbook_rules=playbook_rules,
                )

                if analysis.confidence != Confidence.LOW:
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
                        delta=(spread.short_contract.greeks.delta
                              if spread.short_contract and spread.short_contract.greeks
                              else None),
                        theta=(spread.short_contract.greeks.theta
                              if spread.short_contract and spread.short_contract.greeks
                              else None),
                        thesis=analysis.thesis,
                        confidence=analysis.confidence,
                        suggested_contracts=size_result.contracts,
                        analysis_price=spread.credit,
                    )

                    rec = await db.get_recommendation(rec_id)
                    message_id = await discord.send_recommendation(rec)
                    await db.set_recommendation_discord_message_id(rec_id, message_id)

                    print(f"Sent midday recommendation: {rec_id}")

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
                            spread_order = SpreadOrder(
                                underlying=spread.underlying,
                                short_symbol=short_symbol,
                                long_symbol=long_symbol,
                                contracts=size_result.contracts,
                                limit_price=spread.credit,
                            )
                            order = await alpaca.place_spread_order(spread_order)

                            # Update recommendation status
                            await db.update_recommendation_status(rec_id, RecommendationStatus.APPROVED)

                            # Create trade record with pending_fill status
                            trade_id = await db.create_trade(
                                recommendation_id=rec_id,
                                underlying=spread.underlying,
                                spread_type=spread.spread_type,
                                short_strike=spread.short_strike,
                                long_strike=spread.long_strike,
                                expiration=spread.expiration,
                                entry_credit=spread.credit,
                                contracts=size_result.contracts,
                                broker_order_id=order.id,
                                status=TradeStatus.PENDING_FILL,
                            )

                            # Update Discord message to show order placed
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
                                        {"name": "Contracts", "value": str(size_result.contracts), "inline": True},
                                        {"name": "Order ID", "value": order.id, "inline": True},
                                    ],
                                }],
                                components=[],  # Remove buttons
                            )

                            print(f"Order placed (pending fill): {trade_id}, Order: {order.id}")

                        except Exception as e:
                            print(f"Error auto-approving trade: {e}")

            except ClaudeRateLimitError as e:
                print(f"Claude API rate limit error: {e}")
                await discord.send_api_token_alert("Claude", str(e))
            except Exception as e:
                print(f"Error processing opportunity: {e}")

    print("Midday check complete.")
