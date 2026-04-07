"""Midday check worker - runs at 12:00 PM ET.

Lighter version of morning scan - checks positions and
looks for new setups if capacity available.
"""

from datetime import datetime, timedelta

from core import http
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
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        print(f"Midday check failed: {error_msg}\n{tb}")
        try:
            from core.notifications.discord import DiscordClient
            discord = DiscordClient(
                bot_token=env.DISCORD_BOT_TOKEN,
                public_key=env.DISCORD_PUBLIC_KEY,
                channel_id=env.DISCORD_CHANNEL_ID,
            )
            await discord.send_message(
                content="**Midday Check FAILED**",
                embeds=[{
                    "title": "Midday Error",
                    "color": 0xED4245,
                    "description": f"Midday check crashed.\n\n**Error:** `{error_msg}`",
                    "fields": [
                        {"name": "Traceback (last 500 chars)", "value": f"```\n{tb[-500:]}\n```"},
                    ],
                }],
            )
        except Exception as notify_err:
            print(f"Failed to send error notification: {notify_err}")
        raise
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
                rec_thesis = f"Algorithmic midday trade: {spread.underlying} {spread.spread_type.value}"
                rec_confidence = Confidence.MEDIUM

                if rec_thesis is not None:
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
                        thesis=rec_thesis,
                        confidence=rec_confidence,
                        suggested_contracts=size_result.contracts,
                        analysis_price=spread.credit,
                    )

                    rec = await db.get_recommendation(rec_id)

                    # Build OCC symbols
                    exp_parts = spread.expiration.split("-")
                    exp_str = exp_parts[0][2:] + exp_parts[1] + exp_parts[2]
                    option_type = "P" if spread.spread_type == SpreadType.BULL_PUT else "C"
                    short_symbol = f"{spread.underlying}{exp_str}{option_type}{int(spread.short_strike * 1000):08d}"
                    long_symbol = f"{spread.underlying}{exp_str}{option_type}{int(spread.long_strike * 1000):08d}"

                    spread_order = SpreadOrder(
                        underlying=spread.underlying,
                        short_symbol=short_symbol,
                        long_symbol=long_symbol,
                        contracts=size_result.contracts,
                        limit_price=spread.credit,
                    )

                    # Place order at broker
                    order = None
                    try:
                        order = await alpaca.place_spread_order(spread_order)
                        print(f"Midday order placed at broker: {order.id}")
                    except Exception as e:
                        print(f"Error placing midday order at broker: {e}")

                    if order is not None:
                        # Record in DB (order already placed -- must not lose track)
                        try:
                            await db.update_recommendation_status(rec_id, RecommendationStatus.APPROVED)
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
                            print(f"Midday trade recorded: {trade_id}, Order: {order.id}")

                            try:
                                await discord.send_autonomous_notification(
                                    rec=rec,
                                    v2_confidence=rec_confidence.value if hasattr(rec_confidence, 'value') else 0.5,
                                    v2_thesis=rec_thesis,
                                    order_id=order.id,
                                )
                            except Exception as e:
                                print(f"Error sending midday notification: {e}")

                        except Exception as e:
                            import traceback
                            error_msg = f"{type(e).__name__}: {e}"
                            print(f"CRITICAL: DB write failed after midday order! Order: {order.id}, Error: {error_msg}")
                            print(traceback.format_exc())
                            try:
                                await alpaca.cancel_order(order.id)
                                print(f"Cancelled orphaned midday order: {order.id}")
                            except Exception as cancel_err:
                                print(f"FAILED to cancel orphaned order {order.id}: {cancel_err}")
                            await discord.send_message(
                                content="**GHOST TRADE ALERT**",
                                embeds=[{
                                    "title": "Midday Order Placed But DB Write Failed",
                                    "color": 0xED4245,
                                    "description": f"**Order ID:** `{order.id}`\n**Underlying:** {spread.underlying}\n**Error:** `{error_msg}`",
                                }],
                            )

            except Exception as e:
                print(f"Error processing opportunity: {e}")

    print("Midday check complete.")
