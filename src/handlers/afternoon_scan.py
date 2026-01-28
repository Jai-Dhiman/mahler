"""Afternoon scan worker - runs at 3:30 PM ET.

Final scan before market close. More conservative - only very high
probability setups with excellent IV conditions.
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
)

UNDERLYINGS = ["SPY", "QQQ", "IWM"]


def _create_orchestrator(claude, debate_rounds: int = 2) -> AgentOrchestrator:
    """Create a configured agent orchestrator."""
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
    return orchestrator


def _map_result_to_confidence(result) -> Confidence:
    """Map pipeline result confidence to Confidence enum."""
    if result.confidence >= 0.7:
        return Confidence.HIGH
    elif result.confidence >= 0.4:
        return Confidence.MEDIUM
    else:
        return Confidence.LOW


async def handle_afternoon_scan(env):
    """Run afternoon scan with stricter criteria."""
    print("Starting afternoon scan...")

    # Signal start to heartbeat monitor
    heartbeat_url = getattr(env, "HEARTBEAT_URL", None)
    await http.ping_heartbeat_start(heartbeat_url, "afternoon_scan")

    job_success = False
    try:
        await _run_afternoon_scan(env)
        job_success = True
    finally:
        await http.ping_heartbeat(heartbeat_url, "afternoon_scan", success=job_success)


async def _run_afternoon_scan(env):
    """Internal afternoon scan logic."""

    # Initialize clients
    db = D1Client(env.MAHLER_DB)
    kv = KVClient(env.MAHLER_KV)
    circuit_breaker = CircuitBreaker(kv)

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

    # V2 Multi-Agent System
    debate_rounds = int(getattr(env, "MULTI_AGENT_DEBATE_ROUNDS", "2"))
    orchestrator = _create_orchestrator(claude, debate_rounds=debate_rounds)

    if not await alpaca.is_market_open():
        print("Market is closed, skipping afternoon scan")
        return

    account = await alpaca.get_account()
    positions = await db.get_all_positions()

    sizer = PositionSizer()
    heat = sizer.calculate_portfolio_heat(positions, account.equity)

    if heat["at_limit"]:
        print("Portfolio heat at limit, skipping scan")
        return

    # More conservative config for afternoon
    config = ScreenerConfig(
        min_dte=35,  # Slightly longer DTE
        max_dte=45,
        min_delta=0.15,  # More OTM
        max_delta=0.25,
        min_iv_rank=60.0,  # Higher IV requirement
        min_credit_pct=0.30,  # Better premium requirement
    )

    screener = OptionsScreener(config)
    playbook_rules = await db.get_playbook_rules()

    best_opportunity = None
    best_score = 0

    for symbol in UNDERLYINGS:
        try:
            chain = await alpaca.get_options_chain(symbol)
            if not chain.contracts:
                continue

            atm = [
                c
                for c in chain.contracts
                if abs(c.strike - chain.underlying_price) < chain.underlying_price * 0.02
            ]
            current_iv = atm[0].implied_volatility if atm and atm[0].implied_volatility else 0.20
            iv_metrics = calculate_iv_metrics(current_iv, [current_iv * 0.8, current_iv * 1.2])

            # Stricter IV requirement for afternoon
            if iv_metrics.iv_rank < 60:
                continue

            opportunities = screener.screen_chain(chain, iv_metrics)
            if opportunities and opportunities[0].score > best_score:
                best_score = opportunities[0].score
                best_opportunity = (opportunities[0], chain.underlying_price, iv_metrics)

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    # Only send if we found a high-quality setup
    if best_opportunity and best_score > 0.6:  # Higher score threshold
        opp, underlying_price, iv_metrics = best_opportunity
        spread = opp.spread

        size_result = sizer.calculate_size(
            spread=spread,
            account_equity=account.equity,
            current_positions=positions,
        )

        if size_result.contracts > 0:
            try:
                # Build agent context for V2 analysis
                context = build_agent_context(
                    spread=spread,
                    underlying_price=underlying_price,
                    iv_metrics=iv_metrics,
                    term_structure=None,
                    mean_reversion=None,
                    regime=None,
                    regime_probability=None,
                    current_vix=None,
                    vix_3m=None,
                    price_bars=None,
                    positions=positions,
                    portfolio_greeks=None,
                    account_equity=account.equity,
                    buying_power=account.buying_power,
                    daily_pnl=0,
                    weekly_pnl=0,
                    playbook_rules=playbook_rules,
                    similar_trades=[],
                    scan_type="afternoon",
                )

                # Run V2 multi-agent analysis
                result = await orchestrator.run_pipeline(context)
                confidence = _map_result_to_confidence(result)

                # Only high confidence for afternoon, and not a skip recommendation
                if confidence == Confidence.HIGH and result.recommendation != "skip":
                    rec_id = await db.create_recommendation(
                        underlying=spread.underlying,
                        spread_type=spread.spread_type,
                        short_strike=spread.short_strike,
                        long_strike=spread.long_strike,
                        expiration=spread.expiration,
                        credit=spread.credit,
                        max_loss=spread.max_loss,
                        expires_at=datetime.now() + timedelta(minutes=10),  # Shorter expiry
                        iv_rank=iv_metrics.iv_rank,
                        delta=(spread.short_contract.greeks.delta
                              if spread.short_contract and spread.short_contract.greeks
                              else None),
                        theta=(spread.short_contract.greeks.theta
                              if spread.short_contract and spread.short_contract.greeks
                              else None),
                        thesis=result.thesis,
                        confidence=confidence,
                        suggested_contracts=size_result.contracts,
                        analysis_price=spread.credit,
                    )

                    rec = await db.get_recommendation(rec_id)
                    message_id = await discord.send_recommendation(rec)
                    await db.set_recommendation_discord_message_id(rec_id, message_id)

                    print(f"Sent afternoon recommendation: {rec_id}")

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

    print("Afternoon scan complete.")
