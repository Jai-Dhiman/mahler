"""Position monitor worker - runs every 5 minutes during market hours.

Monitors open positions for exit conditions:
- 50% profit target
- 200% stop loss (configurable based on win rate)
- 21 DTE time exit
"""

from datetime import datetime

from core import http
from core.broker.alpaca import AlpacaClient
from core.db.d1 import D1Client
from core.db.kv import KVClient
from core.notifications.discord import DiscordClient
from core.risk.circuit_breaker import CircuitBreaker, RiskLevel
from core.risk.validators import ExitConfig, ExitValidator


async def handle_position_monitor(env):
    """Monitor positions for exit conditions."""
    print("Starting position monitor...")

    # Signal start to heartbeat monitor
    heartbeat_url = getattr(env, "HEARTBEAT_URL", None)
    await http.ping_heartbeat_start(heartbeat_url, "position_monitor")

    job_success = False
    try:
        await _run_position_monitor(env)
        job_success = True
    finally:
        await http.ping_heartbeat(heartbeat_url, "position_monitor", success=job_success)


async def _run_position_monitor(env):
    """Internal position monitor logic."""

    # Initialize clients
    db = D1Client(env.MAHLER_DB)
    kv = KVClient(env.MAHLER_KV)
    circuit_breaker = CircuitBreaker(kv)

    # Check circuit breaker
    if not await circuit_breaker.is_trading_allowed():
        status = await circuit_breaker.check_status()
        print(f"Trading halted: {status.reason}")
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

    # Check if market is open
    if not await alpaca.is_market_open():
        print("Market is closed, skipping monitor")
        return

    # Get open trades from database
    open_trades = await db.get_open_trades()

    if not open_trades:
        print("No open trades to monitor")
        return

    print(f"Monitoring {len(open_trades)} open trades...")

    exit_validator = ExitValidator()

    # Get account info for circuit breaker checks
    account = await alpaca.get_account()

    # Run graduated circuit breaker checks
    daily_stats = await kv.get_daily_stats()
    daily_starting_equity = daily_stats.get("starting_equity", account.equity)

    weekly_starting_equity = await kv.get_weekly_starting_equity()
    if weekly_starting_equity == 0:
        weekly_starting_equity = account.equity  # Fallback if not initialized

    risk_state = await circuit_breaker.evaluate_all(
        starting_daily_equity=daily_starting_equity,
        starting_weekly_equity=weekly_starting_equity,
        peak_equity=max(daily_starting_equity, weekly_starting_equity),
        current_equity=account.equity,
    )

    # Log risk state
    if risk_state.level != RiskLevel.NORMAL:
        print(f"Risk level: {risk_state.level.value}, size multiplier: {risk_state.size_multiplier}")
        if risk_state.reason:
            print(f"Reason: {risk_state.reason}")

    # Send alert if needed
    if risk_state.should_alert and risk_state.reason:
        await discord.send_circuit_breaker_alert(risk_state.reason)

    # If halted, stop processing
    if risk_state.level == RiskLevel.HALTED:
        print(f"Trading halted: {risk_state.reason}")
        return

    # Process each open trade
    for trade in open_trades:
        try:
            # Get current prices
            chain = await alpaca.get_options_chain(trade.underlying)

            # Find our contracts
            exp_parts = trade.expiration.split("-")
            exp_str = exp_parts[0][2:] + exp_parts[1] + exp_parts[2]
            option_type = "P" if trade.spread_type.value == "bull_put" else "C"

            short_symbol = (
                f"{trade.underlying}{exp_str}{option_type}{int(trade.short_strike * 1000):08d}"
            )
            long_symbol = (
                f"{trade.underlying}{exp_str}{option_type}{int(trade.long_strike * 1000):08d}"
            )

            short_contract = next((c for c in chain.contracts if c.symbol == short_symbol), None)
            long_contract = next((c for c in chain.contracts if c.symbol == long_symbol), None)

            if not short_contract or not long_contract:
                print(f"Could not find contracts for trade {trade.id}")
                continue

            # Calculate current value (cost to close)
            # To close: buy back short, sell long
            current_value = short_contract.mid - long_contract.mid
            unrealized_pnl = (trade.entry_credit - current_value) * trade.contracts * 100

            # Update position in database
            await db.upsert_position(
                trade_id=trade.id,
                underlying=trade.underlying,
                short_strike=trade.short_strike,
                long_strike=trade.long_strike,
                expiration=trade.expiration,
                contracts=trade.contracts,
                current_value=current_value,
                unrealized_pnl=unrealized_pnl,
            )

            # Check exit conditions
            should_exit, exit_reason = exit_validator.check_all_exit_conditions(
                entry_credit=trade.entry_credit,
                current_value=current_value,
                expiration=trade.expiration,
            )

            if should_exit:
                print(f"Exit triggered for {trade.underlying}: {exit_reason}")

                # Send exit alert to Discord
                await discord.send_exit_alert(
                    trade=trade,
                    reason=exit_reason,
                    current_value=current_value,
                    unrealized_pnl=unrealized_pnl,
                )

        except Exception as e:
            print(f"Error monitoring trade {trade.id}: {e}")
            await circuit_breaker.check_api_errors()

    print("Position monitor complete.")
