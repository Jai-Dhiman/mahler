"""Position monitor worker - runs every 5 minutes during market hours.

Monitors open positions for exit conditions:
- IV-adjusted profit target (scales with IV rank)
- Gamma protection (exit at 70% profit when DTE <= 21)
- Gamma explosion (force exit when DTE <= 7)
- 200% stop loss (configurable based on win rate)
"""

from datetime import UTC, datetime

from core import http
from core.analysis.greeks import days_to_expiry
from core.broker.alpaca import AlpacaClient
from core.broker.types import OrderStatus
from core.db.d1 import D1Client
from core.db.kv import KVClient
from core.notifications.discord import DiscordClient
from core.risk.circuit_breaker import CircuitBreaker, RiskLevel
from core.risk.validators import ExitValidator
from core.types import TradeStatus

# Circuit breaker alert deduplication window (1 hour)
CIRCUIT_BREAKER_ALERT_DEDUP_SECONDS = 3600


async def _should_send_circuit_breaker_alert(kv: KVClient, reason: str) -> bool:
    """Check if we should send a circuit breaker alert (dedup within 1 hour).

    Prevents notification spam by tracking the last alert time for each reason.
    Returns True if we should send the alert, False if we should skip it.
    """
    # Create a key based on the reason (normalized)
    reason_key = reason.lower().replace(" ", "_").replace("(", "").replace(")", "")
    key = f"circuit_breaker_alert:{reason_key}"

    # Check if we've already alerted recently
    last_alert = await kv.get(key)
    if last_alert:
        # Already alerted within the dedup window, skip
        return False

    # Record this alert with TTL
    await kv.put(
        key,
        datetime.now(UTC).isoformat(),
        expiration_ttl=CIRCUIT_BREAKER_ALERT_DEDUP_SECONDS,
    )
    return True


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

    # Step 1: Reconcile pending_fill trades (check if orders filled or expired)
    await _reconcile_pending_orders(db, alpaca, discord, kv)

    # Step 2: Reconcile any pending exit orders (orders placed but DB update failed)
    await _reconcile_pending_exit_orders(db, alpaca, discord, kv)

    # Check if market is open
    if not await alpaca.is_market_open():
        print("Market is closed, skipping monitor")
        return

    # Get open trades from database (only trades with confirmed fills)
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
        print(
            f"Risk level: {risk_state.level.value}, size multiplier: {risk_state.size_multiplier}"
        )
        if risk_state.reason:
            print(f"Reason: {risk_state.reason}")

    # Send alert if needed (with deduplication to prevent spam)
    if risk_state.should_alert and risk_state.reason:
        # Check if we've already alerted for this reason recently (1 hour dedup window)
        should_send = await _should_send_circuit_breaker_alert(kv, risk_state.reason)
        if should_send:
            await discord.send_circuit_breaker_alert(risk_state.reason)
        else:
            print(f"Skipping duplicate circuit breaker alert: {risk_state.reason}")

    # If halted, stop processing
    if risk_state.level == RiskLevel.HALTED:
        print(f"Trading halted: {risk_state.reason}")
        return

    # Process each open trade
    for trade in open_trades:
        try:
            # Get current prices - include the trade's expiration in the date range
            # so we can find contracts even when DTE < 25 (the default start)
            chain = await alpaca.get_options_chain(
                trade.underlying,
                expiration_start=trade.expiration,
                expiration_end=trade.expiration,
            )

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

            # Extract current IV from contracts (average of short and long)
            current_iv = None
            if short_contract.implied_volatility and long_contract.implied_volatility:
                current_iv = (
                    short_contract.implied_volatility + long_contract.implied_volatility
                ) / 2
            elif short_contract.implied_volatility:
                current_iv = short_contract.implied_volatility
            elif long_contract.implied_volatility:
                current_iv = long_contract.implied_volatility

            # Fetch IV history for IV-adjusted exits
            iv_history = await db.get_iv_history(trade.underlying, lookback_days=252)

            # Calculate DTE for logging
            dte = days_to_expiry(trade.expiration)

            # Check exit conditions with IV context
            should_exit, exit_reason, iv_rank = exit_validator.check_all_exit_conditions(
                entry_credit=trade.entry_credit,
                current_value=current_value,
                expiration=trade.expiration,
                current_iv=current_iv,
                iv_history=iv_history,
            )

            if should_exit:
                print(
                    f"Exit triggered for {trade.underlying}: {exit_reason} (IV rank={iv_rank}, DTE={dte})"
                )

                # Check if auto-execute is enabled
                auto_execute = getattr(env, "AUTO_APPROVE_TRADES", "false").lower() == "true"

                if auto_execute:
                    # Auto-execute the exit
                    await _auto_execute_exit(
                        trade=trade,
                        short_symbol=short_symbol,
                        long_symbol=long_symbol,
                        current_value=current_value,
                        unrealized_pnl=unrealized_pnl,
                        exit_reason=exit_reason,
                        iv_rank=iv_rank,
                        dte=dte,
                        alpaca=alpaca,
                        db=db,
                        discord=discord,
                        kv=kv,
                    )
                else:
                    # Send exit alert with buttons for manual approval
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


async def _reconcile_pending_orders(db, alpaca, discord, kv):
    """Reconcile pending_fill trades by checking their broker order status.

    This ensures we only track trades that actually filled, and properly
    handle orders that expired or were cancelled.

    Also implements price adjustment for unfilled orders to improve fill rates.
    """
    pending_trades = await db.get_pending_fill_trades()

    if not pending_trades:
        return

    print(f"Reconciling {len(pending_trades)} pending orders...")

    for trade in pending_trades:
        if not trade.broker_order_id:
            print(f"Trade {trade.id} has no broker_order_id, marking as expired")
            await db.update_trade_status(trade.id, TradeStatus.EXPIRED)
            continue

        try:
            order = await alpaca.get_order(trade.broker_order_id)

            if order.status == OrderStatus.FILLED:
                # Order filled - mark trade as open
                print(f"Order {order.id} FILLED - activating trade {trade.id}")
                await db.mark_trade_filled(trade.id)

                # Clean up adjustment tracking (keyed by trade.id)
                await kv.delete(f"order_adjustment:{trade.id}")

                # Update daily stats now that we have a confirmed fill
                await kv.update_daily_stats(trades_delta=1)

                # Update D1 daily performance for accurate daily summaries
                today = datetime.now().strftime("%Y-%m-%d")
                await db.update_daily_performance(today, trades_opened_delta=1)

                # Send Discord notification
                await discord.send_message(
                    content=f"**Trade Filled: {trade.underlying}**",
                    embeds=[
                        {
                            "title": f"Order Filled: {trade.underlying}",
                            "description": "Your order has been filled and position is now active.",
                            "color": 0x57F287,  # Green
                            "fields": [
                                {
                                    "name": "Strategy",
                                    "value": trade.spread_type.value.replace("_", " ").title(),
                                    "inline": True,
                                },
                                {
                                    "name": "Strikes",
                                    "value": f"${trade.short_strike:.2f}/${trade.long_strike:.2f}",
                                    "inline": True,
                                },
                                {
                                    "name": "Credit",
                                    "value": f"${trade.entry_credit:.2f}",
                                    "inline": True,
                                },
                                {
                                    "name": "Contracts",
                                    "value": str(trade.contracts),
                                    "inline": True,
                                },
                            ],
                        }
                    ],
                )

            elif order.status in [OrderStatus.EXPIRED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                # Order did not fill - mark trade as expired
                print(f"Order {order.id} {order.status.value} - expiring trade {trade.id}")
                await db.update_trade_status(trade.id, TradeStatus.EXPIRED)

                # Get adjustment data before cleaning up (to show final price)
                adjustment_data = await kv.get_json(f"order_adjustment:{trade.id}")
                final_price = (
                    adjustment_data.get("current_price", trade.entry_credit)
                    if adjustment_data
                    else trade.entry_credit
                )
                original_price = (
                    adjustment_data.get("original_price", trade.entry_credit)
                    if adjustment_data
                    else trade.entry_credit
                )
                adjustments_made = (
                    adjustment_data.get("adjustments_made", 0) if adjustment_data else 0
                )

                # Clean up adjustment tracking (keyed by trade.id)
                await kv.delete(f"order_adjustment:{trade.id}")

                # Delete any position snapshot for this trade
                await db.delete_position(trade.id)

                # Build fields for Discord notification
                fields = [
                    {
                        "name": "Strategy",
                        "value": trade.spread_type.value.replace("_", " ").title(),
                        "inline": True,
                    },
                    {
                        "name": "Strikes",
                        "value": f"${trade.short_strike:.2f}/${trade.long_strike:.2f}",
                        "inline": True,
                    },
                    {"name": "Final Price", "value": f"${final_price:.2f}", "inline": True},
                ]
                if adjustments_made > 0:
                    fields.append(
                        {
                            "name": "Original Price",
                            "value": f"${original_price:.2f}",
                            "inline": True,
                        }
                    )
                    fields.append(
                        {"name": "Adjustments", "value": str(adjustments_made), "inline": True}
                    )

                # Send Discord notification
                await discord.send_message(
                    content=f"**Order Expired: {trade.underlying}**",
                    embeds=[
                        {
                            "title": f"Order {order.status.value.title()}: {trade.underlying}",
                            "description": "The limit order did not fill before expiration.",
                            "color": 0xED4245,  # Red
                            "fields": fields,
                        }
                    ],
                )

            elif order.status in [
                OrderStatus.NEW,
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED,
            ]:
                # Order still pending - check if we should adjust price
                await _maybe_adjust_order_price(
                    trade=trade,
                    order=order,
                    alpaca=alpaca,
                    db=db,
                    discord=discord,
                    kv=kv,
                )

            else:
                print(f"Order {order.id} in unexpected status: {order.status.value}")

        except Exception as e:
            print(f"Error reconciling order for trade {trade.id}: {e}")


async def _reconcile_pending_exit_orders(db, alpaca, discord, kv):
    """Reconcile trades that have exit orders placed but are still open.

    This is the primary mechanism for confirming exit order fills.
    When _auto_execute_exit places an order, the trade stays open until
    this function confirms the fill and closes it properly.
    """
    trades_with_exits = await db.get_trades_with_pending_exits()

    if not trades_with_exits:
        return

    print(f"Reconciling {len(trades_with_exits)} trades with pending exit orders...")

    for trade in trades_with_exits:
        try:
            order = await alpaca.get_order(trade.exit_order_id)

            if order.status == OrderStatus.FILLED:
                # Exit order filled - close the trade in database
                print(f"Exit order {order.id} FILLED - closing trade {trade.id}")

                # Get the fill price from the order
                # For multi-leg orders, filled_avg_price is the net debit/credit
                exit_debit = abs(order.filled_avg_price) if order.filled_avg_price else 0

                # Get exit metadata stored when order was placed
                exit_metadata = await kv.get_json(f"exit_metadata:{trade.id}")
                exit_reason = exit_metadata.get("exit_reason") if exit_metadata else None
                iv_rank_at_exit = exit_metadata.get("iv_rank_at_exit") if exit_metadata else None
                dte_at_exit = exit_metadata.get("dte_at_exit") if exit_metadata else None

                # Calculate realized P/L
                realized_pnl = (trade.entry_credit - exit_debit) * trade.contracts * 100

                # Close the trade with exit analytics
                await db.close_trade(
                    trade_id=trade.id,
                    exit_debit=exit_debit,
                    exit_reason=exit_reason,
                    iv_rank_at_exit=iv_rank_at_exit,
                    dte_at_exit=dte_at_exit,
                )

                # Delete position snapshot
                await db.delete_position(trade.id)

                # Update daily stats (KV)
                await kv.update_daily_stats(pnl_delta=realized_pnl)

                # Update D1 daily performance for accurate daily summaries
                today = datetime.now().strftime("%Y-%m-%d")
                win_delta = 1 if realized_pnl > 0 else 0
                loss_delta = 1 if realized_pnl < 0 else 0
                await db.update_daily_performance(
                    today,
                    realized_pnl_delta=realized_pnl,
                    trades_closed_delta=1,
                    win_delta=win_delta,
                    loss_delta=loss_delta,
                )

                # Clean up exit metadata
                await kv.delete(f"exit_metadata:{trade.id}")

                # Send Discord notification
                pnl_color = 0x57F287 if realized_pnl > 0 else 0xED4245
                pnl_emoji = "+" if realized_pnl > 0 else ""

                fields = [
                    {
                        "name": "Strategy",
                        "value": trade.spread_type.value.replace("_", " ").title(),
                        "inline": True,
                    },
                    {
                        "name": "Strikes",
                        "value": f"${trade.short_strike:.2f}/${trade.long_strike:.2f}",
                        "inline": True,
                    },
                    {"name": "Entry Credit", "value": f"${trade.entry_credit:.2f}", "inline": True},
                    {"name": "Exit Debit", "value": f"${exit_debit:.2f}", "inline": True},
                    {
                        "name": "Realized P/L",
                        "value": f"{pnl_emoji}${realized_pnl:.2f}",
                        "inline": True,
                    },
                ]
                if exit_reason:
                    fields.insert(0, {"name": "Reason", "value": exit_reason, "inline": False})
                if dte_at_exit is not None:
                    fields.append({"name": "DTE", "value": str(dte_at_exit), "inline": True})
                if iv_rank_at_exit is not None:
                    fields.append(
                        {"name": "IV Rank", "value": f"{iv_rank_at_exit:.0f}", "inline": True}
                    )

                await discord.send_message(
                    content=f"**Position Closed: {trade.underlying}**",
                    embeds=[
                        {
                            "title": f"Position Closed: {trade.underlying}",
                            "color": pnl_color,
                            "fields": fields,
                        }
                    ],
                )

            elif order.status in [OrderStatus.EXPIRED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                # Exit order did not fill - clear the exit order ID so we can try again
                print(
                    f"Exit order {order.id} {order.status.value} - clearing from trade {trade.id}"
                )
                await db.clear_exit_order_id(trade.id)

                # Clean up exit metadata
                await kv.delete(f"exit_metadata:{trade.id}")

                await discord.send_message(
                    content=f"**Exit Order Expired: {trade.underlying}**",
                    embeds=[
                        {
                            "title": f"Exit Order {order.status.value.title()}: {trade.underlying}",
                            "description": "The exit order did not fill. Will retry on next trigger.",
                            "color": 0xF59E0B,  # Amber
                            "fields": [
                                {"name": "Trade ID", "value": trade.id, "inline": True},
                            ],
                        }
                    ],
                )

            elif order.status in [
                OrderStatus.NEW,
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED,
            ]:
                # Exit order still pending - nothing to do
                print(
                    f"Exit order {order.id} still pending ({order.status.value}) for trade {trade.id}"
                )

            else:
                print(f"Exit order {order.id} in unexpected status: {order.status.value}")

        except Exception as e:
            print(f"Error reconciling exit order for trade {trade.id}: {e}")


# Price adjustment schedule for unfilled orders
# Each tuple is (minutes_elapsed, adjustment_cents)
# For credit spreads, we adjust by accepting less credit (worse for us, better fill chance)
# More aggressive schedule to improve fill rates - position_monitor runs every 5 min
PRICE_ADJUSTMENT_SCHEDULE = [
    (5, 0.03),  # After 5 min: adjust 3 cents (total 3 cents)
    (10, 0.04),  # After 10 min: adjust 4 more cents (total 7 cents)
    (15, 0.05),  # After 15 min: adjust 5 more cents (total 12 cents)
    (20, 0.05),  # After 20 min: adjust 5 more cents (total 17 cents)
    (30, 0.05),  # After 30 min: adjust 5 more cents (total 22 cents)
    (45, 0.05),  # After 45 min: adjust 5 more cents (total 27 cents)
]

# Maximum total adjustment - up to 30% of original credit or 30 cents, whichever is higher
MAX_TOTAL_ADJUSTMENT = 0.30

# Minimum credit to accept (never go below 5 cents)
MIN_CREDIT = 0.05


async def _maybe_adjust_order_price(trade, order, alpaca, db, discord, kv):
    """Check if a pending order should have its price adjusted to improve fill rate.

    Uses a time-based schedule to gradually improve the price (accept less credit)
    if the order hasn't filled. This balances getting filled vs. getting the best price.

    Price adjustments are tracked in KV to avoid duplicate adjustments.
    When an order is replaced, Alpaca creates a new order with a new ID - we update
    the trade record and migrate tracking to the new order ID.
    """

    # Get order age in minutes
    order_age_minutes = (datetime.now(UTC) - order.created_at).total_seconds() / 60
    print(f"Order {order.id[:8]}... age: {order_age_minutes:.1f} min, status: {order.status.value}")

    # Get current adjustment state from KV
    # Use trade.id as the key base since order IDs change on replacement
    adjustment_key = f"order_adjustment:{trade.id}"
    adjustment_data = await kv.get_json(adjustment_key)

    if adjustment_data is None:
        adjustment_data = {
            "adjustments_made": 0,
            "original_price": trade.entry_credit,
            "current_price": trade.entry_credit,
            "original_order_id": order.id,
        }

    adjustments_made = adjustment_data.get("adjustments_made", 0)
    original_price = adjustment_data.get("original_price", trade.entry_credit)
    current_price = adjustment_data.get("current_price", trade.entry_credit)

    # Determine if we should make another adjustment based on time elapsed
    target_adjustments = 0
    for minutes_threshold, _ in PRICE_ADJUSTMENT_SCHEDULE:
        if order_age_minutes >= minutes_threshold:
            target_adjustments += 1

    if target_adjustments <= adjustments_made:
        # No new adjustment needed yet
        print(
            f"Order {order.id[:8]}... no adjustment needed (made {adjustments_made}, target {target_adjustments})"
        )
        return

    # Calculate the new price
    total_adjustment = 0.0
    for i in range(target_adjustments):
        total_adjustment += PRICE_ADJUSTMENT_SCHEDULE[i][1]

    # Cap total adjustment
    total_adjustment = min(total_adjustment, MAX_TOTAL_ADJUSTMENT)

    # For credit spreads, reducing the credit means worse for us but better fill chance
    # The limit_price is negative (credit), so we make it less negative (smaller credit)
    new_credit = original_price - total_adjustment
    new_credit = max(new_credit, MIN_CREDIT)  # Don't go below minimum
    new_credit = round(new_credit, 2)

    if new_credit >= current_price:
        # Price would be same or worse, skip
        print(
            f"Order {order.id[:8]}... calculated price ${new_credit:.2f} not better than current ${current_price:.2f}"
        )
        return

    # Adjust the order price
    try:
        # For Alpaca multi-leg orders, limit_price is negative for credits
        new_limit_price = -abs(new_credit)

        print(
            f"Adjusting order {order.id[:8]}... from ${current_price:.2f} to ${new_credit:.2f} credit"
        )

        new_order = await alpaca.replace_order(
            order_id=order.id,
            limit_price=new_limit_price,
        )

        # When Alpaca replaces an order, it creates a new order with a new ID
        # Update the trade record with the new order ID
        if new_order.id != order.id:
            print(f"Order replaced: {order.id[:8]}... -> {new_order.id[:8]}...")
            await db.update_trade_order_id(trade.id, new_order.id)

        # Update adjustment tracking (keyed by trade.id so it persists across order replacements)
        adjustment_data["adjustments_made"] = target_adjustments
        adjustment_data["current_price"] = new_credit
        adjustment_data["current_order_id"] = new_order.id
        adjustment_data["last_adjustment"] = datetime.now(UTC).isoformat()
        await kv.put_json(adjustment_key, adjustment_data, expiration_ttl=24 * 3600)

        # Send Discord notification about the adjustment
        await discord.send_message(
            content=f"**Order Price Adjusted: {trade.underlying}**",
            embeds=[
                {
                    "title": f"Price Adjusted: {trade.underlying}",
                    "description": f"Order unfilled after {int(order_age_minutes)} min - adjusted price to improve fill chance.",
                    "color": 0xF59E0B,  # Amber/warning color
                    "fields": [
                        {
                            "name": "Strategy",
                            "value": trade.spread_type.value.replace("_", " ").title(),
                            "inline": True,
                        },
                        {
                            "name": "Strikes",
                            "value": f"${trade.short_strike:.2f}/${trade.long_strike:.2f}",
                            "inline": True,
                        },
                        {
                            "name": "Original Credit",
                            "value": f"${original_price:.2f}",
                            "inline": True,
                        },
                        {"name": "New Credit", "value": f"${new_credit:.2f}", "inline": True},
                        {
                            "name": "Adjustment",
                            "value": f"-${original_price - new_credit:.2f}",
                            "inline": True,
                        },
                        {"name": "Adjustment #", "value": str(target_adjustments), "inline": True},
                    ],
                }
            ],
        )

        print(
            f"Order {order.id[:8]}... price adjusted successfully. New order: {new_order.id[:8]}..."
        )

    except Exception as e:
        print(f"Error adjusting order {order.id}: {e}")
        # Don't fail the whole reconciliation if one adjustment fails


async def _auto_execute_exit(
    trade,
    short_symbol,
    long_symbol,
    current_value,
    unrealized_pnl,
    exit_reason,
    iv_rank,
    dte,
    alpaca,
    db,
    discord,
    kv,
):
    """Auto-execute an exit when conditions are met.

    Places a closing order and saves exit_order_id. The trade remains OPEN
    until _reconcile_pending_exit_orders confirms the order filled.
    This prevents mismatches between DB and broker state.
    """
    order = None
    try:
        print(f"Auto-executing exit for {trade.underlying}: {exit_reason}")

        # Check if there's already a pending exit order for this trade
        if trade.exit_order_id:
            print(f"Trade {trade.id} already has exit order {trade.exit_order_id}, skipping")
            return

        # Place closing order (buy back short, sell long)
        order = await alpaca.place_close_spread_order(
            short_symbol=short_symbol,
            long_symbol=long_symbol,
            contracts=trade.contracts,
            limit_price=current_value,  # Close at current mid price
        )

        print(f"Exit order placed: {order.id}")

        # Save exit order ID - trade stays OPEN until reconciliation confirms fill
        # This prevents DB/broker mismatch if the limit order doesn't fill immediately
        await db.set_exit_order_id(trade.id, order.id)

        # Store exit metadata for reconciliation to use when closing
        await kv.put_json(
            f"exit_metadata:{trade.id}",
            {
                "exit_reason": exit_reason,
                "iv_rank_at_exit": iv_rank,
                "dte_at_exit": dte,
                "expected_exit_debit": current_value,
            },
            expiration_ttl=7 * 24 * 3600,
        )

        # Calculate expected P/L for notification
        expected_pnl = (trade.entry_credit - current_value) * trade.contracts * 100
        pnl_color = 0xF59E0B  # Amber for pending
        pnl_emoji = "+" if expected_pnl > 0 else ""

        # Build fields with optional IV rank
        fields = [
            {"name": "Reason", "value": exit_reason, "inline": False},
            {
                "name": "Strategy",
                "value": trade.spread_type.value.replace("_", " ").title(),
                "inline": True,
            },
            {
                "name": "Strikes",
                "value": f"${trade.short_strike:.2f}/${trade.long_strike:.2f}",
                "inline": True,
            },
            {"name": "DTE", "value": str(dte), "inline": True},
            {"name": "Entry Credit", "value": f"${trade.entry_credit:.2f}", "inline": True},
            {"name": "Limit Price", "value": f"${current_value:.2f}", "inline": True},
            {"name": "Expected P/L", "value": f"{pnl_emoji}${expected_pnl:.2f}", "inline": True},
        ]
        if iv_rank is not None:
            fields.append({"name": "IV Rank", "value": f"{iv_rank:.0f}", "inline": True})

        await discord.send_message(
            content=f"**Exit Order Placed: {trade.underlying}** - {exit_reason}",
            embeds=[
                {
                    "title": f"Exit Order Placed: {trade.underlying}",
                    "description": "Order placed. Will confirm when filled.",
                    "color": pnl_color,
                    "fields": fields,
                }
            ],
        )

        print(f"Exit order placed for {trade.underlying}, awaiting fill confirmation")

    except Exception as e:
        print(f"Error auto-executing exit for {trade.id}: {e}")
        # Include order ID in error message if order was placed
        error_desc = f"Auto-exit failed: {str(e)}"
        if order:
            error_desc += f"\nExit order {order.id} was placed - will reconcile on next run."

        # Send error notification
        await discord.send_message(
            content=f"**Exit Error: {trade.underlying}**",
            embeds=[
                {
                    "title": f"Exit Failed: {trade.underlying}",
                    "color": 0xED4245,
                    "description": error_desc,
                    "fields": [
                        {"name": "Reason", "value": exit_reason, "inline": False},
                        {"name": "Trade ID", "value": trade.id, "inline": True},
                    ],
                }
            ],
        )
