"""EOD summary worker - runs at 4:15 PM ET.

Generates daily summary with:
- P/L for the day
- Open positions
- AI reflection on closed trades
- Archives data to R2
- Position reconciliation with broker
"""

from datetime import datetime

from core import http
from core.ai.claude import ClaudeClient
from core.broker.alpaca import AlpacaClient
from core.db.d1 import D1Client
from core.db.kv import KVClient
from core.db.r2 import R2Client
from core.notifications.discord import DiscordClient
from core.types import Position, TradeStatus


async def reconcile_positions(
    alpaca: AlpacaClient,
    db_positions: list[Position],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Reconcile broker positions with database positions.

    Returns:
        Tuple of (discrepancies, broker_positions_list, db_positions_list)
    """
    discrepancies = []

    # Get broker option positions
    broker_positions = await alpaca.get_option_positions()

    # Build lookup of broker positions by parsed components
    broker_lookup = {}
    broker_positions_list = []
    for bp in broker_positions:
        parsed = alpaca.parse_occ_symbol(bp.symbol)
        if parsed:
            key = f"{parsed['underlying']}:{parsed['expiration']}:{parsed['strike']}"
            broker_lookup[key] = {
                "symbol": bp.symbol,
                "qty": bp.qty,
                "underlying": parsed["underlying"],
                "expiration": parsed["expiration"],
                "strike": parsed["strike"],
                "option_type": parsed["option_type"],
                "market_value": bp.market_value,
            }
            broker_positions_list.append(broker_lookup[key])

    # Build lookup of DB positions
    db_lookup = {}
    db_positions_list = []
    for dp in db_positions:
        # For spreads, we have both short and long strikes
        # Check short strike
        short_key = f"{dp.underlying}:{dp.expiration}:{dp.short_strike}"
        db_lookup[short_key] = {
            "trade_id": dp.trade_id,
            "underlying": dp.underlying,
            "expiration": dp.expiration,
            "strike": dp.short_strike,
            "contracts": dp.contracts,
            "type": "short",
        }

        # Check long strike
        long_key = f"{dp.underlying}:{dp.expiration}:{dp.long_strike}"
        db_lookup[long_key] = {
            "trade_id": dp.trade_id,
            "underlying": dp.underlying,
            "expiration": dp.expiration,
            "strike": dp.long_strike,
            "contracts": dp.contracts,
            "type": "long",
        }

        db_positions_list.append({
            "trade_id": dp.trade_id,
            "underlying": dp.underlying,
            "expiration": dp.expiration,
            "short_strike": dp.short_strike,
            "long_strike": dp.long_strike,
            "contracts": dp.contracts,
        })

    # Check for positions in broker but not in DB
    for key, bp in broker_lookup.items():
        if key not in db_lookup:
            discrepancies.append({
                "type": "broker_only",
                "message": f"Position in broker not in DB: {bp['underlying']} {bp['expiration']} ${bp['strike']} ({bp['qty']} contracts)",
                "details": bp,
            })

    # Check for positions in DB but not in broker
    for key, dp in db_lookup.items():
        if key not in broker_lookup:
            discrepancies.append({
                "type": "db_only",
                "message": f"Position in DB not in broker: {dp['underlying']} {dp['expiration']} ${dp['strike']} ({dp['type']}, {dp['contracts']} contracts)",
                "details": dp,
            })

    # Check for quantity mismatches
    for key in set(broker_lookup.keys()) & set(db_lookup.keys()):
        bp = broker_lookup[key]
        dp = db_lookup[key]
        # Note: broker qty is signed (negative for short), dp.contracts is always positive
        expected_qty = dp["contracts"] if dp["type"] == "long" else -dp["contracts"]
        if bp["qty"] != expected_qty:
            discrepancies.append({
                "type": "qty_mismatch",
                "message": f"Quantity mismatch for {dp['underlying']} {dp['expiration']} ${dp['strike']}: broker={bp['qty']}, db={expected_qty}",
                "details": {"broker": bp, "db": dp},
            })

    return discrepancies, broker_positions_list, db_positions_list


async def handle_eod_summary(env):
    """Generate end-of-day summary."""
    print("Starting EOD summary...")

    # Signal start to heartbeat monitor
    heartbeat_url = getattr(env, "HEARTBEAT_URL", None)
    await http.ping_heartbeat_start(heartbeat_url, "eod_summary")

    job_success = False
    try:
        await _run_eod_summary(env)
        job_success = True
    finally:
        await http.ping_heartbeat(heartbeat_url, "eod_summary", success=job_success)


async def _run_eod_summary(env):
    """Internal EOD summary logic."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Initialize clients
    db = D1Client(env.MAHLER_DB)
    kv = KVClient(env.MAHLER_KV)
    r2 = R2Client(env.ARCHIVE)

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

    # Get account info
    account = await alpaca.get_account()

    # Capture daily IV for each underlying (including diversification assets)
    underlyings = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    for symbol in underlyings:
        try:
            chain = await alpaca.get_options_chain(symbol)
            if chain.contracts:
                # Find ATM contracts (within 2% of underlying price)
                atm_contracts = [
                    c
                    for c in chain.contracts
                    if abs(c.strike - chain.underlying_price) < chain.underlying_price * 0.02
                    and c.implied_volatility
                ]
                if atm_contracts:
                    # Use average IV of ATM options
                    atm_iv = sum(c.implied_volatility for c in atm_contracts) / len(atm_contracts)
                    await db.save_daily_iv(
                        date=today,
                        underlying=symbol,
                        atm_iv=atm_iv,
                        underlying_price=chain.underlying_price,
                    )
                    print(f"Saved IV for {symbol}: {atm_iv:.2%}")
        except Exception as e:
            print(f"Error capturing IV for {symbol}: {e}")

    # Capture VIX data
    try:
        vix_data = await alpaca.get_vix_snapshot()
        if vix_data:
            await db.save_daily_vix(
                date=today,
                vix_close=vix_data["vix"],
                vix3m_close=vix_data.get("vix3m"),
            )
            print(f"Saved VIX: {vix_data['vix']:.2f}")
    except Exception as e:
        print(f"Error capturing VIX: {e}")

    # Get or create daily performance record
    daily_stats = await kv.get_daily_stats(today)
    starting_balance = daily_stats.get("starting_equity", account.equity)

    performance = await db.get_or_create_daily_performance(
        date=today,
        starting_balance=starting_balance,
    )

    # Update ending balance
    await db.update_daily_performance(
        date=today,
        ending_balance=account.equity,
    )

    # Refresh performance
    performance = await db.get_or_create_daily_performance(
        date=today,
        starting_balance=starting_balance,
    )

    # Get positions
    positions = await db.get_all_positions()
    open_trades = await db.get_open_trades()

    # Get trade stats
    trade_stats = await db.get_trade_stats()

    # Generate reflections for trades closed today
    closed_today = []
    all_trades_result = await db.execute(
        "SELECT * FROM trades WHERE status = 'closed' AND closed_at LIKE ?",
        [f"{today}%"],
    )

    for row in all_trades_result.results:
        trade = db._row_to_trade(row)
        closed_today.append(trade)

    # Generate AI reflections for closed trades
    for trade in closed_today:
        if trade.reflection:
            continue  # Already has reflection

        try:
            # Get original thesis from recommendation
            rec = (
                await db.get_recommendation(trade.recommendation_id)
                if trade.recommendation_id
                else None
            )
            original_thesis = rec.thesis if rec else None

            reflection = await claude.generate_reflection(trade, original_thesis)

            # Update trade with reflection
            await db.run(
                "UPDATE trades SET reflection = ?, lesson = ? WHERE id = ?",
                [reflection.reflection, reflection.lesson, trade.id],
            )

            print(f"Generated reflection for trade {trade.id}")

        except Exception as e:
            print(f"Error generating reflection: {e}")

    # Check for playbook updates if we have enough closed trades
    if len(closed_today) >= 2:
        try:
            # Get recent trades with reflections
            recent_with_reflections = [t for t in closed_today if t.reflection]

            if recent_with_reflections:
                playbook_rules = await db.get_playbook_rules()
                updates = await claude.suggest_playbook_updates(
                    recent_trades=recent_with_reflections,
                    current_rules=playbook_rules,
                )

                for new_rule in updates.new_rules:
                    await db.add_playbook_rule(
                        rule=new_rule["rule"],
                        source="learned",
                        supporting_trade_ids=new_rule.get("supporting_trades", []),
                    )
                    print(f"Added playbook rule: {new_rule['rule']}")

        except Exception as e:
            print(f"Error updating playbook: {e}")

    # Reconcile positions with broker
    try:
        discrepancies, broker_positions_list, db_positions_list = await reconcile_positions(
            alpaca, positions
        )

        if discrepancies:
            print(f"Reconciliation found {len(discrepancies)} discrepancies")
            for d in discrepancies:
                print(f"  - {d['message']}")

            await discord.send_reconciliation_alert(
                discrepancies=discrepancies,
                broker_positions=broker_positions_list,
                db_positions=db_positions_list,
            )

            # Store reconciliation failure in KV for next day check
            await kv.put_json(
                f"reconciliation:{today}",
                {
                    "status": "failed",
                    "discrepancy_count": len(discrepancies),
                    "discrepancies": discrepancies,
                    "acknowledged": False,
                },
                expiration_ttl=7 * 24 * 3600,
            )
        else:
            print("Reconciliation successful - all positions match")
            if positions:
                await discord.send_reconciliation_success(len(positions))

            await kv.put_json(
                f"reconciliation:{today}",
                {"status": "passed", "position_count": len(positions)},
                expiration_ttl=7 * 24 * 3600,
            )

    except Exception as e:
        print(f"Error during reconciliation: {e}")

    # Archive daily snapshot to R2
    try:
        positions_data = [
            {
                "trade_id": p.trade_id,
                "underlying": p.underlying,
                "short_strike": p.short_strike,
                "long_strike": p.long_strike,
                "expiration": p.expiration,
                "contracts": p.contracts,
                "current_value": p.current_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in positions
        ]

        await r2.archive_daily_snapshot(
            date=today,
            positions=positions_data,
            performance={
                "starting_balance": performance.starting_balance,
                "ending_balance": performance.ending_balance,
                "realized_pnl": performance.realized_pnl,
                "trades_opened": performance.trades_opened,
                "trades_closed": performance.trades_closed,
                "win_count": performance.win_count,
                "loss_count": performance.loss_count,
            },
            account={
                "equity": account.equity,
                "cash": account.cash,
                "buying_power": account.buying_power,
            },
            reconciliation={
                "status": "passed" if not discrepancies else "failed",
                "discrepancy_count": len(discrepancies) if discrepancies else 0,
            },
        )
        print(f"Archived daily snapshot for {today}")

    except Exception as e:
        print(f"Error archiving snapshot: {e}")

    # Check AI confidence calibration (weekly on Fridays)
    try:
        weekday = datetime.now().weekday()
        if weekday == 4:  # Friday
            calibration = await db.get_confidence_calibration(lookback_days=90)
            stats = await db.get_rolling_calibration_stats(lookback_days=30)

            # Check for calibration issues
            has_issues = any(
                not data.get("is_calibrated", True)
                for data in calibration.values()
                if data.get("total_trades", 0) >= 5  # Only alert if enough trades
            )

            if has_issues:
                await discord.send_calibration_alert(calibration)
                print("Calibration issues detected - alert sent")

            # Send weekly summary
            if calibration:
                await discord.send_calibration_summary(calibration, stats)
                print("Weekly calibration summary sent")

            # Archive calibration data
            await kv.put_json(
                f"calibration:{today}",
                {"calibration": calibration, "stats": stats},
                expiration_ttl=90 * 24 * 3600,  # Keep 90 days
            )

    except Exception as e:
        print(f"Error checking calibration: {e}")

    # Send Discord summary
    await discord.send_daily_summary(
        performance=performance,
        open_positions=len(open_trades),
        trade_stats=trade_stats,
    )

    # Reset daily KV stats for next day
    tomorrow = datetime.now().replace(hour=0, minute=0, second=0) + __import__(
        "datetime"
    ).timedelta(days=1)
    # Store starting equity for tomorrow
    await kv.put_json(
        f"daily:{tomorrow.strftime('%Y-%m-%d')}",
        {"starting_equity": account.equity, "trades_count": 0, "realized_pnl": 0},
        expiration_ttl=7 * 24 * 3600,
    )

    print("EOD summary complete.")
