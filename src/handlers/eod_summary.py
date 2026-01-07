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


async def _run_weekly_optimization(
    db: D1Client,
    kv: KVClient,
    discord: DiscordClient,
) -> None:
    """Run weekly weight optimization on Friday after EOD.

    Requirements:
    - Only runs on Friday
    - Needs at least 100 closed trades
    - Optimizes scoring weights for each regime
    """
    from datetime import datetime

    # Only run on Friday
    weekday = datetime.now().weekday()
    if weekday != 4:
        return

    print("Running weekly weight optimization...")

    try:
        # Check trade count
        trade_stats = await db.get_trade_stats()
        if trade_stats["closed_trades"] < 100:
            print(
                f"Skipping weight optimization: only {trade_stats['closed_trades']} trades (need 100+)"
            )
            return

        # Import optimizer
        from core.analysis.weight_optimizer import WeightOptimizer

        # Create optimizer from DB
        optimizer = await WeightOptimizer.from_db(db)

        total_trades = optimizer.get_total_trades()
        if total_trades < 100:
            print(f"Skipping optimization: insufficient trades ({total_trades})")
            return

        # Optimize all regimes
        results = optimizer.optimize_all_regimes()

        if not results:
            print("No regimes had sufficient data for optimization")
            return

        # Store optimized weights in D1
        for regime, opt in results.items():
            await db.save_optimized_weights(
                regime=regime,
                weight_iv=opt.weights.iv_weight,
                weight_delta=opt.weights.delta_weight,
                weight_credit=opt.weights.credit_weight,
                weight_ev=opt.weights.ev_weight,
                sharpe_ratio=opt.sharpe_ratio,
                n_trades=opt.n_trades,
            )
            print(f"Saved optimized weights for {regime}: Sharpe={opt.sharpe_ratio:.3f}")

        # Cache in KV for fast access
        weights_for_cache = {
            regime: {
                "iv": opt.weights.iv_weight,
                "delta": opt.weights.delta_weight,
                "credit": opt.weights.credit_weight,
                "ev": opt.weights.ev_weight,
            }
            for regime, opt in results.items()
        }

        await kv.cache_weights(weights_for_cache)

        # Send Discord notification
        regimes_updated = len(results)
        total_sharpe = sum(r.sharpe_ratio for r in results.values()) / regimes_updated
        await discord.send_message(
            f"Weekly weight optimization complete: {regimes_updated} regimes updated (avg Sharpe: {total_sharpe:.2f})"
        )

        print(f"Weight optimization complete: {regimes_updated} regimes updated")

    except Exception as e:
        print(f"Weight optimization failed: {e}")
        import traceback

        print(traceback.format_exc())


async def _run_weekly_rule_validation(
    db: D1Client,
    kv: KVClient,
    discord: DiscordClient,
) -> None:
    """Run weekly playbook rule validation on Friday after EOD.

    Uses Mann-Whitney U test to compare trade outcomes with/without each rule,
    with Benjamini-Hochberg FDR correction for multiple testing.

    Requirements:
    - Only runs on Friday
    - Needs at least 50 closed trades with rule tags
    - Validates all rules with sufficient sample sizes
    """
    from datetime import datetime

    # Only run on Friday
    weekday = datetime.now().weekday()
    if weekday != 4:
        return

    print("Running weekly rule validation...")

    try:
        from core.analysis.rule_validator import TradingRuleValidator

        # Create validator from DB (checks for sufficient trades)
        validator = await TradingRuleValidator.from_db(
            db, min_trades=50, lookback_days=90
        )

        if validator is None:
            print("Skipping rule validation: insufficient trades with rule tags")
            return

        # Validate all rules
        results = validator.validate_all_rules()
        summary = validator.get_validation_summary()

        print(f"Rule validation complete: {summary['total_rules_tested']} rules tested")
        print(f"  Significant positive: {summary['significant_positive']}")
        print(f"  Significant negative: {summary['significant_negative']}")
        print(f"  Non-significant: {summary['non_significant']}")
        print(f"  Insufficient data: {summary['rules_with_insufficient_data']}")

        if not results:
            print("No rules had sufficient data for validation")
            return

        # Store validation results and update playbook status
        for result in results:
            # Save validation result
            await db.save_rule_validation(
                rule_id=result.rule_id,
                trades_with_rule=result.trades_with_rule,
                trades_without_rule=result.trades_without_rule,
                mean_pnl_with=result.mean_pnl_with,
                mean_pnl_without=result.mean_pnl_without,
                win_rate_with=result.win_rate_with,
                win_rate_without=result.win_rate_without,
                u_statistic=result.u_statistic,
                p_value=result.p_value,
                p_value_adjusted=result.p_value_adjusted,
                is_significant=result.is_significant,
                effect_direction=result.effect_direction,
            )

            # Update playbook rule validation status
            # Only mark as "validated" if significant AND positive effect
            is_validated = result.is_significant and result.effect_direction == "positive"
            await db.update_playbook_validation_status(
                rule_id=result.rule_id,
                is_validated=is_validated,
                p_value=result.p_value_adjusted,
            )

            status = "validated" if is_validated else ("rejected" if result.is_significant else "inconclusive")
            print(f"  Rule {result.rule_id[:8]}: {status} (p={result.p_value_adjusted:.3f})")

        # Send Discord notification with validation report
        await discord.send_rule_validation_report(results, summary)

        # Cache validation summary for quick access
        await kv.put_json(
            "validation:latest",
            {
                "validated_at": datetime.now().isoformat(),
                "summary": summary,
                "results": [r.to_dict() for r in results],
            },
            expiration_ttl=7 * 24 * 3600,  # Keep for 1 week
        )

        print("Rule validation complete")

    except Exception as e:
        print(f"Rule validation failed: {e}")
        import traceback

        print(traceback.format_exc())


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
    r2 = R2Client(env.MAHLER_BUCKET)

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

    # Weekly weight optimization (Friday)
    await _run_weekly_optimization(db, kv, discord)

    # Weekly rule validation (Friday)
    await _run_weekly_rule_validation(db, kv, discord)

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
