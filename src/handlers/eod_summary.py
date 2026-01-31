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
from core.ai.claude import ClaudeClient, ClaudeRateLimitError
from core.broker.alpaca import AlpacaClient
from core.db.d1 import D1Client
from core.db.kv import KVClient
from core.db.r2 import R2Client
from core.notifications.discord import DiscordClient
from core.types import Position, TradeStatus

# V2 Reflection Engine imports
from core.reflection import (
    SelfReflectionEngine,
    TradeOutcome,
    PredictedOutcome,
)
from core.memory.vectorize import EpisodicMemoryStore
from core.memory.retriever import MemoryRetriever

# V2 Learning imports
from core.learning import TrajectoryStore, DataSynthesizer

# V2 Strategy Monitoring imports
from core.monitoring import StrategyMonitor, AlertThresholds


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

    # Helper to normalize strike for consistent key format
    # Both Alpaca (float from division) and DB (could be int or float) must match
    def normalize_strike(strike):
        """Normalize strike to consistent format (e.g., 645.0 -> '645.0')."""
        return f"{float(strike)}"

    # Build lookup of broker positions by parsed components
    broker_lookup = {}
    broker_positions_list = []
    for bp in broker_positions:
        parsed = alpaca.parse_occ_symbol(bp.symbol)
        if parsed:
            key = f"{parsed['underlying']}:{parsed['expiration']}:{normalize_strike(parsed['strike'])}"
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

    # Build lookup of DB positions (aggregate contracts for same strikes)
    db_lookup = {}
    db_positions_list = []
    for dp in db_positions:
        # For spreads, we have both short and long strikes
        # Check short strike - aggregate if key already exists
        short_key = f"{dp.underlying}:{dp.expiration}:{normalize_strike(dp.short_strike)}"
        if short_key in db_lookup:
            # Aggregate contracts for same strike
            db_lookup[short_key]["contracts"] += dp.contracts
            db_lookup[short_key]["trade_ids"].append(dp.trade_id)
        else:
            db_lookup[short_key] = {
                "trade_ids": [dp.trade_id],
                "underlying": dp.underlying,
                "expiration": dp.expiration,
                "strike": dp.short_strike,
                "contracts": dp.contracts,
                "type": "short",
            }

        # Check long strike - aggregate if key already exists
        long_key = f"{dp.underlying}:{dp.expiration}:{normalize_strike(dp.long_strike)}"
        if long_key in db_lookup:
            # Aggregate contracts for same strike
            db_lookup[long_key]["contracts"] += dp.contracts
            db_lookup[long_key]["trade_ids"].append(dp.trade_id)
        else:
            db_lookup[long_key] = {
                "trade_ids": [dp.trade_id],
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


async def _run_strategy_monitoring(
    db: D1Client,
    kv: KVClient,
    discord: DiscordClient,
    alpaca: AlpacaClient,
) -> None:
    """Run daily strategy monitoring and send alerts.

    Monitors strategy performance against backtest expectations:
    - Win Rate: 70% (backtest: 69.9%)
    - Profit Factor: 6.0 (backtest: 6.10)
    - Max Drawdown: 4.35% (backtest)
    - Slippage: 66% (ORATS 2-leg)

    Ref: analysis/walkforward_findings_2026-01-30.log
    """
    print("Running strategy monitoring...")

    try:
        # Initialize monitoring with default thresholds from env
        thresholds = AlertThresholds.from_env()
        monitor = StrategyMonitor(
            d1_client=db,
            kv_client=kv,
            discord_client=discord,
            thresholds=thresholds,
        )

        # Get current market context
        vix_data = await alpaca.get_vix_snapshot()
        current_vix = vix_data.get("vix", 20.0) if vix_data else 20.0

        # Get IV percentile for primary underlying (QQQ - best performer per backtest)
        iv_percentile = 50.0  # Default
        try:
            # Get historical IV data from D1
            from core.analysis.iv_rank import calculate_iv_percentile
            today = datetime.now().strftime("%Y-%m-%d")

            # Get historical ATM IV from daily_iv table (populated during EOD)
            iv_history = await db.execute(
                """
                SELECT atm_iv FROM daily_iv
                WHERE underlying = 'QQQ'
                ORDER BY date DESC
                LIMIT 252
                """,
                [],
            )

            if iv_history and iv_history.get("results"):
                historical_ivs = [row["atm_iv"] for row in iv_history["results"] if row.get("atm_iv")]
                if historical_ivs:
                    current_iv = historical_ivs[0] if historical_ivs else 0.20
                    iv_percentile = calculate_iv_percentile(current_iv, historical_ivs)
                    print(f"IV Percentile for QQQ: {iv_percentile:.1f}%")
        except Exception as e:
            print(f"Could not get IV metrics: {e}")

        # Determine market regime from VIX
        if current_vix >= 50:
            market_regime = "crisis"
        elif current_vix >= 40:
            market_regime = "high_vol"
        elif current_vix >= 30:
            market_regime = "elevated"
        elif current_vix >= 20:
            market_regime = "normal"
        else:
            market_regime = "low_vol"

        # Run all monitoring checks
        alerts = await monitor.run_all_checks(
            iv_percentile=iv_percentile,
            vix_level=current_vix,
            market_regime=market_regime,
            underlying="QQQ",  # Primary underlying per backtest findings
        )

        if alerts:
            print(f"Strategy monitoring generated {len(alerts)} alerts")
            message_ids = await monitor.send_alerts(alerts)
            print(f"Sent {len(message_ids)} alerts to Discord")

            for alert in alerts:
                print(f"  - [{alert.severity.value}] {alert.category.value}: {alert.title}")
        else:
            print("Strategy monitoring: no alerts triggered")

    except Exception as e:
        import traceback
        print(f"Strategy monitoring failed: {e}")
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
    print(f"[DEBUG] Capturing IV for {len(underlyings)} underlyings")
    for symbol in underlyings:
        try:
            chain = await alpaca.get_options_chain(symbol)
            print(f"[DEBUG] {symbol}: got {len(chain.contracts) if chain.contracts else 0} contracts, price={chain.underlying_price}")
            if chain.contracts:
                # Find ATM contracts (within 2% of underlying price)
                atm_contracts = [
                    c
                    for c in chain.contracts
                    if abs(c.strike - chain.underlying_price) < chain.underlying_price * 0.02
                    and c.implied_volatility
                ]
                print(f"[DEBUG] {symbol}: found {len(atm_contracts)} ATM contracts with IV")
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
                else:
                    print(f"[DEBUG] {symbol}: no ATM contracts with IV found")
            else:
                print(f"[DEBUG] {symbol}: no contracts in chain")
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

    for row in all_trades_result["results"]:
        trade = db._row_to_trade(row)
        closed_today.append(trade)

    # Process reflections for trades with episodic memory
    if closed_today:
        print("Processing reflections for trades with episodic memory...")

        # Initialize reflection components if bindings available
        episodic_store = None
        memory_retriever = None
        reflection_engine = None

        if hasattr(env, "EPISODIC_MEMORY") and hasattr(env, "AI"):
            episodic_store = EpisodicMemoryStore(
                vectorize_binding=env.EPISODIC_MEMORY,
                ai_binding=env.AI,
                d1_binding=env.MAHLER_DB,
            )
            memory_retriever = MemoryRetriever(
                d1_binding=env.MAHLER_DB,
                episodic_store=episodic_store,
            )
            reflection_engine = SelfReflectionEngine(
                claude=claude,
                memory_retriever=memory_retriever,
                episodic_store=episodic_store,
            )
            print("Reflection engine initialized")
        else:
            print("Skipping reflection (bindings not configured)")

        if reflection_engine:
            for trade in closed_today:
                try:
                    # Check if trade has episodic memory record
                    episodic_result = await db.execute(
                        "SELECT * FROM episodic_memory WHERE trade_id = ?",
                        [trade.id],
                    )

                    if not episodic_result["results"]:
                        print(f"No episodic memory for trade {trade.id[:8]}, skipping")
                        continue

                    episodic_row = episodic_result["results"][0]
                    memory_id = episodic_row["id"]

                    # Skip if already has actual outcome
                    if episodic_row.get("actual_outcome"):
                        print(f"Trade {trade.id[:8]} already has outcome, skipping")
                        continue

                    # Build TradeOutcome from trade
                    entry_date = trade.opened_at.strftime("%Y-%m-%d") if trade.opened_at else today
                    exit_date = trade.closed_at.strftime("%Y-%m-%d") if trade.closed_at else today
                    days_held = (trade.closed_at - trade.opened_at).days if trade.closed_at and trade.opened_at else 0

                    # Determine exit reason based on P/L
                    profit_pct = (trade.profit_loss / (trade.entry_credit * trade.contracts * 100) * 100) if trade.profit_loss and trade.entry_credit else 0
                    if profit_pct >= 50:
                        exit_reason = "profit_target"
                    elif profit_pct <= -100:
                        exit_reason = "stop_loss"
                    elif days_held >= 21:
                        exit_reason = "time_exit"
                    else:
                        exit_reason = "manual"

                    trade_outcome = TradeOutcome(
                        trade_id=trade.id,
                        entry_date=entry_date,
                        exit_date=exit_date,
                        underlying=trade.underlying,
                        spread_type=trade.spread_type.value,
                        entry_credit=trade.entry_credit,
                        exit_debit=trade.exit_debit or 0,
                        profit_loss=trade.profit_loss or 0,
                        profit_loss_percent=profit_pct,
                        was_profitable=(trade.profit_loss or 0) > 0,
                        exit_reason=exit_reason,
                        days_held=days_held,
                    )

                    # Build PredictedOutcome from episodic memory
                    import json
                    predicted_data = json.loads(episodic_row["predicted_outcome"]) if episodic_row.get("predicted_outcome") else {}

                    predicted_outcome = PredictedOutcome(
                        recommendation=predicted_data.get("recommendation", "enter"),
                        confidence=predicted_data.get("confidence", 0.5),
                        expected_profit_probability=predicted_data.get("expected_profit_probability", 0.5),
                        key_bull_points=predicted_data.get("key_bull_points", []),
                        key_bear_points=predicted_data.get("key_bear_points", []),
                        thesis=predicted_data.get("thesis", ""),
                    )

                    # Generate reflection using reflection engine
                    reflection = await reflection_engine.generate_reflection(
                        outcome=trade_outcome,
                        predicted=predicted_outcome,
                        memory_id=memory_id,
                        market_regime=episodic_row.get("market_regime"),
                        iv_rank=episodic_row.get("iv_rank"),
                        vix=episodic_row.get("vix_at_entry"),
                    )

                    print(f"Generated reflection for trade {trade.id[:8]}: prediction_correct={reflection.prediction_correct}")

                    # Process candidate rules for learning
                    rule_ids = await reflection_engine.process_candidate_rules(reflection, trade_outcome)
                    if rule_ids:
                        print(f"Processed {len(rule_ids)} candidate rules for trade {trade.id[:8]}")

                except ClaudeRateLimitError as e:
                    print(f"Claude rate limit during reflection: {e}")
                    await discord.send_api_token_alert("Claude", str(e))
                    break  # Stop processing more reflections
                except Exception as e:
                    import traceback
                    print(f"Error generating reflection for trade {trade.id[:8]}: {e}")
                    print(traceback.format_exc())

    # Record outcomes for closed trades in trajectory store
    if closed_today:
        print("Recording outcomes for trajectories...")
        trajectory_store = TrajectoryStore(env.MAHLER_DB)

        for trade in closed_today:
            try:
                # Look up trajectory by trade ID
                trajectory = await trajectory_store.get_trajectory_by_trade_id(trade.id)
                if not trajectory:
                    print(f"No trajectory found for trade {trade.id[:8]}")
                    continue

                if trajectory.has_outcome:
                    print(f"Trajectory for trade {trade.id[:8]} already has outcome")
                    continue

                # Calculate P/L percentage
                entry_date = trade.opened_at.strftime("%Y-%m-%d") if trade.opened_at else today
                exit_date = trade.closed_at.strftime("%Y-%m-%d") if trade.closed_at else today
                days_held = (trade.closed_at - trade.opened_at).days if trade.closed_at and trade.opened_at else 0

                pnl_pct = 0.0
                if trade.entry_credit and trade.contracts:
                    total_credit = trade.entry_credit * trade.contracts * 100
                    pnl_pct = (trade.profit_loss or 0) / total_credit if total_credit > 0 else 0

                # Determine exit reason
                if pnl_pct >= 0.50:
                    exit_reason = "profit_target"
                elif pnl_pct <= -1.00:
                    exit_reason = "stop_loss"
                elif days_held >= 21:
                    exit_reason = "time_exit"
                else:
                    exit_reason = "manual"

                # Update trajectory with outcome
                await trajectory_store.update_outcome(
                    trajectory_id=trajectory.id,
                    actual_pnl=trade.profit_loss or 0,
                    actual_pnl_percent=pnl_pct,
                    exit_reason=exit_reason,
                    days_held=days_held,
                )
                print(f"Updated outcome for trajectory {trajectory.id[:8]}: P/L={pnl_pct:.1%}")

            except Exception as e:
                print(f"Error recording outcome for trade {trade.id[:8]}: {e}")

        # Run auto-labeling at end of day
        try:
            synthesizer = DataSynthesizer(trajectory_store)
            labeled_count = await synthesizer.label_unlabeled_trajectories(limit=100)
            if labeled_count > 0:
                print(f"Auto-labeled {labeled_count} trajectories")

                # Log label distribution
                distribution = await synthesizer.get_label_distribution()
                print(f"Label distribution: {distribution}")
        except Exception as e:
            print(f"Error during auto-labeling: {e}")

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

    # Strategy monitoring alerts (daily)
    try:
        await _run_strategy_monitoring(db, kv, discord, alpaca)
    except Exception as e:
        print(f"Error running strategy monitoring: {e}")

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
