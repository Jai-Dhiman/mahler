from __future__ import annotations

"""Discord client for notifications with interactive buttons."""

from typing import Any

from core import http
from core.types import DailyPerformance, Recommendation, Trade


class DiscordError(Exception):
    """Discord API error."""

    pass


async def verify_ed25519_signature(public_key_hex: str, message: bytes, signature_hex: str) -> bool:
    """Verify Ed25519 signature using JavaScript SubtleCrypto."""
    try:
        from js import Object, Uint8Array, crypto
        from pyodide.ffi import to_js

        print(
            f"Verifying signature: pk_len={len(public_key_hex)}, sig_len={len(signature_hex)}, msg_len={len(message)}"
        )

        # Convert hex to bytes
        public_key_bytes = bytes.fromhex(public_key_hex)
        signature_bytes = bytes.fromhex(signature_hex)

        print(f"Converted: pk_bytes={len(public_key_bytes)}, sig_bytes={len(signature_bytes)}")

        # Create Uint8Arrays from the bytes
        pk_array = Uint8Array.new(to_js(list(public_key_bytes)))
        sig_array = Uint8Array.new(to_js(list(signature_bytes)))
        msg_array = Uint8Array.new(to_js(list(message)))

        # Import the Ed25519 public key - convert dict to JS object via Object.fromEntries
        algorithm = Object.fromEntries(to_js([["name", "Ed25519"]]))
        print(f"Algorithm object: {algorithm}")

        key = await crypto.subtle.importKey("raw", pk_array, algorithm, False, to_js(["verify"]))
        print(f"Key imported successfully")

        # Verify the signature
        result = await crypto.subtle.verify(algorithm, key, sig_array, msg_array)
        print(f"Verification result: {result}")
        return bool(result)
    except Exception as e:
        import traceback

        print(f"Ed25519 verification error: {e}")
        print(traceback.format_exc())
        return False


def _format_daily_summary_footer(
    trade_stats: dict,
    trade_stats_today: dict | None = None,
) -> str:
    """Format the daily summary footer with all-time and today's stats."""
    pf = trade_stats["profit_factor"]
    pf_str = "N/A" if pf == float("inf") else f"{pf:.2f}"

    parts = [
        f"All-Time: {trade_stats['win_rate']:.0%} WR | PF {pf_str} | ${trade_stats['net_pnl']:,.2f} P/L"
    ]

    if trade_stats_today and trade_stats_today.get("closed_trades", 0) > 0:
        today_pf = trade_stats_today["profit_factor"]
        today_pf_str = "N/A" if today_pf == float("inf") else f"{today_pf:.2f}"
        parts.append(
            f"Today: {trade_stats_today['win_rate']:.0%} WR | PF {today_pf_str} | ${trade_stats_today['net_pnl']:,.2f}"
        )

    return " | ".join(parts)


class DiscordClient:
    """Client for Discord notifications with interactive components."""

    BASE_URL = "https://discord.com/api/v10"

    def __init__(self, bot_token: str, public_key: str, channel_id: str):
        self.bot_token = bot_token
        self.public_key = public_key
        self.channel_id = channel_id

        self._headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, endpoint: str, data: dict | None = None) -> dict:
        """Make a request to Discord API."""
        try:
            url = f"{self.BASE_URL}{endpoint}"
            return await http.request(method, url, headers=self._headers, json_data=data)
        except Exception as e:
            raise DiscordError(f"Discord API error: {str(e)}")

    async def verify_signature(self, body: str, timestamp: str, signature: str) -> bool:
        """Verify Discord interaction signature using Ed25519."""
        message = f"{timestamp}{body}".encode()
        return await verify_ed25519_signature(self.public_key, message, signature)

    # Message sending

    async def send_message(
        self,
        content: str,
        embeds: list[dict] | None = None,
        components: list[dict] | None = None,
    ) -> str:
        """Send a message to the channel. Returns message ID."""
        data = {"content": content}
        if embeds:
            data["embeds"] = embeds
        if components:
            data["components"] = components

        result = await self._request(
            "POST",
            f"/channels/{self.channel_id}/messages",
            data,
        )
        return result["id"]

    async def update_message(
        self,
        message_id: str,
        content: str,
        embeds: list[dict] | None = None,
        components: list[dict] | None = None,
    ) -> None:
        """Update an existing message."""
        data = {"content": content}
        if embeds:
            data["embeds"] = embeds
        # Always include components if provided (even empty list to remove buttons)
        if components is not None:
            data["components"] = components

        await self._request(
            "PATCH",
            f"/channels/{self.channel_id}/messages/{message_id}",
            data,
        )

    async def respond_to_interaction(
        self,
        interaction_id: str,
        interaction_token: str,
        content: str,
        embeds: list[dict] | None = None,
        components: list[dict] | None = None,
        update_message: bool = True,
    ) -> None:
        """Respond to a Discord interaction (button click)."""
        data = {
            "type": 7
            if update_message
            else 4,  # 7 = UPDATE_MESSAGE, 4 = CHANNEL_MESSAGE_WITH_SOURCE
            "data": {"content": content},
        }
        if embeds:
            data["data"]["embeds"] = embeds
        # Always include components if provided (even empty list to remove buttons)
        if components is not None:
            data["data"]["components"] = components

        # Interaction responses use a different endpoint (no auth needed)
        url = f"{self.BASE_URL}/interactions/{interaction_id}/{interaction_token}/callback"
        try:
            await http.request(
                "POST", url, headers={"Content-Type": "application/json"}, json_data=data
            )
        except Exception as e:
            raise DiscordError(f"Discord interaction error: {str(e)}")

    # Trade notification (V2 autonomous mode)

    async def send_autonomous_notification(
        self,
        rec: Recommendation,
        v2_confidence: float | None = None,
        v2_thesis: str | None = None,
        order_id: str | None = None,
    ) -> str:
        """Send an autonomous mode notification (info-only, no buttons).

        Used when AUTONOMOUS_MODE is enabled to show trade execution
        without requiring manual approval.

        Args:
            rec: Recommendation details
            v2_confidence: V2 pipeline confidence (0.0-1.0)
            v2_thesis: V2 synthesis thesis
            order_id: Broker order ID if order was placed

        Returns:
            Message ID
        """
        spread_name = (
            "Bull Put Spread" if rec.spread_type.value == "bull_put" else "Bear Call Spread"
        )
        direction = "Bullish" if rec.spread_type.value == "bull_put" else "Bearish"

        # Use V2 confidence for color if available, otherwise use standard confidence
        if v2_confidence is not None:
            if v2_confidence >= 0.7:
                color = 0x57F287  # Green
            elif v2_confidence >= 0.4:
                color = 0xF97316  # Orange
            else:
                color = 0xFEE75C  # Yellow
        else:
            confidence_color = {
                "low": 0xFEE75C,
                "medium": 0xF97316,
                "high": 0x57F287,
            }
            color = confidence_color.get(rec.confidence.value if rec.confidence else "low", 0x5865F2)

        # Use V2 thesis if available
        description = v2_thesis if v2_thesis else (rec.thesis if rec.thesis else "No analysis provided")

        fields = [
            {"name": "Strategy", "value": spread_name, "inline": True},
            {"name": "Direction", "value": direction, "inline": True},
            {"name": "Expiration", "value": rec.expiration, "inline": True},
            {"name": "Short Strike", "value": f"${rec.short_strike:.2f}", "inline": True},
            {"name": "Long Strike", "value": f"${rec.long_strike:.2f}", "inline": True},
            {"name": "Credit", "value": f"${rec.credit:.2f}", "inline": True},
            {"name": "Max Loss", "value": f"${rec.max_loss:.2f}", "inline": True},
            {"name": "Contracts", "value": str(rec.suggested_contracts or 1), "inline": True},
        ]

        # Add V2 confidence if available
        if v2_confidence is not None:
            fields.append(
                {"name": "V2 Confidence", "value": f"{v2_confidence:.0%}", "inline": True}
            )
        else:
            fields.append(
                {
                    "name": "Confidence",
                    "value": (rec.confidence.value.upper() if rec.confidence else "N/A"),
                    "inline": True,
                }
            )

        if rec.iv_rank:
            fields.append({"name": "IV Rank", "value": f"{rec.iv_rank:.1f}%", "inline": True})
        if rec.delta:
            fields.append({"name": "Delta", "value": f"{rec.delta:.3f}", "inline": True})
        if order_id:
            fields.append({"name": "Order ID", "value": order_id, "inline": True})

        embed = {
            "title": f"Autonomous Trade: {rec.underlying}",
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {
                "text": f"Autonomous Mode | ID: {rec.id[:8]}",
            },
        }

        return await self.send_message(
            content=f"**Autonomous Trade Executed: {rec.underlying}**",
            embeds=[embed],
            components=[],  # No buttons in autonomous mode
        )

    async def send_trade_decision(
        self,
        underlying: str,
        spread_type: str,
        short_strike: float,
        long_strike: float,
        expiration: str,
        credit: float,
        decision: str,  # "approved", "rejected", "skipped"
        reason: str,
        ai_summary: str | None = None,
        confidence: float | None = None,
        iv_rank: float | None = None,
        delta: float | None = None,
    ) -> str:
        """Send notification for any trade decision (approved, rejected, or skipped).

        This provides full traceability of the agentic system's decisions.

        Args:
            underlying: Symbol (SPY, QQQ, etc.)
            spread_type: "bull_put" or "bear_call"
            short_strike: Short leg strike
            long_strike: Long leg strike
            expiration: Expiration date
            credit: Net credit
            decision: Decision type
            reason: Why this decision was made
            ai_summary: AI agent's reasoning/summary
            confidence: Confidence level (0.0-1.0)
            iv_rank: Current IV rank
            delta: Short delta

        Returns:
            Message ID
        """
        spread_name = "Bull Put" if spread_type == "bull_put" else "Bear Call"

        # Color based on decision
        if decision == "approved":
            color = 0x57F287  # Green
            emoji = "CHECK"
            title_prefix = "Trade Approved"
        elif decision == "rejected":
            color = 0xED4245  # Red
            emoji = "X"
            title_prefix = "Trade Rejected"
        else:  # skipped
            color = 0xFEE75C  # Yellow
            emoji = "SKIP"
            title_prefix = "Trade Skipped"

        fields = [
            {"name": "Strategy", "value": spread_name, "inline": True},
            {"name": "Strikes", "value": f"${short_strike:.0f}/${long_strike:.0f}", "inline": True},
            {"name": "Expiration", "value": expiration, "inline": True},
            {"name": "Credit", "value": f"${credit:.2f}", "inline": True},
            {"name": "Decision", "value": decision.upper(), "inline": True},
        ]

        if confidence is not None:
            fields.append({"name": "Confidence", "value": f"{confidence:.0%}", "inline": True})
        if iv_rank is not None:
            fields.append({"name": "IV Rank", "value": f"{iv_rank:.1f}%", "inline": True})
        if delta is not None:
            fields.append({"name": "Delta", "value": f"{delta:.3f}", "inline": True})

        fields.append({"name": "Reason", "value": reason[:200], "inline": False})

        # AI summary is the key part for traceability
        description = ai_summary[:500] if ai_summary else "No AI summary available"

        embed = {
            "title": f"{title_prefix}: {underlying}",
            "description": f"**AI Analysis:**\n{description}",
            "color": color,
            "fields": fields,
            "footer": {"text": f"Decision Agent | {spread_name}"},
        }

        return await self.send_message(
            content=f"**{title_prefix}: {underlying} {spread_name}**",
            embeds=[embed],
        )

    async def send_agent_pipeline_log(
        self,
        underlying: str,
        spread_type: str,
        pipeline_result: dict,
    ) -> str:
        """Send detailed log of all agent decisions in the pipeline.

        Args:
            underlying: Symbol
            spread_type: "bull_put" or "bear_call"
            pipeline_result: Dict with analyst_messages, debate_messages, etc.

        Returns:
            Message ID
        """
        spread_name = "Bull Put" if spread_type == "bull_put" else "Bear Call"
        fields = []

        # Analyst perspectives
        analyst_msgs = pipeline_result.get("analyst_messages", [])
        for msg in analyst_msgs[:4]:  # Limit to 4 analysts
            agent_name = msg.get("agent_id", "Unknown").replace("_agent", "").title()
            content = msg.get("content", "")[:150]
            confidence = msg.get("confidence", 0)
            fields.append({
                "name": f"{agent_name} ({confidence:.0%})",
                "value": content + "..." if len(msg.get("content", "")) > 150 else content,
                "inline": False,
            })

        # Debate summary
        debate_msgs = pipeline_result.get("debate_messages", [])
        if debate_msgs:
            last_bull = None
            last_bear = None
            for msg in debate_msgs:
                if "bull" in msg.get("agent_id", "").lower():
                    last_bull = msg
                elif "bear" in msg.get("agent_id", "").lower():
                    last_bear = msg

            if last_bull:
                rec = last_bull.get("structured_data", {}).get("recommendation", "?")
                fields.append({
                    "name": f"Bull Researcher ({last_bull.get('confidence', 0):.0%})",
                    "value": f"Rec: {rec} - {last_bull.get('content', '')[:100]}...",
                    "inline": False,
                })
            if last_bear:
                rec = last_bear.get("structured_data", {}).get("recommendation", "?")
                fields.append({
                    "name": f"Bear Researcher ({last_bear.get('confidence', 0):.0%})",
                    "value": f"Rec: {rec} - {last_bear.get('content', '')[:100]}...",
                    "inline": False,
                })

        # Final decision
        fund_manager = pipeline_result.get("fund_manager_message")
        if fund_manager:
            rec = fund_manager.get("structured_data", {}).get("action", "?")
            fields.append({
                "name": f"Fund Manager Decision ({fund_manager.get('confidence', 0):.0%})",
                "value": f"**{rec.upper()}** - {fund_manager.get('content', '')[:150]}",
                "inline": False,
            })

        embed = {
            "title": f"Agent Pipeline: {underlying} {spread_name}",
            "color": 0x5865F2,  # Blurple
            "fields": fields[:10],  # Discord limit
            "footer": {"text": "Multi-Agent Analysis Pipeline"},
        }

        return await self.send_message(
            content=f"**Agent Analysis: {underlying}**",
            embeds=[embed],
        )

    async def send_slippage_log(
        self,
        underlying: str,
        expected_price: float,
        filled_price: float,
        contracts: int,
        order_type: str,  # "entry" or "exit"
    ) -> str:
        """Log slippage on order fills.

        Args:
            underlying: Symbol
            expected_price: Limit price or expected fill
            filled_price: Actual fill price
            contracts: Number of contracts
            order_type: "entry" or "exit"

        Returns:
            Message ID
        """
        slippage = filled_price - expected_price
        slippage_pct = (slippage / expected_price * 100) if expected_price else 0
        total_slippage = slippage * contracts * 100

        # Color based on slippage severity
        if abs(slippage_pct) < 1:
            color = 0x57F287  # Green - minimal
        elif abs(slippage_pct) < 3:
            color = 0xFEE75C  # Yellow - moderate
        else:
            color = 0xED4245  # Red - significant

        embed = {
            "title": f"Order Fill: {underlying}",
            "color": color,
            "fields": [
                {"name": "Type", "value": order_type.title(), "inline": True},
                {"name": "Expected", "value": f"${expected_price:.2f}", "inline": True},
                {"name": "Filled", "value": f"${filled_price:.2f}", "inline": True},
                {"name": "Slippage", "value": f"${slippage:.2f} ({slippage_pct:+.1f}%)", "inline": True},
                {"name": "Contracts", "value": str(contracts), "inline": True},
                {"name": "Total Impact", "value": f"${total_slippage:.2f}", "inline": True},
            ],
        }

        return await self.send_message(embeds=[embed])

    # Exit alerts

    async def send_exit_alert(
        self,
        trade: Trade,
        reason: str,
        current_value: float,
        unrealized_pnl: float,
    ) -> str:
        """Send an exit alert for a position."""
        pnl_color = 0x57F287 if unrealized_pnl > 0 else 0xED4245  # Green or Red

        embed = {
            "title": f"Exit Alert: {trade.underlying}",
            "color": pnl_color,
            "fields": [
                {"name": "Reason", "value": reason, "inline": False},
                {"name": "Entry Credit", "value": f"${trade.entry_credit:.2f}", "inline": True},
                {"name": "Current Value", "value": f"${current_value:.2f}", "inline": True},
                {"name": "Unrealized P/L", "value": f"${unrealized_pnl:.2f}", "inline": True},
                {"name": "Contracts", "value": str(trade.contracts), "inline": True},
            ],
        }

        return await self.send_message(
            content=f"**Exit Alert: {trade.underlying}** - {reason}",
            embeds=[embed],
        )

    # Daily summary

    async def send_daily_summary(
        self,
        performance: DailyPerformance,
        open_positions: int,
        trade_stats: dict,
        trade_stats_today: dict | None = None,
        screening_summary: dict | None = None,
        market_context: dict | None = None,
        position_details: list[dict] | None = None,
    ) -> str:
        """Send end-of-day summary.

        Args:
            performance: Daily performance stats
            open_positions: Number of open positions
            trade_stats: All-time trade statistics (win_rate, profit_factor, net_pnl)
            trade_stats_today: Today-only trade statistics
            screening_summary: Optional summary of today's screening results
            market_context: Optional market context (VIX, IV, regime)

        Returns:
            Message ID
        """
        # Color based on account change (ending - starting), not just realized P/L
        account_change = performance.ending_balance - performance.starting_balance
        pnl_color = 0x57F287 if account_change >= 0 else 0xED4245

        # Account change percentage
        change_pct = (account_change / performance.starting_balance * 100) if performance.starting_balance > 0 else 0
        change_sign = "+" if account_change >= 0 else ""

        fields = [
            {
                "name": "Starting Balance",
                "value": f"${performance.starting_balance:,.2f}",
                "inline": True,
            },
            {
                "name": "Ending Balance",
                "value": f"${performance.ending_balance:,.2f}",
                "inline": True,
            },
            {
                "name": "Account Change",
                "value": f"{change_sign}${account_change:,.2f} ({change_sign}{change_pct:.2f}%)",
                "inline": True,
            },
            {
                "name": "Realized P/L",
                "value": f"${performance.realized_pnl:,.2f}",
                "inline": True,
            },
            {"name": "Open Positions", "value": str(open_positions), "inline": True},
            {"name": "Trades Opened", "value": str(performance.trades_opened), "inline": True},
            {"name": "Trades Closed", "value": str(performance.trades_closed), "inline": True},
        ]

        # Add market context if provided
        if market_context:
            vix = market_context.get("vix")
            if vix:
                fields.append({"name": "VIX", "value": f"{vix:.1f}", "inline": True})

            regime = market_context.get("regime")
            if regime:
                regime_display = regime.replace("_", " ").title()
                fields.append({"name": "Regime", "value": regime_display, "inline": True})

            iv_percentile = market_context.get("iv_percentile")
            if iv_percentile:
                if isinstance(iv_percentile, dict):
                    # Average across underlyings
                    avg_iv = sum(iv_percentile.values()) / len(iv_percentile) if iv_percentile else 0
                    fields.append({"name": "Avg IV Pctl", "value": f"{avg_iv:.0f}%", "inline": True})
                else:
                    fields.append({"name": "IV Pctl", "value": f"{iv_percentile:.0f}%", "inline": True})

        # Add screening summary if provided
        if screening_summary:
            scanned = screening_summary.get("total_underlyings_scanned", 0)
            found = screening_summary.get("opportunities_found", 0)
            approved = screening_summary.get("opportunities_approved", 0)
            skip_reasons = screening_summary.get("skip_reasons", {})

            # Build screening summary text
            screening_text = f"Scanned: {scanned} | Found: {found} | Approved: {approved}"
            fields.append({
                "name": "Today's Screening",
                "value": screening_text,
                "inline": False,
            })

            # Add skip reasons if there were any
            if skip_reasons and found > approved:
                reasons_text = " | ".join(
                    f"{reason.replace('_', ' ').title()}: {count}"
                    for reason, count in list(skip_reasons.items())[:3]
                )
                if reasons_text:
                    fields.append({
                        "name": "Skip Reasons",
                        "value": reasons_text,
                        "inline": False,
                    })

        # Add position exit status if provided
        if position_details:
            pos_lines = []
            for pos in position_details:
                profit_str = f"{pos['profit_pct']:.0%}" if pos['profit_pct'] is not None else "N/A"
                pos_lines.append(f"{pos['underlying']}: {profit_str} profit, {pos['dte']} DTE")
            if pos_lines:
                fields.append({
                    "name": "Open Position Status",
                    "value": "\n".join(pos_lines),
                    "inline": False,
                })

        # Add win/loss counts
        fields.append({"name": "Wins", "value": str(performance.win_count), "inline": True})
        fields.append({"name": "Losses", "value": str(performance.loss_count), "inline": True})
        fields.append({"name": "\u200b", "value": "\u200b", "inline": True})  # Empty field for alignment

        embed = {
            "title": f"Daily Summary - {performance.date}",
            "color": pnl_color,
            "fields": fields,
            "footer": {
                "text": _format_daily_summary_footer(trade_stats, trade_stats_today),
            },
        }

        return await self.send_message(
            content=f"**Daily Summary: {performance.date}**",
            embeds=[embed],
        )

    # Circuit breaker

    async def send_circuit_breaker_alert(self, reason: str) -> str:
        """Send circuit breaker activation alert (info-only)."""
        embed = {
            "title": "Circuit Breaker Activated",
            "color": 0xED4245,  # Red
            "description": f"**Reason:** {reason}\n\nTrading has been halted. Use /admin/resume endpoint to restore trading.",
        }

        return await self.send_message(
            content="**CIRCUIT BREAKER ACTIVATED**",
            embeds=[embed],
        )

    async def send_api_token_alert(self, service: str, error_message: str) -> str:
        """Send alert when API tokens are exhausted or rate limited."""
        embed = {
            "title": f"{service} API Token Alert",
            "color": 0xED4245,  # Red
            "description": f"**Error:** {error_message}\n\nPlease add more API credits or wait for rate limits to reset.",
            "fields": [
                {"name": "Service", "value": service, "inline": True},
                {"name": "Action Required", "value": "Add API credits", "inline": True},
            ],
        }

        return await self.send_message(
            content=f"**API TOKEN ALERT: {service}**",
            embeds=[embed],
        )

    # Order updates

    async def send_order_filled(self, trade: Trade, filled_price: float) -> str:
        """Send order fill confirmation."""
        embed = {
            "title": f"Order Filled: {trade.underlying}",
            "color": 0x57F287,  # Green
            "fields": [
                {
                    "name": "Strategy",
                    "value": trade.spread_type.value.replace("_", " ").title(),
                    "inline": True,
                },
                {"name": "Expiration", "value": trade.expiration, "inline": True},
                {
                    "name": "Strikes",
                    "value": f"${trade.short_strike:.2f}/${trade.long_strike:.2f}",
                    "inline": True,
                },
                {"name": "Credit", "value": f"${filled_price:.2f}", "inline": True},
                {"name": "Contracts", "value": str(trade.contracts), "inline": True},
                {
                    "name": "Total Credit",
                    "value": f"${filled_price * trade.contracts * 100:.2f}",
                    "inline": True,
                },
            ],
        }

        return await self.send_message(
            content=f"**Order Filled: {trade.underlying}**",
            embeds=[embed],
        )

    # Reconciliation alerts

    async def send_reconciliation_alert(
        self,
        discrepancies: list[dict],
        broker_positions: list[dict],
        db_positions: list[dict],
    ) -> str:
        """Send reconciliation mismatch alert.

        Args:
            discrepancies: List of discrepancy descriptions
            broker_positions: Positions from broker
            db_positions: Positions from database
        """
        embed = {
            "title": "Position Reconciliation Mismatch",
            "color": 0xED4245,  # Red
            "description": "Discrepancies detected between broker and database positions. Manual review required before next trading day.",
            "fields": [
                {
                    "name": "Discrepancy Count",
                    "value": str(len(discrepancies)),
                    "inline": True,
                },
                {
                    "name": "Broker Positions",
                    "value": str(len(broker_positions)),
                    "inline": True,
                },
                {
                    "name": "DB Positions",
                    "value": str(len(db_positions)),
                    "inline": True,
                },
            ],
        }

        # Add discrepancy details (up to 5)
        discrepancy_text = "\n".join(f"- {d['message']}" for d in discrepancies[:5])
        if len(discrepancies) > 5:
            discrepancy_text += f"\n... and {len(discrepancies) - 5} more"

        embed["fields"].append({
            "name": "Discrepancies",
            "value": discrepancy_text or "None",
            "inline": False,
        })

        return await self.send_message(
            content="**RECONCILIATION ALERT - MANUAL REVIEW REQUIRED**",
            embeds=[embed],
        )

    async def send_reconciliation_success(self, position_count: int) -> str:
        """Send confirmation that reconciliation passed."""
        embed = {
            "title": "Position Reconciliation Complete",
            "color": 0x57F287,  # Green
            "description": f"All {position_count} positions match between broker and database.",
        }

        return await self.send_message(
            content="**Reconciliation: All Clear**",
            embeds=[embed],
        )

    # Kill switch

    async def send_kill_switch_activated(self, reason: str, activated_by: str) -> str:
        """Send kill switch activation alert."""
        embed = {
            "title": "TRADING HALTED - Kill Switch Activated",
            "color": 0xED4245,  # Red
            "fields": [
                {"name": "Reason", "value": reason, "inline": False},
                {"name": "Activated By", "value": activated_by, "inline": True},
            ],
            "description": "All trading has been halted. Use /resume to restore trading.",
        }

        return await self.send_message(
            content="**KILL SWITCH ACTIVATED**",
            embeds=[embed],
        )

    async def send_kill_switch_deactivated(self, deactivated_by: str) -> str:
        """Send kill switch deactivation alert."""
        embed = {
            "title": "Trading Resumed",
            "color": 0x57F287,  # Green
            "fields": [
                {"name": "Resumed By", "value": deactivated_by, "inline": True},
            ],
            "description": "Kill switch has been deactivated. Trading will resume on next scan.",
        }

        return await self.send_message(
            content="**Trading Resumed**",
            embeds=[embed],
        )

    # AI Calibration alerts

    async def send_calibration_alert(self, calibration_data: dict) -> str:
        """Send AI confidence calibration alert when calibration gap exceeds threshold."""
        fields = []
        issues = []

        for confidence, data in calibration_data.items():
            if not data.get("is_calibrated", True):
                gap = data.get("calibration_gap", 0)
                actual = data.get("actual_win_rate", 0)
                expected = data.get("expected_win_rate", 0)
                issues.append(confidence)

                fields.append({
                    "name": f"{confidence.upper()} Confidence",
                    "value": f"Expected: {expected:.0%} | Actual: {actual:.0%} | Gap: {gap:+.0%}",
                    "inline": False,
                })

        if not issues:
            return ""  # No alert needed

        embed = {
            "title": "AI Confidence Calibration Alert",
            "color": 0xF97316,  # Orange
            "description": f"Calibration gap exceeds 10% for {len(issues)} confidence level(s). Consider adjusting AI prompts or reviewing trade selection criteria.",
            "fields": fields,
        }

        return await self.send_message(
            content="**AI Calibration Issue Detected**",
            embeds=[embed],
        )

    async def send_calibration_summary(self, calibration_data: dict, stats: dict) -> str:
        """Send weekly calibration summary."""
        fields = []

        for confidence in ["high", "medium", "low"]:
            if confidence in calibration_data:
                data = calibration_data[confidence]
                actual = data.get("actual_win_rate", 0)
                expected = data.get("expected_win_rate", 0)
                total = data.get("total_trades", 0)
                status = "OK" if data.get("is_calibrated", True) else "MISCALIBRATED"

                fields.append({
                    "name": f"{confidence.upper()} ({total} trades)",
                    "value": f"Win Rate: {actual:.0%} (expected {expected:.0%}) - {status}",
                    "inline": False,
                })

        embed = {
            "title": "AI Confidence Calibration Summary",
            "color": 0x5865F2,  # Blurple
            "fields": fields,
            "footer": {
                "text": f"Overall win rate: {stats.get('overall_win_rate', 0):.0%} | Total: {stats.get('total_trades', 0)} trades",
            },
        }

        return await self.send_message(
            content="**Weekly AI Calibration Report**",
            embeds=[embed],
        )

    # Rule Validation

    async def send_rule_validation_report(
        self,
        results: list,
        summary: dict,
    ) -> str:
        """Send weekly rule validation report.

        Args:
            results: List of RuleValidationResult objects
            summary: Validation summary dict
        """
        fields = []

        # Summary statistics
        fields.append({
            "name": "Rules Tested",
            "value": str(summary.get("total_rules_tested", 0)),
            "inline": True,
        })
        fields.append({
            "name": "Validated (Positive)",
            "value": str(summary.get("significant_positive", 0)),
            "inline": True,
        })
        fields.append({
            "name": "Rejected (Negative)",
            "value": str(summary.get("significant_negative", 0)),
            "inline": True,
        })

        # Significant positive rules (validated)
        positive_rules = [r for r in results if r.is_significant and r.effect_direction == "positive"]
        if positive_rules:
            positive_text = "\n".join(
                f"- {r.rule_text[:50]}... (p={r.p_value_adjusted:.3f})"
                if len(r.rule_text) > 50 else f"- {r.rule_text} (p={r.p_value_adjusted:.3f})"
                for r in positive_rules[:3]
            )
            if len(positive_rules) > 3:
                positive_text += f"\n... and {len(positive_rules) - 3} more"
            fields.append({
                "name": "Validated Rules",
                "value": positive_text,
                "inline": False,
            })

        # Significant negative rules (should consider removing)
        negative_rules = [r for r in results if r.is_significant and r.effect_direction == "negative"]
        if negative_rules:
            negative_text = "\n".join(
                f"- {r.rule_text[:50]}... (p={r.p_value_adjusted:.3f})"
                if len(r.rule_text) > 50 else f"- {r.rule_text} (p={r.p_value_adjusted:.3f})"
                for r in negative_rules[:3]
            )
            if len(negative_rules) > 3:
                negative_text += f"\n... and {len(negative_rules) - 3} more"
            fields.append({
                "name": "Consider Removing",
                "value": negative_text,
                "inline": False,
            })

        # Color based on findings
        if negative_rules:
            color = 0xF97316  # Orange - some rules need attention
        elif positive_rules:
            color = 0x57F287  # Green - rules validated
        else:
            color = 0x5865F2  # Blurple - no significant findings

        embed = {
            "title": "Weekly Rule Validation Report",
            "color": color,
            "description": "Statistical validation of playbook rules using Mann-Whitney U test with FDR correction.",
            "fields": fields,
            "footer": {
                "text": f"Insufficient data: {summary.get('rules_with_insufficient_data', 0)} rules | Non-significant: {summary.get('non_significant', 0)} rules",
            },
        }

        return await self.send_message(
            content="**Weekly Playbook Rule Validation**",
            embeds=[embed],
        )

    # Strategy Monitoring Alerts

    async def send_iv_environment_alert(
        self,
        iv_percentile: float,
        vix_level: float,
        alert_type: str,  # "low", "elevated", "crisis"
        consecutive_days: int | None = None,
        suggested_actions: list[str] | None = None,
    ) -> str:
        """Send IV environment alert.

        Args:
            iv_percentile: Current IV percentile (0-100)
            vix_level: Current VIX level
            alert_type: Type of alert ("low", "elevated", "crisis")
            consecutive_days: Days below/above threshold (for low IV)
            suggested_actions: List of recommended actions
        """
        if alert_type == "crisis":
            color = 0xED4245  # Red
            title = "High Volatility Spike"
            description = (
                f"IV at {iv_percentile:.0f}th percentile (VIX: {vix_level:.1f}). "
                "Review position sizes and consider defensive adjustments."
            )
        elif alert_type == "low":
            color = 0xF97316  # Orange
            title = "Low IV Environment Detected"
            description = (
                f"IV below 30th percentile for {consecutive_days}+ consecutive days. "
                "Consider pausing new entries or accepting lower premium."
            )
        else:  # elevated
            color = 0x57F287  # Green
            title = "Elevated IV Environment"
            description = (
                f"IV at {iv_percentile:.0f}th percentile. "
                "Favorable conditions for premium selling."
            )

        fields = [
            {"name": "IV Percentile", "value": f"{iv_percentile:.0f}%", "inline": True},
            {"name": "VIX", "value": f"{vix_level:.1f}", "inline": True},
        ]

        if consecutive_days:
            fields.append(
                {"name": "Consecutive Days", "value": str(consecutive_days), "inline": True}
            )

        if suggested_actions:
            actions_text = "\n".join(f"- {a}" for a in suggested_actions)
            fields.append({"name": "Suggested Actions", "value": actions_text, "inline": False})

        embed = {
            "title": f"IV Environment: {title}",
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": "Ref: analysis/walkforward_findings_2026-01-30.log"},
        }

        return await self.send_message(
            content=f"**IV Alert: {title}**",
            embeds=[embed],
        )

    async def send_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        expected_value: float,
        window_trades: int,
        recent_results: list[str] | None = None,
        suggested_actions: list[str] | None = None,
    ) -> str:
        """Send performance deviation alert.

        Args:
            metric_name: Name of the metric ("win_rate", "profit_factor", "drawdown")
            current_value: Current metric value
            expected_value: Expected value from backtest
            window_trades: Number of trades in evaluation window
            recent_results: List of "W" or "L" for recent trades
            suggested_actions: List of recommended actions
        """
        deviation = expected_value - current_value

        if metric_name == "drawdown":
            color = 0xED4245 if current_value >= 10 else 0xF97316
            title = "Drawdown Alert"
            description = (
                f"Drawdown at {current_value:.1f}% exceeds historical max ({expected_value:.1f}%)."
            )
            deviation_str = f"+{abs(deviation):.1f}%"
        elif metric_name == "win_rate":
            color = 0xF97316  # Orange
            title = "Win Rate Degradation"
            description = (
                f"Win rate at {current_value:.1f}% over last {window_trades} trades "
                f"(expected: {expected_value:.0f}%)."
            )
            deviation_str = f"-{abs(deviation):.1f}%"
        else:  # profit_factor
            color = 0xF97316  # Orange
            title = "Profit Factor Degradation"
            description = (
                f"Profit factor at {current_value:.2f} over last {window_trades} trades "
                f"(expected: {expected_value:.1f})."
            )
            deviation_str = f"-{abs(deviation):.2f}"

        fields = [
            {"name": "Current", "value": f"{current_value:.2f}", "inline": True},
            {"name": "Expected", "value": f"{expected_value:.2f}", "inline": True},
            {"name": "Deviation", "value": deviation_str, "inline": True},
            {"name": "Window", "value": f"{window_trades} trades", "inline": True},
        ]

        if recent_results:
            results_str = " ".join(recent_results[:10])
            fields.append({"name": "Recent Results", "value": results_str, "inline": False})

        if suggested_actions:
            actions_text = "\n".join(f"- {a}" for a in suggested_actions)
            fields.append({"name": "Suggested Actions", "value": actions_text, "inline": False})

        embed = {
            "title": f"Performance Alert: {title}",
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": "Ref: analysis/walkforward_findings_2026-01-30.log"},
        }

        return await self.send_message(
            content=f"**Performance Alert: {title}**",
            embeds=[embed],
        )

    async def send_slippage_alert(
        self,
        avg_slippage: float,
        worst_slippage: float | None,
        expected_slippage: float,
        lookback_trades: int,
        is_critical: bool = False,
    ) -> str:
        """Send slippage quality alert.

        Args:
            avg_slippage: Average slippage percentage over lookback window
            worst_slippage: Worst single fill slippage (optional)
            expected_slippage: Expected slippage from ORATS (66% for 2-leg)
            lookback_trades: Number of trades evaluated
            is_critical: True if strategy-breaking slippage detected
        """
        color = 0xED4245 if is_critical else 0xF97316

        if is_critical:
            title = "Critical Slippage Detected"
            description = (
                f"Poor fill detected: {worst_slippage:.1f}% slippage. "
                "Strategy profitability at risk (breaks at 85%)."
            )
        else:
            title = "Fill Quality Degrading"
            description = (
                f"Average slippage at {avg_slippage:.1f}% over last {lookback_trades} trades "
                f"(expected: {expected_slippage:.0f}%)."
            )

        fields = [
            {"name": "Avg Slippage", "value": f"{avg_slippage:.1f}%", "inline": True},
            {"name": "Expected", "value": f"{expected_slippage:.0f}%", "inline": True},
        ]

        if worst_slippage:
            fields.append(
                {"name": "Worst Fill", "value": f"{worst_slippage:.1f}%", "inline": True}
            )

        fields.append({"name": "Window", "value": f"{lookback_trades} trades", "inline": True})

        suggested_actions = [
            "Review execution timing",
            "Consider wider bid-ask spread filters",
            "Check liquidity conditions at entry time",
        ]
        actions_text = "\n".join(f"- {a}" for a in suggested_actions)
        fields.append({"name": "Suggested Actions", "value": actions_text, "inline": False})

        embed = {
            "title": f"Slippage Alert: {title}",
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": "Strategy breaks at 85% slippage | Ref: walkforward_findings"},
        }

        return await self.send_message(
            content=f"**Slippage Alert: {title}**",
            embeds=[embed],
        )

    async def send_regime_change_alert(
        self,
        previous_regime: str,
        current_regime: str,
        vix_level: float,
        size_multiplier: float | None = None,
        is_crisis: bool = False,
    ) -> str:
        """Send market regime change alert.

        Args:
            previous_regime: Previous regime classification
            current_regime: Current regime classification
            vix_level: Current VIX level
            size_multiplier: Position size multiplier (0.0-1.0)
            is_crisis: True if entering/exiting crisis mode
        """
        if is_crisis and vix_level >= 50:
            color = 0xED4245  # Red
            title = "Crisis Regime Entered"
            description = (
                f"Market regime changed to {current_regime} (VIX: {vix_level:.1f}). "
                "Position sizing reduced, new entries halted."
            )
        elif is_crisis:
            color = 0x57F287  # Green
            title = "Market Regime Normalized"
            description = (
                f"Market regime changed from {previous_regime} to {current_regime} "
                f"(VIX: {vix_level:.1f}). Full position sizing resumed."
            )
        elif vix_level >= 40:
            color = 0xF97316  # Orange
            title = "High Volatility Regime"
            description = (
                f"Regime changed: {previous_regime} -> {current_regime} "
                f"(VIX: {vix_level:.1f}). Significant reduction in position sizing."
            )
        else:
            color = 0x5865F2  # Blurple
            title = "Market Regime Changed"
            description = (
                f"Regime changed: {previous_regime} -> {current_regime} "
                f"(VIX: {vix_level:.1f})."
            )

        fields = [
            {"name": "Previous", "value": previous_regime, "inline": True},
            {"name": "Current", "value": current_regime, "inline": True},
            {"name": "VIX", "value": f"{vix_level:.1f}", "inline": True},
        ]

        if size_multiplier is not None:
            fields.append(
                {"name": "Size Multiplier", "value": f"{size_multiplier:.0%}", "inline": True}
            )

        embed = {
            "title": f"Regime Alert: {title}",
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": "Ref: circuit_breaker.py | VIX thresholds: 20/30/40/50"},
        }

        return await self.send_message(
            content=f"**Regime Change: {title}**",
            embeds=[embed],
        )

    async def send_no_trade_notification(
        self,
        scan_time: str,  # 'morning', 'midday', 'afternoon'
        underlyings_scanned: int,
        opportunities_found: int,
        opportunities_filtered: int,
        skip_reasons: dict,  # {"iv_too_low": 3, "agent_rejected": 2, ...}
        market_context: dict,  # {"vix": 25.3, "iv_percentile": {"SPY": 45}, "regime": "..."}
        underlying_details: dict | None = None,  # Per-underlying breakdown
    ) -> str:
        """Send notification when scan finds no viable trades.

        Args:
            scan_time: Time of scan ('morning', 'midday', 'afternoon')
            underlyings_scanned: Number of underlyings scanned
            opportunities_found: Total opportunities found
            opportunities_filtered: Opportunities that passed initial filters
            skip_reasons: Dict of skip reason -> count
            market_context: Dict with VIX, IV percentile, regime, etc.
            underlying_details: Optional per-underlying breakdown

        Returns:
            Message ID
        """
        # Determine color based on market conditions
        vix = market_context.get("vix", 0)
        if vix >= 40:
            color = 0xED4245  # Red - high volatility
        elif vix >= 30:
            color = 0xF97316  # Orange - elevated
        else:
            color = 0x5865F2  # Blurple - normal

        # Build description
        if opportunities_found == 0:
            description = (
                f"Scanned {underlyings_scanned} underlyings but found no opportunities "
                "that passed initial screening criteria."
            )
        elif opportunities_filtered == 0:
            description = (
                f"Found {opportunities_found} opportunities across {underlyings_scanned} underlyings, "
                "but none passed the screening filters."
            )
        else:
            description = (
                f"Found {opportunities_found} opportunities, {opportunities_filtered} passed filters, "
                "but none were approved by the multi-agent pipeline."
            )

        fields = []

        # Market context
        fields.append({
            "name": "VIX",
            "value": f"{vix:.1f}" if vix else "N/A",
            "inline": True,
        })

        regime = market_context.get("regime", "unknown")
        if regime:
            regime_display = regime.replace("_", " ").title()
            fields.append({
                "name": "Market Regime",
                "value": regime_display,
                "inline": True,
            })

        # Size multiplier if reduced
        combined_mult = market_context.get("combined_multiplier", 1.0)
        if combined_mult < 1.0:
            fields.append({
                "name": "Size Multiplier",
                "value": f"{combined_mult:.0%}",
                "inline": True,
            })

        # IV Percentile by underlying
        iv_percentiles = market_context.get("iv_percentile", {})
        if iv_percentiles:
            iv_text = " | ".join(f"{sym}: {pct:.0f}%" for sym, pct in iv_percentiles.items())
            fields.append({
                "name": "IV Percentile",
                "value": iv_text,
                "inline": False,
            })

        # Skip reasons breakdown
        if skip_reasons:
            reasons_text = "\n".join(f"- {reason.replace('_', ' ').title()}: {count}" for reason, count in skip_reasons.items())
            fields.append({
                "name": "Skip Reasons",
                "value": reasons_text,
                "inline": False,
            })

        # Per-underlying details (if provided)
        if underlying_details:
            underlying_text = []
            for sym, details in underlying_details.items():
                found = details.get("found", 0)
                passed = details.get("passed", 0)
                reason = details.get("reason", "")
                if found == 0:
                    underlying_text.append(f"**{sym}**: No opportunities")
                elif reason:
                    underlying_text.append(f"**{sym}**: {found} found, {passed} passed - {reason}")
                else:
                    underlying_text.append(f"**{sym}**: {found} found, {passed} passed")

            if underlying_text:
                fields.append({
                    "name": "Per-Underlying",
                    "value": "\n".join(underlying_text[:5]),  # Limit to 5
                    "inline": False,
                })

        embed = {
            "title": f"No Trades - {scan_time.title()} Scan",
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {
                "text": "No action required - market conditions or filters prevented trades",
            },
        }

        return await self.send_message(
            content=f"**{scan_time.title()} Scan Complete - No Viable Trades**",
            embeds=[embed],
        )

    async def send_strategy_recommendation(
        self,
        title: str,
        description: str,
        iv_percentile: float,
        win_rate: float | None,
        vix_level: float,
        suggested_actions: list[str],
        severity: str = "warning",  # "info", "warning", "critical"
    ) -> str:
        """Send strategy switch recommendation.

        Args:
            title: Alert title
            description: Detailed description of the situation
            iv_percentile: Current IV percentile
            win_rate: Rolling win rate (optional)
            vix_level: Current VIX level
            suggested_actions: List of recommended actions
            severity: Alert severity level
        """
        colors = {
            "info": 0x5865F2,  # Blurple
            "warning": 0xF97316,  # Orange
            "critical": 0xED4245,  # Red
        }
        color = colors.get(severity, 0xF97316)

        fields = [
            {"name": "IV Percentile", "value": f"{iv_percentile:.0f}%", "inline": True},
            {"name": "VIX", "value": f"{vix_level:.1f}", "inline": True},
        ]

        if win_rate is not None:
            fields.append({"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True})

        if suggested_actions:
            actions_text = "\n".join(f"{i+1}. {a}" for i, a in enumerate(suggested_actions))
            fields.append({"name": "Suggested Actions", "value": actions_text, "inline": False})

        embed = {
            "title": f"Strategy Recommendation: {title}",
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": "Ref: analysis/walkforward_findings_2026-01-30.log"},
        }

        return await self.send_message(
            content=f"**Strategy Alert: {title}**",
            embeds=[embed],
        )
