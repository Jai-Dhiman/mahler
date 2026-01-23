from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from core.types import (
    Confidence,
    DailyPerformance,
    PlaybookRule,
    Position,
    Recommendation,
    RecommendationStatus,
    SpreadType,
    Trade,
    TradeStatus,
)


def js_to_python(obj):
    """Convert JsProxy objects to Python equivalents."""
    from pyodide.ffi import JsProxy

    if isinstance(obj, JsProxy):
        # Check if it's array-like
        if hasattr(obj, "to_py"):
            return obj.to_py()
        # Check if it's an object with properties
        try:
            return {k: js_to_python(getattr(obj, k)) for k in dir(obj) if not k.startswith("_")}
        except Exception:
            return str(obj)
    return obj


def sanitize_params(params: list | None) -> list | None:
    """Sanitize parameters for D1 binding.

    Ensures all values are valid types that can be bound to D1:
    - Converts JsProxy objects to Python equivalents
    - Ensures numeric types are proper Python floats/ints

    Note: Python None is passed through directly. For queries where None
    values cause issues, use dynamic SQL construction (like close_trade does).
    """
    if params is None:
        return None

    # Check if we're in Pyodide environment
    JsProxy = None
    try:
        from pyodide.ffi import JsProxy as _JsProxy
        # Verify it's actually a class/type (not a mock)
        if isinstance(_JsProxy, type):
            JsProxy = _JsProxy
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    # If not in Pyodide (JsProxy is None), no sanitization needed
    if JsProxy is None:
        return params

    sanitized = []
    for val in params:
        try:
            is_js_proxy = isinstance(val, JsProxy)
        except TypeError:
            # isinstance failed (JsProxy might be invalid type)
            is_js_proxy = False

        if is_js_proxy:
            # Convert JsProxy to Python
            py_val = val.to_py() if hasattr(val, "to_py") else None
            sanitized.append(py_val)
        elif isinstance(val, (str, int, float, bool, bytes)) or val is None:
            sanitized.append(val)
        else:
            # Try to convert to a basic type
            try:
                sanitized.append(str(val) if val is not None else None)
            except Exception:
                sanitized.append(None)
    return sanitized


class D1Client:
    """Client for Cloudflare D1 SQLite database operations."""

    def __init__(self, db_binding: Any):
        self.db = db_binding

    async def execute(self, query: str, params: list | None = None, retries: int = 3) -> Any:
        """Execute a query and return results.

        Includes retry logic for transient D1 failures (per Cloudflare documentation).
        """
        import asyncio

        last_error = None
        for attempt in range(retries):
            try:
                if params:
                    # Sanitize params to ensure valid D1 types
                    safe_params = sanitize_params(params)
                    result = await self.db.prepare(query).bind(*safe_params).all()
                else:
                    result = await self.db.prepare(query).all()

                # Convert the results to Python
                return js_to_python(result)
            except Exception as e:
                last_error = e
                error_msg = str(e) if e else "null"
                error_type = type(e).__name__

                if attempt < retries - 1:
                    print(f"D1 execute error (attempt {attempt + 1}/{retries}): {error_type}: {error_msg}")
                    await asyncio.sleep(0.1 * (2 ** attempt))
                else:
                    print(f"D1 execute error (final): type={error_type}, str={e}")

        raise last_error

    async def run(self, query: str, params: list | None = None, retries: int = 3) -> Any:
        """Execute a query without returning results (INSERT, UPDATE, DELETE).

        Includes retry logic for transient D1 failures (per Cloudflare documentation).
        """
        import asyncio

        last_error = None
        for attempt in range(retries):
            try:
                if params:
                    # Sanitize params to ensure valid D1 types
                    safe_params = sanitize_params(params)
                    return await self.db.prepare(query).bind(*safe_params).run()
                return await self.db.prepare(query).run()
            except Exception as e:
                last_error = e
                # Extract error details - handle both Python and JavaScript exceptions
                error_msg = str(e) if e else "null"
                error_type = type(e).__name__
                # Try to get .message attribute (JavaScript error convention)
                if hasattr(e, "message"):
                    error_msg = f"{error_msg} (message: {e.message})"

                if attempt < retries - 1:
                    print(f"D1 run error (attempt {attempt + 1}/{retries}): {error_type}: {error_msg}")
                    # Exponential backoff: 100ms, 200ms, 400ms...
                    await asyncio.sleep(0.1 * (2 ** attempt))
                else:
                    # Final attempt failed
                    print(f"D1 run error (final): type={error_type}, str={e}, repr={repr(e)}")
                    if params:
                        print(f"D1 query: {query[:100]}...")
                        # Don't log sanitized params as they may contain sensitive data

        # Re-raise the last error
        raise last_error

    # Recommendations

    async def create_recommendation(
        self,
        underlying: str,
        spread_type: SpreadType,
        short_strike: float,
        long_strike: float,
        expiration: str,
        credit: float,
        max_loss: float,
        expires_at: datetime,
        iv_rank: float | None = None,
        delta: float | None = None,
        theta: float | None = None,
        thesis: str | None = None,
        confidence: Confidence | None = None,
        suggested_contracts: int | None = None,
        analysis_price: float | None = None,
    ) -> str:
        """Create a new recommendation and return its ID."""
        rec_id = str(uuid4())
        await self.run(
            """
            INSERT INTO recommendations (
                id, expires_at, underlying, spread_type, short_strike, long_strike,
                expiration, credit, max_loss, iv_rank, delta, theta, thesis,
                confidence, suggested_contracts, analysis_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                rec_id,
                expires_at.isoformat(),
                underlying,
                spread_type.value,
                short_strike,
                long_strike,
                expiration,
                credit,
                max_loss,
                iv_rank,
                delta,
                theta,
                thesis,
                confidence.value if confidence else None,
                suggested_contracts,
                analysis_price,
            ],
        )
        return rec_id

    async def get_recommendation(self, rec_id: str) -> Recommendation | None:
        """Get a recommendation by ID."""
        result = await self.execute("SELECT * FROM recommendations WHERE id = ?", [rec_id])
        if not result["results"]:
            return None
        return self._row_to_recommendation(result["results"][0])

    async def get_pending_recommendations(self) -> list[Recommendation]:
        """Get all pending recommendations."""
        result = await self.execute(
            "SELECT * FROM recommendations WHERE status = 'pending' ORDER BY created_at DESC"
        )
        return [self._row_to_recommendation(row) for row in result["results"]]

    async def update_recommendation_status(self, rec_id: str, status: RecommendationStatus) -> None:
        """Update recommendation status."""
        await self.run(
            "UPDATE recommendations SET status = ? WHERE id = ?",
            [status.value, rec_id],
        )

    async def set_recommendation_discord_message_id(self, rec_id: str, message_id: str) -> None:
        """Set the Discord message ID for a recommendation."""
        await self.run(
            "UPDATE recommendations SET discord_message_id = ? WHERE id = ?",
            [message_id, rec_id],
        )

    def _row_to_recommendation(self, row: dict) -> Recommendation:
        return Recommendation(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            status=RecommendationStatus(row["status"]),
            underlying=row["underlying"],
            spread_type=SpreadType(row["spread_type"]),
            short_strike=row["short_strike"],
            long_strike=row["long_strike"],
            expiration=row["expiration"],
            credit=row["credit"],
            max_loss=row["max_loss"],
            iv_rank=row["iv_rank"],
            delta=row["delta"],
            theta=row["theta"],
            thesis=row["thesis"],
            confidence=Confidence(row["confidence"]) if row["confidence"] else None,
            suggested_contracts=row["suggested_contracts"],
            analysis_price=row["analysis_price"],
            discord_message_id=row["discord_message_id"],
        )

    # Trades

    async def create_trade(
        self,
        recommendation_id: str | None,
        underlying: str,
        spread_type: SpreadType,
        short_strike: float,
        long_strike: float,
        expiration: str,
        entry_credit: float,
        contracts: int,
        broker_order_id: str | None = None,
        status: TradeStatus = TradeStatus.OPEN,
    ) -> str:
        """Create a new trade and return its ID.

        Args:
            status: Initial trade status. Use PENDING_FILL for auto-approved orders
                    that haven't been confirmed filled yet.
        """
        trade_id = str(uuid4())
        await self.run(
            """
            INSERT INTO trades (
                id, recommendation_id, opened_at, status, underlying, spread_type,
                short_strike, long_strike, expiration, entry_credit, contracts, broker_order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                trade_id,
                recommendation_id,
                datetime.now().isoformat(),
                status.value,
                underlying,
                spread_type.value,
                short_strike,
                long_strike,
                expiration,
                entry_credit,
                contracts,
                broker_order_id,
            ],
        )
        return trade_id

    async def get_trade(self, trade_id: str) -> Trade | None:
        """Get a trade by ID."""
        result = await self.execute("SELECT * FROM trades WHERE id = ?", [trade_id])
        if not result["results"]:
            return None
        return self._row_to_trade(result["results"][0])

    async def get_open_trades(self) -> list[Trade]:
        """Get all open trades."""
        result = await self.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY opened_at DESC"
        )
        return [self._row_to_trade(row) for row in result["results"]]

    async def update_trade_status(self, trade_id: str, status: TradeStatus) -> None:
        """Update trade status."""
        await self.run(
            "UPDATE trades SET status = ? WHERE id = ?",
            [status.value, trade_id],
        )

    async def update_trade_order_id(self, trade_id: str, new_order_id: str) -> None:
        """Update the broker order ID for a trade (used when order is replaced)."""
        await self.run(
            "UPDATE trades SET broker_order_id = ? WHERE id = ?",
            [new_order_id, trade_id],
        )

    async def mark_trade_filled(self, trade_id: str) -> None:
        """Mark a pending_fill trade as open (filled)."""
        await self.run(
            "UPDATE trades SET status = 'open', opened_at = ? WHERE id = ?",
            [datetime.now().isoformat(), trade_id],
        )

    async def get_pending_fill_trades(self) -> list[Trade]:
        """Get all trades awaiting fill confirmation."""
        result = await self.execute(
            "SELECT * FROM trades WHERE status = 'pending_fill' ORDER BY opened_at DESC"
        )
        return [self._row_to_trade(row) for row in result["results"]]

    async def set_exit_order_id(self, trade_id: str, exit_order_id: str) -> None:
        """Set the exit order ID for a trade.

        This should be called immediately after placing an exit order,
        before attempting to close the trade. This enables reconciliation
        if the close_trade call fails.
        """
        await self.run(
            "UPDATE trades SET exit_order_id = ? WHERE id = ?",
            [exit_order_id, trade_id],
        )

    async def get_trades_with_pending_exits(self) -> list[Trade]:
        """Get trades that have an exit order but are still open.

        These are trades where an exit order was placed but the database
        update to close the trade failed. Used for reconciliation.
        """
        result = await self.execute(
            """SELECT * FROM trades
               WHERE status = 'open' AND exit_order_id IS NOT NULL
               ORDER BY opened_at DESC"""
        )
        return [self._row_to_trade(row) for row in result["results"]]

    async def clear_exit_order_id(self, trade_id: str) -> None:
        """Clear the exit order ID for a trade.

        Used when an exit order expires/cancels without filling.
        """
        await self.run(
            "UPDATE trades SET exit_order_id = NULL WHERE id = ?",
            [trade_id],
        )

    async def close_trade(
        self,
        trade_id: str,
        exit_debit: float,
        reflection: str | None = None,
        lesson: str | None = None,
        exit_reason: str | None = None,
        iv_rank_at_exit: float | None = None,
        dte_at_exit: int | None = None,
    ) -> None:
        """Close a trade with exit details and analytics.

        Dynamically builds SQL to avoid passing None through FFI (which becomes
        JavaScript undefined instead of null, causing D1_TYPE_ERROR).
        """
        trade = await self.get_trade(trade_id)
        if not trade:
            raise ValueError(f"Trade {trade_id} not found")

        profit_loss = (trade.entry_credit - exit_debit) * trade.contracts * 100

        # Build dynamic SQL to avoid None values (FFI converts None to undefined, not null)
        # Required fields (never None)
        set_clauses = [
            "status = 'closed'",
            "closed_at = ?",
            "exit_debit = ?",
            "profit_loss = ?",
        ]
        params = [
            datetime.now().isoformat(),
            exit_debit,
            profit_loss,
        ]

        # Optional fields - only include if not None
        if reflection is not None:
            set_clauses.append("reflection = ?")
            params.append(reflection)
        if lesson is not None:
            set_clauses.append("lesson = ?")
            params.append(lesson)
        if exit_reason is not None:
            set_clauses.append("exit_reason = ?")
            params.append(exit_reason)
        if iv_rank_at_exit is not None:
            set_clauses.append("iv_rank_at_exit = ?")
            params.append(iv_rank_at_exit)
        if dte_at_exit is not None:
            set_clauses.append("dte_at_exit = ?")
            params.append(dte_at_exit)

        params.append(trade_id)

        query = f"UPDATE trades SET {', '.join(set_clauses)} WHERE id = ?"
        await self.run(query, params)

    def _row_to_trade(self, row: dict) -> Trade:
        return Trade(
            id=row["id"],
            recommendation_id=row["recommendation_id"],
            opened_at=datetime.fromisoformat(row["opened_at"]) if row["opened_at"] else None,
            closed_at=datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None,
            status=TradeStatus(row["status"]),
            underlying=row["underlying"],
            spread_type=SpreadType(row["spread_type"]),
            short_strike=row["short_strike"],
            long_strike=row["long_strike"],
            expiration=row["expiration"],
            entry_credit=row["entry_credit"],
            exit_debit=row["exit_debit"],
            profit_loss=row["profit_loss"],
            contracts=row["contracts"],
            broker_order_id=row["broker_order_id"],
            exit_order_id=row.get("exit_order_id"),
            reflection=row["reflection"],
            lesson=row["lesson"],
        )

    # Positions

    async def upsert_position(
        self,
        trade_id: str,
        underlying: str,
        short_strike: float,
        long_strike: float,
        expiration: str,
        contracts: int,
        current_value: float,
        unrealized_pnl: float,
    ) -> str:
        """Create or update a position snapshot."""
        existing = await self.execute("SELECT id FROM positions WHERE trade_id = ?", [trade_id])
        if existing["results"]:
            pos_id = existing["results"][0]["id"]
            await self.run(
                """
                UPDATE positions
                SET current_value = ?, unrealized_pnl = ?, updated_at = ?
                WHERE id = ?
                """,
                [current_value, unrealized_pnl, datetime.now().isoformat(), pos_id],
            )
            return pos_id

        pos_id = str(uuid4())
        await self.run(
            """
            INSERT INTO positions (
                id, trade_id, underlying, short_strike, long_strike, expiration,
                contracts, current_value, unrealized_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                pos_id,
                trade_id,
                underlying,
                short_strike,
                long_strike,
                expiration,
                contracts,
                current_value,
                unrealized_pnl,
            ],
        )
        return pos_id

    async def delete_position(self, trade_id: str) -> None:
        """Delete position for a closed trade."""
        await self.run("DELETE FROM positions WHERE trade_id = ?", [trade_id])

    async def get_all_positions(self) -> list[Position]:
        """Get all current positions."""
        result = await self.execute("SELECT * FROM positions ORDER BY updated_at DESC")
        return [self._row_to_position(row) for row in result["results"]]

    def _row_to_position(self, row: dict) -> Position:
        return Position(
            id=row["id"],
            trade_id=row["trade_id"],
            underlying=row["underlying"],
            short_strike=row["short_strike"],
            long_strike=row["long_strike"],
            expiration=row["expiration"],
            contracts=row["contracts"],
            current_value=row["current_value"],
            unrealized_pnl=row["unrealized_pnl"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # Daily Performance

    async def get_or_create_daily_performance(
        self, date: str, starting_balance: float
    ) -> DailyPerformance:
        """Get or create daily performance record."""
        result = await self.execute("SELECT * FROM daily_performance WHERE date = ?", [date])
        if result["results"]:
            return self._row_to_daily_performance(result["results"][0])

        await self.run(
            """
            INSERT INTO daily_performance (date, starting_balance, ending_balance, realized_pnl)
            VALUES (?, ?, ?, 0)
            """,
            [date, starting_balance, starting_balance],
        )
        return DailyPerformance(
            date=date,
            starting_balance=starting_balance,
            ending_balance=starting_balance,
            realized_pnl=0,
        )

    async def update_daily_performance(
        self,
        date: str,
        ending_balance: float | None = None,
        realized_pnl_delta: float = 0,
        trades_opened_delta: int = 0,
        trades_closed_delta: int = 0,
        win_delta: int = 0,
        loss_delta: int = 0,
    ) -> None:
        """Update daily performance metrics.

        Creates the row if it doesn't exist (with starting_balance=0, to be filled by EOD).
        """
        print(f"[DEBUG] update_daily_performance called: date={date}, trades_opened_delta={trades_opened_delta}, trades_closed_delta={trades_closed_delta}")

        # Ensure row exists first (INSERT OR IGNORE won't overwrite existing)
        await self.run(
            """
            INSERT OR IGNORE INTO daily_performance
                (date, starting_balance, ending_balance, realized_pnl)
            VALUES (?, 0, 0, 0)
            """,
            [date],
        )

        updates = []
        params = []

        if ending_balance is not None:
            updates.append("ending_balance = ?")
            params.append(ending_balance)
        if realized_pnl_delta:
            updates.append("realized_pnl = realized_pnl + ?")
            params.append(realized_pnl_delta)
        if trades_opened_delta:
            updates.append("trades_opened = trades_opened + ?")
            params.append(trades_opened_delta)
        if trades_closed_delta:
            updates.append("trades_closed = trades_closed + ?")
            params.append(trades_closed_delta)
        if win_delta:
            updates.append("win_count = win_count + ?")
            params.append(win_delta)
        if loss_delta:
            updates.append("loss_count = loss_count + ?")
            params.append(loss_delta)

        if updates:
            params.append(date)
            query = f"UPDATE daily_performance SET {', '.join(updates)} WHERE date = ?"
            print(f"[DEBUG] Running update query: {query} with params: {params}")
            await self.run(query, params)
            print(f"[DEBUG] update_daily_performance completed for {date}")

    def _row_to_daily_performance(self, row: dict) -> DailyPerformance:
        return DailyPerformance(
            date=row["date"],
            starting_balance=row["starting_balance"],
            ending_balance=row["ending_balance"],
            realized_pnl=row["realized_pnl"],
            trades_opened=row["trades_opened"],
            trades_closed=row["trades_closed"],
            win_count=row["win_count"],
            loss_count=row["loss_count"],
        )

    # Playbook

    async def get_playbook_rules(self) -> list[PlaybookRule]:
        """Get all playbook rules."""
        result = await self.execute("SELECT * FROM playbook ORDER BY created_at")
        return [self._row_to_playbook_rule(row) for row in result["results"]]

    async def add_playbook_rule(
        self, rule: str, source: str = "learned", supporting_trade_ids: list[str] | None = None
    ) -> str:
        """Add a new playbook rule."""
        import json

        rule_id = str(uuid4())
        await self.run(
            "INSERT INTO playbook (id, rule, source, supporting_trade_ids) VALUES (?, ?, ?, ?)",
            [rule_id, rule, source, json.dumps(supporting_trade_ids or [])],
        )
        return rule_id

    def _row_to_playbook_rule(self, row: dict) -> PlaybookRule:
        import json

        return PlaybookRule(
            id=row["id"],
            rule=row["rule"],
            source=row["source"],
            supporting_trade_ids=json.loads(row["supporting_trade_ids"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
        )

    # Stats

    async def get_trade_stats(self) -> dict:
        """Get aggregate trade statistics."""
        result = await self.execute(
            """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_trades,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as total_profit,
                SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) ELSE 0 END) as total_loss,
                SUM(profit_loss) as net_pnl
            FROM trades
            """
        )
        row = result["results"][0] if result["results"] else {}
        wins = row.get("wins") or 0
        losses = row.get("losses") or 0
        total_profit = row.get("total_profit") or 0
        total_loss = row.get("total_loss") or 1  # Avoid division by zero

        return {
            "total_trades": row.get("total_trades") or 0,
            "closed_trades": row.get("closed_trades") or 0,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else 0,
            "net_pnl": row.get("net_pnl") or 0,
        }

    # IV History

    async def save_daily_iv(
        self,
        date: str,
        underlying: str,
        atm_iv: float,
        underlying_price: float | None = None,
    ) -> str:
        """Save daily IV observation for an underlying.

        Args:
            date: Date in YYYY-MM-DD format
            underlying: Symbol (e.g., SPY, QQQ)
            atm_iv: At-the-money implied volatility (decimal, e.g., 0.20 for 20%)
            underlying_price: Optional underlying price at observation time
        """
        iv_id = str(uuid4())
        await self.run(
            """
            INSERT INTO iv_history (id, date, underlying, atm_iv, underlying_price)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date, underlying) DO UPDATE SET
                atm_iv = excluded.atm_iv,
                underlying_price = excluded.underlying_price
            """,
            [iv_id, date, underlying, atm_iv, underlying_price],
        )
        return iv_id

    async def get_iv_history(
        self,
        underlying: str,
        lookback_days: int = 252,
    ) -> list[float]:
        """Get historical IV values for an underlying.

        Args:
            underlying: Symbol (e.g., SPY)
            lookback_days: Number of trading days to look back (default 252 = 1 year)

        Returns:
            List of IV values, most recent first
        """
        result = await self.execute(
            """
            SELECT atm_iv FROM iv_history
            WHERE underlying = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            [underlying, lookback_days],
        )
        return [row["atm_iv"] for row in result["results"]]

    async def get_iv_history_count(self, underlying: str) -> int:
        """Get count of IV history records for an underlying."""
        result = await self.execute(
            "SELECT COUNT(*) as count FROM iv_history WHERE underlying = ?",
            [underlying],
        )
        return result["results"][0]["count"] if result["results"] else 0

    # VIX History

    async def save_daily_vix(
        self,
        date: str,
        vix_close: float,
        vix3m_close: float | None = None,
    ) -> str:
        """Save daily VIX observation.

        Dynamically builds SQL to avoid passing None through FFI (which becomes
        JavaScript undefined instead of null, causing D1_TYPE_ERROR).

        Args:
            date: Date in YYYY-MM-DD format
            vix_close: VIX closing value
            vix3m_close: VIX3M closing value (optional, for term structure)
        """
        vix_id = str(uuid4())
        term_structure_ratio = vix_close / vix3m_close if vix3m_close else None

        # Build dynamic SQL to avoid None values (FFI converts None to undefined, not null)
        # Required columns (never None)
        columns = ["id", "date", "vix_close"]
        values = [vix_id, date, vix_close]
        placeholders = ["?", "?", "?"]
        update_clauses = ["vix_close = excluded.vix_close"]

        # Optional columns - only include if not None
        if vix3m_close is not None:
            columns.append("vix3m_close")
            values.append(vix3m_close)
            placeholders.append("?")
            update_clauses.append("vix3m_close = excluded.vix3m_close")

        if term_structure_ratio is not None:
            columns.append("term_structure_ratio")
            values.append(term_structure_ratio)
            placeholders.append("?")
            update_clauses.append("term_structure_ratio = excluded.term_structure_ratio")

        query = f"""
            INSERT INTO vix_history ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT(date) DO UPDATE SET
                {', '.join(update_clauses)}
        """
        await self.run(query, values)
        return vix_id

    async def get_latest_vix(self) -> dict | None:
        """Get the most recent VIX observation."""
        result = await self.execute(
            "SELECT * FROM vix_history ORDER BY date DESC LIMIT 1"
        )
        if not result["results"]:
            return None
        row = result["results"][0]
        return {
            "date": row["date"],
            "vix_close": row["vix_close"],
            "vix3m_close": row["vix3m_close"],
            "term_structure_ratio": row["term_structure_ratio"],
        }

    async def get_vix_history(self, lookback_days: int = 252) -> list[dict]:
        """Get VIX history for lookback period."""
        result = await self.execute(
            """
            SELECT date, vix_close, vix3m_close, term_structure_ratio
            FROM vix_history
            ORDER BY date DESC
            LIMIT ?
            """,
            [lookback_days],
        )
        return [
            {
                "date": row["date"],
                "vix_close": row["vix_close"],
                "vix3m_close": row["vix3m_close"],
                "term_structure_ratio": row["term_structure_ratio"],
            }
            for row in result["results"]
        ]

    # Market Regimes

    async def save_market_regime(
        self,
        symbol: str,
        regime: str,
        probability: float,
        position_multiplier: float,
        features: dict,
        detected_at: str,
    ) -> str:
        """Save market regime detection result.

        Args:
            symbol: Underlying symbol (e.g., SPY)
            regime: Regime name (bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol)
            probability: Confidence in regime (0-1)
            position_multiplier: Position sizing multiplier
            features: Dict of feature values used
            detected_at: ISO timestamp

        Returns:
            Generated regime ID
        """
        import json

        regime_id = str(uuid4())
        await self.run(
            """
            INSERT INTO market_regimes (
                id, symbol, regime, probability, position_multiplier, features, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                regime_id,
                symbol,
                regime,
                probability,
                position_multiplier,
                json.dumps(features),
                detected_at,
            ],
        )
        return regime_id

    async def get_regime_history(
        self,
        symbol: str,
        lookback_days: int = 30,
    ) -> list[dict]:
        """Get regime detection history for analysis.

        Args:
            symbol: Underlying symbol
            lookback_days: Number of days to look back

        Returns:
            List of regime records, most recent first
        """
        import json

        result = await self.execute(
            """
            SELECT * FROM market_regimes
            WHERE symbol = ?
              AND detected_at >= date('now', '-' || ? || ' days')
            ORDER BY detected_at DESC
            """,
            [symbol, lookback_days],
        )

        return [
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "regime": row["regime"],
                "probability": row["probability"],
                "position_multiplier": row["position_multiplier"],
                "features": json.loads(row["features"]) if row["features"] else {},
                "detected_at": row["detected_at"],
            }
            for row in result["results"]
        ]

    async def get_latest_regime(self, symbol: str) -> dict | None:
        """Get the most recent regime detection for a symbol."""
        import json

        result = await self.execute(
            """
            SELECT * FROM market_regimes
            WHERE symbol = ?
            ORDER BY detected_at DESC
            LIMIT 1
            """,
            [symbol],
        )

        if not result["results"]:
            return None

        row = result["results"][0]
        return {
            "id": row["id"],
            "symbol": row["symbol"],
            "regime": row["regime"],
            "probability": row["probability"],
            "position_multiplier": row["position_multiplier"],
            "features": json.loads(row["features"]) if row["features"] else {},
            "detected_at": row["detected_at"],
        }

    # AI Confidence Calibration

    async def get_confidence_calibration(self, lookback_days: int = 90) -> dict:
        """Calculate AI confidence calibration by comparing confidence to actual outcomes.

        Returns:
            Dict with calibration metrics per confidence level
        """
        result = await self.execute(
            """
            SELECT
                r.confidence,
                COUNT(*) as total,
                SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN t.profit_loss < 0 THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN t.profit_loss = 0 THEN 1 ELSE 0 END) as breakeven
            FROM recommendations r
            JOIN trades t ON t.recommendation_id = r.id
            WHERE t.status = 'closed'
                AND t.closed_at >= date('now', '-' || ? || ' days')
                AND r.confidence IS NOT NULL
            GROUP BY r.confidence
            """,
            [lookback_days],
        )

        calibration = {}
        for row in result["results"]:
            confidence = row["confidence"]
            total = row["total"] or 0
            wins = row["wins"] or 0
            losses = row["losses"] or 0

            win_rate = wins / total if total > 0 else 0

            # Expected win rate based on confidence level
            expected_win_rate = {
                "low": 0.50,  # 50% expected
                "medium": 0.65,  # 65% expected
                "high": 0.80,  # 80% expected
            }.get(confidence, 0.50)

            calibration_gap = win_rate - expected_win_rate

            calibration[confidence] = {
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "actual_win_rate": win_rate,
                "expected_win_rate": expected_win_rate,
                "calibration_gap": calibration_gap,
                "is_calibrated": abs(calibration_gap) <= 0.10,  # Within 10%
            }

        return calibration

    async def get_rolling_calibration_stats(self, lookback_days: int = 30) -> dict:
        """Get rolling calibration stats for the last N days.

        Used for detecting calibration drift.
        """
        result = await self.execute(
            """
            SELECT
                r.confidence,
                COUNT(*) as total,
                SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) as wins
            FROM recommendations r
            JOIN trades t ON t.recommendation_id = r.id
            WHERE t.status = 'closed'
                AND t.closed_at >= date('now', '-' || ? || ' days')
                AND r.confidence IS NOT NULL
            GROUP BY r.confidence
            """,
            [lookback_days],
        )

        stats = {
            "period_days": lookback_days,
            "by_confidence": {},
            "overall_win_rate": 0,
            "total_trades": 0,
        }

        total_wins = 0
        total_trades = 0

        for row in result["results"]:
            confidence = row["confidence"]
            total = row["total"] or 0
            wins = row["wins"] or 0

            total_trades += total
            total_wins += wins

            stats["by_confidence"][confidence] = {
                "total": total,
                "wins": wins,
                "win_rate": wins / total if total > 0 else 0,
            }

        stats["total_trades"] = total_trades
        stats["overall_win_rate"] = total_wins / total_trades if total_trades > 0 else 0

        return stats

    # Dynamic Betas

    async def save_dynamic_beta(
        self,
        symbol: str,
        beta_ewma: float,
        beta_rolling_20: float | None,
        beta_rolling_60: float | None,
        beta_blended: float,
        correlation_spy: float | None,
        data_days: int,
    ) -> str:
        """Save dynamic beta calculation result.

        Args:
            symbol: Underlying symbol
            beta_ewma: EWMA beta
            beta_rolling_20: 20-day rolling beta
            beta_rolling_60: 60-day rolling beta
            beta_blended: Blended beta value
            correlation_spy: Correlation with SPY
            data_days: Number of days of data used

        Returns:
            ID of the saved record
        """
        from uuid import uuid4

        beta_id = str(uuid4())
        await self.run(
            """
            INSERT INTO dynamic_betas
            (id, symbol, beta_ewma, beta_rolling_20, beta_rolling_60,
             beta_blended, correlation_spy, data_days, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                beta_id,
                symbol,
                beta_ewma,
                beta_rolling_20,
                beta_rolling_60,
                beta_blended,
                correlation_spy,
                data_days,
            ],
        )
        return beta_id

    async def get_latest_dynamic_beta(self, symbol: str) -> dict | None:
        """Get most recent dynamic beta for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            Beta record or None if not found
        """
        result = await self.execute(
            """
            SELECT * FROM dynamic_betas
            WHERE symbol = ?
            ORDER BY calculated_at DESC
            LIMIT 1
            """,
            [symbol],
        )

        if not result.get("results"):
            return None

        row = result["results"][0]
        return {
            "id": row["id"],
            "symbol": row["symbol"],
            "beta_ewma": row["beta_ewma"],
            "beta_rolling_20": row["beta_rolling_20"],
            "beta_rolling_60": row["beta_rolling_60"],
            "beta_blended": row["beta_blended"],
            "correlation_spy": row["correlation_spy"],
            "data_days": row["data_days"],
            "calculated_at": row["calculated_at"],
        }

    async def get_all_dynamic_betas(self) -> dict[str, dict]:
        """Get most recent dynamic beta for each symbol.

        Returns:
            Dict of symbol -> beta record
        """
        result = await self.execute(
            """
            SELECT DISTINCT symbol,
                FIRST_VALUE(id) OVER w AS id,
                FIRST_VALUE(beta_ewma) OVER w AS beta_ewma,
                FIRST_VALUE(beta_rolling_20) OVER w AS beta_rolling_20,
                FIRST_VALUE(beta_rolling_60) OVER w AS beta_rolling_60,
                FIRST_VALUE(beta_blended) OVER w AS beta_blended,
                FIRST_VALUE(correlation_spy) OVER w AS correlation_spy,
                FIRST_VALUE(data_days) OVER w AS data_days,
                FIRST_VALUE(calculated_at) OVER w AS calculated_at
            FROM dynamic_betas
            WINDOW w AS (PARTITION BY symbol ORDER BY calculated_at DESC)
            """
        )

        return {
            row["symbol"]: {
                "id": row["id"],
                "symbol": row["symbol"],
                "beta_ewma": row["beta_ewma"],
                "beta_rolling_20": row["beta_rolling_20"],
                "beta_rolling_60": row["beta_rolling_60"],
                "beta_blended": row["beta_blended"],
                "correlation_spy": row["correlation_spy"],
                "data_days": row["data_days"],
                "calculated_at": row["calculated_at"],
            }
            for row in result.get("results", [])
        }

    # Optimized Weights

    async def save_optimized_weights(
        self,
        regime: str,
        weight_iv: float,
        weight_delta: float,
        weight_credit: float,
        weight_ev: float,
        sharpe_ratio: float | None,
        n_trades: int,
    ) -> str:
        """Save optimized scoring weights for a regime.

        Args:
            regime: Market regime
            weight_iv: IV score weight
            weight_delta: Delta score weight
            weight_credit: Credit score weight
            weight_ev: Expected value score weight
            sharpe_ratio: Sharpe ratio achieved
            n_trades: Number of trades used

        Returns:
            ID of the saved record
        """
        from uuid import uuid4

        weights_id = str(uuid4())
        await self.run(
            """
            INSERT INTO optimized_weights
            (id, regime, weight_iv, weight_delta, weight_credit, weight_ev,
             sharpe_ratio, n_trades, optimized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                weights_id,
                regime,
                weight_iv,
                weight_delta,
                weight_credit,
                weight_ev,
                sharpe_ratio,
                n_trades,
            ],
        )
        return weights_id

    async def get_latest_optimized_weights(self) -> dict[str, dict]:
        """Get most recent optimized weights for each regime.

        Returns:
            Dict of regime -> weights dict
        """
        result = await self.execute(
            """
            SELECT DISTINCT regime,
                FIRST_VALUE(weight_iv) OVER w AS weight_iv,
                FIRST_VALUE(weight_delta) OVER w AS weight_delta,
                FIRST_VALUE(weight_credit) OVER w AS weight_credit,
                FIRST_VALUE(weight_ev) OVER w AS weight_ev,
                FIRST_VALUE(sharpe_ratio) OVER w AS sharpe_ratio,
                FIRST_VALUE(n_trades) OVER w AS n_trades,
                FIRST_VALUE(optimized_at) OVER w AS optimized_at
            FROM optimized_weights
            WINDOW w AS (PARTITION BY regime ORDER BY optimized_at DESC)
            """
        )

        return {
            row["regime"]: {
                "iv": row["weight_iv"],
                "delta": row["weight_delta"],
                "credit": row["weight_credit"],
                "ev": row["weight_ev"],
                "sharpe_ratio": row["sharpe_ratio"],
                "n_trades": row["n_trades"],
                "optimized_at": row["optimized_at"],
            }
            for row in result.get("results", [])
        }

    # Rule Validation

    async def tag_trade_with_rules(
        self,
        trade_id: str,
        rule_ids: list[str],
    ) -> None:
        """Tag a trade with the playbook rules that influenced it.

        Args:
            trade_id: The trade ID to tag
            rule_ids: List of rule IDs that influenced this trade
        """
        import json

        rule_ids_json = json.dumps(rule_ids)
        await self.run(
            "UPDATE trades SET applied_rule_ids = ? WHERE id = ?",
            [rule_ids_json, trade_id],
        )

    async def get_closed_trades_with_rules(
        self,
        lookback_days: int = 90,
    ) -> list[dict]:
        """Get closed trades with their applied rule IDs.

        Args:
            lookback_days: Number of days to look back

        Returns:
            List of trade dicts with profit_loss and applied_rule_ids
        """
        result = await self.execute(
            """
            SELECT id, profit_loss, applied_rule_ids, closed_at
            FROM trades
            WHERE status = 'closed'
              AND closed_at >= datetime('now', '-' || ? || ' days')
            ORDER BY closed_at DESC
            """,
            [lookback_days],
        )

        return [
            {
                "id": row["id"],
                "profit_loss": row["profit_loss"] or 0.0,
                "applied_rule_ids": row["applied_rule_ids"],
                "closed_at": row["closed_at"],
            }
            for row in result.get("results", [])
        ]

    async def save_rule_validation(
        self,
        rule_id: str,
        trades_with_rule: int,
        trades_without_rule: int,
        mean_pnl_with: float,
        mean_pnl_without: float,
        win_rate_with: float,
        win_rate_without: float,
        u_statistic: float,
        p_value: float,
        p_value_adjusted: float,
        is_significant: bool,
        effect_direction: str,
    ) -> str:
        """Save a rule validation result.

        Args:
            rule_id: The playbook rule ID
            trades_with_rule: Number of trades where rule was applied
            trades_without_rule: Number of trades where rule was not applied
            mean_pnl_with: Mean P/L for trades with rule
            mean_pnl_without: Mean P/L for trades without rule
            win_rate_with: Win rate for trades with rule
            win_rate_without: Win rate for trades without rule
            u_statistic: Mann-Whitney U statistic
            p_value: Raw p-value
            p_value_adjusted: FDR-corrected p-value
            is_significant: Whether the effect is statistically significant
            effect_direction: 'positive', 'negative', or 'neutral'

        Returns:
            ID of the saved validation
        """
        validation_id = str(uuid4())
        await self.run(
            """
            INSERT INTO rule_validations (
                id, rule_id, validated_at, trades_with_rule, trades_without_rule,
                mean_pnl_with, mean_pnl_without, win_rate_with, win_rate_without,
                u_statistic, p_value, p_value_adjusted, is_significant, effect_direction
            ) VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                validation_id,
                rule_id,
                trades_with_rule,
                trades_without_rule,
                mean_pnl_with,
                mean_pnl_without,
                win_rate_with,
                win_rate_without,
                u_statistic,
                p_value,
                p_value_adjusted,
                1 if is_significant else 0,
                effect_direction,
            ],
        )
        return validation_id

    async def update_playbook_validation_status(
        self,
        rule_id: str,
        is_validated: bool,
        p_value: float | None,
    ) -> None:
        """Update the validation status of a playbook rule.

        Args:
            rule_id: The playbook rule ID
            is_validated: Whether the rule has been validated as effective
            p_value: The adjusted p-value from validation
        """
        await self.run(
            """
            UPDATE playbook
            SET is_validated = ?, last_validated_at = datetime('now'), validation_p_value = ?
            WHERE id = ?
            """,
            [1 if is_validated else 0, p_value, rule_id],
        )

    async def get_rule_validations(
        self,
        rule_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get validation history for a rule.

        Args:
            rule_id: The playbook rule ID
            limit: Maximum number of validations to return

        Returns:
            List of validation result dicts
        """
        result = await self.execute(
            """
            SELECT * FROM rule_validations
            WHERE rule_id = ?
            ORDER BY validated_at DESC
            LIMIT ?
            """,
            [rule_id, limit],
        )

        return result.get("results", [])

    async def get_latest_rule_validations(self) -> list[dict]:
        """Get the most recent validation for each rule.

        Returns:
            List of validation result dicts (one per rule)
        """
        result = await self.execute(
            """
            SELECT rv.*, p.rule as rule_text
            FROM rule_validations rv
            JOIN playbook p ON rv.rule_id = p.id
            WHERE rv.validated_at = (
                SELECT MAX(rv2.validated_at)
                FROM rule_validations rv2
                WHERE rv2.rule_id = rv.rule_id
            )
            ORDER BY rv.is_significant DESC, rv.p_value_adjusted ASC
            """
        )

        return result.get("results", [])
