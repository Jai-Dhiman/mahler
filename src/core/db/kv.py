from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from core.types import CircuitBreakerStatus


class KVClient:
    """Client for Cloudflare KV state storage."""

    # Key prefixes
    CIRCUIT_BREAKER_KEY = "circuit_breaker"
    DAILY_KEY_PREFIX = "daily:"
    WEEKLY_KEY_PREFIX = "weekly:"
    RATE_LIMIT_PREFIX = "rate_limit:"
    REGIME_KEY_PREFIX = "regime:"
    LATEST_ENDING_EQUITY_KEY = "latest_ending_equity"

    def __init__(self, kv_binding: Any):
        self.kv = kv_binding

    async def get(self, key: str) -> str | None:
        """Get a value from KV."""
        return await self.kv.get(key)

    async def get_json(self, key: str) -> dict | None:
        """Get a JSON value from KV."""
        value = await self.kv.get(key)
        if value:
            return json.loads(value)
        return None

    async def put(self, key: str, value: str, expiration_ttl: int | None = None) -> None:
        """Put a value into KV with optional TTL in seconds."""
        options = {}
        if expiration_ttl:
            options["expirationTtl"] = expiration_ttl
        await self.kv.put(key, value, options)

    async def put_json(self, key: str, value: dict, expiration_ttl: int | None = None) -> None:
        """Put a JSON value into KV."""
        await self.put(key, json.dumps(value), expiration_ttl)

    async def delete(self, key: str) -> None:
        """Delete a key from KV."""
        await self.kv.delete(key)

    # Circuit Breaker

    async def get_circuit_breaker(self) -> CircuitBreakerStatus:
        """Get current circuit breaker status."""
        data = await self.get_json(self.CIRCUIT_BREAKER_KEY)
        if not data:
            return CircuitBreakerStatus.active()

        return CircuitBreakerStatus(
            halted=data.get("halted", False),
            reason=data.get("reason"),
            triggered_at=datetime.fromisoformat(data["triggered_at"])
            if data.get("triggered_at")
            else None,
        )

    async def trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker."""
        await self.put_json(
            self.CIRCUIT_BREAKER_KEY,
            {
                "halted": True,
                "reason": reason,
                "triggered_at": datetime.now().isoformat(),
            },
        )

    async def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        await self.put_json(self.CIRCUIT_BREAKER_KEY, {"halted": False})

    # Daily Limits

    def _daily_key(self, date: str | None = None) -> str:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return f"{self.DAILY_KEY_PREFIX}{date}"

    async def get_daily_stats(self, date: str | None = None) -> dict:
        """Get daily trading stats.

        Falls back to latest_ending_equity when starting_equity is 0.0
        (e.g., on Mondays when Friday's EOD wrote to Saturday's key).
        """
        defaults = {
            "trades_count": 0,
            "realized_pnl": 0.0,
            "losses_today": 0.0,
            "last_loss_time": None,
            "rapid_loss_amount": 0.0,
            "starting_equity": 0.0,
        }
        data = await self.get_json(self._daily_key(date))
        if data:
            # Merge defaults with existing data to handle schema evolution
            stats = {**defaults, **data}
        else:
            stats = defaults

        # Fall back to latest_ending_equity if starting_equity is 0
        if stats["starting_equity"] == 0.0:
            latest = await self.get_latest_ending_equity()
            if latest is not None and latest > 0:
                stats["starting_equity"] = latest
        return stats

    async def update_daily_stats(
        self,
        trades_delta: int = 0,
        pnl_delta: float = 0.0,
        date: str | None = None,
    ) -> dict:
        """Update daily trading stats."""
        stats = await self.get_daily_stats(date)
        stats["trades_count"] += trades_delta
        stats["realized_pnl"] += pnl_delta

        if pnl_delta < 0:
            stats["losses_today"] += abs(pnl_delta)
            now = datetime.now()
            last_loss = stats.get("last_loss_time")

            # Track rapid losses (within 5 minutes)
            if last_loss:
                last_loss_dt = datetime.fromisoformat(last_loss)
                if now - last_loss_dt < timedelta(minutes=5):
                    stats["rapid_loss_amount"] += abs(pnl_delta)
                else:
                    stats["rapid_loss_amount"] = abs(pnl_delta)
            else:
                stats["rapid_loss_amount"] = abs(pnl_delta)

            stats["last_loss_time"] = now.isoformat()

        # TTL of 7 days for daily stats
        await self.put_json(self._daily_key(date), stats, expiration_ttl=7 * 24 * 3600)
        return stats

    async def get_latest_ending_equity(self) -> float | None:
        """Get the most recent ending equity (survives weekends/holidays)."""
        value = await self.get(self.LATEST_ENDING_EQUITY_KEY)
        if value is not None:
            return float(value)
        return None

    async def set_latest_ending_equity(self, equity: float) -> None:
        """Store the latest ending equity (no TTL - persists until next write)."""
        await self.put(self.LATEST_ENDING_EQUITY_KEY, str(equity))

    async def reset_daily_stats(self, date: str | None = None) -> None:
        """Reset daily stats (for new trading day)."""
        await self.delete(self._daily_key(date))

    # Weekly Limits

    def _get_week_key(self, date: datetime | None = None) -> str:
        """Get the ISO week key for a date (e.g., '2024-W01')."""
        if date is None:
            date = datetime.now()
        year, week, _ = date.isocalendar()
        return f"{self.WEEKLY_KEY_PREFIX}{year}-W{week:02d}"

    def _is_monday(self) -> bool:
        """Check if today is Monday."""
        return datetime.now().weekday() == 0

    async def get_weekly_stats(self, date: datetime | None = None) -> dict:
        """Get weekly trading stats."""
        defaults = {
            "starting_equity": 0.0,
            "trades_count": 0,
            "realized_pnl": 0.0,
            "initialized": False,
        }
        data = await self.get_json(self._get_week_key(date))
        if data:
            # Merge defaults with existing data to handle schema evolution
            return {**defaults, **data}
        return defaults

    async def initialize_weekly_stats(
        self,
        starting_equity: float,
        force: bool = False,
    ) -> dict:
        """Initialize weekly stats with starting equity.

        Should be called on Monday morning to set the baseline.
        If force=False, only initializes if not already set this week.

        Args:
            starting_equity: Account equity at start of week
            force: If True, reinitialize even if already set

        Returns:
            Current weekly stats
        """
        stats = await self.get_weekly_stats()

        if stats.get("initialized") and not force:
            return stats

        stats = {
            "starting_equity": starting_equity,
            "trades_count": 0,
            "realized_pnl": 0.0,
            "initialized": True,
            "initialized_at": datetime.now().isoformat(),
        }

        # TTL of 14 days (covers the week plus some buffer)
        await self.put_json(self._get_week_key(), stats, expiration_ttl=14 * 24 * 3600)
        return stats

    async def update_weekly_stats(
        self,
        trades_delta: int = 0,
        pnl_delta: float = 0.0,
    ) -> dict:
        """Update weekly trading stats."""
        stats = await self.get_weekly_stats()
        stats["trades_count"] += trades_delta
        stats["realized_pnl"] += pnl_delta

        await self.put_json(self._get_week_key(), stats, expiration_ttl=14 * 24 * 3600)
        return stats

    async def get_weekly_starting_equity(self) -> float:
        """Get the starting equity for the current week."""
        stats = await self.get_weekly_stats()
        return stats.get("starting_equity", 0.0)

    # Rate Limiting

    async def check_rate_limit(
        self, service: str, max_requests: int, window_seconds: int = 3600
    ) -> bool:
        """Check if rate limit is exceeded. Returns True if OK to proceed."""
        key = f"{self.RATE_LIMIT_PREFIX}{service}"
        data = await self.get_json(key)

        now = datetime.now()
        if not data:
            await self.put_json(
                key,
                {"count": 1, "window_start": now.isoformat()},
                expiration_ttl=window_seconds,
            )
            return True

        window_start = datetime.fromisoformat(data["window_start"])
        if now - window_start > timedelta(seconds=window_seconds):
            # Window expired, reset
            await self.put_json(
                key,
                {"count": 1, "window_start": now.isoformat()},
                expiration_ttl=window_seconds,
            )
            return True

        if data["count"] >= max_requests:
            return False

        data["count"] += 1
        remaining_ttl = window_seconds - int((now - window_start).total_seconds())
        await self.put_json(key, data, expiration_ttl=max(remaining_ttl, 1))
        return True

    async def increment_error_count(self, window_seconds: int = 60) -> int:
        """Increment API error count and return current count."""
        key = f"{self.RATE_LIMIT_PREFIX}errors"
        data = await self.get_json(key)

        now = datetime.now()
        if not data:
            await self.put_json(
                key,
                {"count": 1, "window_start": now.isoformat()},
                expiration_ttl=window_seconds,
            )
            return 1

        window_start = datetime.fromisoformat(data["window_start"])
        if now - window_start > timedelta(seconds=window_seconds):
            await self.put_json(
                key,
                {"count": 1, "window_start": now.isoformat()},
                expiration_ttl=window_seconds,
            )
            return 1

        data["count"] += 1
        remaining_ttl = window_seconds - int((now - window_start).total_seconds())
        await self.put_json(key, data, expiration_ttl=max(remaining_ttl, 1))
        return data["count"]

    # Market Regime Caching

    def _regime_key(self, symbol: str, date: str, hour: int) -> str:
        """Get cache key for regime detection.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            date: Date string (YYYY-MM-DD)
            hour: Hour of day (0-23)

        Returns:
            Cache key (e.g., "regime:SPY:2024-12-29:10")
        """
        return f"{self.REGIME_KEY_PREFIX}{symbol}:{date}:{hour:02d}"

    async def get_cached_regime(
        self,
        symbol: str,
        date: str,
        hour: int,
    ) -> dict | None:
        """Get cached regime detection result.

        Args:
            symbol: Underlying symbol
            date: Date string (YYYY-MM-DD)
            hour: Hour of day

        Returns:
            Cached regime data or None if not found/expired
        """
        key = self._regime_key(symbol, date, hour)
        return await self.get_json(key)

    async def cache_regime(
        self,
        symbol: str,
        date: str,
        hour: int,
        result: dict,
        ttl_seconds: int = 3600,
    ) -> None:
        """Cache regime detection result.

        Args:
            symbol: Underlying symbol
            date: Date string (YYYY-MM-DD)
            hour: Hour of day
            result: Regime result dict (from RegimeResult.to_dict())
            ttl_seconds: Cache TTL in seconds (default 1 hour)
        """
        key = self._regime_key(symbol, date, hour)
        await self.put_json(key, result, expiration_ttl=ttl_seconds)

    # Dynamic Beta Caching

    BETA_KEY_PREFIX = "beta:"

    def _beta_key(self, symbol: str, date: str) -> str:
        """Get cache key for dynamic beta.

        Args:
            symbol: Underlying symbol (e.g., "QQQ")
            date: Date string (YYYY-MM-DD)

        Returns:
            Cache key (e.g., "beta:QQQ:2024-12-29")
        """
        return f"{self.BETA_KEY_PREFIX}{symbol}:{date}"

    async def get_cached_beta(self, symbol: str, date: str | None = None) -> dict | None:
        """Get cached dynamic beta for a symbol.

        Args:
            symbol: Underlying symbol
            date: Date string (default: today)

        Returns:
            Cached beta data or None if not found/expired
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        key = self._beta_key(symbol, date)
        return await self.get_json(key)

    async def cache_beta(
        self,
        symbol: str,
        beta_result: dict,
        ttl_seconds: int = 24 * 3600,
    ) -> None:
        """Cache dynamic beta result.

        Args:
            symbol: Underlying symbol
            beta_result: Beta result dict (from DynamicBetaResult.to_dict())
            ttl_seconds: Cache TTL in seconds (default 24 hours)
        """
        date = datetime.now().strftime("%Y-%m-%d")
        key = self._beta_key(symbol, date)
        await self.put_json(key, beta_result, expiration_ttl=ttl_seconds)

    # Optimized Weights Caching

    WEIGHTS_KEY = "optimized_weights"

    async def get_cached_weights(self) -> dict | None:
        """Get cached optimized weights for all regimes.

        Returns:
            Dict of regime -> weights or None if not found
        """
        return await self.get_json(self.WEIGHTS_KEY)

    async def cache_weights(
        self,
        weights: dict,
        ttl_seconds: int = 7 * 24 * 3600,
    ) -> None:
        """Cache optimized weights.

        Args:
            weights: Dict of regime -> weight values
            ttl_seconds: Cache TTL in seconds (default 7 days)
        """
        await self.put_json(self.WEIGHTS_KEY, weights, expiration_ttl=ttl_seconds)
