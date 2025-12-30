from __future__ import annotations

"""Pre-trade and pre-execution validators."""

from dataclasses import dataclass
from datetime import datetime

from core.analysis.greeks import days_to_expiry
from core.types import CreditSpread, Recommendation, RecommendationStatus


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    reason: str | None = None


class TradeValidator:
    """Validates trades before execution."""

    # Maximum price drift allowed between analysis and execution
    MAX_PRICE_DRIFT_PCT = 0.01  # 1%

    # Minimum time before expiration to enter
    MIN_DTE_FOR_ENTRY = 21

    def validate_recommendation(
        self,
        rec: Recommendation,
        current_time: datetime | None = None,
    ) -> ValidationResult:
        """Validate a recommendation is still valid for execution.

        Args:
            rec: The recommendation to validate
            current_time: Current time (defaults to now)

        Returns:
            ValidationResult indicating if execution should proceed
        """
        if current_time is None:
            current_time = datetime.now()

        # Check status
        if rec.status != RecommendationStatus.PENDING:
            return ValidationResult(
                valid=False,
                reason=f"Recommendation status is {rec.status.value}, not pending",
            )

        # Check expiration
        if current_time >= rec.expires_at:
            return ValidationResult(
                valid=False,
                reason="Recommendation has expired",
            )

        # Check DTE
        dte = days_to_expiry(rec.expiration)
        if dte < self.MIN_DTE_FOR_ENTRY:
            return ValidationResult(
                valid=False,
                reason=f"DTE ({dte}) is below minimum ({self.MIN_DTE_FOR_ENTRY})",
            )

        return ValidationResult(valid=True)

    def validate_price_drift(
        self,
        rec: Recommendation,
        current_credit: float,
    ) -> ValidationResult:
        """Validate that price hasn't drifted too much since analysis.

        Args:
            rec: The recommendation with original analysis price
            current_credit: Current credit available

        Returns:
            ValidationResult indicating if price is still acceptable
        """
        if rec.analysis_price is None:
            # No price recorded, can't validate
            return ValidationResult(valid=True)

        if rec.analysis_price <= 0:
            return ValidationResult(
                valid=False,
                reason="Invalid analysis price",
            )

        # Calculate drift
        drift = abs(current_credit - rec.analysis_price) / rec.analysis_price

        if drift > self.MAX_PRICE_DRIFT_PCT:
            return ValidationResult(
                valid=False,
                reason=f"Price drift ({drift:.1%}) exceeds maximum ({self.MAX_PRICE_DRIFT_PCT:.0%})",
            )

        return ValidationResult(valid=True)

    def validate_spread(self, spread: CreditSpread) -> ValidationResult:
        """Validate a credit spread is properly constructed.

        Args:
            spread: The spread to validate

        Returns:
            ValidationResult indicating if spread is valid
        """
        # Check credit is positive
        if spread.credit <= 0:
            return ValidationResult(
                valid=False,
                reason="Spread has no credit (or is a debit spread)",
            )

        # Check width
        if spread.width <= 0:
            return ValidationResult(
                valid=False,
                reason="Invalid spread width",
            )

        # Check strikes are ordered correctly
        if spread.spread_type.value == "bull_put":
            if spread.short_strike <= spread.long_strike:
                return ValidationResult(
                    valid=False,
                    reason="Bull put: short strike must be above long strike",
                )
        else:  # bear_call
            if spread.short_strike >= spread.long_strike:
                return ValidationResult(
                    valid=False,
                    reason="Bear call: short strike must be below long strike",
                )

        # Check DTE
        dte = days_to_expiry(spread.expiration)
        if dte <= 0:
            return ValidationResult(
                valid=False,
                reason="Spread has expired",
            )

        return ValidationResult(valid=True)


@dataclass
class ExitConfig:
    """Configurable exit thresholds.

    Research notes:
    - 50% profit target is well-supported by research
    - 200% stop loss requires >80% win rate to be profitable
    - Consider 150% stop if actual win rate is lower
    - IV-adjusted targets capture premium before IV crush in high IV
    - Gamma protection exits early when DTE <= 21 to avoid gamma risk
    """

    profit_target_pct: float = 0.50  # 50% of max profit
    stop_loss_pct: float = 2.00  # 200% of credit (default, conservative)
    time_exit_dte: int = 21  # Close at 21 DTE

    # Win rate thresholds for adjusting stop loss
    # If win rate drops below threshold, use tighter stop
    win_rate_threshold: float = 0.80
    tighter_stop_loss_pct: float = 1.50  # 150% stop when win rate < 80%

    # IV-adjusted exit settings
    iv_adjustment_enabled: bool = True
    iv_high_threshold: float = 70.0  # IV rank above which to reduce targets
    iv_low_threshold: float = 30.0  # IV rank below which to increase targets

    # Gamma protection settings
    gamma_protection_enabled: bool = True
    gamma_protection_pnl: float = 0.70  # Exit at 70% profit when DTE <= 21
    gamma_explosion_dte: int = 7  # Force exit when DTE <= 7


class IVAdjustedExits:
    """IV-adjusted exit logic for dynamic profit targets.

    High IV positions should exit faster (before IV crush).
    Low IV positions can let winners run longer.
    Gamma protection triggers early exits when DTE <= 21.
    """

    def __init__(self, config: ExitConfig):
        self.config = config

    def calculate_iv_rank(
        self,
        current_iv: float,
        iv_52w_high: float,
        iv_52w_low: float,
    ) -> float:
        """Calculate IV rank as percentile between 52-week low and high.

        Args:
            current_iv: Current implied volatility
            iv_52w_high: Highest IV in lookback period
            iv_52w_low: Lowest IV in lookback period

        Returns:
            IV rank from 0-100
        """
        if iv_52w_high <= iv_52w_low:
            return 50.0  # Default to middle if no range

        iv_range = iv_52w_high - iv_52w_low
        if iv_range < 1e-8:
            return 50.0

        rank = (current_iv - iv_52w_low) / iv_range * 100
        return max(0.0, min(100.0, rank))

    def adjusted_profit_target(
        self,
        base_target: float,
        current_iv: float,
        iv_history: list[float],
    ) -> float:
        """Scale profit target based on IV rank.

        High IV (rank > 70): reduce target by up to 15% to capture premium before IV crush
        Low IV (rank < 30): increase target by up to 15% to let winners run

        Args:
            base_target: Base profit target (e.g., 0.50 for 50%)
            current_iv: Current implied volatility
            iv_history: Historical IV values (most recent first)

        Returns:
            Adjusted profit target
        """
        if not iv_history or len(iv_history) < 30:
            return base_target  # Not enough history, use base

        iv_52w_high = max(iv_history[:252]) if len(iv_history) >= 252 else max(iv_history)
        iv_52w_low = min(iv_history[:252]) if len(iv_history) >= 252 else min(iv_history)

        iv_rank = self.calculate_iv_rank(current_iv, iv_52w_high, iv_52w_low)

        if iv_rank > self.config.iv_high_threshold:
            # High IV: reduce target to capture premium before crush
            # Scale: at IV rank 100, reduce by 15%
            reduction = (iv_rank - self.config.iv_high_threshold) / 100 * 0.5
            return base_target * (1 - min(reduction, 0.15))

        elif iv_rank < self.config.iv_low_threshold:
            # Low IV: increase target to let winners run
            # Scale: at IV rank 0, increase by 15%
            increase = (self.config.iv_low_threshold - iv_rank) / 100 * 0.5
            return base_target * (1 + min(increase, 0.15))

        return base_target

    def gamma_aware_exit(
        self,
        pnl_pct: float,
        dte: int,
        target: float,
    ) -> tuple[bool, str | None]:
        """Check for gamma-aware exit conditions.

        Gamma risk accelerates as expiration approaches.
        Exit earlier when DTE is low to avoid gamma explosion.

        Args:
            pnl_pct: Current P/L as percentage of max profit
            dte: Days to expiration
            target: Current profit target

        Returns:
            Tuple of (should_exit, reason)
        """
        if not self.config.gamma_protection_enabled:
            return False, None

        # Force exit when DTE <= 7 (gamma explosion zone)
        if dte <= self.config.gamma_explosion_dte:
            return True, f"gamma_explosion (DTE={dte})"

        # Exit at 70% of target when DTE <= 21 (gamma acceleration)
        if dte <= 21 and pnl_pct >= self.config.gamma_protection_pnl * target:
            return True, f"gamma_protection ({pnl_pct:.0%} profit at {dte} DTE)"

        return False, None


class ExitValidator:
    """Validates exit conditions for positions with configurable thresholds."""

    def __init__(self, config: ExitConfig | None = None):
        self.config = config or ExitConfig()
        self.iv_exits = IVAdjustedExits(self.config)

    def adjust_for_win_rate(self, historical_win_rate: float | None) -> None:
        """Adjust stop loss based on historical win rate.

        Research: 200% stop requires >80% win rate to be profitable.
        If actual win rate is lower, tighten the stop.
        """
        if historical_win_rate is None:
            return

        if historical_win_rate < self.config.win_rate_threshold:
            # Use tighter stop
            self.config.stop_loss_pct = self.config.tighter_stop_loss_pct
            print(
                f"Adjusted stop loss to {self.config.stop_loss_pct:.0%} "
                f"due to win rate ({historical_win_rate:.0%} < {self.config.win_rate_threshold:.0%})"
            )

    def check_profit_target(
        self,
        entry_credit: float,
        current_value: float,
    ) -> ValidationResult:
        """Check if profit target is reached.

        Args:
            entry_credit: Original credit received per spread
            current_value: Current cost to close per spread

        Returns:
            ValidationResult with valid=True if should exit
        """
        if entry_credit <= 0:
            return ValidationResult(valid=False, reason="Invalid entry credit")

        # Profit = entry_credit - current_value
        profit = entry_credit - current_value
        profit_pct = profit / entry_credit

        if profit_pct >= self.config.profit_target_pct:
            return ValidationResult(
                valid=True,
                reason=f"Profit target reached ({profit_pct:.0%} of max)",
            )

        return ValidationResult(valid=False)

    def check_stop_loss(
        self,
        entry_credit: float,
        current_value: float,
    ) -> ValidationResult:
        """Check if stop loss is triggered.

        Args:
            entry_credit: Original credit received per spread
            current_value: Current cost to close per spread

        Returns:
            ValidationResult with valid=True if should exit
        """
        if entry_credit <= 0:
            return ValidationResult(valid=False, reason="Invalid entry credit")

        # Loss occurs when current_value > entry_credit
        # Stop at configured % of credit
        max_loss_before_stop = entry_credit * self.config.stop_loss_pct

        current_loss = current_value - entry_credit
        if current_loss >= max_loss_before_stop:
            return ValidationResult(
                valid=True,
                reason=f"Stop loss triggered (loss = {(current_loss / entry_credit):.0%} of credit)",
            )

        return ValidationResult(valid=False)

    def check_time_exit(self, expiration: str) -> ValidationResult:
        """Check if time-based exit is triggered.

        Args:
            expiration: Expiration date (YYYY-MM-DD)

        Returns:
            ValidationResult with valid=True if should exit
        """
        dte = days_to_expiry(expiration)

        if dte <= self.config.time_exit_dte:
            return ValidationResult(
                valid=True,
                reason=f"Time exit triggered ({dte} DTE <= {self.config.time_exit_dte})",
            )

        return ValidationResult(valid=False)

    def check_all_exit_conditions(
        self,
        entry_credit: float,
        current_value: float,
        expiration: str,
        current_iv: float | None = None,
        iv_history: list[float] | None = None,
    ) -> tuple[bool, str | None, float | None]:
        """Check all exit conditions with IV-adjusted targets.

        Args:
            entry_credit: Original credit received per spread
            current_value: Current cost to close per spread
            expiration: Expiration date (YYYY-MM-DD)
            current_iv: Current implied volatility (optional)
            iv_history: Historical IV values, most recent first (optional)

        Returns:
            Tuple of (should_exit, reason, iv_rank_at_exit)
        """
        dte = days_to_expiry(expiration)
        profit = entry_credit - current_value
        profit_pct = profit / entry_credit if entry_credit > 0 else 0

        # Calculate IV rank if data available
        iv_rank = None
        if current_iv is not None and iv_history and len(iv_history) >= 30:
            iv_52w_high = max(iv_history[:252]) if len(iv_history) >= 252 else max(iv_history)
            iv_52w_low = min(iv_history[:252]) if len(iv_history) >= 252 else min(iv_history)
            iv_rank = self.iv_exits.calculate_iv_rank(current_iv, iv_52w_high, iv_52w_low)

        # Determine profit target (IV-adjusted or base)
        if self.config.iv_adjustment_enabled and current_iv is not None and iv_history:
            profit_target = self.iv_exits.adjusted_profit_target(
                self.config.profit_target_pct,
                current_iv,
                iv_history,
            )
        else:
            profit_target = self.config.profit_target_pct

        # Priority 1: Gamma explosion (force exit when DTE <= 7)
        if self.config.gamma_protection_enabled and dte <= self.config.gamma_explosion_dte:
            return True, f"gamma_explosion (DTE={dte})", iv_rank

        # Priority 2: Gamma protection (exit at 70% of target when DTE <= 21)
        if self.config.gamma_protection_enabled:
            gamma_exit, gamma_reason = self.iv_exits.gamma_aware_exit(profit_pct, dte, profit_target)
            if gamma_exit:
                return True, gamma_reason, iv_rank

        # Priority 3: IV-adjusted profit target
        if profit_pct >= profit_target:
            if iv_rank is not None and profit_target != self.config.profit_target_pct:
                reason = f"iv_adjusted_profit ({profit_pct:.0%} >= {profit_target:.0%} target, IV rank={iv_rank:.0f})"
            else:
                reason = f"profit_target ({profit_pct:.0%} >= {profit_target:.0%})"
            return True, reason, iv_rank

        # Priority 4: Stop loss
        stop_check = self.check_stop_loss(entry_credit, current_value)
        if stop_check.valid:
            return True, stop_check.reason, iv_rank

        # Priority 5: Time exit (redundant with gamma explosion but kept for clarity)
        time_check = self.check_time_exit(expiration)
        if time_check.valid:
            return True, time_check.reason, iv_rank

        return False, None, iv_rank
