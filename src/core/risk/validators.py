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
    """

    profit_target_pct: float = 0.50  # 50% of max profit
    stop_loss_pct: float = 2.00  # 200% of credit (default, conservative)
    time_exit_dte: int = 21  # Close at 21 DTE

    # Win rate thresholds for adjusting stop loss
    # If win rate drops below threshold, use tighter stop
    win_rate_threshold: float = 0.80
    tighter_stop_loss_pct: float = 1.50  # 150% stop when win rate < 80%


class ExitValidator:
    """Validates exit conditions for positions with configurable thresholds."""

    def __init__(self, config: ExitConfig | None = None):
        self.config = config or ExitConfig()

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
    ) -> tuple[bool, str | None]:
        """Check all exit conditions.

        Returns:
            Tuple of (should_exit, reason)
        """
        # Check in order of priority
        profit_check = self.check_profit_target(entry_credit, current_value)
        if profit_check.valid:
            return True, profit_check.reason

        stop_check = self.check_stop_loss(entry_credit, current_value)
        if stop_check.valid:
            return True, stop_check.reason

        time_check = self.check_time_exit(expiration)
        if time_check.valid:
            return True, time_check.reason

        return False, None
