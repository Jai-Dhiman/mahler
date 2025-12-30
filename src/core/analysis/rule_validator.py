"""Trading rule validation using statistical tests.

Uses Mann-Whitney U test to compare trade outcomes with/without each rule,
with Benjamini-Hochberg FDR correction for multiple comparisons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.db.d1 import D1Client


@dataclass
class RuleValidationResult:
    """Result of validating a single trading rule."""

    rule_id: str
    rule_text: str
    trades_with_rule: int
    trades_without_rule: int
    mean_pnl_with: float
    mean_pnl_without: float
    win_rate_with: float
    win_rate_without: float
    u_statistic: float
    p_value: float
    p_value_adjusted: float  # After FDR correction
    is_significant: bool  # p_adjusted < significance_level
    effect_direction: str  # "positive", "negative", "neutral"

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "rule_id": self.rule_id,
            "rule_text": self.rule_text,
            "trades_with_rule": self.trades_with_rule,
            "trades_without_rule": self.trades_without_rule,
            "mean_pnl_with": self.mean_pnl_with,
            "mean_pnl_without": self.mean_pnl_without,
            "win_rate_with": self.win_rate_with,
            "win_rate_without": self.win_rate_without,
            "u_statistic": self.u_statistic,
            "p_value": self.p_value,
            "p_value_adjusted": self.p_value_adjusted,
            "is_significant": self.is_significant,
            "effect_direction": self.effect_direction,
        }


class TradingRuleValidator:
    """Validates playbook rules using statistical testing.

    Workflow:
    1. Partition trades into with/without groups for each rule
    2. Run Mann-Whitney U test to compare P/L distributions
    3. Apply Benjamini-Hochberg FDR correction for multiple testing
    4. Flag rules with significant positive/negative effects

    Example:
        >>> trades = await db.get_closed_trades_with_rules()
        >>> rules = await db.get_playbook_rules()
        >>> validator = TradingRuleValidator(trades, rules)
        >>> results = validator.validate_all_rules()
        >>> for r in results:
        ...     if r.is_significant:
        ...         print(f"{r.rule_id}: {r.effect_direction} (p={r.p_value_adjusted:.3f})")
    """

    # Minimum trades required per group for valid statistical testing
    MIN_TRADES_PER_GROUP = 10

    # Significance level for hypothesis testing
    SIGNIFICANCE_LEVEL = 0.05

    def __init__(self, trades: list[dict], rules: list[dict]):
        """Initialize with trade and rule data.

        Args:
            trades: List of trade dicts with 'profit_loss' and 'applied_rule_ids'
            rules: List of rule dicts with 'id' and 'rule'
        """
        self.trades = trades
        self.rules = {r["id"]: r for r in rules}
        self._results: list[RuleValidationResult] = []

    def validate_rule(self, rule_id: str) -> RuleValidationResult | None:
        """Validate a single rule using Mann-Whitney U test.

        Compares the P/L distribution of trades where the rule was applied
        vs trades where it was not applied.

        Args:
            rule_id: The rule ID to validate

        Returns:
            RuleValidationResult or None if insufficient data
        """
        if rule_id not in self.rules:
            return None

        rule_text = self.rules[rule_id].get("rule", "")

        # Partition trades
        trades_with: list[float] = []
        trades_without: list[float] = []

        for trade in self.trades:
            pnl = trade.get("profit_loss", 0.0)
            applied_rule_ids = trade.get("applied_rule_ids")

            # Parse applied_rule_ids if it's a JSON string
            if isinstance(applied_rule_ids, str):
                try:
                    applied_rule_ids = json.loads(applied_rule_ids)
                except (json.JSONDecodeError, TypeError):
                    applied_rule_ids = []
            elif applied_rule_ids is None:
                applied_rule_ids = []

            if rule_id in applied_rule_ids:
                trades_with.append(pnl)
            else:
                trades_without.append(pnl)

        # Check minimum sample sizes
        if len(trades_with) < self.MIN_TRADES_PER_GROUP or len(trades_without) < self.MIN_TRADES_PER_GROUP:
            return None

        # Convert to numpy arrays
        pnl_with = np.array(trades_with)
        pnl_without = np.array(trades_without)

        # Calculate statistics
        mean_with = float(np.mean(pnl_with))
        mean_without = float(np.mean(pnl_without))
        win_rate_with = float(np.mean(pnl_with > 0))
        win_rate_without = float(np.mean(pnl_without > 0))

        # Mann-Whitney U test
        try:
            from scipy.stats import mannwhitneyu

            # Use two-sided test to detect any difference
            stat, p_value = mannwhitneyu(pnl_with, pnl_without, alternative="two-sided")
        except ImportError:
            # Fallback: simple t-test approximation
            stat, p_value = self._simple_rank_test(pnl_with, pnl_without)

        # Determine effect direction
        if mean_with > mean_without:
            effect_direction = "positive"
        elif mean_with < mean_without:
            effect_direction = "negative"
        else:
            effect_direction = "neutral"

        return RuleValidationResult(
            rule_id=rule_id,
            rule_text=rule_text,
            trades_with_rule=len(trades_with),
            trades_without_rule=len(trades_without),
            mean_pnl_with=mean_with,
            mean_pnl_without=mean_without,
            win_rate_with=win_rate_with,
            win_rate_without=win_rate_without,
            u_statistic=float(stat),
            p_value=float(p_value),
            p_value_adjusted=float(p_value),  # Will be updated by FDR correction
            is_significant=p_value < self.SIGNIFICANCE_LEVEL,
            effect_direction=effect_direction,
        )

    def _simple_rank_test(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Simple rank-based test fallback when scipy is unavailable.

        Uses a normal approximation to the Mann-Whitney U distribution.
        """
        n1, n2 = len(x), len(y)

        # Combine and rank
        combined = np.concatenate([x, y])
        ranks = np.argsort(np.argsort(combined)) + 1

        # U statistic
        r1 = np.sum(ranks[:n1])
        u1 = n1 * n2 + n1 * (n1 + 1) / 2 - r1

        # Normal approximation
        mu = n1 * n2 / 2
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

        if sigma == 0:
            return float(u1), 1.0

        z = (u1 - mu) / sigma

        # Two-tailed p-value using normal approximation
        from math import erf, sqrt

        p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

        return float(u1), float(p_value)

    def validate_all_rules(self) -> list[RuleValidationResult]:
        """Validate all rules and apply FDR correction.

        Returns:
            List of RuleValidationResult with adjusted p-values
        """
        results = []

        # Validate each rule
        for rule_id in self.rules:
            result = self.validate_rule(rule_id)
            if result is not None:
                results.append(result)

        if not results:
            return []

        # Apply FDR correction
        results = self._apply_fdr_correction(results)
        self._results = results

        return results

    def _apply_fdr_correction(
        self, results: list[RuleValidationResult]
    ) -> list[RuleValidationResult]:
        """Apply Benjamini-Hochberg FDR correction to p-values.

        The BH procedure controls the false discovery rate when testing
        multiple hypotheses simultaneously.
        """
        if not results:
            return results

        n = len(results)
        p_values = np.array([r.p_value for r in results])

        # Apply BH correction
        try:
            from statsmodels.stats.multitest import multipletests

            rejected, p_adjusted, _, _ = multipletests(
                p_values, method="fdr_bh", alpha=self.SIGNIFICANCE_LEVEL
            )

            for i, result in enumerate(results):
                result.p_value_adjusted = float(p_adjusted[i])
                result.is_significant = bool(rejected[i])

        except ImportError:
            # Manual BH correction
            sorted_indices = np.argsort(p_values)
            p_adjusted = np.zeros(n)

            for rank, idx in enumerate(sorted_indices, 1):
                # BH adjusted p-value: p * n / rank
                p_adjusted[idx] = min(1.0, p_values[idx] * n / rank)

            # Ensure monotonicity (cumulative minimum from the end)
            for i in range(n - 2, -1, -1):
                sorted_idx = sorted_indices[i]
                next_sorted_idx = sorted_indices[i + 1]
                p_adjusted[sorted_idx] = min(
                    p_adjusted[sorted_idx], p_adjusted[next_sorted_idx]
                )

            for i, result in enumerate(results):
                result.p_value_adjusted = float(p_adjusted[i])
                result.is_significant = p_adjusted[i] < self.SIGNIFICANCE_LEVEL

        return results

    def get_significant_rules(self) -> tuple[list[RuleValidationResult], list[RuleValidationResult]]:
        """Get rules with significant positive and negative effects.

        Returns:
            Tuple of (positive_rules, negative_rules)
        """
        positive = [r for r in self._results if r.is_significant and r.effect_direction == "positive"]
        negative = [r for r in self._results if r.is_significant and r.effect_direction == "negative"]
        return positive, negative

    def get_validation_summary(self) -> dict:
        """Get summary statistics of rule validation.

        Returns:
            dict with summary statistics
        """
        if not self._results:
            return {
                "total_rules_tested": 0,
                "significant_positive": 0,
                "significant_negative": 0,
                "non_significant": 0,
                "rules_with_insufficient_data": len(self.rules) - len(self._results),
            }

        positive, negative = self.get_significant_rules()

        return {
            "total_rules_tested": len(self._results),
            "significant_positive": len(positive),
            "significant_negative": len(negative),
            "non_significant": len(self._results) - len(positive) - len(negative),
            "rules_with_insufficient_data": len(self.rules) - len(self._results),
        }

    @staticmethod
    async def from_db(
        db: D1Client,
        min_trades: int = 50,
        lookback_days: int = 90,
    ) -> TradingRuleValidator | None:
        """Create validator from database.

        Args:
            db: D1Client instance
            min_trades: Minimum closed trades required
            lookback_days: Number of days to look back for trades

        Returns:
            TradingRuleValidator or None if insufficient data
        """
        # Get closed trades with rule tags
        trades = await db.get_closed_trades_with_rules(lookback_days=lookback_days)

        if len(trades) < min_trades:
            return None

        # Get all playbook rules
        rules = await db.get_playbook_rules()

        return TradingRuleValidator(trades, rules)
