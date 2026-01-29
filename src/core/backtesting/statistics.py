"""Statistical validation for strategy performance.

Provides:
- Bootstrap confidence intervals for Sharpe ratio
- Monte Carlo simulation for drawdown distribution
- t-test for strategy vs benchmark comparison
- Statistical significance testing for parameter optimization
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SharpeConfidenceInterval:
    """Confidence interval for Sharpe ratio."""

    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    n_bootstrap: int
    std_error: float

    @property
    def is_significantly_positive(self) -> bool:
        """Check if Sharpe is statistically significantly > 0."""
        return self.lower_bound > 0

    @property
    def interval_width(self) -> float:
        """Width of the confidence interval."""
        return self.upper_bound - self.lower_bound


@dataclass
class DrawdownDistribution:
    """Distribution of drawdowns from Monte Carlo simulation."""

    mean_max_drawdown: float
    median_max_drawdown: float
    percentile_95: float  # 95th percentile (worst 5%)
    percentile_99: float  # 99th percentile (worst 1%)
    std_dev: float
    n_simulations: int

    @property
    def var_95(self) -> float:
        """Value at Risk at 95% confidence."""
        return self.percentile_95

    @property
    def cvar_95(self) -> float:
        """Conditional VaR (Expected Shortfall) at 95%."""
        # This is calculated during simulation, stored as percentile_99
        # as an approximation
        return self.percentile_99


@dataclass
class StrategyComparison:
    """Result of strategy vs benchmark comparison."""

    strategy_mean: float
    benchmark_mean: float
    t_statistic: float
    p_value: float
    outperformance: float  # strategy_mean - benchmark_mean
    is_significant: bool  # p < 0.05
    confidence_level: float


@dataclass
class ParameterSignificance:
    """Statistical significance of a parameter's effect."""

    parameter_name: str
    effect_size: float  # Cohen's d
    p_value: float
    is_significant: bool
    sample_size: int
    test_type: str  # "t-test", "mann-whitney", etc.


class BacktestStatistics:
    """Statistical validation for strategy performance.

    Provides methods for:
    - Bootstrap confidence intervals for Sharpe ratio
    - Monte Carlo simulation for drawdown distribution
    - Hypothesis testing for strategy vs benchmark
    - Parameter significance testing
    """

    def __init__(self, seed: int | None = 42):
        """Initialize statistics calculator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def bootstrap_sharpe_ci(
        self,
        returns: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        annualization_factor: float = np.sqrt(252 / 30),
    ) -> SharpeConfidenceInterval:
        """Calculate bootstrap confidence interval for Sharpe ratio.

        Uses the percentile bootstrap method to estimate the
        sampling distribution of the Sharpe ratio.

        Args:
            returns: Array of period returns (e.g., trade returns)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95%)
            annualization_factor: Factor to annualize Sharpe ratio

        Returns:
            SharpeConfidenceInterval with bounds and statistics
        """
        if len(returns) < 2:
            raise ValueError("Need at least 2 returns for bootstrap")

        returns = np.asarray(returns)

        # Point estimate
        point_sharpe = self._calculate_sharpe(returns, annualization_factor)

        # Bootstrap
        bootstrap_sharpes = np.zeros(n_bootstrap)
        n = len(returns)

        for i in range(n_bootstrap):
            sample_idx = self.rng.choice(n, size=n, replace=True)
            sample_returns = returns[sample_idx]
            bootstrap_sharpes[i] = self._calculate_sharpe(
                sample_returns, annualization_factor
            )

        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = float(np.percentile(bootstrap_sharpes, lower_percentile))
        upper_bound = float(np.percentile(bootstrap_sharpes, upper_percentile))
        std_error = float(np.std(bootstrap_sharpes))

        return SharpeConfidenceInterval(
            point_estimate=point_sharpe,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence,
            n_bootstrap=n_bootstrap,
            std_error=std_error,
        )

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        annualization_factor: float = 1.0,
    ) -> float:
        """Calculate Sharpe ratio from returns.

        Args:
            returns: Array of returns
            annualization_factor: Factor to annualize

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return < 1e-10:
            return 0.0
        return float(mean_return / std_return * annualization_factor)

    def monte_carlo_drawdown(
        self,
        returns: np.ndarray,
        n_simulations: int = 10000,
        path_length: int | None = None,
    ) -> DrawdownDistribution:
        """Monte Carlo simulation for drawdown distribution.

        Shuffles the return sequence to generate alternative equity curves
        and calculates the distribution of maximum drawdowns.

        Args:
            returns: Array of period returns
            n_simulations: Number of Monte Carlo paths
            path_length: Optional custom path length (defaults to len(returns))

        Returns:
            DrawdownDistribution with statistics
        """
        if len(returns) < 2:
            raise ValueError("Need at least 2 returns for simulation")

        returns = np.asarray(returns)
        path_length = path_length or len(returns)

        max_drawdowns = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Shuffle returns to create alternative path
            shuffled = self.rng.permutation(returns)

            # If path_length differs, sample with replacement
            if path_length != len(returns):
                shuffled = self.rng.choice(returns, size=path_length, replace=True)

            # Calculate cumulative returns and drawdown
            cumulative = np.cumprod(1 + shuffled)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / running_max
            max_drawdowns[i] = np.max(drawdowns)

        return DrawdownDistribution(
            mean_max_drawdown=float(np.mean(max_drawdowns)),
            median_max_drawdown=float(np.median(max_drawdowns)),
            percentile_95=float(np.percentile(max_drawdowns, 95)),
            percentile_99=float(np.percentile(max_drawdowns, 99)),
            std_dev=float(np.std(max_drawdowns)),
            n_simulations=n_simulations,
        )

    def strategy_vs_benchmark_ttest(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        alpha: float = 0.05,
    ) -> StrategyComparison:
        """Paired t-test for strategy vs benchmark returns.

        Tests whether the strategy significantly outperforms the benchmark.

        Args:
            strategy_returns: Array of strategy returns
            benchmark_returns: Array of benchmark returns (same periods)
            alpha: Significance level

        Returns:
            StrategyComparison with test results
        """
        strategy_returns = np.asarray(strategy_returns)
        benchmark_returns = np.asarray(benchmark_returns)

        if len(strategy_returns) != len(benchmark_returns):
            raise ValueError("Strategy and benchmark must have same length")

        if len(strategy_returns) < 2:
            raise ValueError("Need at least 2 observations")

        # Paired t-test on the differences
        differences = strategy_returns - benchmark_returns
        t_stat, p_value = stats.ttest_1samp(differences, 0)

        return StrategyComparison(
            strategy_mean=float(np.mean(strategy_returns)),
            benchmark_mean=float(np.mean(benchmark_returns)),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            outperformance=float(np.mean(differences)),
            is_significant=p_value < alpha,
            confidence_level=1 - alpha,
        )

    def mann_whitney_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alternative: str = "two-sided",
    ) -> tuple[float, float]:
        """Mann-Whitney U test for comparing two distributions.

        Non-parametric alternative to t-test, useful for non-normal
        return distributions.

        Args:
            group1: First sample
            group2: Second sample
            alternative: "two-sided", "greater", or "less"

        Returns:
            Tuple of (U statistic, p-value)
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative=alternative
        )

        return float(statistic), float(p_value)

    def parameter_significance(
        self,
        with_parameter: np.ndarray,
        without_parameter: np.ndarray,
        parameter_name: str,
        alpha: float = 0.05,
    ) -> ParameterSignificance:
        """Test if a parameter significantly improves performance.

        Uses Mann-Whitney U test and calculates effect size (Cohen's d).

        Args:
            with_parameter: Returns when parameter is active
            without_parameter: Returns when parameter is not active
            parameter_name: Name of the parameter being tested
            alpha: Significance level

        Returns:
            ParameterSignificance with results
        """
        with_param = np.asarray(with_parameter)
        without_param = np.asarray(without_parameter)

        # Mann-Whitney test
        _, p_value = self.mann_whitney_test(with_param, without_param, "greater")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(with_param) + np.var(without_param)) / 2
        )
        if pooled_std > 0:
            effect_size = (np.mean(with_param) - np.mean(without_param)) / pooled_std
        else:
            effect_size = 0.0

        return ParameterSignificance(
            parameter_name=parameter_name,
            effect_size=float(effect_size),
            p_value=float(p_value),
            is_significant=p_value < alpha,
            sample_size=len(with_param) + len(without_param),
            test_type="mann-whitney",
        )

    def validate_backtest_results(
        self,
        returns: np.ndarray,
        min_trades: int = 50,
        min_sharpe_ci_bound: float = 0.0,
        max_drawdown_95: float = 0.25,
    ) -> dict[str, bool | float | str]:
        """Comprehensive validation of backtest results.

        Checks statistical significance of performance metrics.

        Args:
            returns: Array of trade returns
            min_trades: Minimum number of trades required
            min_sharpe_ci_bound: Minimum lower bound of Sharpe CI
            max_drawdown_95: Maximum acceptable 95th percentile drawdown

        Returns:
            Dict with validation results
        """
        returns = np.asarray(returns)
        n_trades = len(returns)

        results: dict[str, bool | float | str] = {
            "n_trades": n_trades,
            "sufficient_trades": n_trades >= min_trades,
        }

        if n_trades < 2:
            results["validation_status"] = "insufficient_data"
            results["passed"] = False
            return results

        # Sharpe confidence interval
        sharpe_ci = self.bootstrap_sharpe_ci(returns)
        results["sharpe_point"] = sharpe_ci.point_estimate
        results["sharpe_lower"] = sharpe_ci.lower_bound
        results["sharpe_upper"] = sharpe_ci.upper_bound
        results["sharpe_significant"] = sharpe_ci.is_significantly_positive
        results["sharpe_ci_passes"] = sharpe_ci.lower_bound >= min_sharpe_ci_bound

        # Drawdown analysis
        dd_dist = self.monte_carlo_drawdown(returns)
        results["dd_mean"] = dd_dist.mean_max_drawdown
        results["dd_95"] = dd_dist.percentile_95
        results["dd_passes"] = dd_dist.percentile_95 <= max_drawdown_95

        # Overall validation
        passed = (
            results["sufficient_trades"]
            and results["sharpe_ci_passes"]
            and results["dd_passes"]
        )
        results["passed"] = passed
        results["validation_status"] = "passed" if passed else "failed"

        return results
