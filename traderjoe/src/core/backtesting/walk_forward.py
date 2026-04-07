"""Enhanced walk-forward validation with parameter stability tracking.

Standard walk-forward: 6mo train / 1mo validate / 1mo test

Enhancements:
- Parameter stability tracking across periods
- Regime-specific parameter sets
- Alpha decay monitoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from core.backtesting.config import BacktestConfig, MarketRegime


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    train_months: int = 6
    validate_months: int = 1
    test_months: int = 1
    min_trades_per_period: int = 20

    # Parameter stability settings
    track_stability: bool = True
    max_parameter_drift: float = 0.20  # Max 20% change between periods

    # Regime settings
    regime_specific_params: bool = True


@dataclass
class PeriodResult:
    """Results for a single walk-forward period."""

    period_id: int
    train_start: date
    train_end: date
    validate_start: date
    validate_end: date
    test_start: date
    test_end: date

    # Optimized parameters
    optimized_params: dict[str, float]

    # Performance metrics
    train_sharpe: float
    validate_sharpe: float
    test_sharpe: float

    train_trades: int
    validate_trades: int
    test_trades: int

    # Regime during test period
    dominant_regime: str = "normal"


@dataclass
class ParameterStability:
    """Stability metrics for a single parameter."""

    parameter_name: str
    values_over_time: list[float]
    mean: float
    std_dev: float
    coefficient_of_variation: float  # std / mean
    max_drift: float  # Max change between consecutive periods
    is_stable: bool  # CV < threshold


@dataclass
class AlphaDecay:
    """Alpha decay analysis results."""

    months_since_optimization: list[int]
    out_of_sample_sharpe: list[float]
    decay_rate: float  # Slope of regression
    half_life_months: float | None  # Months until alpha halves
    is_decaying: bool


@dataclass
class WalkForwardResults:
    """Complete walk-forward validation results."""

    periods: list[PeriodResult]
    config: WalkForwardConfig

    # Aggregate metrics
    avg_train_sharpe: float
    avg_validate_sharpe: float
    avg_test_sharpe: float

    total_train_trades: int
    total_validate_trades: int
    total_test_trades: int

    # Stability analysis
    parameter_stability: dict[str, ParameterStability] = field(default_factory=dict)

    # Regime-specific optimal parameters
    regime_params: dict[str, dict[str, float]] = field(default_factory=dict)

    # Alpha decay
    alpha_decay: AlphaDecay | None = None

    @property
    def is_robust(self) -> bool:
        """Check if strategy is robust based on walk-forward results."""
        # Sharpe degradation from train to test should be < 50%
        if self.avg_train_sharpe <= 0:
            return False
        degradation = (self.avg_train_sharpe - self.avg_test_sharpe) / self.avg_train_sharpe
        return degradation < 0.50 and self.avg_test_sharpe > 0

    @property
    def sharpe_degradation_pct(self) -> float:
        """Percentage degradation from train to test."""
        if self.avg_train_sharpe <= 0:
            return 100.0
        return (self.avg_train_sharpe - self.avg_test_sharpe) / self.avg_train_sharpe * 100


class WalkForwardValidator:
    """Enhanced walk-forward validation with stability tracking.

    Implements anchored walk-forward where the training window
    expands with each period while maintaining fixed test windows.
    """

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
        optimizer: Callable[[pd.DataFrame], dict[str, float]] | None = None,
    ):
        """Initialize walk-forward validator.

        Args:
            config: WalkForwardConfig with period settings
            optimizer: Function that takes training data and returns optimal params
        """
        self.config = config or WalkForwardConfig()
        self.optimizer = optimizer

    def generate_periods(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
    ) -> list[tuple[date, date, date, date, date, date]]:
        """Generate train/validate/test date ranges.

        Args:
            data: DataFrame with date column
            date_column: Name of date column

        Returns:
            List of (train_start, train_end, val_start, val_end, test_start, test_end)
        """
        dates = pd.to_datetime(data[date_column]).sort_values()
        min_date = dates.min()
        max_date = dates.max()

        periods = []
        current_date = min_date

        # Calculate initial training window end
        train_months = self.config.train_months
        validate_months = self.config.validate_months
        test_months = self.config.test_months
        total_period = train_months + validate_months + test_months

        while True:
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=train_months)
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=validate_months)
            test_start = val_end
            test_end = test_start + pd.DateOffset(months=test_months)

            if test_end > max_date:
                break

            periods.append((
                train_start.date(),
                train_end.date(),
                val_start.date(),
                val_end.date(),
                test_start.date(),
                test_end.date(),
            ))

            # Roll forward by 1 month
            current_date = current_date + pd.DateOffset(months=1)

        return periods

    def run_validation(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
        return_column: str = "return",
    ) -> WalkForwardResults:
        """Run walk-forward validation with stability metrics.

        Args:
            data: DataFrame with trade data
            date_column: Name of date column
            return_column: Name of return column

        Returns:
            WalkForwardResults with all metrics
        """
        if self.optimizer is None:
            raise ValueError("Optimizer function must be provided")

        periods = self.generate_periods(data, date_column)
        period_results: list[PeriodResult] = []
        param_history: dict[str, list[float]] = {}

        for i, (t_start, t_end, v_start, v_end, test_start, test_end) in enumerate(periods):
            # Filter data for each period
            dates = pd.to_datetime(data[date_column])

            train_mask = (dates >= pd.Timestamp(t_start)) & (dates < pd.Timestamp(t_end))
            val_mask = (dates >= pd.Timestamp(v_start)) & (dates < pd.Timestamp(v_end))
            test_mask = (dates >= pd.Timestamp(test_start)) & (dates < pd.Timestamp(test_end))

            train_data = data[train_mask]
            val_data = data[val_mask]
            test_data = data[test_mask]

            # Check minimum trades
            if len(train_data) < self.config.min_trades_per_period:
                continue

            # Optimize on training data
            optimized_params = self.optimizer(train_data)

            # Track parameter history
            for param, value in optimized_params.items():
                if param not in param_history:
                    param_history[param] = []
                param_history[param].append(value)

            # Calculate Sharpe for each period
            train_sharpe = self._calculate_sharpe(train_data[return_column].values)
            validate_sharpe = self._calculate_sharpe(val_data[return_column].values)
            test_sharpe = self._calculate_sharpe(test_data[return_column].values)

            period_results.append(PeriodResult(
                period_id=i,
                train_start=t_start,
                train_end=t_end,
                validate_start=v_start,
                validate_end=v_end,
                test_start=test_start,
                test_end=test_end,
                optimized_params=optimized_params,
                train_sharpe=train_sharpe,
                validate_sharpe=validate_sharpe,
                test_sharpe=test_sharpe,
                train_trades=len(train_data),
                validate_trades=len(val_data),
                test_trades=len(test_data),
            ))

        if not period_results:
            raise ValueError("No valid periods found in data")

        # Calculate aggregate metrics
        avg_train = np.mean([p.train_sharpe for p in period_results])
        avg_validate = np.mean([p.validate_sharpe for p in period_results])
        avg_test = np.mean([p.test_sharpe for p in period_results])

        total_train = sum(p.train_trades for p in period_results)
        total_validate = sum(p.validate_trades for p in period_results)
        total_test = sum(p.test_trades for p in period_results)

        # Parameter stability analysis
        stability_results = {}
        if self.config.track_stability:
            stability_results = self.track_parameter_stability(param_history)

        # Alpha decay analysis
        alpha_decay = self._analyze_alpha_decay(period_results)

        return WalkForwardResults(
            periods=period_results,
            config=self.config,
            avg_train_sharpe=float(avg_train),
            avg_validate_sharpe=float(avg_validate),
            avg_test_sharpe=float(avg_test),
            total_train_trades=total_train,
            total_validate_trades=total_validate,
            total_test_trades=total_test,
            parameter_stability=stability_results,
            alpha_decay=alpha_decay,
        )

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        annualization_factor: float = np.sqrt(12),
    ) -> float:
        """Calculate Sharpe ratio from returns array."""
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret < 1e-10:
            return 0.0
        return float(mean_ret / std_ret * annualization_factor)

    def track_parameter_stability(
        self,
        period_params: dict[str, list[float]],
    ) -> dict[str, ParameterStability]:
        """Track how parameters change across periods.

        Args:
            period_params: Dict mapping parameter names to their values over time

        Returns:
            Dict of ParameterStability for each parameter
        """
        stability_results = {}

        for param_name, values in period_params.items():
            if len(values) < 2:
                continue

            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)

            # Coefficient of variation (relative stability)
            cv = std_val / mean_val if mean_val != 0 else float("inf")

            # Maximum drift between consecutive periods
            diffs = np.abs(np.diff(values_array))
            max_drift = np.max(diffs / np.abs(values_array[:-1])) if len(diffs) > 0 else 0

            # Stability check
            is_stable = cv < 0.30 and max_drift < self.config.max_parameter_drift

            stability_results[param_name] = ParameterStability(
                parameter_name=param_name,
                values_over_time=values,
                mean=float(mean_val),
                std_dev=float(std_val),
                coefficient_of_variation=float(cv),
                max_drift=float(max_drift),
                is_stable=is_stable,
            )

        return stability_results

    def get_regime_specific_params(
        self,
        period_results: list[PeriodResult],
    ) -> dict[str, dict[str, float]]:
        """Return optimal parameters by market regime.

        Groups periods by their dominant regime and returns
        the average optimal parameters for each regime.

        Args:
            period_results: List of PeriodResult from walk-forward

        Returns:
            Dict mapping regime to average optimal params
        """
        regime_params: dict[str, list[dict[str, float]]] = {}

        for period in period_results:
            regime = period.dominant_regime
            if regime not in regime_params:
                regime_params[regime] = []
            regime_params[regime].append(period.optimized_params)

        # Average parameters for each regime
        averaged: dict[str, dict[str, float]] = {}
        for regime, params_list in regime_params.items():
            if not params_list:
                continue

            avg_params: dict[str, float] = {}
            all_keys = set()
            for p in params_list:
                all_keys.update(p.keys())

            for key in all_keys:
                values = [p.get(key, 0) for p in params_list]
                avg_params[key] = float(np.mean(values))

            averaged[regime] = avg_params

        return averaged

    def _analyze_alpha_decay(
        self,
        period_results: list[PeriodResult],
    ) -> AlphaDecay | None:
        """Analyze alpha decay over time.

        Fits a linear regression to out-of-sample Sharpe ratios
        to estimate decay rate and half-life.

        Args:
            period_results: List of PeriodResult

        Returns:
            AlphaDecay analysis or None if insufficient data
        """
        if len(period_results) < 3:
            return None

        months = list(range(len(period_results)))
        sharpes = [p.test_sharpe for p in period_results]

        # Linear regression
        slope, intercept = np.polyfit(months, sharpes, 1)

        # Calculate half-life
        half_life = None
        if slope < 0 and intercept > 0:
            # Time until Sharpe reaches half of initial
            target = intercept / 2
            half_life = (target - intercept) / slope

        return AlphaDecay(
            months_since_optimization=months,
            out_of_sample_sharpe=sharpes,
            decay_rate=float(slope),
            half_life_months=float(half_life) if half_life else None,
            is_decaying=slope < -0.01,  # Threshold for meaningful decay
        )

    def suggest_reoptimization_frequency(
        self,
        alpha_decay: AlphaDecay | None,
    ) -> str:
        """Suggest how often to reoptimize based on alpha decay.

        Args:
            alpha_decay: AlphaDecay analysis results

        Returns:
            Recommendation string
        """
        if alpha_decay is None:
            return "Insufficient data to determine reoptimization frequency"

        if not alpha_decay.is_decaying:
            return "Parameters appear stable. Reoptimize quarterly."

        if alpha_decay.half_life_months is not None:
            half_life = alpha_decay.half_life_months
            if half_life < 3:
                return f"Fast alpha decay (half-life: {half_life:.1f} months). Reoptimize monthly."
            elif half_life < 6:
                return f"Moderate alpha decay (half-life: {half_life:.1f} months). Reoptimize bi-monthly."
            else:
                return f"Slow alpha decay (half-life: {half_life:.1f} months). Reoptimize quarterly."

        return "Alpha is decaying but half-life could not be determined. Monitor closely."
