from __future__ import annotations

"""Implied Volatility Rank calculations and term structure analysis."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.interpolate import UnivariateSpline


@dataclass
class IVMetrics:
    """IV metrics for an underlying."""

    current_iv: float
    iv_rank: float  # Percentile rank over lookback period
    iv_percentile: float  # Percentage of days IV was lower
    iv_high: float  # Highest IV in period
    iv_low: float  # Lowest IV in period


def calculate_iv_rank(
    current_iv: float,
    historical_ivs: list[float],
) -> float:
    """Calculate IV Rank.

    IV Rank = (Current IV - 52-week Low) / (52-week High - 52-week Low) * 100

    Args:
        current_iv: Current implied volatility
        historical_ivs: List of historical IV values (typically 252 trading days)

    Returns:
        IV Rank as percentage (0-100)
    """
    if not historical_ivs:
        return 50.0  # Default to middle if no history

    iv_low = min(historical_ivs)
    iv_high = max(historical_ivs)

    if iv_high == iv_low:
        return 50.0  # Avoid division by zero

    iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
    return max(0.0, min(100.0, iv_rank))


def calculate_iv_percentile(
    current_iv: float,
    historical_ivs: list[float],
) -> float:
    """Calculate IV Percentile.

    IV Percentile = Percentage of days where IV was lower than current

    Args:
        current_iv: Current implied volatility
        historical_ivs: List of historical IV values

    Returns:
        IV Percentile as percentage (0-100)
    """
    if not historical_ivs:
        return 50.0

    days_lower = sum(1 for iv in historical_ivs if iv < current_iv)
    return (days_lower / len(historical_ivs)) * 100


def calculate_iv_metrics(
    current_iv: float,
    historical_ivs: list[float],
) -> IVMetrics:
    """Calculate comprehensive IV metrics.

    Args:
        current_iv: Current implied volatility
        historical_ivs: List of historical IV values (ideally 252 for one year)

    Returns:
        IVMetrics with rank, percentile, high, and low
    """
    if not historical_ivs:
        return IVMetrics(
            current_iv=current_iv,
            iv_rank=50.0,
            iv_percentile=50.0,
            iv_high=current_iv,
            iv_low=current_iv,
        )

    return IVMetrics(
        current_iv=current_iv,
        iv_rank=calculate_iv_rank(current_iv, historical_ivs),
        iv_percentile=calculate_iv_percentile(current_iv, historical_ivs),
        iv_high=max(historical_ivs),
        iv_low=min(historical_ivs),
    )


def is_elevated_iv(iv_rank: float, threshold: float = 50.0) -> bool:
    """Check if IV is elevated enough for selling premium.

    Per PRD: Entry trigger is IV Rank >= 50 (preferably >= 70)
    """
    return iv_rank >= threshold


def get_iv_regime(iv_rank: float) -> str:
    """Categorize the current IV regime."""
    if iv_rank >= 70:
        return "high"
    elif iv_rank >= 50:
        return "elevated"
    elif iv_rank >= 30:
        return "normal"
    else:
        return "low"


# Historical IV storage helpers


@dataclass
class IVDataPoint:
    """Single IV observation."""

    date: str
    iv: float


class IVHistory:
    """Manager for historical IV data."""

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._data: dict[str, list[IVDataPoint]] = {}  # symbol -> data points

    def add_observation(self, symbol: str, date: str, iv: float) -> None:
        """Add an IV observation for a symbol."""
        if symbol not in self._data:
            self._data[symbol] = []

        self._data[symbol].append(IVDataPoint(date=date, iv=iv))

        # Trim to lookback period
        cutoff_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        self._data[symbol] = [dp for dp in self._data[symbol] if dp.date >= cutoff_date]

    def get_historical_ivs(self, symbol: str) -> list[float]:
        """Get historical IV values for a symbol."""
        if symbol not in self._data:
            return []
        return [dp.iv for dp in self._data[symbol]]

    def get_metrics(self, symbol: str, current_iv: float) -> IVMetrics:
        """Get IV metrics for a symbol."""
        return calculate_iv_metrics(current_iv, self.get_historical_ivs(symbol))

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        return {
            symbol: [{"date": dp.date, "iv": dp.iv} for dp in data]
            for symbol, data in self._data.items()
        }

    @classmethod
    def from_dict(cls, data: dict, lookback_days: int = 252) -> "IVHistory":
        """Deserialize from dict."""
        history = cls(lookback_days)
        for symbol, points in data.items():
            for point in points:
                history.add_observation(symbol, point["date"], point["iv"])
        return history


# IV Term Structure Analysis


class TermStructureRegime(str, Enum):
    """IV term structure regime classification.

    Contango: IV increases with DTE (normal market condition, favorable for selling vol)
    Backwardation: IV decreases with DTE (fear in near-term, avoid selling vol)
    Flat: IV roughly equal across tenors (neutral)
    """

    CONTANGO = "contango"
    BACKWARDATION = "backwardation"
    FLAT = "flat"


@dataclass
class TermStructurePoint:
    """Single point on the IV term structure."""

    dte: int  # Days to expiration
    iv: float  # Implied volatility


@dataclass
class TermStructureResult:
    """Result of term structure analysis."""

    regime: TermStructureRegime
    signal: str  # Trading signal description
    ratio_30_90: float  # 30-day IV / 90-day IV ratio
    slope: float  # Term structure slope


class IVTermStructure:
    """IV term structure analysis using spline interpolation.

    Analyzes the shape of the IV curve across different expirations to detect
    whether the market is in contango (normal) or backwardation (stressed).

    Trading implications:
    - Contango: Favorable for selling premium (IV is lower near-term)
    - Backwardation: Avoid selling premium (elevated near-term fear)
    """

    # Thresholds for regime classification
    CONTANGO_THRESHOLD = 0.95  # ratio < 0.95 = contango
    BACKWARDATION_THRESHOLD = 1.05  # ratio > 1.05 = backwardation

    def __init__(self, points: list[TermStructurePoint]):
        """Initialize with IV points at different expirations.

        Args:
            points: List of TermStructurePoint with DTE and IV values.
                   Requires at least 3 points for spline fitting.
        """
        if len(points) < 2:
            raise ValueError("Requires at least 2 points for term structure analysis")

        # Sort by DTE
        sorted_points = sorted(points, key=lambda p: p.dte)
        self.dtes = np.array([p.dte for p in sorted_points], dtype=float)
        self.ivs = np.array([p.iv for p in sorted_points], dtype=float)

        # Fit spline if we have enough points
        self._spline: UnivariateSpline | None = None
        if len(points) >= 3:
            try:
                from scipy.interpolate import UnivariateSpline

                # Use smoothing spline (s > 0 allows some deviation)
                self._spline = UnivariateSpline(
                    self.dtes, self.ivs, k=min(3, len(points) - 1), s=0.001
                )
            except (ImportError, ValueError):
                # Fall back to linear interpolation
                self._spline = None

    @classmethod
    def from_options_chain(
        cls, expirations: list[str], ivs: list[float]
    ) -> "IVTermStructure":
        """Create term structure from options chain data.

        Args:
            expirations: List of expiration dates (YYYY-MM-DD format)
            ivs: List of ATM implied volatilities for each expiration

        Returns:
            IVTermStructure instance
        """
        today = datetime.now()
        points = []

        for exp_str, iv in zip(expirations, ivs):
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = max(1, (exp_date - today).days)
            points.append(TermStructurePoint(dte=dte, iv=iv))

        return cls(points)

    def interpolate_iv(self, dte: int) -> float:
        """Get interpolated IV for any DTE.

        Args:
            dte: Days to expiration

        Returns:
            Interpolated IV value
        """
        if self._spline is not None:
            return float(self._spline(dte))

        # Linear interpolation fallback
        return float(np.interp(dte, self.dtes, self.ivs))

    def detect_regime(self) -> TermStructureResult:
        """Detect term structure regime (contango/backwardation/flat).

        Uses the ratio of 30-day IV to 90-day IV:
        - ratio < 0.95: contango (IV increases with DTE)
        - ratio > 1.05: backwardation (IV decreases with DTE)
        - else: flat

        Returns:
            TermStructureResult with regime, signal, ratio, and slope
        """
        # Get IV at standard tenors (or closest available)
        iv_30 = self.interpolate_iv(30)
        iv_90 = self.interpolate_iv(90)

        # Handle edge case where 90-day IV is zero
        if iv_90 <= 0:
            ratio = 1.0
        else:
            ratio = iv_30 / iv_90

        # Calculate slope using linear regression on log(DTE)
        slope = self.get_slope()

        # Classify regime
        if ratio < self.CONTANGO_THRESHOLD:
            regime = TermStructureRegime.CONTANGO
            signal = "favorable_for_selling_vol"
        elif ratio > self.BACKWARDATION_THRESHOLD:
            regime = TermStructureRegime.BACKWARDATION
            signal = "avoid_selling_vol"
        else:
            regime = TermStructureRegime.FLAT
            signal = "neutral"

        return TermStructureResult(
            regime=regime,
            signal=signal,
            ratio_30_90=ratio,
            slope=slope,
        )

    def get_slope(self) -> float:
        """Calculate term structure slope.

        Uses linear regression on log(DTE) vs IV to get the slope.
        Positive slope = contango (IV increases with time)
        Negative slope = backwardation (IV decreases with time)

        Returns:
            Slope coefficient
        """
        if len(self.dtes) < 2:
            return 0.0

        # Use log(DTE) for better linearity
        log_dtes = np.log(self.dtes + 1)  # +1 to avoid log(0)

        # Simple linear regression: IV = a + b * log(DTE)
        n = len(log_dtes)
        sum_x = np.sum(log_dtes)
        sum_y = np.sum(self.ivs)
        sum_xy = np.sum(log_dtes * self.ivs)
        sum_x2 = np.sum(log_dtes * log_dtes)

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return float(slope)

    def to_dict(self) -> dict:
        """Serialize for caching."""
        result = self.detect_regime()
        return {
            "points": [{"dte": int(d), "iv": float(iv)} for d, iv in zip(self.dtes, self.ivs)],
            "regime": result.regime.value,
            "signal": result.signal,
            "ratio_30_90": result.ratio_30_90,
            "slope": result.slope,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IVTermStructure":
        """Deserialize from dict."""
        points = [TermStructurePoint(dte=p["dte"], iv=p["iv"]) for p in data["points"]]
        return cls(points)


# IV Mean Reversion Analysis


class MeanReversionSignal(str, Enum):
    """IV mean reversion trading signal."""

    SELL_VOL = "sell_vol"  # IV elevated, expect reversion down
    BUY_VOL = "buy_vol"  # IV depressed, expect reversion up
    HOLD = "hold"  # No actionable signal


@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters.

    The OU process models mean-reverting behavior:
    dX = theta * (mu - X) * dt + sigma * dW

    Where:
    - theta: Mean reversion speed (higher = faster reversion)
    - mu: Long-term mean (equilibrium level)
    - sigma: Volatility of the process
    - half_life: Time to revert halfway to the mean
    """

    theta: float  # Mean reversion speed
    mu: float  # Long-term mean
    sigma: float  # Volatility of process
    half_life: float  # Days to half-revert


@dataclass
class MeanReversionResult:
    """Result of mean reversion analysis."""

    signal: MeanReversionSignal
    z_score: float
    current_iv: float
    long_term_mean: float
    is_stationary: bool
    adf_p_value: float | None


class IVMeanReversion:
    """IV mean reversion analysis using Ornstein-Uhlenbeck model.

    Models IV as a mean-reverting process and generates trading signals
    when IV deviates significantly from its long-term mean.

    Trading implications:
    - z-score > 2: SELL_VOL (IV elevated, expect to mean-revert down)
    - z-score < -2: BUY_VOL (IV depressed, expect to mean-revert up)
    - otherwise: HOLD
    """

    # Minimum data points required for analysis
    MIN_HISTORY_DAYS = 60

    def __init__(self, iv_history: list[float], dt: float = 1 / 252):
        """Initialize with historical IV values.

        Args:
            iv_history: List of historical IV values (oldest to newest)
            dt: Time step in years (default 1/252 for daily data)
        """
        if len(iv_history) < self.MIN_HISTORY_DAYS:
            raise ValueError(
                f"Requires at least {self.MIN_HISTORY_DAYS} days of IV history, "
                f"got {len(iv_history)}"
            )

        self.iv = np.array(iv_history, dtype=float)
        self.dt = dt
        self._ou_params: OUParameters | None = None

    def estimate_ou_parameters(self) -> OUParameters:
        """Estimate Ornstein-Uhlenbeck parameters via OLS regression.

        Uses the discrete approximation:
        X(t+1) - X(t) = a + b * X(t) + epsilon

        Where:
        - theta = -b / dt
        - mu = -a / b (or mean of series if b is near zero)
        - sigma = std(residuals) / sqrt(dt)
        - half_life = ln(2) / theta

        Returns:
            OUParameters with estimated values
        """
        if self._ou_params is not None:
            return self._ou_params

        # Calculate changes
        iv_lag = self.iv[:-1]
        iv_change = self.iv[1:] - self.iv[:-1]

        # OLS regression: dX = a + b * X + e
        # Design matrix [1, X]
        n = len(iv_lag)
        X = np.column_stack([np.ones(n), iv_lag])
        y = iv_change

        # Solve normal equations: (X'X)^-1 X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y
            coeffs = np.linalg.solve(XtX, Xty)
            a, b = coeffs[0], coeffs[1]
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            a, b = 0.0, -0.01

        # Calculate residuals and sigma
        residuals = y - (a + b * iv_lag)
        sigma = float(np.std(residuals)) / np.sqrt(self.dt)

        # Extract OU parameters
        # Handle case where b is very small (not mean reverting)
        if abs(b) < 1e-10:
            theta = 0.01  # Small positive value
            mu = float(np.mean(self.iv))
            half_life = float("inf")
        else:
            theta = -b / self.dt
            # Ensure theta is positive for mean reversion
            if theta <= 0:
                theta = abs(theta) if abs(theta) > 0.01 else 0.01
            mu = -a / b if abs(b) > 1e-10 else float(np.mean(self.iv))
            half_life = np.log(2) / theta if theta > 0 else float("inf")

        self._ou_params = OUParameters(
            theta=float(theta),
            mu=float(mu),
            sigma=float(sigma),
            half_life=float(half_life),
        )

        return self._ou_params

    def generate_signal(
        self,
        current_iv: float | None = None,
        z_entry: float = 2.0,
    ) -> MeanReversionResult:
        """Generate trading signal based on z-score from mean.

        Args:
            current_iv: Current IV value (uses last in history if None)
            z_entry: Z-score threshold for generating signal (default 2.0)

        Returns:
            MeanReversionResult with signal and details
        """
        params = self.estimate_ou_parameters()

        # Use last IV if current not provided
        if current_iv is None:
            current_iv = float(self.iv[-1])

        # Calculate z-score
        # For OU process, stationary std = sigma / sqrt(2 * theta)
        if params.theta > 0:
            stationary_std = params.sigma / np.sqrt(2 * params.theta)
        else:
            stationary_std = float(np.std(self.iv))

        if stationary_std > 0:
            z_score = (current_iv - params.mu) / stationary_std
        else:
            z_score = 0.0

        # Generate signal
        if z_score > z_entry:
            signal = MeanReversionSignal.SELL_VOL
        elif z_score < -z_entry:
            signal = MeanReversionSignal.BUY_VOL
        else:
            signal = MeanReversionSignal.HOLD

        # Test stationarity
        adf_result = self.test_mean_reversion()

        return MeanReversionResult(
            signal=signal,
            z_score=float(z_score),
            current_iv=current_iv,
            long_term_mean=params.mu,
            is_stationary=adf_result["is_stationary"],
            adf_p_value=adf_result["p_value"],
        )

    def test_mean_reversion(self) -> dict:
        """Test for mean reversion using Augmented Dickey-Fuller test.

        The ADF test checks if a time series is stationary (mean-reverting).
        A p-value < 0.05 indicates the series is stationary.

        Returns:
            dict with adf_statistic, p_value, is_stationary, critical_values
        """
        try:
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(self.iv, maxlag=None, autolag="AIC")
            return {
                "adf_statistic": float(result[0]),
                "p_value": float(result[1]),
                "is_stationary": result[1] < 0.05,
                "critical_values": {k: float(v) for k, v in result[4].items()},
            }
        except ImportError:
            # statsmodels not available, use simple heuristic
            # Check if variance is bounded (rough stationarity check)
            half_n = len(self.iv) // 2
            first_half_var = float(np.var(self.iv[:half_n]))
            second_half_var = float(np.var(self.iv[half_n:]))

            # If variances are similar, likely stationary
            var_ratio = max(first_half_var, second_half_var) / (
                min(first_half_var, second_half_var) + 1e-10
            )
            is_stationary = var_ratio < 2.0

            return {
                "adf_statistic": None,
                "p_value": None,
                "is_stationary": is_stationary,
                "critical_values": {},
            }

    def to_dict(self) -> dict:
        """Serialize for caching."""
        params = self.estimate_ou_parameters()
        return {
            "theta": params.theta,
            "mu": params.mu,
            "sigma": params.sigma,
            "half_life": params.half_life,
            "iv_mean": float(np.mean(self.iv)),
            "iv_std": float(np.std(self.iv)),
            "n_samples": len(self.iv),
        }
