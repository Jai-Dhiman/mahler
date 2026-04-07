"""Risk management module for the Mahler trading system.

Provides:
- Position sizing with correlation-aware limits
- Circuit breakers with graduated response
- Trade and exit validation
- Three-perspective risk assessment (V2)
- Dynamic exit management (V2)

Usage:
    from core.risk import (
        PositionSizer,
        CircuitBreaker,
        TradeValidator,
        ExitValidator,
        ThreePerspectiveRiskManager,
    )
"""

from core.risk.position_sizer import (
    PositionSizer,
    PositionSizeResult,
    RiskLimits,
)
from core.risk.circuit_breaker import (
    CircuitBreaker,
    GraduatedCircuitBreaker,
    GraduatedConfig,
    RiskLevel,
    RiskState,
)
from core.risk.validators import (
    DynamicExitCalculator,
    DynamicExitResult,
    ExitConfig,
    ExitValidator,
    IVAdjustedExits,
    TradeValidator,
    TradingStyle,
    ValidationResult,
)
from core.risk.three_perspective import (
    PerspectiveAssessment,
    RiskPerspective,
    ThreePerspectiveConfig,
    ThreePerspectiveResult,
    ThreePerspectiveRiskManager,
)

__all__ = [
    # Position Sizer
    "PositionSizer",
    "PositionSizeResult",
    "RiskLimits",
    # Circuit Breaker
    "CircuitBreaker",
    "GraduatedCircuitBreaker",
    "GraduatedConfig",
    "RiskLevel",
    "RiskState",
    # Validators
    "DynamicExitCalculator",
    "DynamicExitResult",
    "ExitConfig",
    "ExitValidator",
    "IVAdjustedExits",
    "TradeValidator",
    "TradingStyle",
    "ValidationResult",
    # Three-Perspective (V2)
    "PerspectiveAssessment",
    "RiskPerspective",
    "ThreePerspectiveConfig",
    "ThreePerspectiveResult",
    "ThreePerspectiveRiskManager",
]
