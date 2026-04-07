"""Model loader for pre-computed parameters from R2.

Loads trained model parameters from R2 storage with optional KV caching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.db.kv import KVClient


@dataclass
class RegimeModelParams:
    """Parameters for regime detection model."""

    version: str
    trained_at: str
    n_samples: int
    scaler_mean: list[float]  # 7 values
    scaler_scale: list[float]  # 7 values
    gmm_means: list[list[float]]  # 4 x 7 matrix
    gmm_covariances: list[list[list[float]]]  # 4 x 7 x 7 tensor
    gmm_weights: list[float]  # 4 values
    regime_mapping: dict[str, str]  # cluster_id (str) -> regime name

    @classmethod
    def from_dict(cls, data: dict) -> RegimeModelParams:
        """Deserialize from dictionary."""
        return cls(
            version=data["version"],
            trained_at=data["trained_at"],
            n_samples=data["n_samples"],
            scaler_mean=data["scaler_mean"],
            scaler_scale=data["scaler_scale"],
            gmm_means=data["gmm_means"],
            gmm_covariances=data["gmm_covariances"],
            gmm_weights=data["gmm_weights"],
            regime_mapping=data["regime_mapping"],
        )


@dataclass
class WeightModelParams:
    """Parameters for optimized weights."""

    version: str
    trained_at: str
    n_samples: int
    weights_by_regime: dict[str, dict[str, float]]  # regime -> {iv, delta, credit, ev}
    sharpe_by_regime: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict) -> WeightModelParams:
        """Deserialize from dictionary."""
        return cls(
            version=data["version"],
            trained_at=data["trained_at"],
            n_samples=data["n_samples"],
            weights_by_regime=data["weights_by_regime"],
            sharpe_by_regime=data["sharpe_by_regime"],
        )


@dataclass
class ExitModelParams:
    """Parameters for optimized exit settings."""

    version: str
    trained_at: str
    n_samples: int
    profit_target: float
    stop_loss: float
    time_exit_dte: int
    sharpe_ratio: float

    @classmethod
    def from_dict(cls, data: dict) -> ExitModelParams:
        """Deserialize from dictionary."""
        return cls(
            version=data["version"],
            trained_at=data["trained_at"],
            n_samples=data["n_samples"],
            profit_target=data["profit_target"],
            stop_loss=data["stop_loss"],
            time_exit_dte=int(data["time_exit_dte"]),
            sharpe_ratio=data["sharpe_ratio"],
        )


class ModelLoader:
    """Loads pre-computed models from R2 with KV caching."""

    # R2 keys
    REGIME_MODEL_KEY = "models/regime/latest.json"
    WEIGHTS_MODEL_KEY = "models/weights/latest.json"
    EXIT_MODEL_KEY = "models/exit/latest.json"

    # Cache TTL (1 hour)
    CACHE_TTL = 3600

    def __init__(self, r2_binding: Any, kv_client: KVClient | None = None):
        """Initialize model loader.

        Args:
            r2_binding: Cloudflare R2 bucket binding
            kv_client: Optional KV client for caching
        """
        self.r2 = r2_binding
        self.kv = kv_client

    async def get_regime_params(self) -> RegimeModelParams | None:
        """Load regime model parameters.

        Returns:
            RegimeModelParams or None if not found
        """
        data = await self._load_with_cache("model:regime", self.REGIME_MODEL_KEY)
        if data is None:
            return None
        return RegimeModelParams.from_dict(data)

    async def get_weight_params(self) -> WeightModelParams | None:
        """Load weight optimization parameters.

        Returns:
            WeightModelParams or None if not found
        """
        data = await self._load_with_cache("model:weights", self.WEIGHTS_MODEL_KEY)
        if data is None:
            return None
        return WeightModelParams.from_dict(data)

    async def get_exit_params(self) -> ExitModelParams | None:
        """Load exit optimization parameters.

        Returns:
            ExitModelParams or None if not found
        """
        data = await self._load_with_cache("model:exit", self.EXIT_MODEL_KEY)
        if data is None:
            return None
        return ExitModelParams.from_dict(data)

    async def _load_with_cache(self, cache_key: str, r2_key: str) -> dict | None:
        """Load from KV cache, fall back to R2.

        Args:
            cache_key: Key for KV cache
            r2_key: Key for R2 object

        Returns:
            Parsed JSON data or None if not found
        """
        # Try KV cache first
        if self.kv:
            cached = await self.kv.get_json(cache_key)
            if cached:
                return cached

        # Load from R2
        obj = await self.r2.get(r2_key)
        if obj is None:
            return None

        # Parse the response - R2 returns an object with body
        body = await obj.text()
        data = json.loads(body)

        # Cache in KV for subsequent requests
        if self.kv:
            await self.kv.put_json(cache_key, data, expiration_ttl=self.CACHE_TTL)

        return data
