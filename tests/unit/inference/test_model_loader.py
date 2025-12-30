"""Tests for model loader."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.inference.model_loader import (
    ExitModelParams,
    ModelLoader,
    RegimeModelParams,
    WeightModelParams,
)


@pytest.fixture
def sample_regime_data():
    """Sample regime model data for testing."""
    return {
        "version": "1.0.0",
        "trained_at": "2025-01-06T10:00:00",
        "n_samples": 1000,
        "scaler_mean": [0.15, 0.01, 0.005, 0.20, 0.04, 1.0, 0.015],
        "scaler_scale": [0.08, 0.03, 0.02, 0.10, 0.03, 0.3, 0.008],
        "gmm_means": [[0.1] * 7, [0.2] * 7, [0.3] * 7, [0.4] * 7],
        "gmm_covariances": [[[0.01] * 7] * 7] * 4,
        "gmm_weights": [0.25, 0.25, 0.25, 0.25],
        "regime_mapping": {
            "0": "bull_low_vol",
            "1": "bull_high_vol",
            "2": "bear_low_vol",
            "3": "bear_high_vol",
        },
    }


@pytest.fixture
def sample_weight_data():
    """Sample weight model data for testing."""
    return {
        "version": "1.0.0",
        "trained_at": "2025-01-06T10:00:00",
        "n_samples": 500,
        "weights_by_regime": {
            "bull_low_vol": {"iv": 0.30, "delta": 0.25, "credit": 0.25, "ev": 0.20},
            "bull_high_vol": {"iv": 0.35, "delta": 0.20, "credit": 0.25, "ev": 0.20},
        },
        "sharpe_by_regime": {
            "bull_low_vol": 1.5,
            "bull_high_vol": 1.2,
        },
    }


@pytest.fixture
def sample_exit_data():
    """Sample exit model data for testing."""
    return {
        "version": "1.0.0",
        "trained_at": "2025-01-06T10:00:00",
        "n_samples": 200,
        "profit_target": 0.45,
        "stop_loss": 0.18,
        "time_exit_dte": 18,
        "sharpe_ratio": 1.8,
    }


@pytest.fixture
def mock_r2_binding(sample_regime_data, sample_weight_data, sample_exit_data):
    """Create mock R2 binding."""

    async def mock_get(key: str):
        data_map = {
            "models/regime/latest.json": sample_regime_data,
            "models/weights/latest.json": sample_weight_data,
            "models/exit/latest.json": sample_exit_data,
        }

        if key not in data_map:
            return None

        mock_obj = MagicMock()
        mock_obj.text = AsyncMock(return_value=json.dumps(data_map[key]))
        return mock_obj

    mock = MagicMock()
    mock.get = mock_get
    return mock


@pytest.fixture
def mock_kv_client():
    """Create mock KV client."""
    cache = {}

    async def get_json(key: str):
        return cache.get(key)

    async def put_json(key: str, value: dict, expiration_ttl: int | None = None):
        cache[key] = value

    mock = MagicMock()
    mock.get_json = get_json
    mock.put_json = put_json
    mock._cache = cache  # Expose for testing
    return mock


class TestRegimeModelParams:
    """Test RegimeModelParams dataclass."""

    def test_from_dict(self, sample_regime_data):
        """Verify from_dict correctly deserializes data."""
        params = RegimeModelParams.from_dict(sample_regime_data)

        assert params.version == "1.0.0"
        assert params.trained_at == "2025-01-06T10:00:00"
        assert params.n_samples == 1000
        assert len(params.scaler_mean) == 7
        assert len(params.scaler_scale) == 7
        assert len(params.gmm_means) == 4
        assert len(params.gmm_covariances) == 4
        assert len(params.gmm_weights) == 4
        assert len(params.regime_mapping) == 4


class TestWeightModelParams:
    """Test WeightModelParams dataclass."""

    def test_from_dict(self, sample_weight_data):
        """Verify from_dict correctly deserializes data."""
        params = WeightModelParams.from_dict(sample_weight_data)

        assert params.version == "1.0.0"
        assert params.n_samples == 500
        assert "bull_low_vol" in params.weights_by_regime
        assert params.weights_by_regime["bull_low_vol"]["iv"] == 0.30
        assert params.sharpe_by_regime["bull_low_vol"] == 1.5


class TestExitModelParams:
    """Test ExitModelParams dataclass."""

    def test_from_dict(self, sample_exit_data):
        """Verify from_dict correctly deserializes data."""
        params = ExitModelParams.from_dict(sample_exit_data)

        assert params.version == "1.0.0"
        assert params.n_samples == 200
        assert params.profit_target == 0.45
        assert params.stop_loss == 0.18
        assert params.time_exit_dte == 18
        assert params.sharpe_ratio == 1.8


class TestModelLoader:
    """Test ModelLoader class."""

    @pytest.mark.asyncio
    async def test_get_regime_params(self, mock_r2_binding, sample_regime_data):
        """Verify get_regime_params loads from R2."""
        loader = ModelLoader(mock_r2_binding)

        params = await loader.get_regime_params()

        assert params is not None
        assert isinstance(params, RegimeModelParams)
        assert params.version == sample_regime_data["version"]

    @pytest.mark.asyncio
    async def test_get_weight_params(self, mock_r2_binding, sample_weight_data):
        """Verify get_weight_params loads from R2."""
        loader = ModelLoader(mock_r2_binding)

        params = await loader.get_weight_params()

        assert params is not None
        assert isinstance(params, WeightModelParams)
        assert params.version == sample_weight_data["version"]

    @pytest.mark.asyncio
    async def test_get_exit_params(self, mock_r2_binding, sample_exit_data):
        """Verify get_exit_params loads from R2."""
        loader = ModelLoader(mock_r2_binding)

        params = await loader.get_exit_params()

        assert params is not None
        assert isinstance(params, ExitModelParams)
        assert params.profit_target == sample_exit_data["profit_target"]

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        """Verify returns None when model not found in R2."""

        async def mock_get(key: str):
            return None

        mock_r2 = MagicMock()
        mock_r2.get = mock_get

        loader = ModelLoader(mock_r2)

        params = await loader.get_regime_params()
        assert params is None

    @pytest.mark.asyncio
    async def test_caches_in_kv(self, mock_r2_binding, mock_kv_client, sample_regime_data):
        """Verify data is cached in KV after loading from R2."""
        loader = ModelLoader(mock_r2_binding, mock_kv_client)

        # First load - should hit R2
        params = await loader.get_regime_params()
        assert params is not None

        # Check KV cache was populated
        assert "model:regime" in mock_kv_client._cache

    @pytest.mark.asyncio
    async def test_uses_kv_cache(self, mock_kv_client, sample_regime_data):
        """Verify uses KV cache when available."""
        # Pre-populate cache
        mock_kv_client._cache["model:regime"] = sample_regime_data

        # Create R2 mock that should not be called
        mock_r2 = MagicMock()
        mock_r2.get = AsyncMock(side_effect=Exception("Should not be called"))

        loader = ModelLoader(mock_r2, mock_kv_client)

        # Should use cache, not call R2
        params = await loader.get_regime_params()

        assert params is not None
        assert params.version == sample_regime_data["version"]
