"""LLM Router for multi-model agent system.

Routes agent requests to appropriate LLM models based on agent type:
- Analysts (IV, Tech, Macro, Greeks): Claude Haiku for fast, structured output
- Researchers (Bull, Bear): Claude Sonnet for complex argumentation
- Decision chain (Facilitator, Trader, Fund Manager): Claude Sonnet for reasoning
- Risk deliberation: Claude Haiku for structured perspective output
- Reflection: Claude Haiku for pattern extraction

This routing strategy balances cost and capability:
- Total cost stays similar to all-Sonnet approach
- Analysts run faster with Haiku
- Critical decision points get full Sonnet reasoning
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.ai.claude import ClaudeClient


class ModelTier(str, Enum):
    """Model tiers for routing."""

    FAST = "fast"  # Claude Haiku - quick, structured output
    REASONING = "reasoning"  # Claude Sonnet - complex reasoning


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    max_tokens: int = 1024
    temperature: float = 0.0
    tier: ModelTier = ModelTier.REASONING

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tier": self.tier.value,
        }


# Model definitions
HAIKU_CONFIG = ModelConfig(
    model_id="claude-haiku-4-5-20251101",
    max_tokens=1024,
    temperature=0.0,
    tier=ModelTier.FAST,
)

SONNET_CONFIG = ModelConfig(
    model_id="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    tier=ModelTier.REASONING,
)


# Agent to model mapping
# Based on research recommendations in the implementation plan:
# - Analysts: Haiku (structured output, parallel calls)
# - Researchers: Sonnet (complex argumentation)
# - Decision chain: Sonnet (critical decision points)
# - Risk agents: Haiku (structured perspective output)
# - Reflection: Haiku (pattern extraction)

DEFAULT_AGENT_MODEL_MAP: dict[str, ModelConfig] = {
    # Analysts - cheaper model for structured output
    "iv_analyst": HAIKU_CONFIG,
    "technical_analyst": HAIKU_CONFIG,
    "macro_analyst": HAIKU_CONFIG,
    "greeks_analyst": HAIKU_CONFIG,

    # Researchers - need complex reasoning for debate
    "bull_researcher": SONNET_CONFIG,
    "bear_researcher": SONNET_CONFIG,

    # Decision chain - critical reasoning points
    "facilitator": SONNET_CONFIG,
    "trader": SONNET_CONFIG,
    "fund_manager": SONNET_CONFIG,

    # Risk deliberation - structured perspective output
    "aggressive_risk": HAIKU_CONFIG,
    "neutral_risk": HAIKU_CONFIG,
    "conservative_risk": HAIKU_CONFIG,
    "risk_facilitator": SONNET_CONFIG,  # Synthesis needs reasoning

    # Reflection - pattern extraction
    "reflection_engine": HAIKU_CONFIG,
}


class LLMRouter:
    """Routes agent requests to appropriate LLM models.

    The router maintains a mapping of agent IDs to model configurations
    and provides appropriately configured clients for each agent.

    Usage:
        router = LLMRouter(api_key="...")
        client = router.get_client("iv_analyst")  # Returns Haiku client
        client = router.get_client("bull_researcher")  # Returns Sonnet client
    """

    def __init__(
        self,
        api_key: str,
        agent_model_map: dict[str, ModelConfig] | None = None,
        default_config: ModelConfig | None = None,
    ):
        """Initialize the LLM router.

        Args:
            api_key: Anthropic API key
            agent_model_map: Custom mapping of agent IDs to models
            default_config: Default model config for unknown agents
        """
        self.api_key = api_key
        self.agent_model_map = agent_model_map or DEFAULT_AGENT_MODEL_MAP.copy()
        self.default_config = default_config or SONNET_CONFIG

        # Client cache to avoid creating multiple clients
        self._client_cache: dict[str, ClaudeClient] = {}

    def get_config(self, agent_id: str) -> ModelConfig:
        """Get model configuration for an agent.

        Args:
            agent_id: ID of the agent requesting a client

        Returns:
            ModelConfig for this agent
        """
        # Normalize agent_id to lowercase for matching
        normalized_id = agent_id.lower()

        # Check exact match first
        if normalized_id in self.agent_model_map:
            return self.agent_model_map[normalized_id]

        # Check for partial matches (e.g., "iv_analyst_v2" matches "iv_analyst")
        for key, config in self.agent_model_map.items():
            if key in normalized_id or normalized_id in key:
                return config

        # Fallback to default
        return self.default_config

    def get_client(self, agent_id: str) -> ClaudeClient:
        """Get an appropriately configured Claude client for an agent.

        Args:
            agent_id: ID of the agent requesting a client

        Returns:
            ClaudeClient configured for this agent's model
        """
        from core.ai.claude import ClaudeClient

        config = self.get_config(agent_id)

        # Check cache
        cache_key = config.model_id
        if cache_key in self._client_cache:
            return self._client_cache[cache_key]

        # Create new client
        client = ClaudeClient(
            api_key=self.api_key,
            model=config.model_id,
            max_tokens=config.max_tokens,
        )

        self._client_cache[cache_key] = client
        return client

    def set_agent_model(self, agent_id: str, config: ModelConfig) -> None:
        """Override the model config for a specific agent.

        Args:
            agent_id: ID of the agent
            config: ModelConfig to use
        """
        self.agent_model_map[agent_id.lower()] = config

        # Clear cache for this model to pick up changes
        if config.model_id in self._client_cache:
            del self._client_cache[config.model_id]

    def set_all_to_model(self, config: ModelConfig) -> None:
        """Set all agents to use the same model.

        Useful for testing or when you want consistent behavior.

        Args:
            config: ModelConfig to use for all agents
        """
        for agent_id in self.agent_model_map:
            self.agent_model_map[agent_id] = config

        self._client_cache.clear()

    def get_model_for_tier(self, tier: ModelTier) -> ModelConfig:
        """Get the model configuration for a tier.

        Args:
            tier: The model tier

        Returns:
            ModelConfig for this tier
        """
        if tier == ModelTier.FAST:
            return HAIKU_CONFIG
        return SONNET_CONFIG

    def get_routing_summary(self) -> dict[str, Any]:
        """Get a summary of current routing configuration."""
        fast_agents = []
        reasoning_agents = []

        for agent_id, config in self.agent_model_map.items():
            if config.tier == ModelTier.FAST:
                fast_agents.append(agent_id)
            else:
                reasoning_agents.append(agent_id)

        return {
            "fast_model": HAIKU_CONFIG.model_id,
            "reasoning_model": SONNET_CONFIG.model_id,
            "fast_agents": sorted(fast_agents),
            "reasoning_agents": sorted(reasoning_agents),
            "default_tier": self.default_config.tier.value,
        }

    def estimate_cost_per_pipeline(
        self,
        tokens_per_call: int = 500,
    ) -> dict[str, float]:
        """Estimate cost per pipeline run.

        Based on 2026 pricing:
        - Haiku: $1/M input, $5/M output
        - Sonnet: $3/M input, $15/M output

        Args:
            tokens_per_call: Average tokens per agent call

        Returns:
            Cost breakdown by tier and total
        """
        haiku_input_cost = 1.0 / 1_000_000
        haiku_output_cost = 5.0 / 1_000_000
        sonnet_input_cost = 3.0 / 1_000_000
        sonnet_output_cost = 15.0 / 1_000_000

        fast_count = sum(
            1 for c in self.agent_model_map.values()
            if c.tier == ModelTier.FAST
        )
        reasoning_count = sum(
            1 for c in self.agent_model_map.values()
            if c.tier == ModelTier.REASONING
        )

        # Assume similar input/output tokens
        haiku_cost = fast_count * tokens_per_call * (haiku_input_cost + haiku_output_cost)
        sonnet_cost = reasoning_count * tokens_per_call * (sonnet_input_cost + sonnet_output_cost)

        return {
            "haiku_calls": fast_count,
            "sonnet_calls": reasoning_count,
            "haiku_cost": haiku_cost,
            "sonnet_cost": sonnet_cost,
            "total_cost": haiku_cost + sonnet_cost,
        }


def create_router_from_env() -> LLMRouter:
    """Create a router using API key from environment.

    Returns:
        LLMRouter configured with environment API key
    """
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    return LLMRouter(api_key=api_key)
