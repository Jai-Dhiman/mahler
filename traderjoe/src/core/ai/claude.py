from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core import http


class ClaudeError(Exception):
    """Claude API error."""

    pass


class ClaudeRateLimitError(ClaudeError):
    """Claude API rate limit or token exhaustion error."""

    pass


@dataclass
class ClaudeResponse:
    """Response from Claude API with optional thinking trace.

    Attributes:
        text: The main response text content
        thinking: Optional extended thinking/reasoning trace (when enabled)
        model: Model that generated the response
        usage: Token usage information
    """

    text: str
    thinking: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "thinking": self.thinking,
            "model": self.model,
            "usage": self.usage,
        }


class ClaudeClient:
    """Client for Claude AI analysis.

    Supports model selection for multi-model routing:
    - Default: claude-sonnet-4-20250514
    - Can be overridden at init or per-request
    """

    BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        """Initialize the Claude client.

        Args:
            api_key: Anthropic API key
            model: Optional model override (default: claude-sonnet-4-20250514)
            max_tokens: Optional max_tokens override (default: 1024)
        """
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def _request(
        self,
        messages: list[dict],
        system: str,
        model_override: str | None = None,
        max_tokens_override: int | None = None,
    ) -> str:
        """Make a request to Claude API.

        Args:
            messages: List of message dictionaries
            system: System prompt
            model_override: Optional model override for this request
            max_tokens_override: Optional max_tokens override for this request

        Returns:
            Response text content
        """
        model = model_override or self.model
        max_tokens = max_tokens_override or self.max_tokens

        try:
            data = await http.request(
                "POST",
                f"{self.BASE_URL}/messages",
                headers=self._headers,
                json_data={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": messages,
                },
            )

            content = data.get("content", [])
            if not content:
                raise ClaudeError("Empty response from Claude")

            return content[0].get("text", "")
        except Exception as e:
            error_str = str(e)
            # Check for rate limit / token exhaustion errors
            if any(
                indicator in error_str.lower()
                for indicator in [
                    "http 429",
                    "rate_limit",
                    "rate limit",
                    "overloaded",
                    "credit balance",
                    "insufficient_quota",
                    "billing",
                ]
            ):
                raise ClaudeRateLimitError(f"Claude API rate limit/token error: {error_str}")
            raise ClaudeError(f"Claude API error: {error_str}")

    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from Claude response, handling markdown code blocks and prose."""
        import re

        text = text.strip()

        # Try direct parse first (bare JSON)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract from fenced code block (```json ... ``` with possible trailing text)
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            return json.loads(fence_match.group(1).strip())

        # Extract first JSON object embedded in prose
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))

        raise json.JSONDecodeError("No JSON found in response", text, 0)

    async def _request_with_thinking(
        self,
        messages: list[dict],
        system: str,
        model_override: str | None = None,
        budget_tokens: int = 10000,
    ) -> ClaudeResponse:
        """Make a request with extended thinking enabled.

        Extended thinking captures Claude's chain-of-thought reasoning
        process, useful for analysis tasks and trajectory logging.

        Note: Extended thinking requires specific models (claude-3-5-sonnet
        or later) and increases token usage. Use sparingly for important
        decisions where the reasoning trace is valuable.

        Args:
            messages: List of message dictionaries
            system: System prompt
            model_override: Optional model override (must support extended thinking)
            budget_tokens: Maximum tokens for thinking (default 10000)

        Returns:
            ClaudeResponse with text and thinking content
        """
        model = model_override or self.model

        # Extended thinking requires larger max_tokens
        max_tokens = max(self.max_tokens, 16000)

        try:
            data = await http.request(
                "POST",
                f"{self.BASE_URL}/messages",
                headers=self._headers,
                json_data={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": messages,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                    },
                },
            )

            content = data.get("content", [])
            if not content:
                raise ClaudeError("Empty response from Claude")

            # Extract text and thinking from response
            text_content = ""
            thinking_content = None

            for block in content:
                if block.get("type") == "thinking":
                    thinking_content = block.get("thinking", "")
                elif block.get("type") == "text":
                    text_content = block.get("text", "")

            return ClaudeResponse(
                text=text_content,
                thinking=thinking_content,
                model=data.get("model"),
                usage=data.get("usage"),
            )

        except Exception as e:
            error_str = str(e)
            # Check for rate limit / token exhaustion errors
            if any(
                indicator in error_str.lower()
                for indicator in [
                    "http 429",
                    "rate_limit",
                    "rate limit",
                    "overloaded",
                    "credit balance",
                    "insufficient_quota",
                    "billing",
                ]
            ):
                raise ClaudeRateLimitError(f"Claude API rate limit/token error: {error_str}")
            # Extended thinking may not be supported - fall back to regular request
            if "thinking" in error_str.lower() or "unsupported" in error_str.lower():
                # Fall back to regular request without thinking
                text = await self._request(messages, system, model_override)
                return ClaudeResponse(text=text, thinking=None, model=model)
            raise ClaudeError(f"Claude API error: {error_str}")
