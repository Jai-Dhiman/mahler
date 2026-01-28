from __future__ import annotations

import json

from core import http


class ClaudeError(Exception):
    """Claude API error."""

    pass


class ClaudeRateLimitError(ClaudeError):
    """Claude API rate limit or token exhaustion error."""

    pass


class ClaudeClient:
    """Client for Claude AI analysis."""

    BASE_URL = "https://api.anthropic.com/v1"
    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 1024

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def _request(self, messages: list[dict], system: str) -> str:
        """Make a request to Claude API."""
        try:
            data = await http.request(
                "POST",
                f"{self.BASE_URL}/messages",
                headers=self._headers,
                json_data={
                    "model": self.MODEL,
                    "max_tokens": self.MAX_TOKENS,
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
        """Parse JSON from Claude response, handling markdown code blocks."""
        # Strip markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            text = "\n".join(lines[1:-1])

        return json.loads(text)
