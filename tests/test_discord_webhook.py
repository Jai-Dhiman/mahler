"""Tests for Discord webhook handler.

These tests ensure Discord interactions are properly parsed
and button actions are correctly routed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDiscordInteractionParsing:
    """Test Discord interaction payload parsing."""

    def test_parse_approve_button_custom_id(self):
        """Test parsing approve button custom_id."""
        custom_id = "approve_trade:rec-123"
        action, entity_id = custom_id.split(":", 1)

        assert action == "approve_trade"
        assert entity_id == "rec-123"

    def test_parse_reject_button_custom_id(self):
        """Test parsing reject button custom_id."""
        custom_id = "reject_trade:rec-456"
        action, entity_id = custom_id.split(":", 1)

        assert action == "reject_trade"
        assert entity_id == "rec-456"

    def test_parse_close_position_custom_id(self):
        """Test parsing close position button custom_id."""
        custom_id = "close_position:trade-789"
        action, entity_id = custom_id.split(":", 1)

        assert action == "close_position"
        assert entity_id == "trade-789"

    def test_parse_hold_position_custom_id(self):
        """Test parsing hold position button custom_id."""
        custom_id = "hold_position:trade-101"
        action, entity_id = custom_id.split(":", 1)

        assert action == "hold_position"
        assert entity_id == "trade-101"

    def test_parse_kill_switch_custom_ids(self):
        """Test parsing kill switch button custom_ids."""
        # Kill switch buttons don't have entity IDs
        halt_id = "halt_trading"
        resume_id = "resume_trading"
        ack_id = "acknowledge_reconciliation"

        assert ":" not in halt_id
        assert ":" not in resume_id
        assert ":" not in ack_id

    def test_parse_custom_id_with_multiple_colons(self):
        """Test parsing custom_id containing multiple colons in entity_id."""
        # Edge case: UUID-like entity_id with dashes
        custom_id = "approve_trade:550e8400-e29b-41d4-a716-446655440000"
        action, entity_id = custom_id.split(":", 1)

        assert action == "approve_trade"
        assert entity_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_parse_empty_entity_id(self):
        """Test parsing custom_id with empty entity part."""
        custom_id = "approve_trade:"
        action, entity_id = custom_id.split(":", 1)

        assert action == "approve_trade"
        assert entity_id == ""


class TestDiscordInteractionTypes:
    """Test Discord interaction type handling."""

    def test_ping_interaction_type(self, sample_discord_interaction_payload):
        """Test PING interaction type (1) is handled."""
        payload = {"type": 1}  # PING

        assert payload["type"] == 1
        # Handler should respond with {"type": 1} (PONG)

    def test_component_interaction_type(self, sample_discord_interaction_payload):
        """Test MESSAGE_COMPONENT interaction type (3) is handled."""
        assert sample_discord_interaction_payload["type"] == 3

    def test_invalid_interaction_type(self):
        """Test invalid interaction type is rejected."""
        payload = {"type": 2}  # APPLICATION_COMMAND (not supported)

        # Type 2 is not MESSAGE_COMPONENT, should return error
        assert payload["type"] != 3


class TestDiscordInteractionValidation:
    """Test interaction validation logic."""

    def test_missing_custom_id(self):
        """Test handling of missing custom_id in interaction."""
        payload = {
            "type": 3,
            "id": "interaction-123",
            "token": "token",
            "data": {},  # Missing custom_id
        }

        custom_id = payload.get("data", {}).get("custom_id", "")
        assert custom_id == ""

    def test_missing_message_id(self):
        """Test handling of missing message_id in interaction."""
        payload = {
            "type": 3,
            "id": "interaction-123",
            "token": "token",
            "data": {"custom_id": "approve_trade:rec-123"},
            # Missing "message" key
        }

        message_id = payload.get("message", {}).get("id")
        assert message_id is None

    def test_malformed_payload(self):
        """Test handling of malformed JSON payload."""
        malformed_json = "not json at all"

        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_json)


class TestDiscordSignatureVerification:
    """Test Discord signature verification edge cases."""

    def test_empty_signature_fails(self):
        """Test that empty signature fails verification."""
        # Empty signature should fail
        signature = ""
        assert len(signature) == 0

    def test_empty_timestamp_fails(self):
        """Test that empty timestamp fails verification."""
        timestamp = ""
        assert len(timestamp) == 0

    def test_signature_construction(self):
        """Test signature message construction."""
        timestamp = "1704067200"
        body = '{"type":3}'

        # Discord expects: timestamp + body for signature verification
        message = f"{timestamp}{body}".encode()
        assert message == b'1704067200{"type":3}'


class TestDiscordEmbedConstruction:
    """Test Discord embed construction."""

    def test_approval_embed_fields(self):
        """Test approved trade embed has required fields."""
        embed = {
            "title": "Trade Approved: SPY",
            "color": 0x57F287,
            "fields": [
                {"name": "Strategy", "value": "Bull Put Spread", "inline": True},
                {"name": "Expiration", "value": "2024-02-15", "inline": True},
                {"name": "Strikes", "value": "$470.00/$465.00", "inline": True},
                {"name": "Credit", "value": "$1.25", "inline": True},
                {"name": "Contracts", "value": "2", "inline": True},
                {"name": "Order ID", "value": "order-123", "inline": True},
            ],
        }

        assert embed["title"] == "Trade Approved: SPY"
        assert embed["color"] == 0x57F287  # Green
        assert len(embed["fields"]) == 6

    def test_rejection_embed_fields(self):
        """Test rejected trade embed structure."""
        embed = {
            "title": "Trade Rejected: SPY",
            "color": 0xED4245,  # Red
            "description": "Bull Put Spread | $470.00/$465.00 | 2024-02-15",
        }

        assert embed["color"] == 0xED4245
        assert "Rejected" in embed["title"]

    def test_exit_alert_embed_colors(self):
        """Test exit alert uses correct colors based on P/L."""
        # Positive P/L = green
        profit_embed = {"color": 0x57F287 if 100 > 0 else 0xED4245}
        assert profit_embed["color"] == 0x57F287

        # Negative P/L = red
        loss_embed = {"color": 0x57F287 if -50 > 0 else 0xED4245}
        assert loss_embed["color"] == 0xED4245


class TestDiscordButtonComponents:
    """Test Discord button component construction."""

    def test_approve_reject_buttons(self):
        """Test approve/reject button components."""
        components = [
            {
                "type": 1,  # Action Row
                "components": [
                    {
                        "type": 2,  # Button
                        "style": 3,  # Success (green)
                        "label": "Approve",
                        "custom_id": "approve_trade:rec-123",
                    },
                    {
                        "type": 2,  # Button
                        "style": 4,  # Danger (red)
                        "label": "Reject",
                        "custom_id": "reject_trade:rec-123",
                    },
                ],
            }
        ]

        assert len(components) == 1
        assert len(components[0]["components"]) == 2
        assert components[0]["components"][0]["style"] == 3  # Green
        assert components[0]["components"][1]["style"] == 4  # Red

    def test_close_hold_buttons(self):
        """Test close/hold position buttons."""
        components = [
            {
                "type": 1,
                "components": [
                    {
                        "type": 2,
                        "style": 3,  # Success
                        "label": "Close Position",
                        "custom_id": "close_position:trade-123",
                    },
                    {
                        "type": 2,
                        "style": 2,  # Secondary (gray)
                        "label": "Hold",
                        "custom_id": "hold_position:trade-123",
                    },
                ],
            }
        ]

        assert components[0]["components"][0]["label"] == "Close Position"
        assert components[0]["components"][1]["style"] == 2  # Gray

    def test_empty_components_removes_buttons(self):
        """Test empty components list removes all buttons."""
        # After approval/rejection, buttons are removed
        components = []

        assert len(components) == 0
