# config/shared/tests/test_honcho_client.py
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

_CFG = {
    "workspace_id": "mahler",
    "ai_peer_id": "mahler",
    "user_peer_id": "jai",
    "api_key": "test-key",
}


def _mock_sdk():
    mock_conclusions = MagicMock()
    mock_peer = MagicMock()
    mock_peer.conclusions_of.return_value = mock_conclusions
    mock_honcho = MagicMock()
    mock_honcho.peer.return_value = mock_peer
    return mock_honcho, mock_conclusions


class TestConclude(unittest.TestCase):

    def test_conclude_writes_content_and_session_id_to_sdk(self):
        mock_honcho, mock_conclusions = _mock_sdk()
        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            honcho_client.conclude("Jai is focused on traderjoe", session_id="project-synthesis")

        mock_conclusions.create.assert_called_once_with([
            {"content": "Jai is focused on traderjoe", "session_id": "project-synthesis"}
        ])

    def test_conclude_raises_runtime_error_on_sdk_exception(self):
        _, mock_conclusions = _mock_sdk()
        mock_conclusions.create.side_effect = Exception("connection refused")
        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            with self.assertRaises(RuntimeError) as ctx:
                honcho_client.conclude("text")
        self.assertIn("Honcho conclude failed", str(ctx.exception))


class TestListConclusions(unittest.TestCase):

    def test_list_conclusions_filters_out_entries_older_than_since_days(self):
        now = datetime.now(timezone.utc)
        old = MagicMock()
        old.content = "old fact"
        old.created_at = (now - timedelta(days=31)).isoformat()
        recent = MagicMock()
        recent.content = "recent fact"
        recent.created_at = (now - timedelta(days=5)).isoformat()

        _, mock_conclusions = _mock_sdk()
        mock_conclusions.list.return_value = [old, recent]

        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            result = honcho_client.list_conclusions(since_days=30)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "recent fact")

    def test_list_conclusions_raises_when_conclusion_missing_created_at(self):
        item = MagicMock()
        item.created_at = None
        _, mock_conclusions = _mock_sdk()
        mock_conclusions.list.return_value = [item]
        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            with self.assertRaises(RuntimeError) as ctx:
                honcho_client.list_conclusions()
        self.assertIn("missing created_at", str(ctx.exception))

    def test_list_conclusions_raises_runtime_error_on_sdk_exception(self):
        _, mock_conclusions = _mock_sdk()
        mock_conclusions.list.side_effect = Exception("API down")
        with (
            patch("honcho_client._load_config", return_value=_CFG),
            patch("honcho_client._build_conclusions_client", return_value=mock_conclusions),
        ):
            import honcho_client
            with self.assertRaises(RuntimeError) as ctx:
                honcho_client.list_conclusions()
        self.assertIn("Honcho list_conclusions failed", str(ctx.exception))


class TestBuildClient(unittest.TestCase):

    def test_build_conclusions_client_initializes_honcho_with_workspace_and_key(self):
        mock_honcho_module = MagicMock()
        mock_honcho_class = MagicMock()
        mock_honcho_module.Honcho = mock_honcho_class
        mock_instance = MagicMock()
        mock_honcho_class.return_value = mock_instance

        with patch.dict("sys.modules", {"honcho": mock_honcho_module}):
            import honcho_client
            import importlib
            importlib.reload(honcho_client)
            honcho_client._build_conclusions_client(_CFG)

        mock_honcho_class.assert_called_with(workspace_id="mahler", api_key="test-key")
        mock_instance.peer.assert_called_with("mahler")
        mock_instance.peer.return_value.conclusions_of.assert_called_with("jai")


if __name__ == "__main__":
    unittest.main()
