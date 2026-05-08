import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLogWin(unittest.TestCase):

    def test_log_win_inserts_win_row_to_d1(self):
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_win
            log_win("mahler", "Shipped kaizen-reflection skill", "abc1234")
        mock_client.insert_project_log.assert_called_once_with(
            "mahler", "win", "Shipped kaizen-reflection skill", "abc1234"
        )

    def test_log_win_propagates_d1_exception(self):
        mock_client = MagicMock()
        mock_client.insert_project_log.side_effect = RuntimeError("D1 timeout")
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_win
            with self.assertRaises(RuntimeError):
                log_win("mahler", "Shipped something", "abc")


class TestLogBlockerNoKeyword(unittest.TestCase):

    def test_does_not_write_when_no_keywords_in_user_messages(self):
        transcript = {
            "cwd": "/Users/jdhiman/Documents/mahler",
            "messages": [
                {"role": "user", "content": "can you add a feature to the morning brief?"},
                {"role": "assistant", "content": "Sure, I will add it."},
            ],
        }
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/Users/jdhiman/Documents/mahler")
        mock_client.insert_project_log.assert_not_called()

    def test_does_not_write_when_messages_list_is_empty(self):
        transcript = {"cwd": "/tmp/project", "messages": []}
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")
        mock_client.insert_project_log.assert_not_called()

    def test_does_not_write_when_keyword_appears_only_in_assistant_message(self):
        transcript = {
            "cwd": "/tmp/project",
            "messages": [
                {"role": "user", "content": "what should I work on next?"},
                {"role": "assistant", "content": "the issue is that you have many options"},
            ],
        }
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client):
            from project_log import log_blocker_if_triggered
            log_blocker_if_triggered(transcript, "/tmp/project")
        mock_client.insert_project_log.assert_not_called()


class TestLogBlockerKeywordMatch(unittest.TestCase):

    def test_writes_blocker_row_when_keyword_matched(self):
        transcript = {
            "cwd": "/Users/jdhiman/Documents/mahler",
            "messages": [
                {
                    "role": "user",
                    "content": "the main issue is that the model isn't giving accurate scores",
                },
                {"role": "assistant", "content": "Let me look into that."},
            ],
        }
        d1 = MagicMock()
        d1.query.return_value = []

        def fake_run(cmd, **kwargs):
            r = MagicMock(); r.returncode = 0
            r.stdout = "The scoring model produces inaccurate outputs, blocking feature completion."
            return r

        with patch("project_log._get_d1_client", return_value=d1), \
             patch("project_log.subprocess.run", side_effect=fake_run), \
             patch("project_log._derive_project_name", return_value="mahler"), \
             patch("project_log._derive_git_ref", return_value="abc1234"):
            import project_log
            project_log.log_blocker_if_triggered(transcript, "/Users/jdhiman/Documents/mahler")

        blocker_inserts = [
            c for c in d1.query.call_args_list
            if "INSERT INTO project_log" in c.args[0] and c.args[1] and "blocker" in c.args[1]
        ]
        self.assertEqual(len(blocker_inserts), 1)
        params = blocker_inserts[0].args[1]
        self.assertEqual(params[0], "mahler")
        self.assertEqual(params[1], "blocker")
        self.assertIn("inaccurate outputs", params[2])
        self.assertEqual(params[3], "abc1234")

    def test_does_not_write_when_claude_returns_empty_string(self):
        transcript = {
            "cwd": "/tmp/project",
            "messages": [{"role": "user", "content": "i am stuck on this issue"}],
        }
        d1 = MagicMock()
        d1.query.return_value = []

        def fake_run(cmd, **kwargs):
            r = MagicMock(); r.returncode = 0; r.stdout = ""
            return r

        with patch("project_log._get_d1_client", return_value=d1), \
             patch("project_log.subprocess.run", side_effect=fake_run), \
             patch("project_log._derive_project_name", return_value="project"), \
             patch("project_log._derive_git_ref", return_value=""):
            import project_log
            project_log.log_blocker_if_triggered(transcript, "/tmp/project")

        blocker_inserts = [
            c for c in d1.query.call_args_list
            if "INSERT INTO project_log" in c.args[0] and c.args[1] and "blocker" in c.args[1]
        ]
        self.assertEqual(len(blocker_inserts), 0)

    def test_uses_transcript_key_as_fallback_for_messages(self):
        transcript = {
            "cwd": "/tmp/project",
            "transcript": [{"role": "user", "content": "blocked on the D1 migration"}],
        }
        d1 = MagicMock()
        d1.query.return_value = []

        def fake_run(cmd, **kwargs):
            r = MagicMock(); r.returncode = 0
            r.stdout = "D1 migration is blocking progress."
            return r

        with patch("project_log._get_d1_client", return_value=d1), \
             patch("project_log.subprocess.run", side_effect=fake_run), \
             patch("project_log._derive_project_name", return_value="project"), \
             patch("project_log._derive_git_ref", return_value=""):
            import project_log
            project_log.log_blocker_if_triggered(transcript, "/tmp/project")

        blocker_inserts = [
            c for c in d1.query.call_args_list
            if "INSERT INTO project_log" in c.args[0] and c.args[1] and "blocker" in c.args[1]
        ]
        self.assertEqual(len(blocker_inserts), 1)
        params = blocker_inserts[0].args[1]
        self.assertEqual(params[1], "blocker")


class TestLogSessionHeartbeat(unittest.TestCase):

    def test_writes_session_row_with_project_ref_branch(self):
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._derive_project_name", return_value="mahler"), \
             patch("project_log._derive_git_ref", return_value="abc1234"), \
             patch("project_log._derive_branch", return_value="main"):
            from project_log import log_session_heartbeat
            log_session_heartbeat("/Users/jdhiman/Documents/mahler")

        mock_client.ensure_tables.assert_called_once()
        mock_client.insert_session_heartbeat.assert_called_once_with("mahler", "abc1234", "main")

    def test_works_outside_git_repo(self):
        mock_client = MagicMock()
        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._derive_project_name", return_value="tmp"), \
             patch("project_log._derive_git_ref", return_value=""), \
             patch("project_log._derive_branch", return_value=""):
            from project_log import log_session_heartbeat
            log_session_heartbeat("/tmp/project")

        mock_client.insert_session_heartbeat.assert_called_once_with("tmp", "", "")

    def test_propagates_d1_exception(self):
        mock_client = MagicMock()
        mock_client.insert_session_heartbeat.side_effect = RuntimeError("D1 error")
        with patch("project_log._get_d1_client", return_value=mock_client), \
             patch("project_log._derive_project_name", return_value="mahler"), \
             patch("project_log._derive_git_ref", return_value="abc"), \
             patch("project_log._derive_branch", return_value="main"):
            from project_log import log_session_heartbeat
            with self.assertRaises(RuntimeError):
                log_session_heartbeat("/Users/jdhiman/Documents/mahler")


class TestSyncLocalToD1Memory(unittest.TestCase):
    def test_inserts_memory_files_as_local_capture_rows(self):
        with tempfile.TemporaryDirectory() as memdir, tempfile.TemporaryDirectory() as repos:
            memdir_path = Path(memdir)
            (memdir_path / "MEMORY.md").write_text("index file")
            (memdir_path / "user_role.md").write_text("solo dev")

            d1 = MagicMock()
            d1.query.return_value = []

            with patch("project_log._get_d1_client", return_value=d1):
                import project_log
                project_log.sync_local_to_d1(memdir_path, Path(repos))

            memory_inserts = [
                c for c in d1.query.call_args_list
                if "INSERT OR IGNORE INTO local_capture" in c.args[0]
                and c.args[1] and c.args[1][0] == "memory"
            ]
            self.assertEqual(len(memory_inserts), 2)
            for call in memory_inserts:
                params = call.args[1]
                self.assertEqual(params[0], "memory")  # source
                self.assertIn(params[1], {"MEMORY.md", "user_role.md"})  # project = filename
                self.assertTrue(params[2].startswith("#"))  # content starts with "# <filename>\n..."
                self.assertTrue(len(params[3]) == 64)  # sha256 hex


class TestSyncLocalToD1Git(unittest.TestCase):
    def test_inserts_recent_git_commits_per_repo(self):
        import tempfile

        with tempfile.TemporaryDirectory() as memdir, tempfile.TemporaryDirectory() as repos:
            d1 = MagicMock()
            d1.query.return_value = []

            git_log_output = "abc1234 First commit subject\ndef5678 Second commit subject\n"

            def fake_run(cmd, **kwargs):
                result = MagicMock()
                result.returncode = 0
                if cmd[0] == "git" and "log" in cmd:
                    result.stdout = git_log_output
                else:
                    result.stdout = ""
                return result

            repo_dir = Path(repos) / "myproject"
            (repo_dir / ".git").mkdir(parents=True)

            with patch("project_log._get_d1_client", return_value=d1), \
                 patch("project_log.subprocess.run", side_effect=fake_run):
                import project_log
                project_log.sync_local_to_d1(Path(memdir), Path(repos))

            git_inserts = [
                c for c in d1.query.call_args_list
                if "INSERT OR IGNORE INTO local_capture" in c.args[0]
                and c.args[1] and c.args[1][0] == "git"
            ]
            self.assertEqual(len(git_inserts), 2)
            for call in git_inserts:
                params = call.args[1]
                self.assertEqual(params[1], "myproject")
                self.assertIn("commit subject", params[2])


class TestBlockerClassifierViaClaude(unittest.TestCase):
    def test_uses_claude_subprocess_and_inserts_blocker_row(self):
        transcript = {
            "messages": [
                {"role": "user", "content": "I'm stuck on the auth migration"},
            ],
            "cwd": "/tmp/fakeproj",
        }
        d1 = MagicMock()
        d1.query.return_value = []

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if cmd and cmd[0] == "claude":
                result.stdout = "Stuck migrating Postgres auth schema due to FK ordering."
            else:
                result.stdout = ""
            return result

        with patch("project_log._get_d1_client", return_value=d1), \
             patch("project_log.subprocess.run", side_effect=fake_run), \
             patch("project_log._derive_project_name", return_value="fakeproj"), \
             patch("project_log._derive_git_ref", return_value="abc1234"):
            import project_log
            project_log.log_blocker_if_triggered(transcript, "/tmp/fakeproj")

        blocker_inserts = [
            c for c in d1.query.call_args_list
            if "INSERT INTO project_log" in c.args[0]
            and c.args[1] and "blocker" in c.args[1]
        ]
        self.assertEqual(len(blocker_inserts), 1)
        params = blocker_inserts[0].args[1]
        self.assertIn("Postgres auth schema", " ".join(str(p) for p in params))


class TestBlockerClassifierMissingCli(unittest.TestCase):
    def test_returns_no_insert_when_claude_cli_missing(self):
        transcript = {
            "messages": [{"role": "user", "content": "I'm stuck on the auth migration"}],
            "cwd": "/tmp/x",
        }
        d1 = MagicMock()
        d1.query.return_value = []

        def fake_run(cmd, **kwargs):
            if cmd and cmd[0] == "claude":
                raise FileNotFoundError("no such file: claude")
            r = MagicMock(); r.returncode = 0; r.stdout = ""
            return r

        with patch("project_log._get_d1_client", return_value=d1), \
             patch("project_log.subprocess.run", side_effect=fake_run), \
             patch("project_log._derive_project_name", return_value="x"), \
             patch("project_log._derive_git_ref", return_value=""):
            import project_log
            project_log.log_blocker_if_triggered(transcript, "/tmp/x")

        blocker_inserts = [
            c for c in d1.query.call_args_list
            if "INSERT INTO project_log" in c.args[0]
            and c.args[1] and "blocker" in c.args[1]
        ]
        self.assertEqual(len(blocker_inserts), 0)
