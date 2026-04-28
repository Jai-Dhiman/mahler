import sys
from pathlib import Path

for _p in (str(Path.home() / ".hermes" / "shared"), str(Path(__file__).resolve().parents[3] / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d1_base import D1Client as _D1Base


class D1Client(_D1Base):
    def get_recent_project_log(self, days: int = 7) -> list[dict]:
        return self.query(
            "SELECT project, entry_type, summary, git_ref, created_at FROM project_log "
            "WHERE created_at >= datetime('now', ? || ' days') ORDER BY created_at DESC",
            [f"-{days}"],
        )

    def insert_project_log(self, project: str, entry_type: str, summary: str, git_ref: str) -> None:
        self.query(
            "INSERT INTO project_log (project, entry_type, summary, git_ref) VALUES (?, ?, ?, ?)",
            [project, entry_type, summary, git_ref],
        )

    def insert_session_heartbeat(self, project: str, git_ref: str, branch: str) -> None:
        self.query(
            "INSERT INTO project_log (project, entry_type, git_ref, branch) VALUES (?, 'session', ?, ?)",
            [project, git_ref, branch],
        )
