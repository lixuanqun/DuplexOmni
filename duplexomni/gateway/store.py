"""SQLite session and task log.

One connection is shared by every session in the process. Tests pass
``:memory:``. ``from_env`` points at a file so a restart can still read
unfinished work.
"""

from __future__ import annotations

import sqlite3
import threading
import time


class SessionStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    goal TEXT NOT NULL DEFAULT '',
                    tool TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    detail TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            self._fail_orphans()
            self._conn.commit()

    def _fail_orphans(self) -> None:
        """A new process cannot resume a task that was running in the last one."""
        now = time.time()
        self._conn.execute(
            """
            UPDATE tasks
            SET status = 'failed',
                detail = '进程重启，任务未完成',
                updated_at = ?
            WHERE status IN ('running', 'queued', 'tool')
            """,
            (now,),
        )

    def ensure_session(self, session_id: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions(session_id, created_at, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, now, now),
            )
            self._conn.commit()

    def upsert_task(
        self,
        session_id: str,
        task_id: str,
        *,
        status: str,
        goal: str = "",
        tool: str = "",
        summary: str = "",
        detail: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT goal, tool, summary, detail FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is not None:
                goal = goal or row["goal"]
                tool = tool or row["tool"]
                summary = summary or row["summary"]
                detail = detail or row["detail"]
            self._conn.execute(
                """
                INSERT INTO tasks(
                    task_id, session_id, status, goal, tool, summary, detail, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status = excluded.status,
                    goal = excluded.goal,
                    tool = excluded.tool,
                    summary = excluded.summary,
                    detail = excluded.detail,
                    updated_at = excluded.updated_at
                """,
                (task_id, session_id, status, goal, tool, summary, detail, now),
            )
            self._conn.commit()

    def append_turn(self, session_id: str, role: str, text: str) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns(session_id, role, text, created_at) VALUES(?, ?, ?, ?)",
                (session_id, role, cleaned[:4000], now),
            )
            self._conn.commit()

    def list_turns(self, session_id: str, *, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, text FROM turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "text": row["text"]} for row in reversed(rows)]

    def list_tasks(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT task_id, status, goal, tool, summary, detail
                FROM tasks WHERE session_id = ? ORDER BY updated_at
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def interrupt_running(self, session_id: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks SET status = 'interrupted', updated_at = ?
                WHERE session_id = ? AND status IN ('running', 'queued', 'tool')
                """,
                (now, session_id),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
