# db.py
"""SQLite persistence layer for AgentFlow.

Defines the schema used by all agents and provides a tiny helper
`log_event(task_id, agent, type, payload)` that writes an ``Event`` row.
The module lazily creates the database file ``agentflow.db`` in the
project root if it does not exist.
"""

import json
import os
import sqlite3
from typing import Any, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "agentflow.db")

# Ensure the directory exists and initialise the DB on first import.
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_cur = _conn.cursor()

# Create tables if they are missing.
_cur.executescript(
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        created_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL,
        agent TEXT NOT NULL,
        type TEXT NOT NULL,
        payload TEXT NOT NULL,
        timestamp REAL NOT NULL,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    );

    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL,
        vendor TEXT NOT NULL,
        attribute TEXT NOT NULL,
        value TEXT NOT NULL,
        source_url TEXT NOT NULL,
        snippet TEXT NOT NULL,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    );

    CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        expires_at REAL
    );

    CREATE TABLE IF NOT EXISTS approvals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL,
        approved INTEGER NOT NULL,
        approved_by TEXT,
        timestamp REAL NOT NULL,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    );

    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL,
        start_ts REAL NOT NULL,
        end_ts REAL,
        status TEXT,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
    );
    """
)
_conn.commit()


def _serialize(payload: Dict[str, Any]) -> str:
    """Serialize a payload dict to JSON for storage."""
    return json.dumps(payload, ensure_ascii=False)


def log_event(task_id: int, agent: str, type: str, payload: Dict[str, Any]) -> None:
    """Insert a new event row.

    Parameters
    ----------
    task_id: int
        Identifier of the task this event belongs to.
    agent: str
        Name of the agent emitting the event (e.g. "research").
    type: str
        Short machine‑readable event type.
    payload: dict
        Arbitrary JSON‑serialisable data describing the event.
    """
    timestamp = __import__("time").time()
    _cur.execute(
        "INSERT INTO events (task_id, agent, type, payload, timestamp) VALUES (?, ?, ?, ?, ?)",
        (task_id, agent, type, _serialize(payload), timestamp),
    )
    _conn.commit()


# ---------------------------------------------------------------------------
# Task History Save / Load
# ---------------------------------------------------------------------------
def create_task(description: str) -> int:
    """Create a new task record and return its ID."""
    created_at = __import__("time").time()
    _cur.execute(
        "INSERT INTO tasks (description, created_at) VALUES (?, ?)",
        (description, created_at),
    )
    _conn.commit()
    return _cur.lastrowid  # type: ignore


def record_run_start(task_id: int) -> int:
    """Record the beginning of a plan execution run for task_id."""
    start_ts = __import__("time").time()
    _cur.execute(
        "INSERT INTO runs (task_id, start_ts, status) VALUES (?, ?, ?)",
        (task_id, start_ts, "running"),
    )
    _conn.commit()
    return _cur.lastrowid  # type: ignore


def record_run_end(run_id: int, status: str) -> None:
    """Record completion of a plan execution run."""
    end_ts = __import__("time").time()
    _cur.execute(
        "UPDATE runs SET end_ts = ?, status = ? WHERE id = ?",
        (end_ts, status, run_id),
    )
    _conn.commit()


def list_runs(limit: int = 50) -> list[Dict[str, Any]]:
    """List past runs with task description, timestamps, and status."""
    _cur.execute(
        """
        SELECT r.id, r.task_id, t.description, r.start_ts, r.end_ts, r.status
        FROM runs r
        LEFT JOIN tasks t ON r.task_id = t.id
        ORDER BY r.start_ts DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = _cur.fetchall()
    return [
        {
            "id": row[0],
            "task_id": row[1],
            "description": row[2] or "",
            "start_ts": row[3],
            "end_ts": row[4],
            "status": row[5],
        }
        for row in rows
    ]


def load_run_events(task_id: int) -> list[Dict[str, Any]]:
    """Load all events for a given task_id ordered chronologically."""
    _cur.execute(
        """
        SELECT id, task_id, agent, type, payload, timestamp
        FROM events
        WHERE task_id = ?
        ORDER BY id ASC
        """,
        (task_id,),
    )
    rows = _cur.fetchall()
    results = []
    for row in rows:
        try:
            payload = json.loads(row[4])
        except Exception:
            payload = {"raw": row[4]}
        results.append(
            {
                "id": row[0],
                "task_id": row[1],
                "agent": row[2],
                "type": row[3],
                "payload": payload,
                "timestamp": row[5],
            }
        )
    return results


def list_tasks(limit: int = 50) -> list[Dict[str, Any]]:
    """List recent tasks."""
    _cur.execute(
        "SELECT id, description, created_at FROM tasks ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = _cur.fetchall()
    return [
        {"id": row[0], "description": row[1], "created_at": row[2]}
        for row in rows
    ]


# Helper to fetch a DB cursor for other modules (read‑only or write).
def get_connection() -> sqlite3.Connection:
    """Return the shared SQLite connection."""
    return _conn


# Expose functions.
__all__ = [
    "log_event",
    "create_task",
    "record_run_start",
    "record_run_end",
    "list_runs",
    "load_run_events",
    "list_tasks",
    "get_connection",
    "DB_PATH",
]
