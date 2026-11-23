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

# Helper to fetch a DB cursor for other modules (read‑only or write).
def get_connection() -> sqlite3.Connection:
    """Return the shared SQLite connection."""
    return _conn

# Expose the connection for tests that need direct queries.
__all__ = ["log_event", "get_connection", "DB_PATH"]
