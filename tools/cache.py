"""tools/cache.py - SQLite cache keyed by URL or query using db.py's cache table."""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Optional

from db import DB_PATH


def _get_connection() -> sqlite3.Connection:
    """Return an open connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            expires_at REAL
        );
        """
    )
    return conn


def get_cached(key: str) -> Optional[str]:
    """Retrieve a cached value by key (URL or query).
    
    Returns None on cache miss or expired TTL.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,))
        row = cur.fetchone()
        if not row:
            return None

        val, expires_at = row
        now = time.time()
        if expires_at is not None and expires_at <= now:
            # Expired: clean up
            cur.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return None

        return str(val)
    finally:
        conn.close()


def set_cached(key: str, value: str, ttl: Optional[float] = None) -> None:
    """Store a key-value pair in cache with an optional TTL in seconds."""
    expires_at = (time.time() + ttl) if ttl is not None else None
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO cache (key, value, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                expires_at = excluded.expires_at
            """,
            (key, str(value), expires_at),
        )
        conn.commit()
    finally:
        conn.close()


def delete_cached(key: str) -> None:
    """Delete a key from the cache."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()


def clear_all_cache() -> None:
    """Clear all entries from the cache table (primarily for testing)."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM cache")
        conn.commit()
    finally:
        conn.close()
