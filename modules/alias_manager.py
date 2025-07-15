from __future__ import annotations

import os
import sqlite3
from typing import List, Dict, Any, Optional


class AliasManager:
    """Light-weight wrapper around SQLite for storing user-defined command aliases.

    The aliases are persisted in a table called ``aliases`` inside the given
    SQLite database file (default: ``jarvis.db`` in the current working
    directory). Each record has the following columns:

        id INTEGER PRIMARY KEY AUTOINCREMENT
        name TEXT UNIQUE            -- the alias name the user will type
        command TEXT NOT NULL       -- the command (or command chain) to run
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    Only three operations are required for now: add/update, remove and look-up.
    """

    def __init__(self, db_path: Optional[str] = None):
        # Store DB in the current working dir if path not supplied
        self.db_path = db_path or os.path.join(os.getcwd(), "jarvis.db")
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def add_alias(self, name: str, command: str) -> None:
        """Create or replace an alias."""
        name = name.strip()
        command = command.strip()
        if not name or not command:
            raise ValueError("Alias name and command must be non-empty")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO aliases (name, command) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET command = excluded.command",
                (name, command),
            )

    def remove_alias(self, name: str) -> bool:
        """Delete an alias. Returns True if an alias was removed."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM aliases WHERE name = ?", (name.strip(),))
            return cur.rowcount > 0

    def get_alias(self, name: str) -> Optional[str]:
        """Return the command mapped to *name* or **None** if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT command FROM aliases WHERE name = ?", (name.strip(),))
            row = cur.fetchone()
            return row[0] if row else None

    def list_aliases(self) -> List[Dict[str, Any]]:
        """Return a list of all aliases as dicts: {id, name, command}."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT id, name, command FROM aliases ORDER BY id")
            return [
                {"id": rid, "name": n, "command": cmd} for rid, n, cmd in cur.fetchall()
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        """Create the *aliases* table if it doesn't yet exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    command TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            ) 