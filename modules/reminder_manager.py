from __future__ import annotations

"""Reminder Manager
~~~~~~~~~~~~~~~~~~~~
Light-weight wrapper around the existing `SQLiteManager` for creating, listing
and deleting simple textual reminders.  Each reminder has an *id*, the text and
an ISO-formatted datetime string indicating when the reminder should fire.

This module does **not** implement any scheduling / background notifications –
that will be handled by the caller (e.g. JarvisBot) if desired.  For now it
exposes a minimal CRUD API so commands such as ::

    reminder add 2024-12-31T23:59 "Happy New Year!"
    reminder list
    reminder remove 3

can be wired up easily.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import os

from modules.sqlite_manager import SQLiteManager


class ReminderManager:
    """High-level helper for the *reminders* table defined inside
    :pymod:`modules.sqlite_manager`.
    """

    def __init__(self, db_path: Optional[str] = None):
        # Default to the same location as *AliasManager* (pwd/jarvis.db)
        self.db_path = db_path or os.path.join(os.getcwd(), "jarvis.db")
        self._db = SQLiteManager(self.db_path)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def add(self, when_iso: str, text: str) -> Dict[str, Any]:
        """Insert a new reminder.

        Parameters
        ----------
        when_iso: str
            ISO-8601 (or any parsable) datetime string such as
            ``2024-12-31T23:59``.  The string is stored verbatim – the caller is
            responsible for correct formatting.
        text: str
            Reminder text.
        """
        if not text.strip():
            return {"success": False, "error": "Reminder text cannot be empty"}
        if not when_iso.strip():
            return {"success": False, "error": "Reminder time cannot be empty"}
        try:
            # Validate – *datetime.fromisoformat* accepts both date and datetime
            # strings, raise ValueError if it fails so we can return nice error.
            datetime.fromisoformat(when_iso)
        except ValueError:
            return {
                "success": False,
                "error": "Invalid datetime format. Use ISO e.g. 2024-12-31T23:59",
            }
        try:
            self._db.add_reminder(text, when_iso)
            return {"success": True}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list(self) -> Dict[str, Any]:
        """Return **all** reminders ordered by *id*."""
        try:
            rows = self._db.get_reminders()
            return {"success": True, "reminders": rows, "count": len(rows)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def remove(self, reminder_id: int) -> Dict[str, Any]:
        """Delete a reminder by primary key."""
        try:
            ok = self._db.delete_reminder(int(reminder_id))
            if ok:
                return {"success": True}
            return {"success": False, "error": f"Reminder {reminder_id} not found"}
        except Exception as exc:
            return {"success": False, "error": str(exc)} 