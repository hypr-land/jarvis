import sqlite3
import os
import atexit
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, List, Dict, Any

class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn_pool = []
        self._max_pool_size = 5
        self._lock = threading.Lock()
        atexit.register(self._cleanup)
        
        try:
            self._initialize_database()
        except sqlite3.Error as e:
            raise Exception(f"Failed to initialize database at {db_path}: {str(e)}")

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a connection from pool or create a new one"""
        conn = None
        try:
            with self._lock:
                if self._conn_pool:
                    conn = self._conn_pool.pop()
            
            if conn is None:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,  # 30 second timeout
                    isolation_level=None,  # Autocommit mode
                    check_same_thread=False  # Allow multiple threads
                )
                conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
                conn.execute('PRAGMA busy_timeout=30000')  # 30 second busy timeout
                
            yield conn
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except:
                    pass
            raise
        else:
            # Return connection to pool if there's room
            with self._lock:
                if len(self._conn_pool) < self._max_pool_size:
                    self._conn_pool.append(conn)
                else:
                    try:
                        conn.close()
                    except:
                        pass

    def _cleanup(self):
        """Clean up all database connections"""
        with self._lock:
            for conn in self._conn_pool:
                try:
                    conn.close()
                except:
                    pass
            self._conn_pool.clear()

    def _initialize_database(self):
        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        with self._get_connection() as conn:
            with conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS reminders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        reminder_text TEXT NOT NULL,
                        reminder_time TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                # Add any necessary indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_reminder_time ON reminders(reminder_time)')

    def add_reminder(self, reminder_text: str, reminder_time: str) -> None:
        with self._get_connection() as conn:
            with conn:
                conn.execute('''
                    INSERT INTO reminders (reminder_text, reminder_time)
                    VALUES (?, ?)
                ''', (reminder_text, reminder_time))

    def get_reminders(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM reminders ORDER BY id')
                return [dict(row) for row in cursor.fetchall()]

    def delete_reminder(self, reminder_id: int) -> bool:
        with self._get_connection() as conn:
            with conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM reminders WHERE id = ?', (reminder_id,))
                if not cursor.fetchone():
                    return False
                conn.execute('DELETE FROM reminders WHERE id = ?', (reminder_id,))
                return True
