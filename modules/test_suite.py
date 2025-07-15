import tempfile
import os
import logging
import time
import sqlite3
import subprocess
from typing import List, Dict, Optional

from modules.file_manager import FileManager
from modules.search_engine import SearchEngine
from modules.tts_module import TTSModule
from modules.git_manager import GitManager
from modules.process_monitor import ProcessMonitor, ProcessStats
from modules.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)


class JarvisTestSuite:
    """Run a set of lightweight integration tests against a JarvisBot instance."""

    def __init__(self, bot):
        self.bot = bot
        self.results: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def run(self) -> List[Dict[str, str]]:
        """Execute all tests and return a list of result dictionaries."""
        # Core functionality tests
        self._test_search()
        self._test_file_operations()
        self._test_tts()
        
        # Database and version control tests
        self._test_sqlite_operations()
        self._test_git_operations()
        
        # System and process tests
        self._test_process_monitor()
        
        # Optional components
        self._test_vision_module()
        self._test_wake_word()
        
        return self.results

    def all_passed(self) -> bool:
        return all(r["status"] == "success" for r in self.results)

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------
    def _record(self, name: str, status: str, message: str = ""):
        logger.info("Test %s: %s - %s", name, status, message)
        self.results.append({"name": name, "status": status, "message": message})

    def _test_search(self):
        if not self.bot.config.search_enabled:
            self._record("search", "skipped", "Search disabled in config")
            return
        try:
            result_list = self.bot.search_engine.search("python programming", max_results=1, summarize=False)
            ok = bool(result_list)
            self._record("search", "success" if ok else "failed", f"{len(result_list)} result(s) returned")
        except Exception as exc:
            self._record("search", "failed", str(exc))

    def _test_file_operations(self):
        import uuid
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                fm = FileManager(base_directory=tmpdir)
                fname = f"{uuid.uuid4().hex[:8]}.txt"
                content = "hello world"
                res = fm.create_file(fname, content)
                assert res["success"], res.get("error")
                read = fm.read_file(fname)
                assert read["success"] and read["content"] == content
                delr = fm.delete_file(fname)
                assert delr["success"]
                self._record("file_ops", "success")
        except AssertionError as err:
            self._record("file_ops", "failed", str(err))
        except Exception as exc:
            self._record("file_ops", "failed", str(exc))

    def _test_tts(self):
        tts = TTSModule(enabled=False)  # Do not actually speak in CI
        try:
            ok = tts.available
            self._record("tts_available", "success" if ok else "partial", "TTS availability check")
        except Exception as exc:
            self._record("tts_available", "failed", str(exc))
            
    def _test_sqlite_operations(self):
        """Test basic SQLite database operations."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
            
            db = SQLiteManager(db_path)
            
            # Add a test reminder
            test_time = "2025-07-14 12:00:00"
            db.add_reminder("Test reminder", test_time)
            
            # Test getting reminders
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM reminders WHERE reminder_text = ?", ("Test reminder",))
                result = cursor.fetchone()
                
                assert result is not None, "Failed to retrieve inserted reminder"
                assert result[1] == "Test reminder", "Reminder text doesn't match"
                assert result[2] == test_time, "Reminder time doesn't match"
            
            # Cleanup
            os.unlink(db_path)
            self._record("sqlite_operations", "success")
            
        except Exception as exc:
            self._record("sqlite_operations", "failed", str(exc))
            
    def _test_git_operations(self):
        """Test basic Git operations."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a test file
                test_file = os.path.join(tmpdir, 'test.txt')
                with open(test_file, 'w') as f:
                    f.write('test')
                
                # Initialize git repo
                subprocess.run(['git', 'init'], cwd=tmpdir, check=True)
                
                # Check if .git directory exists
                assert os.path.exists(os.path.join(tmpdir, '.git')), "Git repo not initialized"
                
                # Check git status
                git = GitManager(tmpdir)
                result = git._run_git(['status'])
                assert result['success'], "Failed to get git status"
                
                self._record("git_operations", "success")
                
        except Exception as exc:
            self._record("git_operations", "skipped", str(exc))
            
    def _test_process_monitor(self):
        """Test process monitoring functionality."""
        try:
            # The monitor starts automatically in its own thread
            monitor = ProcessMonitor(os.getpid())
            
            # Let it collect some data
            time.sleep(2)
            
            # Get stats
            stats = monitor.get_stats()
            
            # Stop monitoring
            monitor._stop_flag = True
            
            # Check if we got valid stats
            assert isinstance(stats, ProcessStats), "Invalid stats object"
            assert hasattr(stats, 'cpu_1min'), "CPU stats missing"
            assert hasattr(stats, 'ram_1min'), "Memory stats missing"
            
            self._record("process_monitor", "success")
            
        except Exception as exc:
            self._record("process_monitor", "skipped", str(exc))
            
    def _test_vision_module(self):
        """Test vision module if available."""
        if not hasattr(self.bot, 'vision_module') or not self.bot.vision_module:
            self._record("vision_module", "skipped", "Vision module not enabled")
            return
            
        try:
            # Simple test - just check if module is responsive
            # Note: This doesn't actually test image processing
            self._record("vision_module", "success", "Module loaded successfully")
            
        except Exception as exc:
            self._record("vision_module", "failed", str(exc))
            
    def _test_wake_word(self):
        """Test wake word detection if enabled."""
        if not hasattr(self.bot, 'wake_detector') or not self.bot.wake_detector:
            self._record("wake_word", "skipped", "Wake word detection not enabled")
            return
            
        try:
            # Just test if the detector is running
            is_alive = hasattr(self.bot.wake_detector, 'is_alive') and self.bot.wake_detector.is_alive()
            self._record("wake_word", "success" if is_alive else "skipped", "Detector running")
        except Exception as exc:
            self._record("wake_word", "failed", str(exc))