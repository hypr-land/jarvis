#!/usr/bin/env python3


from __future__ import annotations

import os
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GitManager:
    """Light-weight Git helper for basic version-control actions."""

    def __init__(self, repo_path: str | None = None):
        # Default to current working directory
        self.repo_path = os.path.abspath(repo_path or os.getcwd())
        logger.debug(f"GitManager initialized with repo path: {self.repo_path}")


    def _run_git(self, args: list[str], timeout: int = 30) -> Dict[str, Any]:
        """Run a git command and capture its output."""
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if completed.returncode == 0:
                logger.debug("git %s succeeded", " ".join(args))
                return {"success": True, "output": completed.stdout.strip()}
            else:
                logger.error("git %s failed: %s", " ".join(args), completed.stderr)
                return {
                    "success": False,
                    "error": (completed.stderr or completed.stdout).strip(),
                }
        except FileNotFoundError:
            return {"success": False, "error": "git executable not found in PATH."}
        except Exception as exc:  # Catch-all for subprocess errors
            return {"success": False, "error": str(exc)}

    def _in_git_repo(self) -> bool:
        return os.path.isdir(os.path.join(self.repo_path, ".git"))

    def commit_and_push(self, file_path: str, message: str) -> Dict[str, Any]:
        """Stage `file_path`, commit with `message`, and push to current branch."""
        if not self._in_git_repo():
            return {"success": False, "error": "Current directory is not a git repository."}

        abs_file = os.path.abspath(file_path)
        if not os.path.exists(abs_file):
            return {"success": False, "error": f"File not found: {file_path}"}

        # Stage file
        add_res = self._run_git(["add", abs_file])
        if not add_res["success"]:
            return add_res

        # Commit – allow empty to gracefully handle "nothing to commit".
        commit_res = self._run_git(["commit", "-m", message])
        if not commit_res["success"] and "nothing to commit" not in commit_res.get("error", "").lower():
            return commit_res

        # Push to remote
        push_res = self._run_git(["push"])
        if not push_res["success"]:
            return push_res

        # Everything OK
        return {
            "success": True,
            "message": "File committed and pushed successfully.",
            "details": {
                "add": add_res.get("output", ""),
                "commit": commit_res.get("output", commit_res.get("error", "")),
                "push": push_res.get("output", push_res.get("error", "")),
            },
        }

    def push(self) -> Dict[str, Any]:
        """Push current branch to remote."""
        if not self._in_git_repo():
            return {"success": False, "error": "Current directory is not a git repository."}
        return self._run_git(["push"]) 

    def pull(self) -> Dict[str, Any]:
        """Pull updates for the current branch from remote."""
        if not self._in_git_repo():
            return {"success": False, "error": "Current directory is not a git repository."}
        return self._run_git(["pull"]) 

    def add_all(self) -> Dict[str, Any]:
        """Stage all new, modified and deleted files (git add .)."""
        if not self._in_git_repo():
            return {"success": False, "error": "Current directory is not a git repository."}
        return self._run_git(["add", "."]) 