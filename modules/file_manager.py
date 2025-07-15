#!/usr/bin/env python3


import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import mimetypes

class FileManager:
    """Handles file operations for JARVIS"""

    def __init__(self, base_directory: str = None):
        self.base_directory = Path(base_directory) if base_directory else Path.cwd()
        self.allowed_extensions = {
            '.txt', '.md', '.py', '.js', '.html', '.css', '.json',
            '.yaml', '.yml', '.xml', '.csv', '.log', '.sh', '.bat'
        }
        self.max_file_size = 10 * 1024 * 1024  # 10MB limit

    def create_file(self, filename: str, content: str = "", overwrite: bool = False) -> Dict[str, Any]:
        """Create a new file with optional content"""
        try:
            file_path = self.base_directory / filename

            # Security check - ensure file is within base directory
            if not self._is_safe_path(file_path):
                return {"success": False, "error": "Invalid file path - outside allowed directory"}

            # Check file extension
            if file_path.suffix.lower() not in self.allowed_extensions:
                return {"success": False, "error": f"File extension not allowed. Allowed: {', '.join(self.allowed_extensions)}"}

            # Check if file exists
            if file_path.exists() and not overwrite:
                return {"success": False, "error": "File already exists. Use overwrite=True to replace"}

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                "success": True,
                "message": f"File created: {file_path}",
                "path": str(file_path),
                "size": len(content.encode('utf-8'))
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to create file: {str(e)}"}

    def read_file(self, filename: str) -> Dict[str, Any]:
        """Read content from a file"""
        try:
            file_path = self.base_directory / filename

            if not self._is_safe_path(file_path):
                return {"success": False, "error": "Invalid file path"}

            if not file_path.exists():
                return {"success": False, "error": "File does not exist"}

            if file_path.stat().st_size > self.max_file_size:
                return {"success": False, "error": "File too large to read"}

            # Determine if file is text or binary
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and not mime_type.startswith('text'):
                return {"success": False, "error": "Cannot read binary files"}

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": file_path.stat().st_size
            }

        except UnicodeDecodeError:
            return {"success": False, "error": "File contains non-text content"}
        except Exception as e:
            return {"success": False, "error": f"Failed to read file: {str(e)}"}

    def delete_file(self, filename: str) -> Dict[str, Any]:
        """Delete a file"""
        try:
            file_path = self.base_directory / filename

            if not self._is_safe_path(file_path):
                return {"success": False, "error": "Invalid file path"}

            if not file_path.exists():
                return {"success": False, "error": "File does not exist"}

            if file_path.is_dir():
                return {"success": False, "error": "Cannot delete directories with this method"}

            file_path.unlink()

            return {
                "success": True,
                "message": f"File deleted: {file_path}",
                "path": str(file_path)
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to delete file: {str(e)}"}

    def list_files(self, pattern: str = "*") -> Dict[str, Any]:
        """List files in the base directory"""
        try:
            files = []
            for file_path in self.base_directory.glob(pattern):
                if file_path.is_file():
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path.relative_to(self.base_directory)),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime
                    })

            return {
                "success": True,
                "files": files,
                "count": len(files)
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to list files: {str(e)}"}

    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe (within base directory)"""
        try:
            path.resolve().relative_to(self.base_directory.resolve())
            return True
        except ValueError:
            return False

    # Add these methods to the JarvisBot class:

    def init_file_manager(self):
        """Initialize file manager - add this to JarvisBot.__init__"""
        self.file_manager = FileManager()

    async def handle_file_operations(self, user_input: str) -> Optional[str]:
        """Handle file operation commands"""
        parts = user_input.lower().strip().split()

        if not parts:
            return None

        command = parts[0]

        if command == "create" and len(parts) >= 2:
            filename = parts[1]
            content = ""

            # Check if content is provided
            if len(parts) > 2:
                content = " ".join(parts[2:])
            else:
                # Interactive content input
                Colors.print_command_output("Enter file content (press Ctrl+D or Ctrl+Z when finished):")
                content_lines = []
                try:
                    while True:
                        line = input()
                        content_lines.append(line)
                except EOFError:
                    content = "\n".join(content_lines)

            result = self.file_manager.create_file(filename, content)

            if result["success"]:
                return f"File created successfully: {result['path']} ({result['size']} bytes)"
            else:
                return f"Failed to create file: {result['error']}"

        elif command == "read" and len(parts) >= 2:
            filename = parts[1]
            result = self.file_manager.read_file(filename)

            if result["success"]:
                return f"Content of {filename}:\n{'='*50}\n{result['content']}\n{'='*50}\nFile size: {result['size']} bytes"
            else:
                return f"Failed to read file: {result['error']}"

        elif command == "delete" and len(parts) >= 2:
            filename = parts[1]

            # Ask for confirmation
            Colors.print_warning(f"Are you sure you want to delete '{filename}'? (y/N): ")
            confirmation = input().strip().lower()

            if confirmation not in ['y', 'yes']:
                return "File deletion cancelled."

            result = self.file_manager.delete_file(filename)

            if result["success"]:
                return f"File deleted successfully: {result['path']}"
            else:
                return f"Failed to delete file: {result['error']}"

        elif command == "list" or command == "ls":
            pattern = parts[1] if len(parts) > 1 else "*"
            result = self.file_manager.list_files(pattern)

            if result["success"]:
                if result["count"] == 0:
                    return "No files found."

                file_list = f"Found {result['count']} files:\n"
                for file_info in result["files"]:
                    file_list += f"  {file_info['name']} ({file_info['size']} bytes)\n"

                return file_list.strip()
            else:
                return f"Failed to list files: {result['error']}"

        return None

# Update available_commands list - add this to JarvisBot.__init__:
# self.available_commands.extend(['create', 'read', 'delete', 'list', 'ls'])

# Add to help display:
def display_file_help(self):
    """Display file operation help"""
    help_text = """
File Operations:
  create <filename> [content]  - Create a new file with optional content
  read <filename>             - Read and display file content
  delete <filename>           - Delete a file (with confirmation)
  list [pattern]              - List files (default: all files)
  ls [pattern]                - Alias for list

Examples:
  create hello.txt "Hello World"
  create script.py
  read hello.txt
  delete old_file.txt
  list *.py
"""
    return help_text
