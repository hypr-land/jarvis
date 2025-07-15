#!/usr/bin/env python3

import os
import subprocess
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FileOpener:
    """File opener for opening files and applications with appropriate programs"""
    
    def __init__(self):
        # Default application mappings
        self.app_mappings = {
            # Code editors
            '.py': 'kate',
            '.rs': 'kate', 
            '.toml': 'kate',
            '.js': 'kate',
            '.ts': 'kate',
            '.html': 'kate',
            '.css': 'kate',
            '.json': 'kate',
            '.xml': 'kate',
            '.yaml': 'kate',
            '.yml': 'kate',
            '.md': 'kate',
            '.txt': 'kate',
            '.sh': 'kate',
            '.bash': 'kate',
            '.zsh': 'kate',
            '.fish': 'kate',
            '.c': 'kate',
            '.cpp': 'kate',
            '.h': 'kate',
            '.hpp': 'kate',
            '.java': 'kate',
            '.kt': 'kate',
            '.go': 'kate',
            '.php': 'kate',
            '.rb': 'kate',
            '.pl': 'kate',
            '.lua': 'kate',
            '.sql': 'kate',
            '.r': 'kate',
            '.m': 'kate',
            '.scala': 'kate',
            '.swift': 'kate',
            '.dart': 'kate',
            '.vue': 'kate',
            '.svelte': 'kate',
            '.jsx': 'kate',
            '.tsx': 'kate',
            
            # Web browsers
            '.html': 'google-chrome',
            '.htm': 'google-chrome',
            '.xhtml': 'google-chrome',
            
            # Documents
            '.pdf': 'okular',
            '.doc': 'libreoffice',
            '.docx': 'libreoffice',
            '.xls': 'libreoffice',
            '.xlsx': 'libreoffice',
            '.ppt': 'libreoffice',
            '.pptx': 'libreoffice',
            '.odt': 'libreoffice',
            '.ods': 'libreoffice',
            '.odp': 'libreoffice',
            
            # Images
            '.jpg': 'gwenview',
            '.jpeg': 'gwenview',
            '.png': 'gwenview',
            '.gif': 'gwenview',
            '.bmp': 'gwenview',
            '.svg': 'gwenview',
            '.webp': 'gwenview',
            '.ico': 'gwenview',
            '.tiff': 'gwenview',
            '.tif': 'gwenview',
            
            # Audio
            '.mp3': 'vlc',
            '.wav': 'vlc',
            '.flac': 'vlc',
            '.ogg': 'vlc',
            '.m4a': 'vlc',
            '.aac': 'vlc',
            
            # Video
            '.mp4': 'vlc',
            '.avi': 'vlc',
            '.mkv': 'vlc',
            '.mov': 'vlc',
            '.wmv': 'vlc',
            '.flv': 'vlc',
            '.webm': 'vlc',
            '.m4v': 'vlc',
            
            # Archives
            '.zip': 'ark',
            '.tar': 'ark',
            '.gz': 'ark',
            '.bz2': 'ark',
            '.7z': 'ark',
            '.rar': 'ark',
            
            # Executables and applications
            '.bin': None,  # Will be executed directly
            '.exe': None,  # Will be executed directly
            '': None,      # No extension - will be executed directly
        }
    
    def open_file(self, file_path: str) -> Dict[str, Any]:
        """Open a file with the appropriate application"""
        try:
            # Resolve the file path
            resolved_path = os.path.abspath(file_path)
            
            if not os.path.exists(resolved_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Get file extension
            file_ext = Path(resolved_path).suffix.lower()
            
            # Check if it's an executable (no extension or .bin/.exe)
            if not file_ext or file_ext in ['.bin', '.exe']:
                return self._execute_file(resolved_path)
            
            # Get the appropriate application
            app = self.app_mappings.get(file_ext)
            
            if not app:
                return {
                    "success": False,
                    "error": f"No application configured for file type: {file_ext}"
                }
            
            # Try to open with the specified application
            return self._open_with_app(resolved_path, app)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error opening file: {str(e)}"
            }
    
    def _open_with_app(self, file_path: str, app: str) -> Dict[str, Any]:
        """Open file with specified application"""
        try:
            # Try to run the application
            result = subprocess.run(
                [app, file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Opened {file_path} with {app}",
                    "app": app,
                    "file": file_path
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to open with {app}: {result.stderr}"
                }
                
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Application not found: {app}"
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Application timed out: {app}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error running {app}: {str(e)}"
            }
    
    def _execute_file(self, file_path: str) -> Dict[str, Any]:
        """Execute a binary file directly"""
        try:
            # Make sure the file is executable
            os.chmod(file_path, 0o755)
            
            # Execute the file
            result = subprocess.run(
                [file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": True,
                "message": f"Executed: {file_path}",
                "file": file_path,
                "return_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr
            }
            
        except PermissionError:
            return {
                "success": False,
                "error": f"Permission denied: {file_path}"
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out: {file_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing {file_path}: {str(e)}"
            }
    
    def launch_app(self, app_name: str) -> Dict[str, Any]:
        """Launch an application by name"""
        try:
            # Common application names and their commands
            app_commands = {
                'chrome': 'google-chrome',
                'firefox': 'firefox',
                'discord': 'discord',
                'spotify': 'spotify',
                'steam': 'steam',
                'telegram': 'telegram-desktop',
                'slack': 'slack',
                'code': 'code',
                'vscode': 'code',
                'kate': 'kate',
                'kwrite': 'kwrite',
                'gedit': 'gedit',
                'nano': 'nano',
                'vim': 'vim',
                'neovim': 'nvim',
                'emacs': 'emacs',
                'thunderbird': 'thunderbird',
                'evolution': 'evolution',
                'konsole': 'konsole',
                'gnome-terminal': 'gnome-terminal',
                'xterm': 'xterm',
                'vlc': 'vlc',
                'mpv': 'mpv',
                'gwenview': 'gwenview',
                'okular': 'okular',
                'libreoffice': 'libreoffice',
                'ark': 'ark',
                'dolphin': 'dolphin',
                'nautilus': 'nautilus',
                'nemo': 'nemo',
                'pcmanfm': 'pcmanfm',
                'thunar': 'thunar',
            }
            
            # Get the command for the app
            command = app_commands.get(app_name.lower(), app_name)
            
            # Try to launch the application
            result = subprocess.run(
                [command],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Launched: {app_name}",
                    "app": app_name,
                    "command": command
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to launch {app_name}: {result.stderr}"
                }
                
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Application not found: {app_name}"
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Launch timed out: {app_name}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error launching {app_name}: {str(e)}"
            }
    
    def get_supported_extensions(self) -> Dict[str, str]:
        """Get list of supported file extensions and their applications"""
        return self.app_mappings.copy() 