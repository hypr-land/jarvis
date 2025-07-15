#!/usr/bin/env python3


import os
import sys
from typing import Optional, Any, Callable
from functools import wraps
import time


class TerminalColors:
    """
    Terminal color utilities with fallback for non-color terminals.
    
    Features:
    - Rich library integration when available
    - ANSI color codes as fallback
    - Consistent color scheme across applications
    - Easy to extend with new color methods
    """
    
    # Color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m'
    }
    
    def __init__(self, use_colors: bool = True, speak_callback: Optional[Callable[[str], None]] = None, streaming_enabled: bool = False, streaming_speed: float = 0.02):
        """
        Initialize terminal colors.
        
        Args:
            use_colors: Whether to use colors (default: True)
            speak_callback: Optional function(text:str) that will be invoked
                            after printing, e.g., to feed text into a TTS
                            engine.
        """
        self.use_colors = use_colors and self._supports_colors()
        self.speak_callback = speak_callback
        self.rich_available = self._check_rich_available()
        self.streaming_enabled = streaming_enabled
        self.streaming_speed = streaming_speed
        
        if self.rich_available:
            try:
                from rich.console import Console
                from rich.text import Text
                self.console = Console()
            except ImportError:
                self.rich_available = False
    
    def _supports_colors(self) -> bool:
        """Check if terminal supports colors."""
        return (
            hasattr(sys.stdout, 'isatty') and 
            sys.stdout.isatty() and 
            os.environ.get('TERM') != 'dumb'
        )
    
    def _check_rich_available(self) -> bool:
        """Check if rich library is available."""
        try:
            import rich
            return True
        except ImportError:
            return False
    
    def print_prompt(self, text: str, end: str = ""):
        """Print prompt text in yellow."""
        if self.rich_available:
            self.console.print(text, style="bold yellow", end=end)
        elif self.use_colors:
            print(f"{self.COLORS['bold']}{self.COLORS['yellow']}{text}{self.COLORS['reset']}", end=end)
        else:
            print(text, end=end)
    
    def print_success(self, text: str):
        """Print success message in green."""
        if self.rich_available:
            self.console.print(text, style="green")
        elif self.use_colors:
            print(f"{self.COLORS['green']}{text}{self.COLORS['reset']}")
        else:
            print(text)
        if self.speak_callback:
            self.speak_callback(text)
    
    def print_error(self, text: str):
        """Print error message in red."""
        if self.rich_available:
            self.console.print(text, style="bold red")
        elif self.use_colors:
            print(f"{self.COLORS['bold']}{self.COLORS['red']}{text}{self.COLORS['reset']}")
        else:
            print(text)
        if self.speak_callback:
            self.speak_callback(text)
    
    def print_warning(self, text: str):
        """Print warning message in yellow."""
        if self.rich_available:
            self.console.print(text, style="yellow")
        elif self.use_colors:
            print(f"{self.COLORS['yellow']}{text}{self.COLORS['reset']}")
        else:
            print(text)
        if self.speak_callback:
            self.speak_callback(text)
    
    def print_info(self, text: str):
        """Print info message in cyan."""
        if self.rich_available:
            self.console.print(text, style="cyan")
        elif self.use_colors:
            print(f"{self.COLORS['cyan']}{text}{self.COLORS['reset']}")
        else:
            print(text)
        if self.speak_callback:
            self.speak_callback(text)
    
    def print_highlight(self, text: str):
        """Print highlighted text in bright white."""
        if self.rich_available:
            self.console.print(text, style="bold white")
        elif self.use_colors:
            print(f"{self.COLORS['bold']}{self.COLORS['bright_white']}{text}{self.COLORS['reset']}")
        else:
            print(text)
        if self.speak_callback:
            self.speak_callback(text)
    
    def print_command_output(self, text: str):
        """Print command output in default color."""
        if self.rich_available:
            self.console.print(text)
        else:
            print(text)
        if self.speak_callback:
            self.speak_callback(text)
    
    def print_ai_response(self, text: str):
        """Print AI response in cyan/blue color."""
        if self.streaming_enabled:
            # Stream text character by character
            prefix = ""
            suffix = ""
            if not self.rich_available and self.use_colors:
                prefix = f"{self.COLORS['cyan']}"
                suffix = f"{self.COLORS['reset']}"
            sys.stdout.write(prefix)
            sys.stdout.flush()
            try:
                for idx, ch in enumerate(text):
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                    time.sleep(self.streaming_speed)
            except KeyboardInterrupt:
                # On Ctrl+C, flush the rest of the text immediately
                remaining = text[idx+1:] if 'idx' in locals() else text
                sys.stdout.write(remaining)
            finally:
                sys.stdout.write(suffix + "\n")
                sys.stdout.flush()
            if self.speak_callback:
                self.speak_callback(text)
            return
        if self.rich_available:
            self.console.print(text, style="cyan")
        elif self.use_colors:
            print(f"{self.COLORS['cyan']}{text}{self.COLORS['reset']}")
        else:
            print(text)
        if self.speak_callback:
            self.speak_callback(text)
    
    def print_table(self, headers: list, rows: list):
        """Print a formatted table."""
        if self.rich_available:
            from rich.table import Table
            table = Table()
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
        else:
            # Simple table formatting
            col_widths = [len(str(header)) for header in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            
            # Print headers
            header_str = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
            print(header_str)
            print("-" * len(header_str))
            
            # Print rows
            for row in rows:
                row_str = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                print(row_str)


class TerminalUtils:
    """
    General terminal utilities.
    """
    
    @staticmethod
    def clear_screen():
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def get_terminal_size() -> tuple:
        """Get terminal dimensions."""
        try:
            import shutil
            return shutil.get_terminal_size()
        except:
            return (80, 24)  # Default fallback
    
    @staticmethod
    def print_progress_bar(current: int, total: int, width: int = 50, char: str = "█"):
        """Print a progress bar."""
        progress = int(width * current / total)
        bar = char * progress + "░" * (width - progress)
        percentage = current / total * 100
        print(f"\r[{bar}] {percentage:.1f}%", end="", flush=True)
    
    @staticmethod
    def print_separator(char: str = "=", length: Optional[int] = None):
        """Print a separator line."""
        if length is None:
            length = TerminalUtils.get_terminal_size()[0]
        print(char * length)
    
    @staticmethod
    def print_centered(text: str, char: str = " "):
        """Print centered text."""
        width = int(TerminalUtils.get_terminal_size()[0])
        padding = (width - len(text)) // 2
        print(char * padding + text + char * padding)


def timeit(func):
    """
    Decorator to measure function execution time.
    
    Usage:
        @timeit
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"Function '{func.__name__}' took {elapsed:.4f} seconds")
        return result
    return wrapper


def confirm_action(prompt: str = "Continue? (y/n): ", default: bool = True) -> bool:
    """
    Get user confirmation for an action.
    
    Args:
        prompt: Confirmation prompt
        default: Default answer if user just presses Enter
        
    Returns:
        True if confirmed, False otherwise
    """
    while True:
        response = input(prompt).strip().lower()
        if response in ['', 'y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def get_user_input(prompt: str, default: str = "", required: bool = False) -> str:
    """
    Get user input with validation.
    
    Args:
        prompt: Input prompt
        default: Default value if user just presses Enter
        required: Whether input is required
        
    Returns:
        User input string
    """
    while True:
        response = input(prompt).strip()
        if response:
            return response
        elif default and not required:
            return default
        elif required:
            print("This field is required.")
        else:
            return ""


class Spinner:
    """
    Simple terminal spinner for long-running operations.
    """
    
    def __init__(self, message: str = "Loading...", chars: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        self.message = message
        self.chars = chars
        self.current = 0
        self.running = False
    
    def start(self):
        """Start the spinner."""
        self.running = True
        print(f"{self.chars[self.current]} {self.message}", end="", flush=True)
    
    def update(self):
        """Update the spinner."""
        if self.running:
            print(f"\r{self.chars[self.current]} {self.message}", end="", flush=True)
            self.current = (self.current + 1) % len(self.chars)
    
    def stop(self, message: str = "Done"):
        """Stop the spinner."""
        self.running = False
        print(f"\r{message}")


# Example usage
if __name__ == "__main__":
    colors = TerminalColors()
    
    colors.print_success("Success message")
    colors.print_error("Error message")
    colors.print_warning("Warning message")
    colors.print_info("Info message")
    colors.print_highlight("Highlighted text")
    
    TerminalUtils.print_separator()
    TerminalUtils.print_centered("Centered Text")
    
    # Progress bar example
    for i in range(101):
        TerminalUtils.print_progress_bar(i, 100)
        time.sleep(0.05)
    print()  # New line after progress bar 