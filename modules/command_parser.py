import re
import asyncio
import io
import sys
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class CommandOperator(Enum):
    PIPE = "|"           # command1 | command2 (pass output of cmd1 to cmd2)
    AND = "&"            # command1 & command2 (run both concurrently)
    SEQUENTIAL = ";"     # command1 ; command2 (run cmd2 after cmd1)
    CONDITIONAL_AND = "&&"  # command1 && command2 (run cmd2 only if cmd1 succeeds)
    CONDITIONAL_OR = "||"   # command1 || command2 (run cmd2 only if cmd1 fails)
    BACKGROUND = "&"     # command & (run in background - same as AND but different context)

@dataclass
class Command:
    command: str
    args: List[str]
    raw_input: str

@dataclass
class CommandChain:
    commands: List[Command]
    operators: List[CommandOperator]

class CommandParser:
    """Parse and execute command chains with shell-like operators"""

    def __init__(self, jarvis_bot):
        self.bot = jarvis_bot
        self.operator_patterns = {
            '||': CommandOperator.CONDITIONAL_OR,
            '&&': CommandOperator.CONDITIONAL_AND,
            '|': CommandOperator.PIPE,
            ';': CommandOperator.SEQUENTIAL,
            '&': CommandOperator.AND,
        }

    def parse_command_line(self, input_line: str) -> CommandChain:
        """Parse a command line with operators into a CommandChain"""
        # First, split by operators while preserving the operators
        parts = self._split_by_operators(input_line)
        
        commands = []
        operators = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Even indices are commands
                if part.strip():
                    commands.append(self._create_command(part.strip()))
            else:  # Odd indices are operators
                operator = self._get_operator(part.strip())
                if operator:
                    operators.append(operator)
        
        return CommandChain(commands, operators)

    def _split_by_operators(self, input_line: str) -> List[str]:
        """Split input by operators while preserving the operators"""
        # Create a regex pattern that matches operators
        operator_pattern = r'(\|\||&&|\||;|&)'
        
        # Split by operators and keep the operators
        parts = re.split(operator_pattern, input_line)
        
        # Filter out empty parts
        return [part for part in parts if part.strip()]

    def _get_operator(self, token: str) -> Optional[CommandOperator]:
        """Check if token is an operator"""
        return self.operator_patterns.get(token)

    def _create_command(self, command_str: str) -> Command:
        """Create a Command object from a command string"""
        if not command_str:
            return Command("", [], "")

        # Split the command string into tokens, handling quotes
        tokens = self._tokenize_command(command_str)
        
        if not tokens:
            return Command("", [], "")

        command = tokens[0]
        args = tokens[1:] if len(tokens) > 1 else []
        raw_input = command_str

        return Command(command, args, raw_input)

    def _tokenize_command(self, command_str: str) -> List[str]:
        """Tokenize a command string, handling quoted arguments"""
        tokens = []
        current_token = ""
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(command_str):
            char = command_str[i]
            
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_token += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_token += char
            elif char.isspace() and not in_quotes:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
            
            i += 1
        
        if current_token:
            tokens.append(current_token)
        
        return tokens

    async def execute_command_chain(self, command_chain: CommandChain) -> Dict[str, Any]:
        """Execute a command chain with proper operator handling"""
        if not command_chain.commands:
            return {"success": False, "error": "No commands to execute"}

        results = []
        previous_output = None
        previous_success = True

        for i, command in enumerate(command_chain.commands):
            # Determine if we should execute this command
            should_execute = True

            if i > 0:  # Not the first command
                operator = command_chain.operators[i-1]

                if operator == CommandOperator.CONDITIONAL_AND and not previous_success:
                    should_execute = False
                elif operator == CommandOperator.CONDITIONAL_OR and previous_success:
                    should_execute = False

            if should_execute:
                # Handle piped input
                if i > 0 and command_chain.operators[i-1] == CommandOperator.PIPE:
                    # Modify command to include piped input
                    if previous_output:
                        command.raw_input = f"{command.raw_input} {previous_output}"

                # Execute the command
                if i > 0 and command_chain.operators[i-1] == CommandOperator.AND:
                    # Execute concurrently (background)
                    result = await self._execute_command_async(command)
                else:
                    # Execute sequentially
                    result = await self._execute_command(command)

                results.append(result)
                previous_output = result.get("output", "")
                previous_success = result.get("success", False)
            else:
                # Command was skipped due to conditional operator
                results.append({
                    "success": True,
                    "output": f"Command skipped due to conditional operator",
                    "command": command.raw_input
                })

        return {
            "success": True,
            "results": results,
            "total_commands": len(command_chain.commands)
        }

    async def _execute_command(self, command: Command) -> Dict[str, Any]:
        """Execute a single command and capture its output"""
        try:
            # Capture stdout to get the actual output
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                # Determine if this should be treated as a built-in command.
                processed_input = command.raw_input.lstrip()
                is_system_command = False

                if processed_input.startswith('!'):
                    # Strip the prefix and run through the bot's command handler
                    processed_input = processed_input[1:].lstrip()
                    is_system_command = await self.bot.process_command(processed_input)
                # Otherwise, leave is_system_command = False so it will be handled as an AI query
                
                # Get the captured output
                output = captured_output.getvalue().strip()
                
                if is_system_command:
                    return {
                        "success": True,
                        "output": output if output else f"System command '{command.command}' executed successfully",
                        "command": command.raw_input,
                        "type": "system"
                    }
                else:
                    # It's an AI query
                    response = await self.bot.generate_response(command.raw_input)
                    return {
                        "success": True,
                        "output": response,
                        "command": command.raw_input,
                        "type": "ai"
                    }
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command.raw_input,
                "type": "error"
            }

    async def _execute_command_async(self, command: Command) -> Dict[str, Any]:
        """Execute a command asynchronously (for background/concurrent execution)"""
        # Create a task for async execution
        task = asyncio.create_task(self._execute_command(command))
        try:
            result = await task
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command.raw_input,
                "type": "async_error"
            }

"""
1. Pipe operator (|):
   help | stats
   search python | config

2. Sequential operator (;):
   clear ; help ; stats

3. Conditional AND (&&):
   create test.txt "hello" && read test.txt

4. Conditional OR (||):
   read nonexistent.txt || create nonexistent.txt "default content"

5. Background/Concurrent (&):
   search python & search javascript

6. Mixed operators:
   clear ; create test.txt "hello world" && read test.txt | stats
"""
