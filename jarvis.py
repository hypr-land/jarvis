#   Copyright 2025 hypr-land

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


# put dbus-send --session --type=method_call --dest=org.jarvis.Service /org/jarvis/Service org.jarvis.Service.TriggerScreenshotAnalysis     as a command for shortcut to get jarvis intergration with spectacle
import os
import sys
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime, date
current_date = date.today()
import psutil
import time
from collections import deque
import threading
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse, parse_qs, unquote
import re
# keep local imports under this
from modules.command_parser import CommandParser
from modules.config_manager import ConfigManager, BotConfig
from modules.tts_module import TTSModule
from modules.file_opener import FileOpener
from modules.file_manager import FileManager
from modules.process_monitor import ProcessMonitor, SystemMonitor
from modules.search_engine import SearchEngine, SearchConfig
from modules.terminal_utils import TerminalColors, TerminalUtils, timeit
from modules.url_utils import URLUtils
from modules.git_manager import GitManager
from modules.stt_module import STTManager, AudioConfig, list_input_devices, record_test
from modules.wake_word import WakeWordDetector
from modules.test_suite import JarvisTestSuite
from modules.email_manager import EmailManager
from modules.alias_manager import AliasManager
from modules.reminder_manager import ReminderManager

try:
    import readline
    import atexit
    import pathlib
    # this does nothing, was a todo thing
    histfile = os.path.join(pathlib.Path.home(), ".jarvis_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)
    readline.set_history_length(1000)
except ImportError:
    readline = None
    print("Install 'readline' for arrow key navigation and command history.")

# Vision
import base64
from PIL import Image
import io
import subprocess
from pathlib import Path
import argparse

search_integer = 0


pid = os.getpid()

try:
    import groq  #
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None  # type: ignore
    GROQ_AVAILABLE = False
    print("groq package not available – will use Ollama if configured.")

# Ollama availability flag (optional)
try:
    import ollama  # noqa: F401  # Only to test availability; actual import later
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from rich.console import Console
    from rich.text import Text
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better formatting: pip install rich")
    print("Or, if the text under this is colored, you can ignore this, i guess?")

try:
    from difflib import get_close_matches
    DIFFLIB_AVAILABLE = True
except ImportError:
    DIFFLIB_AVAILABLE = False

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colors = {
            logging.ERROR: '\033[31m',    # Red
            logging.WARNING: '\033[33m',  # Yellow
            logging.INFO: '\033[37m',     # White
            logging.DEBUG: '\033[36m',    # Cyan
            'RESET': '\033[0m'
        }

    def format(self, record):
        # Add color to the level name
        if record.levelno in self.colors:
            color = self.colors[record.levelno]
            record.levelname = f"{color}{record.levelname}{self.colors['RESET']}"
            record.msg = f"{color}{record.msg}{self.colors['RESET']}"
        return super().format(record)

# Set up colored logging
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

GROQ_API_TOKEN = os.getenv("GROQ_API_KEY")
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def normalize_url(url):
    if url.startswith("//"):
        return "https:" + url
    elif url.startswith("http://") or url.startswith("https://"):
        return url
    else:
        return url

def extract_real_url(redirect_url):
    try:
        parsed = urlparse(redirect_url)
        qs = parse_qs(parsed.query)
        if 'uddg' in qs:
            real_url = qs['uddg'][0]
            return unquote(real_url)
        else:
            return redirect_url
    except Exception:
        return redirect_url

def scrape_page_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = text.strip()
        return text
    except Exception as e:
        return f"[Error scraping: {e}]"

def summarize_with_groq(text):
    if not GROQ_API_TOKEN:
        return "[Error: GROQ_API_TOKEN not set]"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_TOKEN}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following content briefly:\n\n{text}"}
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }
    try:
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        summary = data["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        return f"[Error summarizing: {e}]"

def search_duckduckgo(S_query, max_results=1, timeout=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(S_query)}"
    response = requests.get(url, headers=headers, timeout=timeout)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select("a.result__a")[:max_results]:
        title = result.get_text()
        raw_link = result["href"]
        link = extract_real_url(raw_link)
        link = normalize_url(link)
        content = scrape_page_content(link)
        truncated_content = content[:8192]  # Approx 2048 tokens
        summary = summarize_with_groq(truncated_content)
        results.append({
            "title": title,
            "link": link,
            "content": content,
            "summary": summary
        })
    return results

def perform_search(query: str, config: Optional[BotConfig] = None) -> str:
    """Perform search using DuckDuckGo and return formatted results"""
    if config and not config.search_enabled:
        return "Search functionality is disabled in configuration."
    
    try:
        max_results = config.search_max_results if config else 3
        timeout = config.search_timeout if config else 5
        results = search_duckduckgo(query, max_results=max_results, timeout=timeout)
        if not results:
            return f"No search results found for: {query}"

        formatted_results = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result['title']}\n"
            formatted_results += f"   URL: {result['link']}\n"
            formatted_results += f"   Summary: {result['summary']}\n\n"

        return formatted_results.strip()
    except Exception as e:
        return f"Search error: {str(e)}"

def is_search_available() -> bool:
    """Check if search functionality is available"""
    return GROQ_API_TOKEN is not None



class ConversationHistory:
    """Manages conversation history"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only the last max_history messages (excluding system message)
        if len(self.messages) > self.max_history + 1:  # +1 for system message
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for Groq API"""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]

    def clear(self):
        """Clear conversation history except system message"""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

class STTProcessor:
    """Speech-to-Text processor using Groq's Whisper API"""

    def __init__(self, config: BotConfig, engine: str = "sr"):
        self.config = config
        self.model = config.stt_model
        self.stt_manager = None
        
        if config.stt_enabled:
            try:
                # Initialize STT manager with config
                audio_config = AudioConfig(
                    sample_rate=16000,
                    channels=1,
                    blocksize=8000,
                    input_gain=1.0,
                    silence_threshold=0.03,
                    silence_duration=0.5
                )
                
                self.stt_manager = STTManager(
                    engine=engine,  # Use SpeechRecognition instead of Groq Whisper
                    model=self.model,
                    api_key=config.api_key,
                    audio_config=audio_config
                )
                logger.info(f"STT processor initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize STT: {e}")

    async def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file to text"""
        if not self.stt_manager:
            raise RuntimeError("STT not initialized")
        
        try:
            result = self.stt_manager.transcribe_file(audio_file_path)
            return result.text
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            raise

    def start_realtime(self):
        """Start real-time transcription"""
        if not self.stt_manager:
            raise RuntimeError("STT not initialized")
        self.stt_manager.start_realtime()

    def stop_realtime(self):
        """Stop real-time transcription"""
        if self.stt_manager:
            self.stt_manager.stop_realtime()

    def is_available(self) -> bool:
        """Check if STT functionality is available"""
        return self.stt_manager is not None



class JarvisBot:
    """Main JARVIS AI Bot class"""

    def __init__(self, config: BotConfig, server_mode: bool = False):
        self.config = config
        self.server_mode = server_mode
        
        # Initialise AI client (Groq or Ollama)
        try:
            provider = getattr(self.config, 'api_provider', 'groq').lower()
        except AttributeError:
            provider = 'groq'

        self.client: Any = None
        self.main_model: str = self.config.main_model

        if provider == 'ollama':
            try:
                self.client = self._init_ollama_client()
                self.main_model = getattr(self.config, 'ollama_main_model', self.config.main_model)
            except Exception as e:
                logger.warning(f"Failed to initialise Ollama client: {e}. Falling back to Groq…")
                try:
                    self.client = self._init_groq_client()
                    self.main_model = getattr(self.config, 'groq_main_model', self.config.main_model)
                except Exception as g_exc:
                    logger.error(f"Failed to initialise Groq client as fallback: {g_exc}")
                    raise
        else:
            try:
                self.client = self._init_groq_client()
                self.main_model = getattr(self.config, 'groq_main_model', self.config.main_model)
            except Exception as e:
                logger.warning(f"Failed to initialise Groq client: {e}. Attempting Ollama fallback…")
                try:
                    self.client = self._init_ollama_client()
                    self.main_model = getattr(self.config, 'ollama_main_model', self.config.main_model)
                except Exception as o_exc:
                    logger.error(f"Failed to initialise Ollama client as fallback: {o_exc}")
                    raise

        if not self.client:
            raise RuntimeError("Unable to initialise any AI client (Groq or Ollama)")

        self.history = ConversationHistory(max_history=config.max_history)
        try:
            self.stt_processor = STTProcessor(config, engine=getattr(config, 'stt_engine', 'groq'))
        except Exception as e:
            logger.warning(f"Failed to initialize STT: {e}")
            self.stt_processor = None
            
        self.running = False
        
        pid = os.getpid()
        self.monitor = ProcessMonitor(pid, interval=config.monitoring_interval)
        self.colors = TerminalColors(use_colors=config.colors_enabled, streaming_enabled=getattr(config, 'streaming_enabled', False), streaming_speed=getattr(config, 'streaming_speed', 0.02))
        self.search_engine = SearchEngine(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",  # Default user agent
            timeout=config.search_timeout,
            max_content_length=8192
        )
        self.url_utils = URLUtils(timeout=10)
        # Store the current working directory path for context
        self.current_path = os.getcwd()

        # Initialize vision module if enabled
        self.vision_module = None
        if not server_mode and config.vision_enabled:
            try:
                from modules.vision_module import VisionModule
                self.vision_module = VisionModule(self.client, self.config.vision_model)
                logger.info("Vision module initialized successfully")
            except ImportError:
                logger.warning("Vision module not available")
            except Exception as e:
                logger.error(f"Failed to initialize vision module: {e}")

        # Initialize desktop integration if enabled
        self.desktop_integration = None
        if not server_mode and config.desktop_integration_enabled:
            try:
                from modules.desktop_integration import DesktopIntegration
                self.desktop_integration = DesktopIntegration(config, self.vision_module, self)
                self.desktop_integration.start()
                logger.info("Desktop integration started successfully")
            except ImportError as e:
                logger.warning(f"Desktop integration not available: {e}")
            except Exception as e:
                logger.error(f"Failed to start desktop integration: {e}")

        # Initialize DBus handler if enabled
        self.dbus_handler = None
        if not server_mode and config.dbus_enabled:
            try:
                from modules.dbus_handler import DBusHandler, is_dbus_available
                if is_dbus_available():
                    self.dbus_handler = DBusHandler(config=config, bot=self)
                    # Start DBus handler in a separate thread
                    import threading
                    self.dbus_thread = threading.Thread(target=self.dbus_handler.run, daemon=True)
                    self.dbus_thread.start()
                    logger.info("DBus handler started successfully")
                else:
                    logger.warning("DBus support not available")
            except ImportError as e:
                logger.warning(f"DBus support not available: {e}")
            except Exception:
                # Silently handle any other errors - DBus is optional
                pass

        # Initialize other components
        self.file_manager = FileManager()
        # Initialize Git manager
        self.git_manager = GitManager()

        # Initialize Email manager
        self.email_manager = EmailManager()
        self.tts_module = TTSModule(
            language=getattr(self.config, 'tts_language', 'en'),
            tld=getattr(self.config, 'tts_tld', 'com'),
            enabled=False  # Off by default; use 'tts enable' command to activate
        )
        self.alias_manager = AliasManager(os.path.join(os.getcwd(), "jarvis.db"))

        # Initialize wake-word detector (Porcupine)
        self.wake_detector = None
        if getattr(config, 'wake_word_enabled', False):
            try:
                keyword_paths = getattr(config, 'wake_word_keywords', [])
                if keyword_paths and keyword_paths[0] != 'path/to/your/ppn/here':
                    self.wake_detector = WakeWordDetector(
                        keyword_paths=keyword_paths,
                        sensitivities=[getattr(config, 'wake_word_sensitivity', 0.5)] * len(keyword_paths),
                        detected_callback=self._on_wake_word_detected,
                        access_key=(getattr(config, 'wake_word_access_key', None) or os.getenv('PICOVOICE_ACCESS_KEY')),
                    )
                    self.wake_detector.start()
                    logger.info("Wake-word detector initialised")
                else:
                    logger.warning("Wake-word enabled but no keyword paths provided or default path is used.")
            except Exception as exc:
                logger.error(f"Failed to start wake-word detector: {exc}")

        self.file_opener = None
        if not server_mode and hasattr(config, 'file_opening_enabled') and config.file_opening_enabled:
            self.file_opener = FileOpener()

        # Set up system message
        self.history.add_message("system", self.config.system_prompt)

        self.reminder_manager = ReminderManager(os.path.join(os.getcwd(), "jarvis.db"))

    def _init_groq_client(self) -> 'Groq':
        """Initialize Groq client with API key. Requires the ``groq`` python package
        and a valid ``GROQ_API_KEY``. If either is missing an ``ImportError`` is
        raised so the caller can attempt another provider."""

        if not GROQ_AVAILABLE or Groq is None:
            raise ImportError("groq python package is not available")

        api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable or config api_key required")

        try:
            client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise

    def _init_ollama_client(self):

        try:
            import ollama  # Local model serving – https://github.com/ollama/ollama
        except ImportError as exc:
            raise RuntimeError("ollama python package not found. Install with: pip install ollama") from exc

        class _OllamaResponse:  # Mimics the response object shape used by Groq SDK
            class _Message:
                def __init__(self, content: str):
                    self.content = content

            class _Choice:
                def __init__(self, content: str):
                    self.message = _OllamaResponse._Message(content)

            def __init__(self, content: str):
                # Build a minimal structure: response.choices[0].message.content
                self.choices = [self._Choice(content)]

        class _OllamaCompletions:
            def create(self, model: str, messages: list, max_tokens: int | None = None, temperature: float = 0.7):  # noqa: D401,E501
                # ``ollama.chat`` ignores max_tokens if not supplied – it streams until stop.
                try:
                    res = ollama.chat(model=model, messages=messages, options={"temperature": temperature})
                    content: str = res.get('message', {}).get('content', '')
                except Exception as e:
                    logger.error(f"Ollama chat completion failed: {e}")
                    raise
                return _OllamaResponse(content)

        class _OllamaChat:
            def __init__(self):
                self.completions = _OllamaCompletions()

        class OllamaClientWrapper:
            def __init__(self):
                self.chat = _OllamaChat()

        logger.info("Ollama client wrapper initialised successfully")
        return OllamaClientWrapper()

    def should_search_for_topic(self, topic: str) -> bool:
        """
        Helper method to determine if a topic requires searching
        """
        current_info_keywords = [
            'news', 'latest', 'recent', 'current', 'today', 'now', 'update',
            'weather', 'stock', 'price', 'score', 'result', 'announcement',
            'breaking', 'development', 'release', 'launch', 'event'
        ]

        stable_knowledge_keywords = [
            'history', 'definition', 'explain', 'how to', 'what is', 'who was',
            'concept', 'theory', 'principle', 'tutorial', 'guide', 'example',
            'difference', 'comparison', 'meaning', 'origin', 'causes', 'effects'
        ]

        topic_lower = topic.lower()

        # Check for current info needs
        if any(keyword in topic_lower for keyword in current_info_keywords):
            return True

        # Check for stable knowledge
        if any(keyword in topic_lower for keyword in stable_knowledge_keywords):
            return False

        # Default to not searching for ambiguous cases
        return False

    async def generate_response(self, user_input: str) -> str:
        """Generate response with optimized single API call and file creation"""
        if not self.client:
            return "AI client is not initialized."
        global search_integer
        try:
            self.history.add_message("user", user_input)
            current_date_iso = date.today().isoformat()
            enhanced_system_prompt = self.config.system_prompt + f"\n\nCURRENT_WORKING_PATH: {self.current_path}\nCURRENT_DATE: {current_date_iso}\n\n" + """

            SEARCH CAPABILITY:
            If you need current information to answer the user's question, you can request a search by including:
            <SEARCH>your search query here</SEARCH>

            Only use search for:
            - Current events, news, or recent developments
            - Real-time data (weather, stocks, scores)
            - Information that changes frequently
            - When user explicitly asks to "search" or "look up"

            Do NOT search for:
            - Historical facts or established knowledge
            - General explanations or definitions
            - Creative requests or personal questions
            - Technical concepts or tutorials

            FILE CREATION CAPABILITY:
            If the user asks you to create a file, save something to a file, or write content to a file, you can do this by including:
            <CREATE_FILE>
            filename: desired_filename.ext
            content: the content to write to the file
            </CREATE_FILE>

            Only use this when the user explicitly asks to create, save, or write to a file.

            GIT COMMIT CAPABILITY:
            If the user asks you to commit & push a file to git, include:
            <GIT_COMMIT>
            file: path/to/file.ext
            message: concise commit message
            </GIT_COMMIT>

            Only use this when the user explicitly asks to push/commit something to git or version control.

            GIT ADD CAPABILITY:
            If the user asks you to stage all current changes (e.g. "git add .", "stage all files"), include:
            <GIT_ADD></GIT_ADD>

            Only use this when the user explicitly asks to stage/add all changes.

            EMAIL CAPABILITY:
            To send an email, include a block like:
            <SEND_EMAIL>
            to: recipient@example.com
            subject: Brief subject line here
            body:
            Multi-line body text goes here. Keep it concise.
            </SEND_EMAIL>

            Only generate this block when the user clearly instructs to send an email (e.g. "email Bob about the meeting tomorrow").

            After requesting search or file creation, wait for the results and then provide your response using that information.

            REMINDER CAPABILITY:
            If the user asks to **set**, **read/list** or **remove** reminders you can use the
            following blocks:

            1. Add a reminder (e.g. "Remind me to submit the report at 5 pm")

               <REMINDER_ADD>
               time: ISO_DATETIME
               text: reminder text here
               </REMINDER_ADD>

            2. List all reminders (e.g. "What reminders do I have?")

               <REMINDER_LIST></REMINDER_LIST>

            3. Remove a reminder by ID (e.g. "Delete reminder 3")

               <REMINDER_REMOVE>
               id: 3
               </REMINDER_REMOVE>

            Use ISO-8601 date-time whenever possible (e.g. 2024-12-31T17:00).
            
            """

            # Ensure exactly one system prompt in the conversation history
            if self.history.messages:
                if self.history.messages[0]["role"] == "system":
                    # Update the existing system prompt
                    self.history.messages[0]["content"] = enhanced_system_prompt
                else:
                    # Prepend a new system prompt if the first message is not a system one
                    self.history.messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
            else:
                # History is empty – start it with the system prompt
                self.history.messages.append({"role": "system", "content": enhanced_system_prompt})

            # Build the message list for the API call
            messages = self.history.get_messages_for_api()
            
            # First, try to get a response that might include search request or file creation
            initial_response = self.client.chat.completions.create(  # type: ignore[arg-type]
                model=self.main_model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            assistant_content = ""
            if initial_response.choices[0].message.content:
                assistant_content = initial_response.choices[0].message.content.strip()

            # Track if we've already processed file creation to prevent duplicates
            file_creation_processed = False
            search_processed = False
            git_processed = False  # for commit
            git_add_processed = False
            email_processed = False

            # Check if search was requested
            if '<SEARCH>' in assistant_content and '</SEARCH>' in assistant_content and not search_processed:
                search_processed = True
                # Extract search query
                search_match = re.search(r'<SEARCH>(.*?)</SEARCH>', assistant_content, re.DOTALL)
                if search_match:
                    search_query = search_match.group(1).strip()

                    if search_query and len(search_query) > 2:
                        self.colors.print_info(f"🔍 Searching for: {search_query}")

                        try:
                            search_results = perform_search(search_query, self.config)
                            search_integer += 1

                            if search_results and "No search results found" not in search_results:
                                self.colors.print_info("✓ Search results obtained")

                                # Create follow-up message with search results
                                search_context = f"Search results for '{search_query}':\n{search_results}"

                                # Add search results to conversation without duplicating the search tag
                                messages.append({"role": "user", "content": f"Here are the search results:\n{search_context}\n\nNow please provide your response based on this information."})

                                # Generate final response with search results
                                final_response = self.client.chat.completions.create(  # type: ignore[arg-type]
                                    model=self.main_model,
                                    messages=messages,  # type: ignore[arg-type]
                                    max_tokens=self.config.max_tokens,
                                    temperature=self.config.temperature
                                )

                                if final_response.choices[0].message.content:
                                    assistant_content = final_response.choices[0].message.content.strip()

                            else:
                                self.colors.print_warning("⚠ No useful search results found")
                                # Remove search tags and continue with original response
                                assistant_content = re.sub(r'<SEARCH>.*?</SEARCH>', '', assistant_content, flags=re.DOTALL).strip()

                        except Exception as search_error:
                            self.colors.print_error(f"❌ Search failed: {search_error}")
                            # Remove search tags and continue with original response
                            assistant_content = re.sub(r'<SEARCH>.*?</SEARCH>', '', assistant_content, flags=re.DOTALL).strip()

            if ('<CREATE_FILE>' in assistant_content and '</CREATE_FILE>' in assistant_content and
                not file_creation_processed and self.is_file_creation_request(user_input)):
                file_creation_processed = True
                # Extract file creation request
                file_match = re.search(r'<CREATE_FILE>(.*?)</CREATE_FILE>', assistant_content, re.DOTALL)
                if file_match:
                    file_content = file_match.group(1).strip()

                    # Parse filename and content
                    lines = file_content.split('\n')
                    filename = None
                    content = None

                    for line in lines:
                        if line.strip().startswith('filename:'):
                            filename = line.split('filename:', 1)[1].strip()
                        elif line.strip().startswith('content:'):
                            content = line.split('content:', 1)[1].strip()
                            # Get all remaining lines as content
                            content_start_idx = lines.index(line)
                            remaining_lines = lines[content_start_idx + 1:]
                            if remaining_lines:
                                content += '\n' + '\n'.join(remaining_lines)

                    if filename and content is not None:
                        self.colors.print_info(f"📁 Creating file: {filename}")

                        try:
                            # Check if file already exists to prevent accidental overwrites
                            if os.path.exists(filename):
                                self.colors.print_warning(f"⚠ File '{filename}' already exists. Skipping creation.")
                                file_creation_result = "File already exists and was not overwritten."
                            else:
                                result = self.file_manager.create_file(filename, content)
                                if result["success"]:
                                    self.colors.print_info(f"✓ File created successfully: {result['path']}")
                                    file_creation_result = f"File '{filename}' has been created successfully at {result['path']}"
                                else:
                                    self.colors.print_error(f"❌ Failed to create file: {result['error']}")
                                    file_creation_result = f"Failed to create file: {result['error']}"

                            # Add file creation result to conversation without duplicating the creation tag
                            messages.append({"role": "user", "content": file_creation_result})

                            # Generate final response with file creation confirmation
                            final_response = self.client.chat.completions.create(  # type: ignore[arg-type]
                                model=self.main_model,
                                messages=messages,  # type: ignore[arg-type]
                                max_tokens=self.config.max_tokens,
                                temperature=self.config.temperature
                            )

                            if final_response.choices[0].message.content:
                                assistant_content = final_response.choices[0].message.content.strip()






                        except Exception as file_error:
                            self.colors.print_error(f"❌ File creation failed: {file_error}")
                            # Remove file creation tags and continue with original response
                            assistant_content = re.sub(r'<CREATE_FILE>.*?</CREATE_FILE>', '', assistant_content, flags=re.DOTALL).strip()
            if ('<REMINDER_ADD>' in assistant_content and '</REMINDER_ADD>' in assistant_content):
                rem_match = re.search(r'<REMINDER_ADD>(.*?)</REMINDER_ADD>', assistant_content, re.DOTALL)
                if rem_match:
                    rem_block = rem_match.group(1).strip()
                    time_val = None
                    text_val = None
                    for line in rem_block.split('\n'):
                        if line.strip().startswith('time:'):
                            time_val = line.split(':', 1)[1].strip()
                        elif line.strip().startswith('text:'):
                            text_val = line.split(':', 1)[1].strip()
                    if time_val and text_val:
                        rem_res = self.reminder_manager.add(time_val, text_val)
                        rep_msg = "Reminder saved successfully." if rem_res.get('success') else f"Failed to save reminder: {rem_res.get('error')}"
                    else:
                        rep_msg = "Failed to parse reminder parameters."

                    # Replace the tag with result message
                    assistant_content = re.sub(r'<REMINDER_ADD>.*?</REMINDER_ADD>', rep_msg, assistant_content, flags=re.DOTALL).strip()

            # Handle reminder list tag
            if '<REMINDER_LIST>' in assistant_content and '</REMINDER_LIST>' in assistant_content:
                list_res = self.reminder_manager.list()
                if list_res.get('success'):
                    if list_res['count'] == 0:
                        list_text = 'You have no reminders.'
                    else:
                        lines = [f"{r['id']}. [{r['reminder_time']}] {r['reminder_text']}" for r in list_res['reminders']]
                        list_text = '\n'.join(lines)
                else:
                    list_text = f"Failed to list reminders: {list_res.get('error')}"

                assistant_content = re.sub(r'<REMINDER_LIST>.*?</REMINDER_LIST>', list_text, assistant_content, flags=re.DOTALL).strip()

            # Handle reminder remove tag
            if '<REMINDER_REMOVE>' in assistant_content and '</REMINDER_REMOVE>' in assistant_content:
                remrm_match = re.search(r'<REMINDER_REMOVE>(.*?)</REMINDER_REMOVE>', assistant_content, re.DOTALL)
                remove_msg = ''
                if remrm_match:
                    block = remrm_match.group(1).strip()
                    rid = None
                    for line in block.split('\n'):
                        if line.strip().startswith('id:'):
                            try:
                                rid = int(line.split(':',1)[1].strip())
                            except ValueError:
                                rid = None
                    if rid is not None:
                        rm_res = self.reminder_manager.remove(rid)
                        remove_msg = 'Reminder removed.' if rm_res.get('success') else f"Failed to remove: {rm_res.get('error')}"
                    else:
                        remove_msg = 'Invalid reminder id.'

                assistant_content = re.sub(r'<REMINDER_REMOVE>.*?</REMINDER_REMOVE>', remove_msg, assistant_content, flags=re.DOTALL).strip()

            # Check if git commit was requested
            if ('<GIT_COMMIT>' in assistant_content and '</GIT_COMMIT>' in assistant_content and not git_processed):
                git_processed = True
                git_match = re.search(r'<GIT_COMMIT>(.*?)</GIT_COMMIT>', assistant_content, re.DOTALL)
                if git_match:
                    git_block = git_match.group(1).strip()
                    lines = [l.strip() for l in git_block.split('\n') if l.strip()]
                    file_to_commit = None
                    commit_msg = None
                    for l in lines:
                        if l.lower().startswith('file:'):
                            file_to_commit = l.split(':',1)[1].strip()
                        elif l.lower().startswith('message:'):
                            commit_msg = l.split(':',1)[1].strip()

                    if file_to_commit:
                        commit_msg = commit_msg or f"Update {file_to_commit}"
                        self.colors.print_info(f"🔄 Committing {file_to_commit} to git …")
                        git_res = self.git_manager.commit_and_push(file_to_commit, commit_msg)
                        if git_res.get('success'):
                            self.colors.print_info("✓ Git commit & push successful")
                            git_result_msg = f"Git operation successful for {file_to_commit}."
                        else:
                            self.colors.print_error(f"❌ Git error: {git_res.get('error')}")
                            git_result_msg = f"Git operation failed: {git_res.get('error')}"

                        # Add result to context
                        messages.append({"role":"user","content": git_result_msg})

                        # Re-generate final response
                        final_response = self.client.chat.completions.create(  # type: ignore[arg-type]
                            model=self.main_model,
                            messages=messages,  # type: ignore[arg-type]
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature
                        )
                        if final_response.choices[0].message.content:
                            assistant_content = final_response.choices[0].message.content.strip()

                # Remove tag from assistant_content in any case
                assistant_content = re.sub(r'<GIT_COMMIT>.*?</GIT_COMMIT>', '', assistant_content, flags=re.DOTALL).strip()

            # Check if git add (stage all) was requested
            if ('<GIT_ADD>' in assistant_content and '</GIT_ADD>' in assistant_content and not git_add_processed):
                git_add_processed = True
                self.colors.print_info("🔄 Staging all changes (git add .)…")
                git_res = self.git_manager.add_all()
                if git_res.get('success'):
                    self.colors.print_info("✓ All changes staged")
                    git_result_msg = "All changes have been staged successfully."
                else:
                    self.colors.print_error(f"❌ Git error: {git_res.get('error')}")
                    git_result_msg = f"Git staging failed: {git_res.get('error')}"

                # Add result to context
                messages.append({"role": "user", "content": git_result_msg})

                # Re-generate final response to reflect staging result
                final_response = self.client.chat.completions.create(  # type: ignore[arg-type]
                    model=self.main_model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                if final_response.choices[0].message.content:
                    assistant_content = final_response.choices[0].message.content.strip()

                # Remove tag
                assistant_content = re.sub(r'<GIT_ADD>.*?</GIT_ADD>', '', assistant_content, flags=re.DOTALL).strip()

            # Check if email was requested
            if ('<SEND_EMAIL>' in assistant_content and '</SEND_EMAIL>' in assistant_content and not email_processed):
                email_processed = True
                email_match = re.search(r'<SEND_EMAIL>(.*?)</SEND_EMAIL>', assistant_content, re.DOTALL)
                if email_match:
                    email_block = email_match.group(1).strip()
                    to_addr = None
                    subject = None
                    body = None

                    # Split by lines and parse fields
                    lines = [l.strip() for l in email_block.split('\n')]
                    collecting_body = False
                    body_lines = []
                    for line in lines:
                        if collecting_body:
                            body_lines.append(line)
                        elif line.lower().startswith('to:'):
                            to_addr = line.split(':',1)[1].strip()
                        elif line.lower().startswith('subject:'):
                            subject = line.split(':',1)[1].strip()
                        elif line.lower().startswith('body:'):
                            collecting_body = True
                            maybe_first_body = line.split(':',1)[1].lstrip()
                            if maybe_first_body:
                                body_lines.append(maybe_first_body)

                    body = '\n'.join(body_lines).strip() if body_lines else ''

                    if to_addr and subject is not None and body is not None:
                        self.colors.print_info(f"📧 Sending email to {to_addr} …")
                        email_res = self.email_manager.send_email(to_addr, subject, body)
                        if email_res.get('success'):
                            self.colors.print_info("✓ Email sent successfully")
                            email_result_msg = f"Email successfully sent to {to_addr}."
                        else:
                            self.colors.print_error(f"❌ Email failed: {email_res.get('error')}")
                            email_result_msg = f"Failed to send email: {email_res.get('error')}."

                        # Add result back into context for final response
                        messages.append({"role": "user", "content": email_result_msg})

                        final_response = self.client.chat.completions.create(
                            model=self.main_model,
                            messages=messages,
                            max_tokens=self.config.max_tokens,
                            temperature=self.config.temperature
                        )
                        if final_response.choices[0].message.content:
                            assistant_content = final_response.choices[0].message.content.strip()

                # Strip email tag regardless
                assistant_content = re.sub(r'<SEND_EMAIL>.*?</SEND_EMAIL>', '', assistant_content, flags=re.DOTALL).strip()

            # Clean up any remaining search or file creation tags
            assistant_content = re.sub(r'<SEARCH>.*?</SEARCH>', '', assistant_content, flags=re.DOTALL).strip()
            assistant_content = re.sub(r'<CREATE_FILE>.*?</CREATE_FILE>', '', assistant_content, flags=re.DOTALL).strip()
            assistant_content = re.sub(r'<GIT_COMMIT>.*?</GIT_COMMIT>', '', assistant_content, flags=re.DOTALL).strip()
            assistant_content = re.sub(r'<GIT_ADD>.*?</GIT_ADD>', '', assistant_content, flags=re.DOTALL).strip()
            assistant_content = re.sub(r'<SEND_EMAIL>.*?</SEND_EMAIL>', '', assistant_content, flags=re.DOTALL).strip()

            # Handle file opening requests
            if '<OPEN_FILE>' in assistant_content and '</OPEN_FILE>' in assistant_content:
                assistant_content = await self._handle_file_opening(assistant_content)

            # Handle app launching requests
            if '<LAUNCH_APP>' in assistant_content and '</LAUNCH_APP>' in assistant_content:
                assistant_content = await self._handle_app_launching(assistant_content)

            # Add final response to history
            self.history.add_message("assistant", assistant_content)
            
            # Speak the response if TTS is enabled
            if self.tts_module.is_enabled():
                try:
                    # Speak the response asynchronously so it doesn't block
                    asyncio.create_task(self._speak_response(assistant_content))
                except Exception as e:
                    logger.error(f"TTS error: {e}")
            
            return assistant_content

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Apologies, I've encountered a problem: {str(e)}"


    def is_file_creation_request(self, user_input: str) -> bool:
        """Check if user input is explicitly requesting file creation"""
        file_creation_keywords = [
            'create', 'make', 'save', 'write', 'generate'
        ]
        file_indicators = [
            'file', '.py', '.txt', '.js', '.html', '.css', '.json', '.md'
        ]

        user_lower = user_input.lower()

        # Check for explicit file creation commands
        if any(keyword in user_lower for keyword in file_creation_keywords):
            if any(indicator in user_lower for indicator in file_indicators):
                return True

        # Check for direct file creation patterns
        file_creation_patterns = [
            r'create\s+\w+\.\w+',
            r'make\s+\w+\.\w+',
            r'save\s+.*\s+to\s+file',
            r'write\s+.*\s+to\s+file'
        ]

        for pattern in file_creation_patterns:
            if re.search(pattern, user_lower):
                return True

        return False

    @property
    def available_commands(self) -> List[str]:
        """Get list of all available commands for command suggestion"""
        commands = []
        
        # System commands
        commands.extend(['help', 'h', 'clear', 'c', 'history', 'hist', 'config', 'cfg',
                        'stats', 'statistics', 'quit', 'exit', 'q', 'voice', 'debug',
                        'search', 'rerun', 'vision', 'cls', 'tts'])
        
        # File commands
        commands.extend(['create', 'read', 'delete', 'list', 'ls'])
        
        # TTS commands
        commands.extend(['tts enable', 'tts disable', 'tts status', 'tts speak'])
        # Git commands
        commands.extend(['git', 'git publish', 'git push', 'git pull'])
        # Mic commands
        commands.extend(['mic', 'mic test', 'mic list'])

        # Email commands
        commands.extend(['email', 'email send'])
        # Alias management
        commands.extend(['alias', 'unalias', 'aliases'])
        # Reminder commands
        commands.extend([
            'reminder', 'reminder list', 'reminder remove', 'reminder delete'
        ])
 
        return commands

    def display_welcome_message(self):
        """Display welcome message"""
        search_status = "Available" if is_search_available() else "Not Available"
        
        # Display ASCII art if configured
        if self.config.ascii_art:
            self.colors.print_info(self.config.ascii_art)
            self.colors.print_info("")  # Empty line after ASCII art
        
        welcome_text = f"""
{'='*60}
  {self.config.bot_name}
{'='*60}
Model: {self.config.main_model}
STT Model: {self.config.stt_model} (Not yet implemented)
Search: {search_status} (Custom DuckDuckGo + Groq)
Type 'help' for commands, 'quit' to exit
{'='*60}
"""
        self.colors.print_info(welcome_text)

    def display_help(self):
        """Display help information with command chaining"""
        search_help = "  search <query>  - Search the web using DuckDuckGo" if is_search_available() else "  search          - Search functionality not available"

        help_text = f"""
    Available Commands:
    !help         - Show this help message
    clear         - Clear conversation history
    !history      - Show conversation history
    config        - Show current configuration
    stats         - Show session statistics
    {search_help.replace('search', '!search')}
    debug         - Show process information
    !quit/!exit   - Exit the program
    rerun         - Restarts the program and applies any new changes
    cls           - clears screen

    File Operations:
    !create <filename> [content] - Create file
    !read <filename>             - Read file content
    !delete <filename>           - Delete a file
    !list [pattern]              - List files (default: *)
    !ls [pattern]                - Alias for list

    Alias Commands:
    !alias <name> <command_chain>   - Create/update an alias
    !alias remove <name>            - Remove an alias
    !aliases                        - List all aliases

    Git:
    !git publish <file> "msg"   - Commit the file and push with message
    !git push                   - Push current branch to remote
    !git pull                   - Pull latest changes from remote
"""

        # Add conditional features based on mode and config
        if not self.server_mode and self.config.vision_enabled and self.vision_module:
            help_text += """
    Vision:
    vision <image> [prompt]     - Analyze image with AI
"""

        help_text += """
    TTS (Text-to-Speech):
    tts enable    - Enable TTS functionality
    tts disable   - Disable TTS functionality
    tts status    - Show TTS status
    tts speak <text> - Speak the given text

    Voice Commands:
    voice         - Start voice input mode (Ctrl+C to stop)
        """
        
        if self.server_mode:
            help_text += "\n    [SERVER MODE] - GUI features disabled for server environments"
            
        self.colors.print_command_output(help_text)
        self.colors.print_info(f"Process ID (PID): {pid}")

    @timeit
    def rerun(self):
        print("Restarting script to apply changes...")
        print("")
        print("")
        print("")
        # future me, yes this is on purpose, LET THEM STAY EMPTY!
        os.execv(sys.executable, [sys.executable] + sys.argv)

    @timeit
    def debug(self):
        print(f"Process ID (PID): {self.monitor.p.pid}")
        stats = self.monitor.get_stats()
        print(f"Usage rates:\nCPU: ({stats.cpu_1min:.2f}, {stats.cpu_5min:.2f}, {stats.cpu_10min:.2f}, {stats.cpu_15min:.2f})")
        print(f"RAM: ({stats.ram_1min:.2f}, {stats.ram_5min:.2f}, {stats.ram_10min:.2f}, {stats.ram_15min:.2f})")
        
        # Rough token usage estimate (characters / 4 ≈ tokens)
        total_chars = sum(len(msg["content"]) for msg in self.history.messages if msg["role"] != "system")
        approx_tokens = total_chars // 4  # Simple heuristic
        print(f"Approx. tokens in memory: {approx_tokens} (heuristic)")

        import sys, traceback
        threads = threading.enumerate()
        frames = sys._current_frames()

        descriptors = []
        for t in threads:
            tid = t.ident
            frame = frames.get(tid) if tid is not None else None
            if frame:
                stack = traceback.extract_stack(frame)
                last = stack[-1] if stack else None
                loc = f"{os.path.basename(last.filename)}:{last.lineno}" if last else "n/a"
            else:
                loc = "n/a"
            desc = f"[{t.ident}] {t.name}{' (daemon)' if t.daemon else ''} @ {loc}"
            descriptors.append(desc)

        # Print descriptors two per line (2x2 grid style)
        for i in range(0, len(descriptors), 2):
            left = descriptors[i]
            right = descriptors[i + 1] if i + 1 < len(descriptors) else ""
            # Pad left column to 60 characters for alignment
            print(f"{left:<60}{right}")

    @timeit
    def display_history(self):
        """Display conversation history"""
        self.colors.print_command_output("\n--- Conversation History ---")
        for msg in self.history.messages:
            if msg["role"] != "system":
                role_display = msg["role"].upper()
                self.colors.print_command_output(f"{role_display}: {msg['content']}")
        self.colors.print_command_output("--- End History ---\n")

    @timeit
    async def handle_vision(self, user_input: str) -> Optional[str]:
        """
        Usage: vision <image_path|url> [prompt]
        """
        if not self.vision_module:
            return "Vision module is not enabled or failed to initialize."
        parts = user_input.strip().split()
        if len(parts) < 2:
            return "Usage: vision <image_path|url> [prompt]"

        image_source = parts[1]
        user_prompt = " ".join(parts[2:]) if len(parts) > 2 else "Whats in this image? Respond as jarvis from the iron man movies would."

        try:
            # First, use the vision model to analyze the image
            vision_analysis = self.vision_module.analyze_image(image_source, user_prompt)
            
            # Then, feed the vision analysis to the main model for a more refined response
            vision_context = f"Image Analysis: {vision_analysis}\n\n Respond as jarvis. User's question about the image: {user_prompt}"
            
            # Create a conversation context for the main model
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": vision_context}
            ]
            
            # Get response from main model
            response = self.client.chat.completions.create(  # type: ignore[arg-type]
                model=self.main_model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            if response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return ""
            
        except Exception as e:
            return f"Error processing image: {e}"

    @timeit
    def display_config(self):
        """Display current configuration"""
        config_text = f"""
--- Configuration ---
Main Model: {self.config.main_model}
Vision Model: {self.config.vision_model}
STT Model: {self.config.stt_model}
AI Provider: {getattr(self.config, 'api_provider', 'groq')}
"""
        self.colors.print_command_output(config_text)

    @timeit
    def display_stats(self):
        """Display session statistics"""
        total_messages = len([msg for msg in self.history.messages if msg["role"] != "system"])
        user_messages = len([msg for msg in self.history.messages if msg["role"] == "user"])
        assistant_messages = len([msg for msg in self.history.messages if msg["role"] == "assistant"])

        stats_text = f"""
--- Session Statistics ---
Total Messages: {total_messages}
User Messages: {user_messages}
Assistant Messages: {assistant_messages}
--- End Statistics ---
"""
        self.colors.print_command_output(stats_text)

    def suggest_command(self, user_input: str) -> Optional[str]:
        """Suggest similar built-in command, preserving required '!' prefix."""
        if not DIFFLIB_AVAILABLE:
            return None

        # Remove leading '!' (if present) and any surrounding whitespace for matching
        cleaned_input = user_input.lstrip('!').strip().lower()

        # Separate the base command from any arguments (if provided)
        if " " in cleaned_input:
            base_cmd, remainder = cleaned_input.split(" ", 1)
        else:
            base_cmd, remainder = cleaned_input, ""

        # Attempt to find the closest match amongst the available commands
        matches = get_close_matches(
            base_cmd,
            self.available_commands,
            n=1,
            cutoff=0.8,
        )

        if not matches:
            return None

        suggested_base = matches[0]

        suggestion = f"!{suggested_base}"
        if remainder:
            suggestion += f" {remainder}"

        return suggestion

    def ask_confirmation(self, suggested_command: str) -> bool:
        """Ask user for confirmation on suggested command"""
        self.colors.print_warning(f"Perhaps you meant '{suggested_command}'?")
        self.colors.print_command_output("(y/n or just press Enter for yes): ")

        try:
            response = input().strip().lower()
            return response in ['', 'y', 'yes']
        except KeyboardInterrupt:
            return False

    def clear_screen(self):
        os.system("clear")
        config = BotConfig()


    async def process_command(self, command: str) -> bool:
        """Process system commands. Returns True if command was handled."""
        command_parts = command.lower().strip().split()
        base_command = command_parts[0] if command_parts else ""

        if base_command in ['help', 'h']:
            self.display_help()
            return True
        elif base_command in ['clear', 'c']:
            self.history.clear()
            self.history.add_message("system", self.config.system_prompt)
            self.colors.print_info("Conversation history cleared.")
            return True
        elif base_command in ['history', 'hist']:
            self.display_history()
            return True
        elif base_command in ['config', 'cfg']:
            self.display_config()
            return True
        elif base_command in ['stats', 'statistics']:
            self.display_stats()
            return True
        elif base_command in ['quit', 'exit', 'q']:
            self.colors.print_info(f"Goodbye! {self.config.bot_name} is shutting down.")
            # Stop main loop even when invoked from aliases/command chains
            self.running = False
            # In interactive mode, we simply mark running False; in unit tests or
            # chained execution it helps to raise SystemExit so upper layers can
            # abort further processing cleanly. We'll catch it in the run() loop.
            raise SystemExit
        elif base_command == 'voice':
            if self.stt_processor.is_available():
                try:
                    self.colors.print_info("Starting voice input mode...")
                    self.stt_processor.start_realtime()
                    
                    # Add callback to handle transcribed text
                    def handle_transcription(result):
                        if result and result.text:
                            self.colors.print_info(f"Transcribed: {result.text}")
                            # Process the transcribed text as a command or query
                            asyncio.create_task(self.process_command_line(result.text))
                    
                    if self.stt_processor.stt_manager:
                        self.stt_processor.stt_manager.add_callback(handle_transcription)
                    
                    # Let it run for a while (user can press Ctrl+C to stop)
                    self.colors.print_info("Listening... (Press Ctrl+C to stop)")
                    try:
                        while True:
                            await asyncio.sleep(0.1)
                    except KeyboardInterrupt:
                        self.colors.print_info("\nStopping voice input...")
                    finally:
                        self.stt_processor.stop_realtime()
                        
                except Exception as e:
                    self.colors.print_error(f"Voice input error: {e}")
            else:
                self.colors.print_command_output("Voice input not available. Make sure STT is enabled and configured.")
            return True
        elif base_command == 'tts':
            return await self.handle_tts_command(command_parts)
        elif base_command == 'debug':
            self.debug()
            return True

        elif base_command in ['alias', 'unalias', 'aliases']:
            return await self._handle_alias_command(base_command, command)

        # Execute stored alias if it matches
        alias_expansion = self.alias_manager.get_alias(base_command)
        if alias_expansion:
            await self.process_command_line(alias_expansion)
            return True

        if base_command == 'mic':
            return await self.handle_mic_command(command_parts)

        elif base_command == 'rerun':
            self.rerun()
            return True

        elif base_command == 'cls':
            self.clear_screen()
            return True

        elif base_command == 'git':
            return await self.handle_git_command(command_parts)

        elif base_command == 'email':
            return await self.handle_email_command(command_parts)

        elif base_command == 'search':
            if not is_search_available():
                self.colors.print_error("Search functionality is not available. Please check your GROQ_API_KEY.")
                return True

            if len(command_parts) < 2:
                self.colors.print_command_output("Usage: search <query>")
                return True

            query = " ".join(command_parts[1:])
            self.colors.print_info(f"Searching for: {query}")

            try:
                results = perform_search(query, self.config)
                self.colors.print_command_output(f"\nSearch Results:\n{results}")
            except Exception as e:
                self.colors.print_error(f"Search error: {e}")

            return True

        elif base_command in ['create', 'read', 'delete', 'list', 'ls']:
            file_result = await self.handle_file_operations(command)
            if file_result:
                self.colors.print_command_output(file_result)
            return True

        elif base_command == 'vision':
            if self.server_mode or not self.config.vision_enabled:
                self.colors.print_error("Vision module is disabled in server mode or not configured.")
                return True
            if self.vision_module is None:
                self.colors.print_error("Vision module not available.")
                return True
            output = await self.handle_vision(command)
            if output:
                self.colors.print_ai_response(output)
            return True

        elif base_command == 'reminder':
            return await self.handle_reminder_command(command_parts)

        # No match – not a built-in command
        return False





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
                self.colors.print_command_output("Enter file content (press Ctrl+D or Ctrl+Z when finished):")
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
            self.colors.print_warning(f"Are you sure you want to delete '{filename}'? (y/N): ")
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

    @timeit
    async def process_command_line(self, user_input: str) -> bool:
        """Process a command line that might contain operators"""
        if not hasattr(self, 'command_parser'):
            self.command_parser = CommandParser(self)

        # Check if this is an alias-management command – bypass operator parsing so
        # that any ';' or other symbols inside the alias definition are preserved.
        _stripped = user_input.lstrip()
        alias_mgmt_keywords = ("alias", "unalias", "aliases")
        is_alias_mgmt = False

        if _stripped.startswith('!'):
            bang_removed = _stripped[1:].lstrip()
            is_alias_mgmt = any(bang_removed.startswith(k) for k in alias_mgmt_keywords)
            if is_alias_mgmt:
                return await self.process_command(bang_removed)
        else:
            is_alias_mgmt = any(_stripped.startswith(k) for k in alias_mgmt_keywords)
            if is_alias_mgmt:
                return await self.process_command(_stripped)
 
        # Check if input contains any operators
        has_operators = any(op in user_input for op in ['||', '&&', '|', ';', '&'])

        if not has_operators:
            # Require '!' prefix for built-in commands
            if user_input.startswith('!'):
                return await self.process_command(user_input[1:].lstrip())
            else:
                # Treat as AI prompt (no built-in handling)
                return False

        # Parse and execute command chain
        try:
            command_chain = self.command_parser.parse_command_line(user_input)
            result = await self.command_parser.execute_command_chain(command_chain)

            if result["success"]:
                self.colors.print_info(f"Executed {result['total_commands']} commands:")
                for i, cmd_result in enumerate(result["results"]):
                    self.colors.print_command_output(f"  {i+1}. {cmd_result['command']}")
                    if cmd_result["success"]:
                        if cmd_result.get("type") == "ai":
                            self.colors.print_ai_response(f"     → {cmd_result['output']}")
                        else:
                            self.colors.print_command_output(f"     → {cmd_result['output']}")
                    else:
                        self.colors.print_error(f"     → Error: {cmd_result.get('error', 'Unknown error')}")
            else:
                self.colors.print_error(f"Command chain execution failed: {result.get('error', 'Unknown error')}")

            return True

        except Exception as e:
            self.colors.print_error(f"Error processing command chain: {e}")
            return False


    async def handle_tts_command(self, command_parts: List[str]) -> bool:
        """Handle TTS sub-commands (enable/disable/status/speak)."""
        if len(command_parts) < 2:
            self.colors.print_command_output("Usage: tts <enable|disable|status|speak> [text]")
            return True
        sub = command_parts[1].lower()
        if sub == "enable":
            res = self.tts_module.enable()
            if res["success"]:
                self.colors.print_info(res.get("message", "TTS enabled successfully"))
        elif sub == "disable":
            res = self.tts_module.disable()
            self.colors.print_info(res.get("message") if res["success"] else res.get("error", "TTS disable failed"))
        elif sub == "status":
            status = self.tts_module.get_status()
            self.colors.print_command_output(
                f"TTS Status: {status['status']}\nAvailable: {status['available']}\nLanguage: {status['language']}\nTLD: {status['tld']}"
            )
        elif sub == "speak":
            if len(command_parts) < 3:
                self.colors.print_command_output("Usage: tts speak <text>")
                return True
            text = " ".join(command_parts[2:])
            res = self.tts_module.speak(text)
            if res["success"]:
                message = res.get("message")
                if message:
                    self.colors.print_info(message)
            else:
                error_msg = res.get("error", "TTS speak failed")
                if error_msg:
                    self.colors.print_error(error_msg)
        else:
            self.colors.print_error(f"Unknown TTS command: {sub}")
        return True

    async def handle_git_command(self, command_parts: List[str]) -> bool:
        """git publish <file> "commit msg"  |  git push | git pull | git add"""
        if len(command_parts) < 2:
            self.colors.print_command_output("Usage: git publish <file_path> \"commit msg\" | git push | git pull"
                                " | git add")
            return True
        sub = command_parts[1].lower()
        if sub in {"publish", "commit"}:
            if len(command_parts) < 4:
                self.colors.print_command_output("Usage: git publish <file_path> \"commit msg\"")
                return True
            file_path = command_parts[2]
            commit_msg = " ".join(command_parts[3:]).strip('"')
            res = self.git_manager.commit_and_push(file_path, commit_msg)
            if res["success"]:
                self.colors.print_info("✓ Commit & push successful")
            else:
                error_msg = res.get('error')
                if error_msg:
                    self.colors.print_error(f"❌ {error_msg}")
        elif sub == "push":
            res = self.git_manager.push()
            if res["success"]:
                self.colors.print_info("✓ Push successful")
            else:
                error_msg = res.get('error')
                if error_msg:
                    self.colors.print_error(f"❌ {error_msg}")
        elif sub == "add":
            res = self.git_manager.add_all()
            if res["success"]:
                self.colors.print_info("✓ All changes staged (git add .)")
            else:
                error_msg = res.get('error')
                if error_msg:
                    self.colors.print_error(f"❌ {error_msg}")
        elif sub == "pull":
            res = self.git_manager.pull()
            if res["success"]:
                self.colors.print_info("✓ Pull successful")
            else:
                error_msg = res.get('error')
                if error_msg:
                    self.colors.print_error(f"❌ {error_msg}")
        else:
            self.colors.print_error(f"Unknown git command: {sub}")
        return True

    async def handle_email_command(self, command_parts: List[str]) -> bool:
        """Email command handler.

        Usage:
            email send <recipient_email> "subject" "body text"
        """
        if len(command_parts) < 2:
            self.colors.print_command_output("Usage: email send <to> \"subject\" \"body\"")
            return True

        sub = command_parts[1].lower()

        if sub != "send":
            self.colors.print_error("Unknown email command. Supported: send")
            return True

        if len(command_parts) < 5:
            self.colors.print_command_output("Usage: email send <to> \"subject\" \"body\"")
            return True

        to_addr = command_parts[2]
        subject = command_parts[3].strip('\'\"')
        body_parts = command_parts[4:]
        body = " ".join(body_parts).strip('\'\"')

        self.colors.print_info(f"Sending e-mail to {to_addr} …")

        res = self.email_manager.send_email(to_addr, subject, body)

        if res.get("success"):
            self.colors.print_info("✓ E-mail sent successfully")
        else:
            self.colors.print_error(f"❌ Failed to send e-mail: {res.get('error')}")

        return True

    async def handle_mic_command(self, command_parts: List[str]) -> bool:
        """Mic diagnostic commands: 'mic list' or 'mic test <index> [seconds]'."""
        if len(command_parts) < 2:
            self.colors.print_command_output("Usage: mic <list|test> [index] [seconds]")
            return True

        sub = command_parts[1].lower()
        if sub == 'list':
            devices = list_input_devices()
            if not devices:
                self.colors.print_error("No input devices found.")
            else:
                self.colors.print_command_output("Available input devices:")
                for d in devices:
                    self.colors.print_command_output(f"  {d['index']}: {d['name']} (channels={d['input_channels']}, sr={d['default_samplerate']})")
        elif sub == 'test':
            if len(command_parts) < 3:
                self.colors.print_command_output("Usage: mic test <device_index> [seconds]")
                return True
            try:
                device_idx = int(command_parts[2])
                seconds = int(command_parts[3]) if len(command_parts) > 3 else 3
            except ValueError:
                self.colors.print_error("Device index and seconds must be integers.")
                return True
            res = record_test(device=device_idx, seconds=seconds)
            if res['success']:
                peak = res['peak']
                file_path = res['file']
                self.colors.print_command_output(f"Recorded {seconds}s – peak amplitude: {peak}. File: {file_path}")
            else:
                self.colors.print_error(f"Mic test failed: {res.get('error')}")
        else:
            self.colors.print_command_output("Unknown mic command. Use 'mic list' or 'mic test'.")
        return True

    async def _handle_alias_command(self, base_command: str, full_input: str) -> bool:
        """Internal helper to process alias-related commands.

        Supported syntaxes:
            alias <name> <command_chain>
            alias "complex name" <command_chain>
            alias remove <name>
            unalias <name>
            aliases                         (lists all aliases)
        """

        if base_command == 'aliases':
            rows = self.alias_manager.list_aliases()
            if not rows:
                self.colors.print_command_output("No aliases defined.")
            else:
                self.colors.print_command_output("Defined aliases:")
                for r in rows:
                    self.colors.print_command_output(f"  {r['name']} -> {r['command']}")
            return True

        # Normalize for easier parsing
        lowered = full_input.lower().strip()

        # 2) Remove alias ---------------------------------------------------
        remove_prefixes = [
            'alias remove', 'alias delete', 'alias rm', 'unalias'
        ]
        for pref in remove_prefixes:
            if lowered.startswith(pref):
                # Compute original-case slice to preserve alias case
                alias_name = full_input[len(pref):].strip().strip('"\'')
                if not alias_name:
                    self.colors.print_command_output("Usage: alias remove <name> | unalias <name>")
                    return True
                removed = self.alias_manager.remove_alias(alias_name)
                if removed:
                    self.colors.print_info(f"Removed alias '{alias_name}'.")
                else:
                    self.colors.print_error(f"Alias '{alias_name}' not found.")
                return True

        # 3) Create / update alias -----------------------------------------
        # Strip the leading 'alias' keyword
        if lowered.startswith('alias '):
            definition = full_input[5:].strip()
        else:
            definition = full_input.strip()

        if not definition:
            self.colors.print_command_output("Usage: alias <name> <command_chain>")
            return True

        import re as _re
        pattern = r"(?:[\"'](?P<name>[^\"']+)[\"']|(?P<name_noquote>\S+))\s+(?P<cmd>.+)"
        m = _re.match(pattern, definition)
        if not m:
            self.colors.print_command_output("Invalid syntax. Example: alias \"close\" !cls ; !exit")
            return True

        alias_name = (m.group('name') or m.group('name_noquote')).strip()
        cmd_chain = m.group('cmd').strip()

        try:
            self.alias_manager.add_alias(alias_name, cmd_chain)
            self.colors.print_info(f"Alias '{alias_name}' → '{cmd_chain}' saved.")
        except Exception as exc:
            self.colors.print_error(f"Failed to save alias: {exc}")
        return True

    async def _handle_file_opening(self, assistant_content: str) -> str:
        if not self.file_opener:
            return assistant_content.replace('<OPEN_FILE>', '').replace('</OPEN_FILE>', '')
        matches = re.findall(r'<OPEN_FILE>(.*?)</OPEN_FILE>', assistant_content, re.DOTALL)
        for fp in matches:
            path = fp.strip()
            res = self.file_opener.open_file(path)
            msg = "Opened" if res["success"] else f"Failed to open ({res.get('error')})"
            assistant_content = assistant_content.replace(f'<OPEN_FILE>{fp}</OPEN_FILE>', msg)
        return assistant_content

    async def _handle_app_launching(self, assistant_content: str) -> str:
        if not self.file_opener:
            return assistant_content.replace('<LAUNCH_APP>', '').replace('</LAUNCH_APP>', '')
        matches = re.findall(r'<LAUNCH_APP>(.*?)</LAUNCH_APP>', assistant_content, re.DOTALL)
        for app in matches:
            app_name = app.strip()
            res = self.file_opener.launch_app(app_name)
            msg = "Launched" if res["success"] else f"Failed to launch ({res.get('error')})"
            assistant_content = assistant_content.replace(f'<LAUNCH_APP>{app}</LAUNCH_APP>', msg)
        return assistant_content

    async def _speak_response(self, text: str):
        """Speak text without blocking the main input loop."""
        if not self.tts_module.is_enabled():
            return
        clean = self._clean_text_for_tts(text)
        if not clean:
            return
        loop = asyncio.get_running_loop()
        # Offload the blocking speak() to default ThreadPool
        loop.run_in_executor(None, self.tts_module.speak, clean)

    def _clean_text_for_tts(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'#+\s*(.*)', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'<(OPEN_FILE|LAUNCH_APP|CREATE_FILE|SEARCH|GIT_COMMIT|GIT_ADD|SEND_EMAIL)>.*?</\1>', '', text, flags=re.DOTALL)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > getattr(self.config, 'tts_max_text_length', 500):
            text = text[: self.config.tts_max_text_length] + '...'
        return text

    def _speak_via_tts(self, text: str):
        """Speak arbitrary text if TTS is enabled."""
        if self.tts_module.is_enabled():
            clean = self._clean_text_for_tts(text)
            if clean:
                try:
                    self.tts_module.speak(clean)
                except Exception as exc:
                    logger.error(f"TTS speak failed: {exc}")

    def _on_wake_word_detected(self):
        """Callback invoked by the wake-word detector."""
        try:
            self.colors.print_info("[Wake-Word] Detected – awaiting your command…")
        except Exception:
            print("Wake-word detected")
    async def handle_reminder_command(self, command_parts: List[str]) -> bool:
        """Reminder command handler.

        Syntax::
            reminder add <ISO_TIME> <text>
            reminder list
            reminder remove <id>
            reminder delete <id>
        """
        if len(command_parts) < 2:
            self.colors.print_command_output(
                "Usage: reminder <add|list|remove> ...")
            return True

        sub = command_parts[1].lower()

        if sub == "add":
            # Explicit 'add' is deprecated – inform user.
            self.colors.print_command_output(
                "Adding reminders is now done through natural language. "
                "Just ask: 'Remind me to drink water at 3pm'.")
            return True

        elif sub == "list":
            res = self.reminder_manager.list()
            if res["success"]:
                if res["count"] == 0:
                    self.colors.print_command_output("No reminders.")
                else:
                    self.colors.print_command_output("Reminders:")
                    for r in res["reminders"]:
                        rid = r['id']
                        rtime = r['reminder_time']
                        text = r['reminder_text']
                        self.colors.print_command_output(
                            f"  {rid}. [{rtime}] {text}")
            else:
                self.colors.print_error(f"❌ {res['error']}")

        elif sub in {"remove", "delete"}:
            if len(command_parts) < 3:
                self.colors.print_command_output(
                    "Usage: reminder remove <id>")
                return True
            try:
                rid = int(command_parts[2])
            except ValueError:
                self.colors.print_error("Reminder id must be an integer")
                return True
            res = self.reminder_manager.remove(rid)
            if res["success"]:
                self.colors.print_info("✓ Reminder removed")
            else:
                self.colors.print_error(f"❌ {res['error']}")

        else:
            self.colors.print_error(f"Unknown reminder command: {sub}")

        return True

    async def run(self):
        """Main bot loop with command chaining and TTS support."""
        self.running = True
        self.display_welcome_message()

        try:
            while self.running:
                try:
                    # Prompt user
                    self.colors.print_prompt(f"{self.config.bot_name}> ")
                    user_input = input().strip()

                    if not user_input:
                        continue

                    # Handle built-in / chained commands
                    if await self.process_command_line(user_input):
                        # If an explicit exit command was executed, self.running will be set to False.
                        if not self.running:
                            break
                        continue

                    # Suggest corrections for built-in commands (only when prefixed with '!')
                    suggested = self.suggest_command(user_input) if user_input.startswith('!') else None

                    if suggested:
                        if self.ask_confirmation(suggested):
                            # User accepted suggestion – execute it
                            if await self.process_command_line(suggested):
                                if suggested.lower() in ["!quit", "!exit", "!q"]:
                                    break
                                continue  # Processed; prompt next input
                        else:
                            # User declined suggestion – proceed with original input
                            self.colors.print_command_output("Running command as entered…")

                    # Otherwise treat as AI prompt
                    response = await self.generate_response(user_input)
                    self.colors.print_ai_response(f"\n{self.config.bot_name}: {response}")

                    # Speak response if TTS is enabled
                    if self.tts_module.is_enabled():
                        await self._speak_response(response)

                    print()

                except KeyboardInterrupt:
                    self.colors.print_info(f"\n\n{self.config.bot_name}: Interrupted by user. Goodbye!")
                    break
        finally:
            # Stop background services gracefully
            if self.desktop_integration:
                self.desktop_integration.stop()
            if self.dbus_handler:
                self.dbus_handler.stop()
            if self.wake_detector:
                self.wake_detector.stop()




async def main():
    """Main entry point for JARVIS"""
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. AI Assistant")
    parser.add_argument("--config", type=str, default="setup/jarvis_config.yaml", help="Path to configuration file")
    parser.add_argument("--query", type=str, default=None, help="Run a single query and exit")
    parser.add_argument("--server", action="store_true", help="Run in server mode")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (only in server mode)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (only in server mode)")
    parser.add_argument('--test', action='store_true', help='Run built-in tests and exit')

    args = parser.parse_args()

    # Load configuration
    config_path = "setup/jarvis_config_server.yaml" if args.server else args.config
    config_manager = ConfigManager(config_path)
    config = config_manager.load_config()

    if not config.api_key and not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is required")
        print("Set it with: export GROQ_API_KEY='your_api_key_here'")
        sys.exit(1)

    # One-shot CLI mode: quickly answer a single question and exit
    if args.query is not None:
        try:
            bot = JarvisBot(config, server_mode=True)
            answer = await bot.generate_response(args.query)
            # Print plain text so callers can capture easily
            print(answer)
            sys.exit(0)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    # Initialize JARVIS bot
    try:
        jarvis_bot = JarvisBot(config, server_mode=args.server)
    except Exception as e:
        print(f"Failed to initialize JARVIS: {e}")
        sys.exit(1)

    if args.test:
        test_suite = JarvisTestSuite(jarvis_bot)
        test_suite.run()
        return

    # Server mode
    if args.server:
        try:
            from modules.server_module import JarvisServer
            server = JarvisServer(jarvis_bot, config, host=args.host, port=args.port)
            server.run()
        except ImportError:
            print("Error: Server mode requires additional dependencies.")
            print("Install them with: pip install -r requirements.txt")
            sys.exit(1)
        except Exception as e:
            print(f"Error starting server: {e}")
            sys.exit(1)
    else:
        # Interactive CLI mode
        await jarvis_bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Graceful shutdown message without ugly traceback
        print("\nJARVIS: Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
