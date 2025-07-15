#!/usr/bin/env python3


import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BotConfig:
    """Configuration class for JARVIS"""
    # Bot Identity
    bot_name: str = "JARVIS"
    
    # Models
    # Default model (kept for backward-compatibility). When api_provider is
    # 'groq' this is overridden by groq_main_model; when 'ollama' by
    # ollama_main_model.
    main_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    groq_main_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    ollama_main_model: str = "llama3"  # local default
    vision_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    stt_model: str = "distil-whisper-large-v3-en"
    search_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    # API Configuration
    api_provider: str = "groq"  # 'groq' or 'ollama'
    api_key: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    
    # Desktop Integration
    desktop_integration_enabled: bool = True
    dbus_enabled: bool = True
    dbus_service_name: str = "org.jarvis.Service"
    dbus_object_path: str = "/org/jarvis/Service"
    screenshot_enabled: bool = True
    screenshot_tool: str = "spectacle"
    screenshot_shortcut: str = "Super+Shift+C"
    screenshot_save_directory: str = "~/Pictures/Screenshots"
    screenshot_auto_delete: bool = True
    screenshot_timeout: int = 30
    hotkeys_enabled: bool = True
    hotkey_screenshot_analysis: str = "Super+Shift+C"
    hotkey_quick_command: str = "Super+Space"
    notifications_enabled: bool = True
    notifications_timeout: int = 5000
    notifications_icon: str = "jarvis"
    
    # Search Configuration
    search_enabled: bool = True
    search_max_results: int = 3
    search_timeout: int = 5
    
    # Database Configuration
    database_type: str = "sqlite"
    database_name: str = "jarvis.db"
    database_location: str = str(Path.home() / ".config" / "jarvis")
    
    # Conversation Settings
    max_history: int = 10
    clear_on_start: bool = False
    
    # Process Monitoring
    monitoring_enabled: bool = True
    monitoring_interval: int = 5
    
    # File Operations
    file_opening_enabled: bool = True
    
    # STT Configuration
    stt_enabled: bool = True
    stt_engine: str = "groq"
    
    # TTS Configuration
    tts_enabled: bool = False
    tts_model_path: str = "piper_models/en_GB-alan-medium.onnx"
    tts_max_text_length: int = 500
    
    # Face Detection
    face_detection_enabled: bool = True
    face_camera_index: int = 0
    face_detection_interval: float = 0.1
    face_min_size: int = 20
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    
    # Vision Module
    vision_enabled: bool = True
    vision_supported_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "bmp", "tiff"])
    vision_max_size: int = 5242880  # 5MB
    
    # Display Settings
    colors_enabled: bool = True
    rich_available: bool = True
    prompt_symbol: str = ">"
    welcome_message: bool = True
    help_on_start: bool = False
    ascii_art: Optional[str] = None
    streaming_enabled: bool = False
    streaming_speed: float = 0.02  # seconds per character
    
    # System Commands
    system_commands: List[str] = field(default_factory=lambda: [
        "help", "clear", "history", "config", "stats", "quit", "exit",
        "voice", "tts", "debug", "rerun", "cls"
    ])
    
    # File Commands
    file_commands: List[str] = field(default_factory=lambda: [
        "create", "read", "delete", "list", "ls"
    ])
    
    # Detection Commands
    detection_commands: List[str] = field(default_factory=lambda: [
        "detect", "camera", "stop", "status"
    ])
    
    # System Prompt
    system_prompt: str = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), a highly advanced AI assistant..."""

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Wake-Word Detection
    wake_word_enabled: bool = False
    wake_word_keywords: List[str] = field(default_factory=list)  # list of .ppn paths
    wake_word_sensitivity: float = 0.5
    wake_word_access_key: Optional[str] = None

    # Discord Bot
    discord_bot_enabled: bool = False
    discord_token: Optional[str] = None
    discord_channel_id: Optional[str] = None
    
    @property
    def desktop_integration(self) -> Dict[str, Any]:
        """Get desktop integration settings as a dictionary"""
        return {
            'enabled': self.desktop_integration_enabled,
            'dbus': {
                'enabled': self.dbus_enabled,
                'service_name': self.dbus_service_name,
                'object_path': self.dbus_object_path
            },
            'screenshot': {
                'enabled': self.screenshot_enabled,
                'tool': self.screenshot_tool,
                'shortcut': self.screenshot_shortcut,
                'save_directory': self.screenshot_save_directory,
                'auto_delete': self.screenshot_auto_delete,
                'timeout': self.screenshot_timeout
            },
            'hotkeys': {
                'enabled': self.hotkeys_enabled,
                'bindings': {
                    'screenshot_analysis': self.hotkey_screenshot_analysis,
                    'quick_command': self.hotkey_quick_command
                }
            },
            'notifications': {
                'enabled': self.notifications_enabled,
                'timeout': self.notifications_timeout,
                'icon': self.notifications_icon
            }
        }

    @property
    def dbus(self) -> Dict[str, Any]:
        """Get DBus settings as a dictionary"""
        return {
            'enabled': self.dbus_enabled,
            'service_name': self.dbus_service_name,
            'object_path': self.dbus_object_path
        }


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: str = "jarvis_config.yaml"):
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        self.bot_config: Optional[BotConfig] = None
    
    def load_config(self) -> BotConfig:
        """Load configuration from YAML file"""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_path):
                print(f"Configuration file '{self.config_path}' not found. Creating default configuration...")
                self._create_default_config()
            
            # Load YAML file
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config_data = yaml.safe_load(file)
            
            # Validate and convert to BotConfig
            self.bot_config = self._convert_to_bot_config()
            
            print(f"Configuration loaded successfully from '{self.config_path}'")
            return self.bot_config
            
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            print("Using default configuration...")
            return BotConfig()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration...")
            return BotConfig()
    
    def _create_default_config(self):
        """Create a default configuration file"""
        default_config = {
            'bot': {
                'name': 'JARVIS',
                'personality': {
                    'voice_tone': 'British-accented, polite, refined, eloquent',
                    'characteristics': 'Unfailingly loyal, calm under pressure, highly observant, logically efficient',
                    'humor': 'Subtle dry humor when appropriate'
                }
            },
            'models': {
                'main_model': 'meta-llama/llama-4-maverick-17b-128e-instruct',
                'vision_model': 'meta-llama/llama-4-maverick-17b-128e-instruct',
                'stt_model': 'distil-whisper-large-v3-en',
                'search_model': 'meta-llama/llama-4-scout-17b-16e-instruct'
            },
            'api': {
                'provider': 'groq',
                'max_tokens': 1024,
                'temperature': 0.7
            },
            'search': {
                'enabled': True,
                'max_results': 3,
                'timeout': 5,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            },
            'conversation': {
                'max_history': 10,
                'clear_on_start': False
            },
            'monitoring': {
                'enabled': True,
                'interval': 5
            },
            'file_operations': {
                'default_directory': '.',
                'allowed_extensions': ['.txt', '.py', '.js', '.html', '.css', '.json', '.md', '.yaml', '.yml'],
                'max_file_size': 10485760
            },
            'face_detection': {
                'enabled': True,
                'camera_index': 0,
                'detection_interval': 0.1,
                'min_face_size': 20,
                'scale_factor': 1.1,
                'min_neighbors': 5
            },
            'vision': {
                'enabled': True,
                'supported_formats': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                'max_image_size': 5242880
            },
            'display': {
                'colors_enabled': True,
                'rich_available': True,
                'prompt_symbol': '>',
                'welcome_message': True,
                'help_on_start': False
            },
            'system_prompt': """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), a highly advanced AI assistant...""",
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': None
            },
            'commands': {
                'system_commands': ['help', 'h', 'clear', 'c', 'history', 'hist', 'config', 'cfg', 
                                  'stats', 'statistics', 'quit', 'exit', 'q', 'voice', 'debug', 
                                  'search', 'rerun', 'vision', 'cls'],
                'file_commands': ['create', 'read', 'delete', 'list', 'ls'],
                'detection_commands': ['detect', 'camera', 'stop', 'status']
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(default_config, file, default_flow_style=False, allow_unicode=True)
        
        print(f"Default configuration created at '{self.config_path}'")
    
    def _convert_to_bot_config(self) -> BotConfig:
        """Convert loaded YAML data to BotConfig object"""
        config = BotConfig()
        
        # Bot identity
        if 'bot' in self.config_data:
            bot_data = self.config_data['bot']
            config.bot_name = bot_data.get('name', config.bot_name)
        
        # Models
        if 'models' in self.config_data:
            models_data = self.config_data['models']
            config.main_model = models_data.get('main_model', config.main_model)
            config.groq_main_model = models_data.get('groq_main_model', config.groq_main_model)
            config.ollama_main_model = models_data.get('ollama_main_model', config.ollama_main_model)
            config.vision_model = models_data.get('vision_model', config.vision_model)
            config.stt_model = models_data.get('stt_model', config.stt_model)
            config.search_model = models_data.get('search_model', config.search_model)
        
        # API settings
        if 'api' in self.config_data:
            api_data = self.config_data['api']
            config.max_tokens = api_data.get('max_tokens', config.max_tokens)
            config.temperature = api_data.get('temperature', config.temperature)
            config.api_key = api_data.get('api_key', config.api_key)
            config.api_provider = api_data.get('provider', config.api_provider)
        
        # Desktop Integration
        if 'desktop_integration' in self.config_data:
            desktop_data = self.config_data['desktop_integration']
            config.desktop_integration_enabled = desktop_data.get('enabled', config.desktop_integration_enabled)
            
            # DBus settings
            if 'dbus' in desktop_data:
                dbus_data = desktop_data['dbus']
                config.dbus_enabled = dbus_data.get('enabled', config.dbus_enabled)
                config.dbus_service_name = dbus_data.get('service_name', config.dbus_service_name)
                config.dbus_object_path = dbus_data.get('object_path', config.dbus_object_path)
            
            # Screenshot settings
            if 'screenshot' in desktop_data:
                screenshot_data = desktop_data['screenshot']
                config.screenshot_enabled = screenshot_data.get('enabled', config.screenshot_enabled)
                config.screenshot_tool = screenshot_data.get('tool', config.screenshot_tool)
                config.screenshot_shortcut = screenshot_data.get('shortcut', config.screenshot_shortcut)
                config.screenshot_save_directory = screenshot_data.get('save_directory', config.screenshot_save_directory)
                config.screenshot_auto_delete = screenshot_data.get('auto_delete', config.screenshot_auto_delete)
                config.screenshot_timeout = screenshot_data.get('timeout', config.screenshot_timeout)
            
            # Hotkeys settings
            if 'hotkeys' in desktop_data:
                hotkeys_data = desktop_data['hotkeys']
                config.hotkeys_enabled = hotkeys_data.get('enabled', config.hotkeys_enabled)
                if 'bindings' in hotkeys_data:
                    bindings_data = hotkeys_data['bindings']
                    config.hotkey_screenshot_analysis = bindings_data.get('screenshot_analysis', config.hotkey_screenshot_analysis)
                    config.hotkey_quick_command = bindings_data.get('quick_command', config.hotkey_quick_command)
            
            # Notifications settings
            if 'notifications' in desktop_data:
                notifications_data = desktop_data['notifications']
                config.notifications_enabled = notifications_data.get('enabled', config.notifications_enabled)
                config.notifications_timeout = notifications_data.get('timeout', config.notifications_timeout)
                config.notifications_icon = notifications_data.get('icon', config.notifications_icon)
        
        # Search settings
        if 'search' in self.config_data:
            search_data = self.config_data['search']
            config.search_enabled = search_data.get('enabled', config.search_enabled)
            config.search_max_results = search_data.get('max_results', config.search_max_results)
            config.search_timeout = search_data.get('timeout', config.search_timeout)
        
        # Conversation settings
        if 'conversation' in self.config_data:
            conv_data = self.config_data['conversation']
            config.max_history = conv_data.get('max_history', config.max_history)
            config.clear_on_start = conv_data.get('clear_on_start', config.clear_on_start)
        
        # Monitoring settings
        if 'monitoring' in self.config_data:
            monitor_data = self.config_data['monitoring']
            config.monitoring_enabled = monitor_data.get('enabled', config.monitoring_enabled)
            config.monitoring_interval = monitor_data.get('interval', config.monitoring_interval)
        
        # File operations
        if 'file_operations' in self.config_data:
            file_data = self.config_data['file_operations']
            config.file_opening_enabled = file_data.get('enabled', config.file_opening_enabled)
        
        # STT settings
        if 'stt' in self.config_data:
            stt_data = self.config_data['stt']
            config.stt_enabled = stt_data.get('enabled', config.stt_enabled)
            config.stt_engine = stt_data.get('engine', config.stt_engine)
        
        # TTS settings
        if 'tts' in self.config_data:
            tts_data = self.config_data['tts']
            config.tts_enabled = tts_data.get('enabled', config.tts_enabled)
            config.tts_model_path = tts_data.get('model_path', config.tts_model_path)
            config.tts_max_text_length = tts_data.get('max_text_length', config.tts_max_text_length)
        
        # Face detection
        if 'face_detection' in self.config_data:
            face_data = self.config_data['face_detection']
            config.face_detection_enabled = face_data.get('enabled', config.face_detection_enabled)
            config.face_camera_index = face_data.get('camera_index', config.face_camera_index)
            config.face_detection_interval = face_data.get('detection_interval', config.face_detection_interval)
            config.face_min_size = face_data.get('min_face_size', config.face_min_size)
            config.face_scale_factor = face_data.get('scale_factor', config.face_scale_factor)
            config.face_min_neighbors = face_data.get('min_neighbors', config.face_min_neighbors)
        
        # Vision
        if 'vision' in self.config_data:
            vision_data = self.config_data['vision']
            config.vision_enabled = vision_data.get('enabled', config.vision_enabled)
            config.vision_supported_formats = vision_data.get('supported_formats', config.vision_supported_formats)
            config.vision_max_size = vision_data.get('max_image_size', config.vision_max_size)
        
        # Display
        if 'display' in self.config_data:
            display_data = self.config_data['display']
            config.colors_enabled = display_data.get('colors_enabled', config.colors_enabled)
            config.rich_available = display_data.get('rich_available', config.rich_available)
            config.prompt_symbol = display_data.get('prompt_symbol', config.prompt_symbol)
            config.welcome_message = display_data.get('welcome_message', config.welcome_message)
            config.help_on_start = display_data.get('help_on_start', config.help_on_start)
            config.streaming_enabled = display_data.get('streaming_enabled', config.streaming_enabled)
            config.streaming_speed = display_data.get('streaming_speed', config.streaming_speed)
        
        # System prompt
        if 'system_prompt' in self.config_data:
            config.system_prompt = self.config_data['system_prompt']
        
        # Logging
        if 'logging' in self.config_data:
            log_data = self.config_data['logging']
            config.log_level = log_data.get('level', config.log_level)
            config.log_format = log_data.get('format', config.log_format)
            config.log_file = log_data.get('file', config.log_file)
        
        # Commands
        if 'commands' in self.config_data:
            cmd_data = self.config_data['commands']
            config.system_commands = cmd_data.get('system_commands', config.system_commands)
            config.file_commands = cmd_data.get('file_commands', config.file_commands)
            config.detection_commands = cmd_data.get('detection_commands', config.detection_commands)
        
        # Wake-word detection
        if 'wake_word' in self.config_data:
            ww_data = self.config_data['wake_word']
            config.wake_word_enabled = ww_data.get('enabled', config.wake_word_enabled)
            config.wake_word_keywords = ww_data.get('keywords', config.wake_word_keywords)
            config.wake_word_sensitivity = ww_data.get('sensitivity', config.wake_word_sensitivity)
            config.wake_word_access_key = ww_data.get('access_key', config.wake_word_access_key)
        
        # Discord Bot
        if 'discord_bot' in self.config_data:
            discord_data = self.config_data['discord_bot']
            config.discord_bot_enabled = discord_data.get('enabled', config.discord_bot_enabled)
            config.discord_token = discord_data.get('token', config.discord_token)
            config.discord_channel_id = discord_data.get('channel_id', config.discord_channel_id)
        
        return config
    
    def save_config(self, config: BotConfig):
        """Save current configuration to YAML file"""
        try:
            config_dict = {
                'bot': {
                    'name': config.bot_name
                },
                'models': {
                    'main_model': config.main_model,
                    'groq_main_model': config.groq_main_model,
                    'ollama_main_model': config.ollama_main_model,
                    'vision_model': config.vision_model,
                    'stt_model': config.stt_model,
                    'search_model': config.search_model
                },
                'api': {
                    'provider': config.api_provider,
                    'max_tokens': config.max_tokens,
                    'temperature': config.temperature
                },
                'desktop_integration': {
                    'enabled': config.desktop_integration_enabled,
                    'dbus': {
                        'enabled': config.dbus_enabled,
                        'service_name': config.dbus_service_name,
                        'object_path': config.dbus_object_path
                    },
                    'screenshot': {
                        'enabled': config.screenshot_enabled,
                        'tool': config.screenshot_tool,
                        'shortcut': config.screenshot_shortcut,
                        'save_directory': config.screenshot_save_directory,
                        'auto_delete': config.screenshot_auto_delete,
                        'timeout': config.screenshot_timeout
                    },
                    'hotkeys': {
                        'enabled': config.hotkeys_enabled,
                        'bindings': {
                            'screenshot_analysis': config.hotkey_screenshot_analysis,
                            'quick_command': config.hotkey_quick_command
                        }
                    },
                    'notifications': {
                        'enabled': config.notifications_enabled,
                        'timeout': config.notifications_timeout,
                        'icon': config.notifications_icon
                    }
                },
                'search': {
                    'enabled': config.search_enabled,
                    'max_results': config.search_max_results,
                    'timeout': config.search_timeout
                },
                'conversation': {
                    'max_history': config.max_history,
                    'clear_on_start': config.clear_on_start
                },
                'monitoring': {
                    'enabled': config.monitoring_enabled,
                    'interval': config.monitoring_interval
                },
                'file_operations': {
                    'enabled': config.file_opening_enabled
                },
                'stt': {
                    'enabled': config.stt_enabled,
                    'engine': config.stt_engine
                },
                'tts': {
                    'enabled': config.tts_enabled,
                    'model_path': config.tts_model_path,
                    'max_text_length': config.tts_max_text_length
                },
                'face_detection': {
                    'enabled': config.face_detection_enabled,
                    'camera_index': config.face_camera_index,
                    'detection_interval': config.face_detection_interval,
                    'min_face_size': config.face_min_size,
                    'scale_factor': config.face_scale_factor,
                    'min_neighbors': config.face_min_neighbors
                },
                'vision': {
                    'enabled': config.vision_enabled,
                    'supported_formats': config.vision_supported_formats,
                    'max_image_size': config.vision_max_size
                },
                'display': {
                    'colors_enabled': config.colors_enabled,
                    'rich_available': config.rich_available,
                    'prompt_symbol': config.prompt_symbol,
                    'welcome_message': config.welcome_message,
                    'help_on_start': config.help_on_start,
                    'streaming_enabled': config.streaming_enabled,
                    'streaming_speed': config.streaming_speed
                },
                'system_prompt': config.system_prompt,
                'logging': {
                    'level': config.log_level,
                    'format': config.log_format,
                    'file': config.log_file
                },
                'commands': {
                    'system_commands': config.system_commands,
                    'file_commands': config.file_commands,
                    'detection_commands': config.detection_commands
                },
                'wake_word': {
                    'enabled': config.wake_word_enabled,
                    'keywords': config.wake_word_keywords,
                    'sensitivity': config.wake_word_sensitivity,
                    'access_key': config.wake_word_access_key
                },
                'discord_bot': {
                    'enabled': config.discord_bot_enabled,
                    'token': config.discord_token,
                    'channel_id': config.discord_channel_id
                }
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False, allow_unicode=True)
            
            print(f"Configuration saved to '{self.config_path}'")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")


    
    def get_all_commands(self) -> list:
        """Get all available commands for fuzzy matching"""
        if not self.bot_config:
            return []
        
        all_commands = []
        all_commands.extend(self.bot_config.system_commands)
        all_commands.extend(self.bot_config.file_commands)
        all_commands.extend(self.bot_config.detection_commands)
        return all_commands 