#!/usr/bin/env python3


import os
import yaml
import json
import logging
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

logger = logging.getLogger(__name__)

@dataclass
class ConfigValue:
    """Container for configuration values with metadata"""
    value: Any
    source: str  # 'default', 'file', 'env', 'override'
    type: str
    description: Optional[str] = None
    validation: Optional[str] = None

class ConfigValidator:
    """Validates configuration values against schemas and rules"""
    
    @staticmethod
    def validate_type(value: Any, expected_type: str) -> bool:
        """Validate value against expected type"""
        type_map = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict
        }
        return isinstance(value, type_map.get(expected_type, object))

    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[float] = None, max_val: Optional[float] = None) -> bool:
        """Validate numeric value within range"""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    @staticmethod
    def validate_pattern(value: str, pattern: str) -> bool:
        """Validate string against regex pattern"""
        import re
        return bool(re.match(pattern, value))

class ConfigWatcher(FileSystemEventHandler):
    """Watches for configuration file changes"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.last_reload = 0
        self.reload_delay = 1  # Minimum seconds between reloads

    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path == str(self.config_manager.config_file):
            current_time = time.time()
            if current_time - self.last_reload > self.reload_delay:
                logger.info(f"Configuration file changed: {event.src_path}")
                self.config_manager.reload()
                self.last_reload = current_time

class ConfigManager:
    """
    Configuration manager with support for multiple formats and sources.
    
    Features:
    - Load from YAML, JSON, or INI files
    - Environment variable override
    - Hot reload support
    - Validation rules
    - Default values
    - Type conversion
    """
    
    def __init__(self, 
                 config_file: Union[str, Path],
                 schema_file: Optional[Union[str, Path]] = None,
                 env_prefix: str = "",
                 auto_reload: bool = False):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
            schema_file: Optional path to schema file
            env_prefix: Prefix for environment variables
            auto_reload: Whether to watch for file changes
        """
        self.config_file = Path(config_file)
        self.schema_file = Path(schema_file) if schema_file else None
        self.env_prefix = env_prefix
        self.auto_reload = auto_reload
        
        # Initialize storage
        self.config: Dict[str, ConfigValue] = {}
        self.schema: Dict[str, Dict[str, Any]] = {}
        self.defaults: Dict[str, Any] = {}
        
        # Load configuration
        self._load_schema()
        self._load_config()
        self._load_env_vars()
        
        # Set up file watcher if auto_reload is enabled
        self.observer = None
        if auto_reload:
            self._setup_file_watcher()

    def _setup_file_watcher(self):
        """Set up configuration file watcher"""
        self.observer = Observer()
        handler = ConfigWatcher(self)
        self.observer.schedule(handler, str(self.config_file.parent), recursive=False)
        self.observer.start()

    def _load_schema(self):
        """Load configuration schema if available"""
        if not self.schema_file or not self.schema_file.exists():
            return
        
        try:
            with open(self.schema_file) as f:
                self.schema = yaml.safe_load(f)
            logger.info(f"Loaded schema from {self.schema_file}")
        except Exception as e:
            logger.error(f"Error loading schema: {e}")

    def _load_config(self):
        """Load configuration from file"""
        if not self.config_file.exists():
            logger.warning(f"Configuration file not found: {self.config_file}")
            return
        
        try:
            with open(self.config_file) as f:
                if self.config_file.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif self.config_file.suffix == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_file.suffix}")
            
            self._process_config_data(data)
            logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")

    def _process_config_data(self, data: Dict[str, Any], prefix: str = ""):
        """Process configuration data recursively"""
        for key, value in data.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                self._process_config_data(value, f"{full_key}.")
            else:
                self.config[full_key] = ConfigValue(
                    value=value,
                    source='file',
                    type=type(value).__name__
                )

    def _load_env_vars(self):
        """Load configuration from environment variables"""
        prefix = f"{self.env_prefix}_" if self.env_prefix else ""
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit() and value.count('.') == 1:
                        value = float(value)
                except ValueError:
                    pass
                
                self.config[config_key] = ConfigValue(
                    value=value,
                    source='env',
                    type=type(value).__name__
                )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if key in self.config:
            return self.config[key].value
        return default

    def set(self, key: str, value: Any, source: str = 'override'):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            source: Source of the value
        """
        self.config[key] = ConfigValue(
            value=value,
            source=source,
            type=type(value).__name__
        )

    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        self._load_env_vars()

    def save(self):
        """Save current configuration to file"""
        # Convert config to plain dictionary
        data = {}
        for key, config_value in self.config.items():
            parts = key.split('.')
            current = data
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = config_value.value
        
        # Save to file
        try:
            with open(self.config_file, 'w') as f:
                if self.config_file.suffix in ['.yaml', '.yml']:
                    yaml.safe_dump(data, f, default_flow_style=False)
                elif self.config_file.suffix == '.json':
                    json.dump(data, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def validate(self) -> List[str]:
        """
        Validate current configuration against schema.
        
        Returns:
            List of validation errors
        """
        errors = []
        validator = ConfigValidator()
        
        for key, schema in self.schema.items():
            if key not in self.config:
                if schema.get('required', False):
                    errors.append(f"Missing required key: {key}")
                continue
            
            value = self.config[key].value
            
            # Type validation
            if 'type' in schema:
                if not validator.validate_type(value, schema['type']):
                    errors.append(f"Invalid type for {key}: expected {schema['type']}")
            
            # Range validation
            if schema.get('type') in ['int', 'float']:
                min_val = schema.get('min')
                max_val = schema.get('max')
                if not validator.validate_range(value, min_val, max_val):
                    errors.append(f"Value for {key} out of range [{min_val}, {max_val}]")
            
            # Pattern validation
            if schema.get('type') == 'str' and 'pattern' in schema:
                if not validator.validate_pattern(value, schema['pattern']):
                    errors.append(f"Value for {key} does not match pattern: {schema['pattern']}")
        
        return errors

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Set configuration value using dictionary syntax"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return key in self.config

    def __del__(self):
        """Cleanup file watcher"""
        if self.observer:
            self.observer.stop()
            self.observer.join()

# Example usage
if __name__ == "__main__":
    # Create configuration manager
    config = ConfigManager(
        config_file="config.yaml",
        schema_file="schema.yaml",
        env_prefix="APP",
        auto_reload=True
    )
    
    # Access configuration
    debug = config.get('app.debug', False)
    port = config['app.port']
    
    # Set configuration
    config.set('app.name', 'MyApp')
    config['app.version'] = '1.0.0'
    
    # Save configuration
    config.save()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:", errors) 