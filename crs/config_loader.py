"""Configuration loader for CRS Knowledge system."""
import os
import toml
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


class ConfigLoader:
    """Loads and manages configuration for the CRS system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the config loader.
        
        Args:
            config_path: Path to config file. If None, uses default config.toml
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.toml"
        
        self.config_path = Path(config_path)
        self._config = None
    
    @lru_cache(maxsize=1)
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file with caching."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = toml.load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the loaded configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-related configuration."""
        return self.config.get("models", {})
    
    def get_default_model(self) -> str:
        """Get the default model name."""
        return self.get_model_config().get("default_model", "gpt-4.1-nano")
    
    def get_available_models(self) -> list:
        """Get list of available models from connectors module."""
        try:
            from crs.agents.connectors import available_models
            return list(available_models.keys())
        except ImportError:
            return ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"]
    
    def get_chat_config(self) -> Dict[str, Any]:
        """Get chat-related configuration."""
        return self.config.get("chat", {})
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval-related configuration."""
        return self.config.get("retrieval", {})
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI-related configuration."""
        return self.config.get("ui", {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., "models.default_model")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set a configuration value by dot notation key.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self):
        """Save the current configuration back to file."""
        try:
            with open(self.config_path, 'w') as f:
                toml.dump(self.config, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {e}")


# Global config loader instance
_config_loader = None

def get_config_loader() -> ConfigLoader:
    """Get the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def get_config() -> Dict[str, Any]:
    """Get the global configuration."""
    return get_config_loader().config

def get_default_model() -> str:
    """Get the default model name from config."""
    return get_config_loader().get_default_model()

def get_available_models() -> list:
    """Get available models from config."""
    return get_config_loader().get_available_models()
