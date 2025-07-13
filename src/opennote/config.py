"""
Configuration management for OpenNote.
Handles loading settings from environment variables, config files, and defaults.
"""
import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class OpenNoteConfig:
    """Configuration settings for OpenNote application."""
    
    # LLM Settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:latest"
    
    # Vector Database Settings
    chroma_embedding_model: str = "all-MiniLM-L6-v2"
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_chunk_min_size: int = 200
    
    # Agent Settings
    default_temperature: float = 0.7
    default_top_k: int = 5
    max_history_length: int = 10
    
    # Performance Settings
    request_timeout: int = 60
    batch_size: int = 50
    
    # UI Settings
    debug_mode: bool = False
    show_progress: bool = True
    
    @classmethod
    def from_env(cls) -> 'OpenNoteConfig':
        """Create configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "deepseek-coder:latest"),
            chroma_embedding_model=os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            default_chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", "1000")),
            default_chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200")),
            default_chunk_min_size=int(os.getenv("DEFAULT_CHUNK_MIN_SIZE", "200")),
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "10")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
            batch_size=int(os.getenv("BATCH_SIZE", "50")),
            debug_mode=os.getenv("DEBUG_MODE", "False").lower() == "true",
            show_progress=os.getenv("SHOW_PROGRESS", "True").lower() == "true"
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'OpenNoteConfig':
        """Create configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Merge with environment variables (env takes precedence)
            env_config = cls.from_env()
            
            # Override with file values where env is not set
            for key, value in config_data.items():
                if hasattr(env_config, key) and getattr(env_config, key) is None:
                    setattr(env_config, key, value)
            
            return env_config
            
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using environment/default values.")
            return cls.from_env()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}. Using environment/default values.")
            return cls.from_env()
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        if self.default_chunk_size <= 0:
            errors.append("default_chunk_size must be positive")
        
        if self.default_chunk_overlap < 0:
            errors.append("default_chunk_overlap must be non-negative")
        
        if self.default_chunk_overlap >= self.default_chunk_size:
            errors.append("default_chunk_overlap must be less than default_chunk_size")
        
        if self.default_temperature < 0 or self.default_temperature > 2:
            errors.append("default_temperature must be between 0 and 2")
        
        if self.default_top_k <= 0:
            errors.append("default_top_k must be positive")
        
        if self.max_history_length < 0:
            errors.append("max_history_length must be non-negative")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")


# Global configuration instance
_config: Optional[OpenNoteConfig] = None

def get_config() -> OpenNoteConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        config_path = os.path.join("configs", "config.yaml")
        _config = OpenNoteConfig.from_file(config_path)
        _config.validate()
    return _config

def set_config(config: OpenNoteConfig) -> None:
    """Set the global configuration instance."""
    global _config
    config.validate()
    _config = config