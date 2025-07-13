"""
Tests for the configuration management system.
"""
import os
import tempfile
import unittest
from unittest.mock import patch
import yaml

from src.opennote.config import OpenNoteConfig, get_config, set_config


class TestOpenNoteConfig(unittest.TestCase):
    """Test cases for OpenNoteConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OpenNoteConfig()
        
        self.assertEqual(config.openai_model, "gpt-3.5-turbo")
        self.assertEqual(config.ollama_base_url, "http://localhost:11434")
        self.assertEqual(config.default_chunk_size, 1000)
        self.assertEqual(config.default_chunk_overlap, 200)
        self.assertEqual(config.default_temperature, 0.7)
        self.assertEqual(config.default_top_k, 5)
        self.assertFalse(config.debug_mode)

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MODEL': 'gpt-4',
        'DEFAULT_CHUNK_SIZE': '1500',
        'DEBUG_MODE': 'true'
    })
    def test_from_env(self):
        """Test configuration from environment variables."""
        config = OpenNoteConfig.from_env()
        
        self.assertEqual(config.openai_api_key, 'test-key')
        self.assertEqual(config.openai_model, 'gpt-4')
        self.assertEqual(config.default_chunk_size, 1500)
        self.assertTrue(config.debug_mode)

    def test_from_file(self):
        """Test configuration from YAML file."""
        config_data = {
            'openai_model': 'gpt-4-turbo',
            'default_chunk_size': 2000,
            'debug_mode': True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = OpenNoteConfig.from_file(temp_path)
            self.assertEqual(config.openai_model, 'gpt-4-turbo')
            self.assertEqual(config.default_chunk_size, 2000)
            self.assertTrue(config.debug_mode)
        finally:
            os.unlink(temp_path)

    def test_validation_success(self):
        """Test successful configuration validation."""
        config = OpenNoteConfig(
            default_chunk_size=1000,
            default_chunk_overlap=200,
            default_temperature=0.7,
            default_top_k=5
        )
        # Should not raise any exception
        config.validate()

    def test_validation_failures(self):
        """Test configuration validation failures."""
        # Test negative chunk size
        config = OpenNoteConfig(default_chunk_size=-1)
        with self.assertRaises(ValueError):
            config.validate()
        
        # Test overlap >= chunk size
        config = OpenNoteConfig(default_chunk_size=100, default_chunk_overlap=100)
        with self.assertRaises(ValueError):
            config.validate()
        
        # Test invalid temperature
        config = OpenNoteConfig(default_temperature=3.0)
        with self.assertRaises(ValueError):
            config.validate()
        
        # Test invalid top_k
        config = OpenNoteConfig(default_top_k=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_global_config_management(self):
        """Test global configuration management functions."""
        # Create a test config
        test_config = OpenNoteConfig(debug_mode=True, default_chunk_size=500)
        
        # Set the global config
        set_config(test_config)
        
        # Get the global config
        retrieved_config = get_config()
        
        self.assertEqual(retrieved_config.debug_mode, True)
        self.assertEqual(retrieved_config.default_chunk_size, 500)


if __name__ == "__main__":
    unittest.main()