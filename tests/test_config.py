"""
Tests for the CRS configuration system.

This module tests:
1. Loading configuration from config.toml
2. Getting default model and available models
3. Updating configuration programmatically
4. Using different config sections (models, chat, retrieval, ui)
"""

import pytest

from crs.config_loader import (
    get_available_models,
    get_config_loader,
    get_default_model,
)


class TestConfigLoader:
    """Test cases for the ConfigLoader class."""

    def test_get_config_loader_singleton(self):
        """Test that get_config_loader returns the same instance."""
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2

    def test_get_default_model(self):
        """Test getting the default model."""
        model = get_default_model()
        assert isinstance(model, str)
        assert len(model) > 0
        # Should be the value from config.toml
        assert model == "gpt-4.1-nano"

    def test_get_available_models(self):
        """Test getting available models."""
        models = get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # Default model should be in available models
        assert get_default_model() in models

    def test_model_config(self):
        """Test model configuration section."""
        config_loader = get_config_loader()
        model_config = config_loader.get_model_config()

        assert isinstance(model_config, dict)
        assert "default_model" in model_config
        assert model_config["default_model"] == "gpt-4.1-nano"

    def test_chat_config(self):
        """Test chat configuration section."""
        config_loader = get_config_loader()
        chat_config = config_loader.get_chat_config()

        assert isinstance(chat_config, dict)
        assert "max_history_length" in chat_config
        assert "enable_streaming" in chat_config
        assert chat_config["max_history_length"] == 20
        assert chat_config["enable_streaming"] is True

    def test_retrieval_config(self):
        """Test retrieval configuration section."""
        config_loader = get_config_loader()
        retrieval_config = config_loader.get_retrieval_config()

        assert isinstance(retrieval_config, dict)
        assert "top_k" in retrieval_config
        assert "similarity_threshold" in retrieval_config
        assert retrieval_config["top_k"] == 10
        assert retrieval_config["similarity_threshold"] == 0.7

    def test_ui_config(self):
        """Test UI configuration section."""
        config_loader = get_config_loader()
        ui_config = config_loader.get_ui_config()

        assert isinstance(ui_config, dict)
        assert "page_title" in ui_config
        assert "show_model_name" in ui_config
        assert "theme" in ui_config
        assert ui_config["page_title"] == "CRS Knowledge"
        assert ui_config["show_model_name"] is True
        assert ui_config["theme"] == "light"

    def test_dot_notation_access(self):
        """Test accessing config values using dot notation."""
        config_loader = get_config_loader()

        # Test existing keys
        assert config_loader.get("models.default_model") == "gpt-4.1-nano"
        assert config_loader.get("chat.enable_streaming") is True
        assert config_loader.get("ui.page_title") == "CRS Knowledge"

        # Test nonexistent keys with defaults
        assert config_loader.get("nonexistent.key", "default") == "default"
        assert config_loader.get("models.nonexistent", None) is None

    def test_config_modification(self):
        """Test modifying configuration values."""
        config_loader = get_config_loader()

        # Get original value
        original_model = config_loader.get("models.default_model")

        # Modify value
        config_loader.set("models.default_model", "gpt-4o")
        assert config_loader.get("models.default_model") == "gpt-4o"

        # Reset back to original
        config_loader.set("models.default_model", original_model)
        assert config_loader.get("models.default_model") == original_model


def demo_config_usage():
    """Demonstrate configuration usage (can be called manually for debugging)."""
    print("=== CRS Configuration System Demo ===\n")

    # Get the config loader instance
    config_loader = get_config_loader()

    # 1. Model configuration
    print("1. Model Configuration:")
    print(f"   Default model: {get_default_model()}")
    print(f"   Available models: {get_available_models()}")
    print()

    # 2. Chat configuration
    print("2. Chat Configuration:")
    chat_config = config_loader.get_chat_config()
    print(f"   Max history length: {chat_config.get('max_history_length')}")
    print(f"   Enable streaming: {chat_config.get('enable_streaming')}")
    print()

    # 3. Retrieval configuration
    print("3. Retrieval Configuration:")
    retrieval_config = config_loader.get_retrieval_config()
    print(f"   Top K: {retrieval_config.get('top_k')}")
    print(
        f"   Similarity threshold: {retrieval_config.get('similarity_threshold')}"
    )
    print()

    # 4. UI configuration
    print("4. UI Configuration:")
    ui_config = config_loader.get_ui_config()
    print(f"   Page title: {ui_config.get('page_title')}")
    print(f"   Show model name: {ui_config.get('show_model_name')}")
    print(f"   Theme: {ui_config.get('theme')}")
    print()

    # 5. Using dot notation to get specific values
    print("5. Using Dot Notation:")
    print(
        f"   models.default_model: {config_loader.get('models.default_model')}"
    )
    print(
        f"   chat.enable_streaming: {config_loader.get('chat.enable_streaming')}"
    )
    print(f"   ui.page_title: {config_loader.get('ui.page_title')}")
    print(
        f"   nonexistent.key (with default): {config_loader.get('nonexistent.key', 'default_value')}"
    )
    print()

    # 6. Temporarily modify configuration
    print("6. Configuration Modification:")
    original_model = config_loader.get("models.default_model")
    print(f"   Original model: {original_model}")

    config_loader.set("models.default_model", "gpt-4o")
    print(f"   Modified model: {config_loader.get('models.default_model')}")

    # Reset back
    config_loader.set("models.default_model", original_model)
    print(f"   Reset model: {config_loader.get('models.default_model')}")
    print()

    print("=== Demo Complete ===")


if __name__ == "__main__":
    # If run directly, show the demo
    demo_config_usage()
    """Demonstrate configuration usage."""
    print("=== CRS Configuration System Demo ===\n")

    # Get the config loader instance
    config_loader = get_config_loader()

    # 1. Model configuration
    print("1. Model Configuration:")
    print(f"   Default model: {get_default_model()}")
    print(f"   Available models: {get_available_models()}")
    print()

    # 2. Chat configuration
    print("2. Chat Configuration:")
    chat_config = config_loader.get_chat_config()
    print(f"   Max history length: {chat_config.get('max_history_length')}")
    print(f"   Enable streaming: {chat_config.get('enable_streaming')}")
    print()

    # 3. Retrieval configuration
    print("3. Retrieval Configuration:")
    retrieval_config = config_loader.get_retrieval_config()
    print(f"   Top K: {retrieval_config.get('top_k')}")
    print(
        f"   Similarity threshold: {retrieval_config.get('similarity_threshold')}"
    )
    print()

    # 4. UI configuration
    print("4. UI Configuration:")
    ui_config = config_loader.get_ui_config()
    print(f"   Page title: {ui_config.get('page_title')}")
    print(f"   Show model name: {ui_config.get('show_model_name')}")
    print(f"   Theme: {ui_config.get('theme')}")
    print()

    # 5. Using dot notation to get specific values
    print("5. Using Dot Notation:")
    print(
        f"   models.default_model: {config_loader.get('models.default_model')}"
    )
    print(
        f"   chat.enable_streaming: {config_loader.get('chat.enable_streaming')}"
    )
    print(f"   ui.page_title: {config_loader.get('ui.page_title')}")
    print(
        f"   nonexistent.key (with default): {config_loader.get('nonexistent.key', 'default_value')}"
    )
    print()

    # 6. Temporarily modify configuration
    print("6. Configuration Modification:")
    original_model = config_loader.get("models.default_model")
    print(f"   Original model: {original_model}")

    config_loader.set("models.default_model", "gpt-4o")
    print(f"   Modified model: {config_loader.get('models.default_model')}")

    # Reset back
    config_loader.set("models.default_model", original_model)
    print(f"   Reset model: {config_loader.get('models.default_model')}")
    print()

    print("=== Demo Complete ===")


if __name__ == "__main__":
    # If run directly, show the demo
    demo_config_usage()
