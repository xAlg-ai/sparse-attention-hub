"""Unit tests for ModelServer configuration."""

import pytest

from sparse_attention_hub.adapters.utils.config import ModelServerConfig


@pytest.mark.unit
class TestModelServerConfig:
    """Test the ModelServerConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelServerConfig()
        
        assert config.delete_on_zero_reference is False
        assert config.enable_stats_logging is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ModelServerConfig(
            delete_on_zero_reference=True,
            enable_stats_logging=False
        )
        
        assert config.delete_on_zero_reference is True
        assert config.enable_stats_logging is False

    def test_config_immutability(self) -> None:
        """Test that config behaves as expected after creation."""
        config = ModelServerConfig()
        
        # Modify values
        config.delete_on_zero_reference = True
        config.enable_stats_logging = False
        
        assert config.delete_on_zero_reference is True
        assert config.enable_stats_logging is False

    def test_config_repr(self) -> None:
        """Test string representation of config."""
        config = ModelServerConfig(
            delete_on_zero_reference=True,
            enable_stats_logging=False
        )
        
        repr_str = repr(config)
        assert "ModelServerConfig" in repr_str
        assert "delete_on_zero_reference=True" in repr_str
        assert "enable_stats_logging=False" in repr_str

    def test_config_equality(self) -> None:
        """Test equality comparison of configs."""
        config1 = ModelServerConfig(delete_on_zero_reference=True, enable_stats_logging=True)
        config2 = ModelServerConfig(delete_on_zero_reference=True, enable_stats_logging=True)
        config3 = ModelServerConfig(delete_on_zero_reference=False, enable_stats_logging=True)
        
        assert config1 == config2
        assert config1 != config3
