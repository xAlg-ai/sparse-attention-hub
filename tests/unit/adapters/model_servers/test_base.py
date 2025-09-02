"""Unit tests for the base ModelServer class."""

import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

from sparse_attention_hub.adapters.model_servers.base import (
    ModelEntry,
    ModelServer,
    TokenizerEntry,
)
from sparse_attention_hub.adapters.utils.config import ModelServerConfig
from sparse_attention_hub.adapters.utils.exceptions import (
    ModelCreationError,
    ReferenceCountError,
    ResourceCleanupError,
    TokenizerCreationError,
)


class ConcreteModelServer(ModelServer):
    """Concrete implementation for testing."""

    def __init__(self, config: Optional[ModelServerConfig] = None) -> None:
        super().__init__(config)
        self.created_models = []
        self.created_tokenizers = []
        self.deleted_models = []
        self.deleted_tokenizers = []

    def _create_model(self, model_name: str, gpu_id: Optional[int], model_kwargs: Dict[str, Any]) -> Any:
        """Mock model creation."""
        # Only fail during creation if model name ends with "fail" (not delete_fail)
        if model_name.endswith("fail") and not "delete_fail" in model_name:
            raise ValueError(f"Failed to create model {model_name}")
        
        model = Mock()
        model.name = model_name
        model.gpu_id = gpu_id
        self.created_models.append((model_name, gpu_id, model_kwargs))
        return model

    def _create_tokenizer(self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any]) -> Any:
        """Mock tokenizer creation."""
        if "fail" in tokenizer_name:
            raise ValueError(f"Failed to create tokenizer {tokenizer_name}")
        
        tokenizer = Mock()
        tokenizer.name = tokenizer_name
        self.created_tokenizers.append((tokenizer_name, tokenizer_kwargs))
        return tokenizer

    def _delete_model(self, model: Any, gpu_id: Optional[int]) -> None:
        """Mock model deletion."""
        if hasattr(model, "name") and "delete_fail" in model.name:
            raise RuntimeError(f"Failed to delete model {model.name}")
        
        self.deleted_models.append((model, gpu_id))

    def _delete_tokenizer(self, tokenizer: Any) -> None:
        """Mock tokenizer deletion."""
        if hasattr(tokenizer, "name") and "delete_fail" in tokenizer.name:
            raise RuntimeError(f"Failed to delete tokenizer {tokenizer.name}")
        
        self.deleted_tokenizers.append(tokenizer)


@pytest.mark.unit
class TestModelEntry:
    """Test the ModelEntry dataclass."""

    def test_model_entry_creation(self) -> None:
        """Test ModelEntry creation with all fields."""
        model = Mock()
        kwargs = {"torch_dtype": "float16"}
        
        entry = ModelEntry(
            model=model,
            reference_count=1,
            gpu_id=0,
            model_kwargs=kwargs
        )
        
        assert entry.model == model
        assert entry.reference_count == 1
        assert entry.gpu_id == 0
        assert entry.model_kwargs == kwargs
        assert isinstance(entry.creation_time, datetime)
        assert isinstance(entry.last_access_time, datetime)

    def test_model_entry_defaults(self) -> None:
        """Test ModelEntry creation with default timestamps."""
        model = Mock()
        kwargs = {"torch_dtype": "float16"}
        
        entry = ModelEntry(
            model=model,
            reference_count=1,
            gpu_id=None,  # CPU
            model_kwargs=kwargs
        )
        
        assert entry.gpu_id is None
        assert entry.creation_time is not None
        assert entry.last_access_time is not None

    def test_model_entry_timestamp_update(self) -> None:
        """Test updating last access time."""
        model = Mock()
        entry = ModelEntry(
            model=model,
            reference_count=1,
            gpu_id=0,
            model_kwargs={}
        )
        
        original_time = entry.last_access_time
        time.sleep(0.01)  # Small delay
        entry.last_access_time = datetime.now()
        
        assert entry.last_access_time > original_time


@pytest.mark.unit
class TestTokenizerEntry:
    """Test the TokenizerEntry dataclass."""

    def test_tokenizer_entry_creation(self) -> None:
        """Test TokenizerEntry creation with all fields."""
        tokenizer = Mock()
        kwargs = {"padding_side": "left"}
        
        entry = TokenizerEntry(
            tokenizer=tokenizer,
            reference_count=2,
            tokenizer_kwargs=kwargs
        )
        
        assert entry.tokenizer == tokenizer
        assert entry.reference_count == 2
        assert entry.tokenizer_kwargs == kwargs
        assert isinstance(entry.creation_time, datetime)
        assert isinstance(entry.last_access_time, datetime)


@pytest.mark.unit
class TestModelServerSingleton:
    """Test ModelServer singleton behavior."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        # Reset the singleton instance
        ModelServer._instance = None

    def teardown_method(self) -> None:
        """Clean up after each test."""
        # Reset the singleton instance
        ModelServer._instance = None

    def test_singleton_pattern(self) -> None:
        """Test that ModelServer follows singleton pattern."""
        server1 = ConcreteModelServer()
        server2 = ConcreteModelServer()
        
        assert server1 is server2

    def test_singleton_with_different_configs(self) -> None:
        """Test singleton behavior with different configs."""
        config1 = ModelServerConfig(enable_stats_logging=True)
        config2 = ModelServerConfig(enable_stats_logging=False)
        
        server1 = ConcreteModelServer(config1)
        
        # Should raise ValueError when trying to create with different config
        with pytest.raises(ValueError, match="Trying to create ModelServer with different config"):
            server2 = ConcreteModelServer(config2)

    def test_singleton_thread_safety(self) -> None:
        """Test singleton thread safety."""
        instances = []
        
        def create_instance():
            instance = ConcreteModelServer()
            instances.append(instance)
        
        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All instances should be the same
        assert len(set(id(instance) for instance in instances)) == 1


@pytest.mark.unit
class TestModelServerInitialization:
    """Test ModelServer initialization."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        with ModelServer._lock:
            ModelServer._instance = None
        assert ModelServer._instance is None

    def teardown_method(self) -> None:
        """Clean up after each test."""
        with ModelServer._lock:
            ModelServer._instance = None
        assert ModelServer._instance is None

    def test_default_initialization(self) -> None:
        """Test initialization with default config."""
        assert ModelServer._instance is None
        server = ConcreteModelServer()
        
        assert isinstance(server.config, ModelServerConfig)
        assert server.config.delete_on_zero_reference is False
        assert server.config.enable_stats_logging is True
        assert len(server._models) == 0
        assert len(server._tokenizers) == 0

    def test_custom_config_initialization(self) -> None:
        """Test initialization with custom config."""
        assert ModelServer._instance is None
        config = ModelServerConfig(
            delete_on_zero_reference=True,
            enable_stats_logging=False
        )
        server = ConcreteModelServer(config)
        
        assert server.config == config
        assert server.config.delete_on_zero_reference is True
        assert server.config.enable_stats_logging is False

    @patch('sparse_attention_hub.adapters.model_servers.base.logging.getLogger')
    def test_logger_initialization(self, mock_logger) -> None:
        """Test logger initialization."""
        assert ModelServer._instance is None
        server = ConcreteModelServer()
        
        mock_logger.assert_called_once()
        assert hasattr(server, 'logger')


@pytest.mark.unit
class TestModelManagement:
    """Test model management functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ConcreteModelServer()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    def test_get_model_creates_new(self) -> None:
        """Test getting a model creates new instance when not cached."""
        model = self.server.get_model("test-model", 0, {"torch_dtype": "float16"})
        
        assert model is not None
        assert model.name == "test-model"
        assert len(self.server._models) == 1
        assert len(self.server.created_models) == 1

    def test_get_model_returns_cached(self) -> None:
        """Test getting a model returns cached instance."""
        kwargs = {"torch_dtype": "float16"}
        
        model1 = self.server.get_model("test-model", 0, kwargs)
        model2 = self.server.get_model("test-model", 0, kwargs)
        
        assert model1 is model2
        assert len(self.server._models) == 1
        assert len(self.server.created_models) == 1  # Only created once

    def test_get_model_increments_reference_count(self) -> None:
        """Test that getting a model increments reference count."""
        kwargs = {"torch_dtype": "float16"}
        
        self.server.get_model("test-model", 0, kwargs)
        self.server.get_model("test-model", 0, kwargs)
        
        # Find the model entry
        from sparse_attention_hub.adapters.utils.model_utils import generate_model_key
        key = generate_model_key("test-model", 0, kwargs)
        assert key in self.server._models
        assert self.server._models[key].reference_count == 2

    def test_get_model_different_configs_different_instances(self) -> None:
        """Test that different configs create different model instances."""
        model1 = self.server.get_model("test-model", 0, {"torch_dtype": "float16"})
        model2 = self.server.get_model("test-model", 0, {"torch_dtype": "float32"})
        
        assert model1 is not model2
        assert len(self.server._models) == 2
        assert len(self.server.created_models) == 2

    def test_get_model_different_gpus_different_instances(self) -> None:
        """Test that different GPUs create different model instances."""
        kwargs = {"torch_dtype": "float16"}
        
        model1 = self.server.get_model("test-model", 0, kwargs)
        model2 = self.server.get_model("test-model", 1, kwargs)
        
        assert model1 is not model2
        assert len(self.server._models) == 2

    def test_get_model_cpu_vs_gpu(self) -> None:
        """Test CPU vs GPU model placement."""
        kwargs = {"torch_dtype": "float16"}
        
        model_gpu = self.server.get_model("test-model", 0, kwargs)
        model_cpu = self.server.get_model("test-model", None, kwargs)
        
        assert model_gpu is not model_cpu
        assert len(self.server._models) == 2

    def test_get_model_creation_failure(self) -> None:
        """Test model creation failure handling."""
        with pytest.raises(ModelCreationError) as exc_info:
            self.server.get_model("model-fail", 0, {})
        
        assert "Failed to create model: model-fail" in str(exc_info.value)
        assert exc_info.value.model_name == "model-fail"
        assert exc_info.value.gpu_id == 0
        assert len(self.server._models) == 0

    def test_clear_model_decrements_reference(self) -> None:
        """Test clearing model decrements reference count."""
        kwargs = {"torch_dtype": "float16"}
        
        # Get model twice
        self.server.get_model("test-model", 0, kwargs)
        self.server.get_model("test-model", 0, kwargs)
        
        # Clear once
        result = self.server.clear_model("test-model", 0, kwargs)
        
        assert result is True
        models = list(self.server._models.values())
        assert models[0].reference_count == 1
        assert len(self.server.deleted_models) == 0  # Not deleted yet

    def test_clear_model_with_zero_reference_lazy_mode(self) -> None:
        """Test clearing model to zero reference in lazy mode."""
        kwargs = {"torch_dtype": "float16"}
        
        # Get and clear model
        self.server.get_model("test-model", 0, kwargs)
        result = self.server.clear_model("test-model", 0, kwargs)
        
        assert result is True
        models = list(self.server._models.values())
        assert models[0].reference_count == 0
        assert len(self.server.deleted_models) == 0  # Lazy mode - not deleted

    def test_clear_model_with_zero_reference_aggressive_mode(self) -> None:
        """Test clearing model to zero reference in aggressive mode."""
        config = ModelServerConfig(delete_on_zero_reference=True)
        ModelServer._instance = None
        server = ConcreteModelServer(config)
        
        kwargs = {"torch_dtype": "float16"}
        
        # Get and clear model
        server.get_model("test-model", 0, kwargs)
        result = server.clear_model("test-model", 0, kwargs)
        
        assert result is True
        assert len(server._models) == 0  # Should be deleted
        assert len(server.deleted_models) == 1

    def test_clear_nonexistent_model(self) -> None:
        """Test clearing nonexistent model."""
        result = self.server.clear_model("nonexistent", 0, {})
        
        assert result is False

    def test_clear_model_negative_reference_count_error(self) -> None:
        """Test that clearing model with zero references raises error."""
        kwargs = {"torch_dtype": "float16"}
        
        self.server.get_model("test-model", 0, kwargs)
        self.server.clear_model("test-model", 0, kwargs)
        
        # Try to clear again - should raise error
        with pytest.raises(ReferenceCountError):
            self.server.clear_model("test-model", 0, kwargs)


@pytest.mark.unit
class TestTokenizerManagement:
    """Test tokenizer management functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ConcreteModelServer()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    def test_get_tokenizer_creates_new(self) -> None:
        """Test getting a tokenizer creates new instance when not cached."""
        tokenizer = self.server.get_tokenizer("test-tokenizer", {"padding_side": "left"})
        
        assert tokenizer is not None
        assert tokenizer.name == "test-tokenizer"
        assert len(self.server._tokenizers) == 1
        assert len(self.server.created_tokenizers) == 1

    def test_get_tokenizer_returns_cached(self) -> None:
        """Test getting a tokenizer returns cached instance."""
        kwargs = {"padding_side": "left"}
        
        tokenizer1 = self.server.get_tokenizer("test-tokenizer", kwargs)
        tokenizer2 = self.server.get_tokenizer("test-tokenizer", kwargs)
        
        assert tokenizer1 is tokenizer2
        assert len(self.server._tokenizers) == 1
        assert len(self.server.created_tokenizers) == 1

    def test_get_tokenizer_creation_failure(self) -> None:
        """Test tokenizer creation failure handling."""
        with pytest.raises(TokenizerCreationError) as exc_info:
            self.server.get_tokenizer("fail-tokenizer", {})
        
        assert "Failed to create tokenizer: fail-tokenizer" in str(exc_info.value)
        assert exc_info.value.tokenizer_name == "fail-tokenizer"
        assert len(self.server._tokenizers) == 0

    def test_clear_tokenizer_decrements_reference(self) -> None:
        """Test clearing tokenizer decrements reference count."""
        kwargs = {"padding_side": "left"}
        
        # Get tokenizer twice
        self.server.get_tokenizer("test-tokenizer", kwargs)
        self.server.get_tokenizer("test-tokenizer", kwargs)
        
        # Clear once
        result = self.server.clear_tokenizer("test-tokenizer", kwargs)
        
        assert result is True
        tokenizers = list(self.server._tokenizers.values())
        assert tokenizers[0].reference_count == 1


@pytest.mark.unit
class TestMemoryCleanup:
    """Test memory cleanup functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ConcreteModelServer()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    def test_release_memory_and_clean_single_model(self) -> None:
        """Test releasing memory for a single model."""
        kwargs = {"torch_dtype": "float16"}
        
        # Create model and clear reference
        self.server.get_model("test-model", 0, kwargs)
        self.server.clear_model("test-model", 0, kwargs)
        
        # Now clean it up
        result = self.server.release_memory_and_clean("test-model", 0, kwargs)
        
        assert result is True
        assert len(self.server._models) == 0
        assert len(self.server.deleted_models) == 1

    def test_release_memory_and_clean_with_active_references(self) -> None:
        """Test releasing memory with active references."""
        kwargs = {"torch_dtype": "float16"}
        
        # Create model but don't clear reference
        self.server.get_model("test-model", 0, kwargs)
        
        # Try to clean - should fail without force
        result = self.server.release_memory_and_clean("test-model", 0, kwargs, force_delete=False)
        
        assert result is False
        assert len(self.server._models) == 1
        assert len(self.server.deleted_models) == 0

    def test_release_memory_and_clean_force_delete(self) -> None:
        """Test force deleting model with active references."""
        kwargs = {"torch_dtype": "float16"}
        
        # Create model but don't clear reference
        self.server.get_model("test-model", 0, kwargs)
        
        # Force clean
        result = self.server.release_memory_and_clean("test-model", 0, kwargs, force_delete=True)
        
        assert result is True
        assert len(self.server._models) == 0
        assert len(self.server.deleted_models) == 1

    def test_release_memory_and_clean_all(self) -> None:
        """Test releasing all models."""
        # Create multiple models
        self.server.get_model("model1", 0, {})
        self.server.get_model("model2", 1, {})
        self.server.get_model("model3", None, {})
        
        # Clear references for some
        self.server.clear_model("model1", 0, {})
        self.server.clear_model("model2", 1, {})
        # model3 still has active reference
        
        # Clean all
        stats = self.server.release_memory_and_clean_all(force_delete=False)
        
        assert stats["deleted"] == 2
        assert stats["skipped"] == 1
        assert len(self.server._models) == 1  # model3 still there

    def test_release_memory_and_clean_all_force(self) -> None:
        """Test force releasing all models."""
        # Create multiple models
        self.server.get_model("model1", 0, {})
        self.server.get_model("model2", 1, {})
        
        # Don't clear any references
        
        # Force clean all
        stats = self.server.release_memory_and_clean_all(force_delete=True)
        
        assert stats["deleted"] == 2
        assert stats["skipped"] == 0
        assert len(self.server._models) == 0

    def test_cleanup_unused(self) -> None:
        """Test cleaning up unused resources."""
        # Create models and tokenizers
        self.server.get_model("model1", 0, {})
        self.server.get_model("model2", 1, {})
        self.server.get_tokenizer("tokenizer1", {})
        
        # Clear some references
        self.server.clear_model("model1", 0, {})
        self.server.clear_tokenizer("tokenizer1", {})
        
        # Clean unused
        stats = self.server.cleanup_unused()
        
        assert stats["models"]["deleted"] == 1
        assert stats["models"]["skipped"] == 1
        assert stats["tokenizers"]["deleted"] == 1
        assert stats["tokenizers"]["skipped"] == 0

    def test_deletion_failure_handling(self) -> None:
        """Test handling of deletion failures."""
        # Create model that will fail to delete (using name that doesn't trigger creation failure)
        self.server.get_model("model_delete_fail", 0, {})
        self.server.clear_model("model_delete_fail", 0, {})
        
        # Try to clean - should raise ResourceCleanupError
        with pytest.raises(ResourceCleanupError):
            self.server.release_memory_and_clean("model_delete_fail", 0, {})


@pytest.mark.unit
class TestStatistics:
    """Test statistics functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ConcreteModelServer()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    def test_get_model_stats_empty(self) -> None:
        """Test model statistics with no models."""
        stats = self.server.get_model_stats()
        
        assert stats["total_models"] == 0
        assert stats["total_references"] == 0
        assert stats["zero_reference_models"] == 0
        assert stats["model_keys"] == []

    def test_get_model_stats_with_models(self) -> None:
        """Test model statistics with models."""
        # Create models
        self.server.get_model("model1", 0, {})
        self.server.get_model("model1", 0, {})  # Same model, increment reference
        self.server.get_model("model2", 1, {})
        
        # Clear one reference
        self.server.clear_model("model2", 1, {})
        
        stats = self.server.get_model_stats()
        
        assert stats["total_models"] == 2
        assert stats["total_references"] == 2  # model1: 2, model2: 0
        assert stats["zero_reference_models"] == 1  # model2
        assert len(stats["model_keys"]) == 2

    def test_get_tokenizer_stats(self) -> None:
        """Test tokenizer statistics."""
        # Create tokenizers
        self.server.get_tokenizer("tokenizer1", {})
        self.server.get_tokenizer("tokenizer2", {"padding": "left"})
        
        stats = self.server.get_tokenizer_stats()
        
        assert stats["total_tokenizers"] == 2
        assert stats["total_references"] == 2
        assert stats["zero_reference_tokenizers"] == 0
        assert len(stats["tokenizer_keys"]) == 2


@pytest.mark.unit
class TestThreadSafety:
    """Test thread safety of ModelServer operations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        ModelServer._instance = None
        self.server = ConcreteModelServer()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        ModelServer._instance = None

    def test_concurrent_model_access(self) -> None:
        """Test concurrent model access is thread-safe."""
        kwargs = {"torch_dtype": "float16"}
        models = []
        
        def get_model():
            model = self.server.get_model("test-model", 0, kwargs)
            models.append(model)
        
        threads = [threading.Thread(target=get_model) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All should be the same model instance
        assert len(set(id(model) for model in models)) == 1
        # Reference count should be 10
        model_entries = list(self.server._models.values())
        assert len(model_entries) == 1
        assert model_entries[0].reference_count == 10

    def test_concurrent_cleanup(self) -> None:
        """Test concurrent cleanup operations."""
        kwargs = {"torch_dtype": "float16"}
        
        # Create model
        self.server.get_model("test-model", 0, kwargs)
        self.server.clear_model("test-model", 0, kwargs)
        
        results = []
        
        def cleanup_model():
            result = self.server.release_memory_and_clean("test-model", 0, kwargs)
            results.append(result)
        
        threads = [threading.Thread(target=cleanup_model) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Only one should succeed
        assert sum(results) == 1
        assert len(self.server._models) == 0
