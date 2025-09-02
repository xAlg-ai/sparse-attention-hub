"""Unit tests for ModelServer utility functions."""

import pytest

from sparse_attention_hub.adapters.utils.key_generation import hash_kwargs
from sparse_attention_hub.adapters.utils.model_utils import (
    generate_model_key,
    generate_tokenizer_key,
)


@pytest.mark.unit
class TestHashKwargs:
    """Test the hash_kwargs utility function."""

    def test_empty_kwargs(self) -> None:
        """Test hashing empty kwargs."""
        result = hash_kwargs({})
        assert result == "empty"

    def test_simple_kwargs(self) -> None:
        """Test hashing simple kwargs."""
        kwargs = {"arg1": "value1", "arg2": 42}
        result = hash_kwargs(kwargs)
        
        assert isinstance(result, str)
        assert len(result) == 8  # MD5 hash truncated to 8 chars
        assert result != "empty"

    def test_kwargs_order_independence(self) -> None:
        """Test that kwargs order doesn't affect hash."""
        kwargs1 = {"arg1": "value1", "arg2": 42, "arg3": True}
        kwargs2 = {"arg3": True, "arg1": "value1", "arg2": 42}
        
        hash1 = hash_kwargs(kwargs1)
        hash2 = hash_kwargs(kwargs2)
        
        assert hash1 == hash2

    def test_different_kwargs_different_hash(self) -> None:
        """Test that different kwargs produce different hashes."""
        kwargs1 = {"arg1": "value1", "arg2": 42}
        kwargs2 = {"arg1": "value2", "arg2": 42}
        
        hash1 = hash_kwargs(kwargs1)
        hash2 = hash_kwargs(kwargs2)
        
        assert hash1 != hash2

    def test_nested_kwargs(self) -> None:
        """Test hashing kwargs with nested structures."""
        kwargs = {
            "model_config": {"hidden_size": 768, "num_layers": 12},
            "training_config": {"batch_size": 32, "lr": 0.001}
        }
        result = hash_kwargs(kwargs)
        
        assert isinstance(result, str)
        assert len(result) == 8
        assert result != "empty"

    def test_kwargs_with_none_values(self) -> None:
        """Test hashing kwargs with None values."""
        kwargs = {"arg1": None, "arg2": "value", "arg3": None}
        result = hash_kwargs(kwargs)
        
        assert isinstance(result, str)
        assert len(result) == 8
        assert result != "empty"

    def test_consistent_hashing(self) -> None:
        """Test that hashing is consistent across calls."""
        kwargs = {"model_name": "gpt2", "torch_dtype": "float16"}
        
        hash1 = hash_kwargs(kwargs)
        hash2 = hash_kwargs(kwargs)
        hash3 = hash_kwargs(kwargs)
        
        assert hash1 == hash2 == hash3

    def test_complex_types(self) -> None:
        """Test hashing with complex types (should fallback gracefully)."""
        # This tests the fallback mechanism for non-sortable items
        kwargs = {"arg1": "value", "arg2": {"nested": [1, 2, 3]}}
        result = hash_kwargs(kwargs)
        
        assert isinstance(result, str)
        assert len(result) == 8


@pytest.mark.unit
class TestGenerateModelKey:
    """Test the generate_model_key utility function."""

    def test_basic_model_key(self) -> None:
        """Test basic model key generation."""
        key = generate_model_key("gpt2", 0, {"torch_dtype": "float16"})
        
        assert isinstance(key, str)
        assert "gpt2" in key
        assert "0" in key
        assert "|" in key  # Separator

    def test_model_key_with_cpu(self) -> None:
        """Test model key generation with CPU (None gpu_id)."""
        key = generate_model_key("bert-base", None, {"torch_dtype": "float32"})
        
        assert isinstance(key, str)
        assert "bert-base" in key
        assert "cpu" in key
        assert "|" in key

    def test_model_key_with_empty_kwargs(self) -> None:
        """Test model key generation with empty kwargs."""
        key = generate_model_key("model-name", 1, {})
        
        assert isinstance(key, str)
        assert "model-name" in key
        assert "1" in key
        assert "empty" in key  # From hash_kwargs

    def test_model_key_consistency(self) -> None:
        """Test that model key generation is consistent."""
        kwargs = {"torch_dtype": "float16", "device_map": "auto"}
        
        key1 = generate_model_key("gpt2", 0, kwargs)
        key2 = generate_model_key("gpt2", 0, kwargs)
        
        assert key1 == key2

    def test_different_models_different_keys(self) -> None:
        """Test that different models produce different keys."""
        kwargs = {"torch_dtype": "float16"}
        
        key1 = generate_model_key("gpt2", 0, kwargs)
        key2 = generate_model_key("bert", 0, kwargs)
        
        assert key1 != key2

    def test_different_gpus_different_keys(self) -> None:
        """Test that different GPUs produce different keys."""
        kwargs = {"torch_dtype": "float16"}
        
        key1 = generate_model_key("gpt2", 0, kwargs)
        key2 = generate_model_key("gpt2", 1, kwargs)
        
        assert key1 != key2

    def test_different_kwargs_different_keys(self) -> None:
        """Test that different kwargs produce different keys."""
        key1 = generate_model_key("gpt2", 0, {"torch_dtype": "float16"})
        key2 = generate_model_key("gpt2", 0, {"torch_dtype": "float32"})
        
        assert key1 != key2

    def test_gpu_vs_cpu_different_keys(self) -> None:
        """Test that GPU vs CPU placement produces different keys."""
        kwargs = {"torch_dtype": "float16"}
        
        key_gpu = generate_model_key("gpt2", 0, kwargs)
        key_cpu = generate_model_key("gpt2", None, kwargs)
        
        assert key_gpu != key_cpu
        assert "0" in key_gpu
        assert "cpu" in key_cpu

    def test_model_key_format(self) -> None:
        """Test the format of generated model keys."""
        key = generate_model_key("test-model", 2, {"arg": "value"})
        parts = key.split("|")
        
        assert len(parts) == 3
        assert parts[0] == "test-model"
        assert parts[1] == "2"
        assert len(parts[2]) == 8  # Hash length


@pytest.mark.unit
class TestGenerateTokenizerKey:
    """Test the generate_tokenizer_key utility function."""

    def test_basic_tokenizer_key(self) -> None:
        """Test basic tokenizer key generation."""
        key = generate_tokenizer_key("gpt2", {"padding_side": "left"})
        
        assert isinstance(key, str)
        assert "gpt2" in key
        assert "|" in key  # Separator

    def test_tokenizer_key_with_empty_kwargs(self) -> None:
        """Test tokenizer key generation with empty kwargs."""
        key = generate_tokenizer_key("bert-base", {})
        
        assert isinstance(key, str)
        assert "bert-base" in key
        assert "empty" in key  # From hash_kwargs

    def test_tokenizer_key_consistency(self) -> None:
        """Test that tokenizer key generation is consistent."""
        kwargs = {"padding_side": "left", "truncation": True}
        
        key1 = generate_tokenizer_key("gpt2", kwargs)
        key2 = generate_tokenizer_key("gpt2", kwargs)
        
        assert key1 == key2

    def test_different_tokenizers_different_keys(self) -> None:
        """Test that different tokenizers produce different keys."""
        kwargs = {"padding_side": "left"}
        
        key1 = generate_tokenizer_key("gpt2", kwargs)
        key2 = generate_tokenizer_key("bert", kwargs)
        
        assert key1 != key2

    def test_different_kwargs_different_keys(self) -> None:
        """Test that different kwargs produce different keys."""
        key1 = generate_tokenizer_key("gpt2", {"padding_side": "left"})
        key2 = generate_tokenizer_key("gpt2", {"padding_side": "right"})
        
        assert key1 != key2

    def test_tokenizer_key_format(self) -> None:
        """Test the format of generated tokenizer keys."""
        key = generate_tokenizer_key("test-tokenizer", {"arg": "value"})
        parts = key.split("|")
        
        assert len(parts) == 2
        assert parts[0] == "test-tokenizer"
        assert len(parts[1]) == 8  # Hash length

    def test_kwargs_order_independence(self) -> None:
        """Test that kwargs order doesn't affect tokenizer key."""
        kwargs1 = {"padding_side": "left", "truncation": True, "max_length": 512}
        kwargs2 = {"max_length": 512, "padding_side": "left", "truncation": True}
        
        key1 = generate_tokenizer_key("gpt2", kwargs1)
        key2 = generate_tokenizer_key("gpt2", kwargs2)
        
        assert key1 == key2

    def test_complex_tokenizer_kwargs(self) -> None:
        """Test tokenizer key with complex kwargs."""
        kwargs = {
            "padding_side": "left",
            "special_tokens": {"pad_token": "<PAD>", "eos_token": "<EOS>"},
            "truncation_config": {"max_length": 512, "strategy": "longest_first"}
        }
        
        key = generate_tokenizer_key("complex-tokenizer", kwargs)
        
        assert isinstance(key, str)
        assert "complex-tokenizer" in key
        assert "|" in key


@pytest.mark.unit
class TestKeyUniqueness:
    """Test uniqueness properties of generated keys."""

    def test_model_key_uniqueness(self) -> None:
        """Test that different model configurations produce unique keys."""
        configs = [
            ("gpt2", 0, {"torch_dtype": "float16"}),
            ("gpt2", 1, {"torch_dtype": "float16"}),
            ("gpt2", 0, {"torch_dtype": "float32"}),
            ("bert", 0, {"torch_dtype": "float16"}),
            ("gpt2", None, {"torch_dtype": "float16"}),
        ]
        
        keys = [generate_model_key(*config) for config in configs]
        
        # All keys should be unique
        assert len(keys) == len(set(keys))

    def test_tokenizer_key_uniqueness(self) -> None:
        """Test that different tokenizer configurations produce unique keys."""
        configs = [
            ("gpt2", {"padding_side": "left"}),
            ("gpt2", {"padding_side": "right"}),
            ("gpt2", {}),
            ("bert", {"padding_side": "left"}),
            ("gpt2", {"padding_side": "left", "truncation": True}),
        ]
        
        keys = [generate_tokenizer_key(*config) for config in configs]
        
        # All keys should be unique
        assert len(keys) == len(set(keys))

    def test_mixed_key_collision_avoidance(self) -> None:
        """Test that model and tokenizer keys don't accidentally collide."""
        # Even if we use similar names, the structure should prevent collisions
        model_key = generate_model_key("test-model", 0, {})
        tokenizer_key = generate_tokenizer_key("test-model", {})
        
        # Keys should be different due to different structure (3 vs 2 parts)
        assert model_key != tokenizer_key
        assert len(model_key.split("|")) == 3
        assert len(tokenizer_key.split("|")) == 2
