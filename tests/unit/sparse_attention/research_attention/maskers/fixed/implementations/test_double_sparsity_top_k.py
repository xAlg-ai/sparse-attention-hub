import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.double_sparsity_top_k import (
    DoubleSparsityTopKMasker,
    DoubleSparsityTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


def _get_doublesparse_path() -> str:
    """Get the path to DoubleSparse repository, downloading it if necessary."""
    # Create a temporary directory for DoubleSparse
    temp_dir = tempfile.gettempdir()
    doublesparse_path = os.path.join(temp_dir, "DoubleSparse")

    # Check if DoubleSparse already exists
    if os.path.exists(doublesparse_path):
        return doublesparse_path

    # Download DoubleSparse from GitHub
    print(f"Downloading DoubleSparse from GitHub to {doublesparse_path}...")
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/andy-yang-1/DoubleSparse.git",
                doublesparse_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✅ Successfully downloaded DoubleSparse to {doublesparse_path}")
        return doublesparse_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download DoubleSparse: {e.stderr}") from e


# Import the original implementation from DoubleSparse
_doublesparse_path = _get_doublesparse_path()
_meta_llama_config = (
    _doublesparse_path + "/config/meta-llama/Llama-3.1-8B-Instruct.json"
)


@pytest.mark.unit
class TestDoubleSparsityTopKMaskerImplementation:
    """Test class for DoubleSparsityTopKMasker implementation."""

    def test_double_sparsity_top_k_masker_config_creation(self):
        """Test that double sparsity top k masker config can be created."""
        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=256,
            sorted_channel_file=_meta_llama_config,
        )
        assert config is not None
        assert config.heavy_size == 256
        assert config.group_factor == 16  # Default value
        assert config.label_bits == 4  # Default value

    def test_double_sparsity_top_k_masker_config_with_custom_values(self):
        """Test config creation with custom group_factor and label_bits."""
        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=128,
            sorted_channel_file=_meta_llama_config,
            group_factor=8,
            label_bits=8,
        )
        assert config.group_factor == 8
        assert config.label_bits == 8

    def test_double_sparsity_top_k_masker_config_validation(self):
        """Test config validation raises appropriate errors."""
        # Test invalid group_factor
        with pytest.raises(ValueError, match="group_factor must be > 0"):
            DoubleSparsityTopKMaskerConfig(
                heavy_size=256,
                sorted_channel_file=_meta_llama_config,
                group_factor=0,
            )

        # Test invalid label_bits
        with pytest.raises(ValueError, match="label_bits must be in range \\(0, 16\\]"):
            DoubleSparsityTopKMaskerConfig(
                heavy_size=256,
                sorted_channel_file=_meta_llama_config,
                label_bits=17,
            )

    def test_double_sparsity_top_k_masker_creation(self):
        """Test that double sparsity top k masker can be created."""
        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=256,
            sorted_channel_file=_meta_llama_config,
        )
        masker = DoubleSparsityTopKMasker.create_from_config(config)
        assert isinstance(masker, DoubleSparsityTopKMasker)
        assert masker.config == config

    def test_double_sparsity_top_k_masker_inheritance(self):
        """Test that double sparsity top k masker inherits from TopKMasker."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMasker,
        )

        assert issubclass(DoubleSparsityTopKMasker, TopKMasker)

    def test_double_sparsity_top_k_masker_config_inheritance(self):
        """Test that double sparsity top k masker config inherits from TopKMaskerConfig."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
            TopKMaskerConfig,
        )

        assert issubclass(DoubleSparsityTopKMaskerConfig, TopKMaskerConfig)

    def test_double_sparsity_top_k_masker_registration(self):
        """Test that the masker is properly registered."""
        from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
            MaskerRegistry,
        )

        # Check that the masker is registered (registry uses config classes as keys)
        assert DoubleSparsityTopKMaskerConfig in MaskerRegistry._registry
        assert (
            MaskerRegistry._registry[DoubleSparsityTopKMaskerConfig]
            == DoubleSparsityTopKMasker
        )

    def test_double_sparsity_top_k_masker_with_temp_file(self):
        """Test masker creation with temporary channel file."""
        # Create temporary JSON file with test data
        test_data = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "model.layers.0.self_attn.q_proj": [[8, 9, 10, 11, 12, 13, 14, 15]],
            "model.layers.0.self_attn.qk_proj": [[8, 9, 10, 11, 12, 13, 14, 15]],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(test_data, f)
            temp_file = f.name

        try:
            config = DoubleSparsityTopKMaskerConfig(
                heavy_size=256,
                sorted_channel_file=temp_file,
                channel_selection="k_proj",
            )
            masker = DoubleSparsityTopKMasker.create_from_config(config)
            assert isinstance(masker, DoubleSparsityTopKMasker)
            # Initialize sorted channels by calling _ensure_sorted_channel_is_loaded
            masker._ensure_sorted_channel_is_loaded(0, torch.device("cpu"))
            assert len(masker.sorted_channels) == 3
        finally:
            Path(temp_file).unlink()

    def test_double_sparsity_top_k_masker_pseudo_quantization(self):
        """Test pseudo-quantization functionality."""
        config = DoubleSparsityTopKMaskerConfig(
            heavy_size=256,
            sorted_channel_file=_meta_llama_config,
            label_bits=4,
        )
        masker = DoubleSparsityTopKMasker.create_from_config(config)

        # Create test tensor
        test_tensor = torch.randn(2, 8, 1024, 8)
        quantized_tensor = masker._pseudo_quantize(test_tensor, 4)

        # Shape should remain the same
        assert quantized_tensor.shape == test_tensor.shape

        # Values should be quantized (different from original)
        assert not torch.allclose(quantized_tensor, test_tensor)

    def test_double_sparsity_top_k_masker_full_pipeline(self):
        """Test the full mask generation pipeline."""
        # Create temporary JSON file with valid channel indices for head_dim=8
        test_data = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "model.layers.0.self_attn.q_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(test_data, f)
            temp_file = f.name

        try:
            config = DoubleSparsityTopKMaskerConfig(
                heavy_size=128,
                sorted_channel_file=temp_file,
                group_factor=4,
                label_bits=4,
            )
            masker = DoubleSparsityTopKMasker.create_from_config(config)

            # Create test tensors
            batch_size = 2
            num_heads = 8
            seq_len = 512
            head_dim = 8

            keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
            queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
            values = torch.randn(batch_size, num_heads, seq_len, head_dim)

            # Create empty previous mask
            previous_mask = Mask.create_empty_mask(
                shape=(batch_size, num_heads, seq_len, seq_len),
                dtype=torch.float32,
                device=keys.device,
            )

            # Create sparse meta data
            sparse_meta_data = {}

            # Test full pipeline
            new_mask = masker.add_mask(
                keys=keys,
                queries=queries,
                values=values,
                attention_mask=None,
                scaling=1.0,
                dropout=0.0,
                sparse_meta_data=sparse_meta_data,
                previous_mask=previous_mask,
                layer_idx=0,
            )

            # Check that mask was created
            assert new_mask is not None
            assert new_mask.shape == (batch_size, num_heads, seq_len, seq_len)

            # Note: Channel cache is not currently implemented in the add_mask method
            # The cache functionality exists but is not being used in the current implementation

        finally:
            Path(temp_file).unlink()
