import json
import tempfile
from pathlib import Path

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils.double_sparsity_utils import (
    extract_layer_channels,
    load_sorted_channels_from_file,
    validate_channel_data,
)


@pytest.mark.unit
class TestDoubleSparsityUtils:
    """Test class for double sparsity utility functions."""

    def test_load_sorted_channels_from_file_valid_json(self):
        """Test loading valid JSON file with sorted channels."""
        # Create temporary JSON file
        test_data = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "model.layers.0.self_attn.q_proj": [[8, 9, 10, 11, 12, 13, 14, 15]],
            "model.layers.1.self_attn.k_proj": [[16, 17, 18, 19, 20, 21, 22, 23]],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            result = load_sorted_channels_from_file(temp_file)
            assert result == test_data
        finally:
            Path(temp_file).unlink()

    def test_load_sorted_channels_from_file_invalid_json(self):
        """Test loading invalid JSON file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_sorted_channels_from_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_load_sorted_channels_from_file_nonexistent(self):
        """Test loading nonexistent file raises ValueError."""
        with pytest.raises(ValueError, match="Channel file not found"):
            load_sorted_channels_from_file("nonexistent_file.json")

    def test_load_sorted_channels_from_file_invalid_structure(self):
        """Test loading file with invalid structure raises ValueError."""
        # Test with non-dict root
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)  # List instead of dict
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError, match="Channel file must contain a JSON object"
            ):
                load_sorted_channels_from_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_load_sorted_channels_from_file_invalid_channel_data(self):
        """Test loading file with invalid channel data structure."""
        # Test with non-list channel data
        test_data = {"model.layers.0.self_attn.k_proj": "not_a_list"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError,
                match="Channel data for model.layers.0.self_attn.k_proj must be a list",
            ):
                load_sorted_channels_from_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_load_sorted_channels_from_file_invalid_channel_indices(self):
        """Test loading file with invalid channel indices."""
        # Test with non-integer channel indices
        test_data = {"model.layers.0.self_attn.k_proj": [["not", "integers"]]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError,
                match="Channel indices for model.layers.0.self_attn.k_proj\\[0\\] must be integers",
            ):
                load_sorted_channels_from_file(temp_file)
        finally:
            Path(temp_file).unlink()

    def test_extract_layer_channels_valid(self):
        """Test extracting layer channels with valid data."""
        sorted_channels = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "model.layers.0.self_attn.q_proj": [[8, 9, 10, 11, 12, 13, 14, 15]],
            "model.layers.1.self_attn.k_proj": [[16, 17, 18, 19, 20, 21, 22, 23]],
        }

        result = extract_layer_channels(
            sorted_channels, 0, "k_proj", torch.device("cpu")
        )
        assert torch.equal(result, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]))

        result = extract_layer_channels(
            sorted_channels, 0, "q_proj", torch.device("cpu")
        )
        assert torch.equal(result, torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]]))

        result = extract_layer_channels(
            sorted_channels, 1, "k_proj", torch.device("cpu")
        )
        assert torch.equal(result, torch.tensor([[16, 17, 18, 19, 20, 21, 22, 23]]))

    def test_extract_layer_channels_nonexistent_layer(self):
        """Test extracting channels for nonexistent layer raises ValueError."""
        sorted_channels = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
        }

        with pytest.raises(
            ValueError,
            match="No sorted channels found for model.layers.1.self_attn.k_proj",
        ):
            extract_layer_channels(sorted_channels, 1, "k_proj", torch.device("cpu"))

    def test_extract_layer_channels_nonexistent_projection(self):
        """Test extracting channels for nonexistent projection type raises ValueError."""
        sorted_channels = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
        }

        with pytest.raises(
            ValueError,
            match="No sorted channels found for model.layers.0.self_attn.q_proj",
        ):
            extract_layer_channels(sorted_channels, 0, "q_proj", torch.device("cpu"))

    def test_extract_layer_channels_empty_data(self):
        """Test extracting channels from empty data raises ValueError."""
        sorted_channels = {
            "model.layers.0.self_attn.k_proj": [],
        }

        with pytest.raises(
            ValueError, match="Empty channel data for model.layers.0.self_attn.k_proj"
        ):
            extract_layer_channels(sorted_channels, 0, "k_proj", torch.device("cpu"))

    def test_extract_layer_channels_default_projection_type(self):
        """Test extracting channels with default projection type."""
        sorted_channels = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
        }

        result = extract_layer_channels(
            sorted_channels, 0, "k_proj", torch.device("cpu")
        )  # Default is "k_proj"
        assert torch.equal(result, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]))

    def test_validate_channel_data_valid(self):
        """Test validating valid channel data structure."""
        valid_data = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "model.layers.0.self_attn.q_proj": [[8, 9, 10, 11, 12, 13, 14, 15]],
        }

        # Should not raise any exception
        validate_channel_data(valid_data)

    def test_validate_channel_data_invalid_root(self):
        """Test validating invalid root structure raises ValueError."""
        with pytest.raises(ValueError, match="Channel data must be a dictionary"):
            validate_channel_data("not_a_dict")

    def test_validate_channel_data_invalid_channel_data(self):
        """Test validating invalid channel data structure."""
        invalid_data = {"model.layers.0.self_attn.k_proj": "not_a_list"}

        with pytest.raises(
            ValueError,
            match="Channel data for model.layers.0.self_attn.k_proj must be a list",
        ):
            validate_channel_data(invalid_data)

    def test_validate_channel_data_invalid_channel_list(self):
        """Test validating invalid channel list structure."""
        invalid_data = {"model.layers.0.self_attn.k_proj": [["not", "integers"]]}

        with pytest.raises(
            ValueError,
            match="Channel indices for model.layers.0.self_attn.k_proj\\[0\\] must be integers",
        ):
            validate_channel_data(invalid_data)

    def test_validate_channel_data_mixed_valid_invalid(self):
        """Test validating data with mixed valid and invalid entries."""
        mixed_data = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3]],  # Valid
            "model.layers.0.self_attn.q_proj": "invalid",  # Invalid
        }

        with pytest.raises(
            ValueError,
            match="Channel data for model.layers.0.self_attn.q_proj must be a list",
        ):
            validate_channel_data(mixed_data)

    def test_validate_channel_data_empty_dict(self):
        """Test validating empty dictionary."""
        # Empty dict should be valid
        validate_channel_data({})

    def test_validate_channel_data_nested_structure(self):
        """Test validating nested structure with multiple heads."""
        nested_data = {
            "model.layers.0.self_attn.k_proj": [
                [0, 1, 2, 3, 4, 5, 6, 7],  # Head 0
                [8, 9, 10, 11, 12, 13, 14, 15],  # Head 1
            ],
            "model.layers.0.self_attn.q_proj": [[16, 17, 18, 19, 20, 21, 22, 23]],
        }

        # Should not raise any exception
        validate_channel_data(nested_data)

    def test_integration_load_and_extract(self):
        """Test integration between load and extract functions."""
        # Create comprehensive test data
        test_data = {
            "model.layers.0.self_attn.k_proj": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "model.layers.0.self_attn.q_proj": [[8, 9, 10, 11, 12, 13, 14, 15]],
            "model.layers.1.self_attn.k_proj": [[16, 17, 18, 19, 20, 21, 22, 23]],
            "model.layers.1.self_attn.q_proj": [[24, 25, 26, 27, 28, 29, 30, 31]],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # Load the data
            loaded_data = load_sorted_channels_from_file(temp_file)
            assert loaded_data == test_data

            # Extract specific channels
            layer_0_k = extract_layer_channels(
                loaded_data, 0, "k_proj", torch.device("cpu")
            )
            assert torch.equal(layer_0_k, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]))

            layer_1_q = extract_layer_channels(
                loaded_data, 1, "q_proj", torch.device("cpu")
            )
            assert torch.equal(
                layer_1_q, torch.tensor([[24, 25, 26, 27, 28, 29, 30, 31]])
            )

            # Validate the loaded data
            validate_channel_data(loaded_data)

        finally:
            Path(temp_file).unlink()

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with single channel
        single_channel_data = {"model.layers.0.self_attn.k_proj": [[0]]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(single_channel_data, f)
            temp_file = f.name

        try:
            loaded_data = load_sorted_channels_from_file(temp_file)
            channels = extract_layer_channels(
                loaded_data, 0, "k_proj", torch.device("cpu")
            )
            assert torch.equal(channels, torch.tensor([[0]]))
            validate_channel_data(loaded_data)
        finally:
            Path(temp_file).unlink()

        # Test with large channel indices
        large_indices_data = {
            "model.layers.0.self_attn.k_proj": [[1000, 2000, 3000, 4000]]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(large_indices_data, f)
            temp_file = f.name

        try:
            loaded_data = load_sorted_channels_from_file(temp_file)
            channels = extract_layer_channels(
                loaded_data, 0, "k_proj", torch.device("cpu")
            )
            assert torch.equal(channels, torch.tensor([[1000, 2000, 3000, 4000]]))
            validate_channel_data(loaded_data)
        finally:
            Path(temp_file).unlink()
