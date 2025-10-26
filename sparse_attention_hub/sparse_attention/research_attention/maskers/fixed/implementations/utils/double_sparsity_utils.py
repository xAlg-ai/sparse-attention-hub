"""Double sparsity utility functions."""
import torch
import json
from typing import Dict, List


def load_sorted_channels_from_file(file_path: str) -> Dict[str, List[List[int]]]:
    """Load sorted channel data from JSON file.
    
    Args:
        file_path: Path to the JSON file containing sorted channel indices
        
    Returns:
        Dictionary mapping layer keys to sorted channel indices
        
    Raises:
        ValueError: If file format is invalid
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate that data is a dictionary
        if not isinstance(data, dict):
            raise ValueError("Channel file must contain a JSON object")
        
        # Validate structure - should have keys like "model.layers.X.self_attn.Y"
        for key, value in data.items():
            if not isinstance(value, list):
                raise ValueError(f"Channel data for {key} must be a list")
            for i, channel_list in enumerate(value):
                if not isinstance(channel_list, list):
                    raise ValueError(f"Channel data for {key}[{i}] must be a list")
                if not all(isinstance(x, int) for x in channel_list):
                    raise ValueError(f"Channel indices for {key}[{i}] must be integers")
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in channel file: {e}")
    except FileNotFoundError:
        raise ValueError(f"Channel file not found: {file_path}")


def extract_layer_channels(
    sorted_channels: Dict[str, List[List[int]]], 
    layer_idx: int, 
    projection_type: str,
    device: torch.device
) -> torch.Tensor:
    """Extract sorted channel indices for a specific layer and projection type.
    
    Args:
        sorted_channels: Dictionary containing sorted channel data
        layer_idx: Layer index
        projection_type: Type of projection ("k_proj", "q_proj", "qk_proj")
        
        Returns:
            Tensor of sorted channel indices
        
    Raises:
        ValueError: If layer or projection type not found
    """
    key = f"model.layers.{layer_idx}.self_attn.{projection_type}"
    
    if key not in sorted_channels:
        raise ValueError(f"No sorted channels found for {key}")
    
    # Return the first list of sorted channels (assuming single head for now)
    channel_data = sorted_channels[key]
    if not channel_data:
        raise ValueError(f"Empty channel data for {key}")
    
    return torch.tensor(channel_data, dtype=torch.long, device=device)


def validate_channel_data(data: Dict[str, List[List[int]]]) -> None:
    """Validate channel data structure.
    
    Args:
        data: Channel data dictionary
        
    Raises:
        ValueError: If data structure is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("Channel data must be a dictionary")
    
    for key, value in data.items():
        if not isinstance(value, list):
            raise ValueError(f"Channel data for {key} must be a list")
        for i, channel_list in enumerate(value):
            if not isinstance(channel_list, list):
                raise ValueError(f"Channel data for {key}[{i}] must be a list")
            if not all(isinstance(x, int) for x in channel_list):
                raise ValueError(f"Channel indices for {key}[{i}] must be integers")
