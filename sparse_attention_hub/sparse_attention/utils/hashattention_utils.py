"""Utilities for converting and loading HashAttention weights.

This module provides functions to convert USA (Universal Sparse Attention) 
checkpoint weights to HashAttention format and load them for model use.

Note:
    Currently only supports 3-layered MLPs. Support for other configurations 
    needs to be implemented.
"""

import pickle
from typing import Dict, List

import torch


def convert_usa_weights_to_hash_attention(
    usa_checkpoint_path: str,
    num_layers: int = 32,
    num_heads: int = 32,
    num_mlp_layers: int = 3,
    device: str = "cpu",
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """Convert USA module weights to HashAttentionTopKMasker format.

    Args:
        usa_checkpoint_path: Path to the USA checkpoint file.
        num_layers: Number of transformer layers in the model. Defaults to 32.
        num_heads: Number of attention heads per layer. Defaults to 32.
        num_mlp_layers: Number of MLP layers in the hash transformation. Defaults to 3.
        device: Device to load the weights on ("cpu", "cuda", etc.). Defaults to "cpu".

    Returns:
        Dictionary mapping layer indices to their corresponding weights in HashAttention format.
        Each layer contains "query_matrix", "query_bias", "key_matrix", and "key_bias" lists.
    """
    print(f"Loading USA weights from {usa_checkpoint_path}")
    usa_state_dict = torch.load(usa_checkpoint_path, map_location=device)

    hat_weights = {}
    for layer_idx in range(num_layers):
        layer_weights = _convert_layer_weights(
            usa_state_dict, layer_idx, num_heads, num_mlp_layers
        )
        hat_weights[layer_idx] = layer_weights

    print(
        f"✅ Converted weights for {num_layers} layers, {num_heads} heads, {num_mlp_layers} MLP layers"
    )
    return hat_weights


def _convert_layer_weights(
    usa_state_dict: Dict[str, torch.Tensor],
    layer_idx: int,
    num_heads: int,
    num_mlp_layers: int,
) -> Dict[str, List[torch.Tensor]]:
    """Convert weights for a single layer.

    Args:
        usa_state_dict: USA checkpoint state dictionary.
        layer_idx: Index of the layer to process.
        num_heads: Number of attention heads.
        num_mlp_layers: Number of MLP layers.

    Returns:
        Dictionary containing converted weights for the layer.
    """
    layer_weights = {
        "query_matrix": [],
        "query_bias": [],
        "key_matrix": [],
        "key_bias": [],
    }

    # Collect weights for all heads in this layer
    query_matrices_per_layer = [[] for _ in range(num_mlp_layers)]
    query_biases_per_layer = [[] for _ in range(num_mlp_layers)]
    key_matrices_per_layer = [[] for _ in range(num_mlp_layers)]
    key_biases_per_layer = [[] for _ in range(num_mlp_layers)]

    for head_idx in range(num_heads):
        _extract_head_weights(
            usa_state_dict=usa_state_dict,
            layer_idx=layer_idx,
            head_idx=head_idx,
            num_mlp_layers=num_mlp_layers,
            query_matrices_per_layer=query_matrices_per_layer,
            query_biases_per_layer=query_biases_per_layer,
            key_matrices_per_layer=key_matrices_per_layer,
            key_biases_per_layer=key_biases_per_layer,
        )

    # Stack all heads for each MLP layer
    _stack_head_weights(
        layer_weights=layer_weights,
        num_mlp_layers=num_mlp_layers,
        query_matrices_per_layer=query_matrices_per_layer,
        query_biases_per_layer=query_biases_per_layer,
        key_matrices_per_layer=key_matrices_per_layer,
        key_biases_per_layer=key_biases_per_layer,
    )

    return layer_weights


def _extract_head_weights(
    usa_state_dict: Dict[str, torch.Tensor],
    layer_idx: int,
    head_idx: int,
    num_mlp_layers: int,
    query_matrices_per_layer: List[List[torch.Tensor]],
    query_biases_per_layer: List[List[torch.Tensor]],
    key_matrices_per_layer: List[List[torch.Tensor]],
    key_biases_per_layer: List[List[torch.Tensor]],
) -> None:
    """Extract weights for a single attention head.

    Args:
        usa_state_dict: USA checkpoint state dictionary.
        layer_idx: Index of the current layer.
        head_idx: Index of the current attention head.
        num_mlp_layers: Number of MLP layers.
        query_matrices_per_layer: Storage for query weight matrices.
        query_biases_per_layer: Storage for query bias vectors.
        key_matrices_per_layer: Storage for key weight matrices.
        key_biases_per_layer: Storage for key bias vectors.
    """
    query_prefix = f"{layer_idx}.learning_to_hash_transformation_q.{head_idx}"
    key_prefix = f"{layer_idx}.learning_to_hash_transformation_k.{head_idx}"

    # Extract weights from MLP layers (linear layers at indices 0, 2, 4, ...)
    linear_indices = [i * 2 for i in range(num_mlp_layers)]

    for i, linear_idx in enumerate(linear_indices):
        _extract_query_weights(
            usa_state_dict,
            query_prefix,
            linear_idx,
            i,
            query_matrices_per_layer,
            query_biases_per_layer,
        )
        _extract_key_weights(
            usa_state_dict,
            key_prefix,
            linear_idx,
            i,
            key_matrices_per_layer,
            key_biases_per_layer,
        )


def _extract_query_weights(
    usa_state_dict: Dict[str, torch.Tensor],
    query_prefix: str,
    linear_idx: int,
    mlp_layer_idx: int,
    query_matrices_per_layer: List[List[torch.Tensor]],
    query_biases_per_layer: List[List[torch.Tensor]],
) -> None:
    """Extract query weights for a specific MLP layer.

    Args:
        usa_state_dict: USA checkpoint state dictionary.
        query_prefix: Prefix for query weight keys.
        linear_idx: Index of the linear layer.
        mlp_layer_idx: Index of the MLP layer.
        query_matrices_per_layer: Storage for query weight matrices.
        query_biases_per_layer: Storage for query bias vectors.
    """
    weight_key = f"{query_prefix}.{linear_idx}.weight"
    bias_key = f"{query_prefix}.{linear_idx}.bias"

    if weight_key in usa_state_dict:
        # Transpose to (in_features, out_features)
        weight = usa_state_dict[weight_key].t()
        query_matrices_per_layer[mlp_layer_idx].append(weight)

        if bias_key in usa_state_dict:
            query_biases_per_layer[mlp_layer_idx].append(usa_state_dict[bias_key])
        else:
            # Create zero bias if not present
            zero_bias = torch.zeros(usa_state_dict[weight_key].shape[0])
            query_biases_per_layer[mlp_layer_idx].append(zero_bias)


def _extract_key_weights(
    usa_state_dict: Dict[str, torch.Tensor],
    key_prefix: str,
    linear_idx: int,
    mlp_layer_idx: int,
    key_matrices_per_layer: List[List[torch.Tensor]],
    key_biases_per_layer: List[List[torch.Tensor]],
) -> None:
    """Extract key weights for a specific MLP layer.

    Args:
        usa_state_dict: USA checkpoint state dictionary.
        key_prefix: Prefix for key weight keys.
        linear_idx: Index of the linear layer.
        mlp_layer_idx: Index of the MLP layer.
        key_matrices_per_layer: Storage for key weight matrices.
        key_biases_per_layer: Storage for key bias vectors.
    """
    weight_key = f"{key_prefix}.{linear_idx}.weight"
    bias_key = f"{key_prefix}.{linear_idx}.bias"

    if weight_key in usa_state_dict:
        # Transpose to (in_features, out_features)
        weight = usa_state_dict[weight_key].t()
        key_matrices_per_layer[mlp_layer_idx].append(weight)

        if bias_key in usa_state_dict:
            key_biases_per_layer[mlp_layer_idx].append(usa_state_dict[bias_key])
        else:
            # Create zero bias if not present
            zero_bias = torch.zeros(usa_state_dict[weight_key].shape[0])
            key_biases_per_layer[mlp_layer_idx].append(zero_bias)


def _stack_head_weights(
    layer_weights: Dict[str, List[torch.Tensor]],
    num_mlp_layers: int,
    query_matrices_per_layer: List[List[torch.Tensor]],
    query_biases_per_layer: List[List[torch.Tensor]],
    key_matrices_per_layer: List[List[torch.Tensor]],
    key_biases_per_layer: List[List[torch.Tensor]],
) -> None:
    """Stack weights from all heads for each MLP layer.

    Args:
        layer_weights: Dictionary to store the stacked weights.
        num_mlp_layers: Number of MLP layers.
        query_matrices_per_layer: Query weight matrices for all heads.
        query_biases_per_layer: Query bias vectors for all heads.
        key_matrices_per_layer: Key weight matrices for all heads.
        key_biases_per_layer: Key bias vectors for all heads.
    """
    for i in range(num_mlp_layers):
        if query_matrices_per_layer[i]:
            layer_weights["query_matrix"].append(
                torch.stack(query_matrices_per_layer[i])
            )
            layer_weights["query_bias"].append(torch.stack(query_biases_per_layer[i]))
            layer_weights["key_matrix"].append(torch.stack(key_matrices_per_layer[i]))
            layer_weights["key_bias"].append(torch.stack(key_biases_per_layer[i]))


def create_hat_weights_file_from_usa(
    usa_checkpoint_path: str,
    target_hat_path: str,
    num_layers: int = 32,
    num_heads: int = 32,
    num_mlp_layers: int = 3,
    device: str = "cpu",
) -> None:
    """Create HashAttention weights file from USA checkpoint.

    This function converts USA checkpoint weights to HashAttention format
    and saves them as a pickle file for later use.

    Args:
        usa_checkpoint_path: Path to the input USA checkpoint file.
        target_hat_path: Path where the HashAttention weights file will be saved.
        num_layers: Number of transformer layers in the model. Defaults to 32.
        num_heads: Number of attention heads per layer. Defaults to 32.
        num_mlp_layers: Number of MLP layers in the hash transformation. Defaults to 3.
        device: Device to load the weights on ("cpu", "cuda", etc.). Defaults to "cpu".
    """
    print("Creating HAT weights file from USA checkpoint...")

    hat_weights = convert_usa_weights_to_hash_attention(
        usa_checkpoint_path=usa_checkpoint_path,
        num_layers=num_layers,
        num_heads=num_heads,
        num_mlp_layers=num_mlp_layers,
        device=device,
    )

    _save_weights_to_file(hat_weights, target_hat_path)
    print(f"✅ HAT weights saved to {target_hat_path}")


def _save_weights_to_file(
    hat_weights: Dict[int, Dict[str, List[torch.Tensor]]], target_path: str
) -> None:
    """Save HashAttention weights to a pickle file.

    Args:
        hat_weights: Dictionary of HashAttention weights to save.
        target_path: Path where the weights file will be saved.
    """
    with open(target_path, "wb") as f:
        pickle.dump(hat_weights, f)


def load_hat_weights(
    hat_weights_path: str, device: str = "cpu"
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """Load HashAttention weights from a pickle file.

    This function loads previously saved HashAttention weights and moves
    them to the specified device.

    Args:
        hat_weights_path: Path to the HashAttention weights pickle file.
        device: Device to move the loaded weights to ("cpu", "cuda", etc.). Defaults to "cpu".

    Returns:
        Dictionary mapping layer indices to their corresponding weights in HashAttention format.
        Each layer contains "query_matrix", "query_bias", "key_matrix", and "key_bias" lists.
    """
    print(f"Loading HAT weights from {hat_weights_path}")

    hat_weights = _load_weights_from_file(hat_weights_path)
    _move_weights_to_device(hat_weights, device)

    print(f"✅ Loaded HAT weights for {len(hat_weights)} layers")
    return hat_weights


def _load_weights_from_file(file_path: str) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """Load weights from a pickle file.

    Args:
        file_path: Path to the pickle file.

    Returns:
        Dictionary of loaded weights.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _move_weights_to_device(
    hat_weights: Dict[int, Dict[str, List[torch.Tensor]]], device: str
) -> None:
    """Move all weights to the specified device.

    Args:
        hat_weights: Dictionary of HashAttention weights to move.
        device: Target device for the weights.
    """
    for layer_idx, layer_weights in hat_weights.items():
        for key, tensor_list in layer_weights.items():
            hat_weights[layer_idx][key] = [tensor.to(device) for tensor in tensor_list]
