import torch
import torch.nn as nn


def extract_weights_from_usa(
    usa_module, layer_idx=0
):
    """
    Extract weights from USA module and format them for
    HashAttentionTopKMasker.

    Args:
        usa_module: USA module instance
        layer_idx: Layer index for the weights dictionary

    Returns:
        Dictionary formatted for HashAttentionTopKMaskerConfig.hat_weights
    """
    num_heads = usa_module.num_heads

    # Initialize weight structure
    hat_weights = {
        layer_idx: {
            "key_matrix": [],
            "key_bias": [],
            "query_matrix": [],
            "query_bias": [],
        }
    }

    # Extract weights for each MLP layer
    # We need to stack weights across heads to get (H, d_in, d_out)
    # tensors

    # Get the number of linear layers (excluding activations)
    k_linear_layers = [
        layer
        for layer in usa_module.learning_to_hash_transformation_k[0]
        if isinstance(layer, nn.Linear)
    ]
    q_linear_layers = [
        layer
        for layer in usa_module.learning_to_hash_transformation_q[0]
        if isinstance(layer, nn.Linear)
    ]

    print(
        f"Found {len(k_linear_layers)} linear layers in key "
        f"transformation"
    )
    print(
        f"Found {len(q_linear_layers)} linear layers in query "
        f"transformation"
    )

    # Extract key matrices and biases
    for layer_idx_mlp in range(len(k_linear_layers)):
        # Stack weights and biases across heads
        key_weights = []
        key_biases = []

        for head_idx in range(num_heads):
            # Get the actual linear layer from the sequential module
            actual_layer = None
            layer_count = 0
            for module in usa_module.learning_to_hash_transformation_k[
                head_idx
            ]:
                if isinstance(module, nn.Linear):
                    if layer_count == layer_idx_mlp:
                        actual_layer = module
                        break
                    layer_count += 1

            if actual_layer is not None:
                key_weights.append(
                    actual_layer.weight.detach().clone().T
                )  # Transpose for correct shape
                key_biases.append(actual_layer.bias.detach().clone())

        # Stack to get (H, d_in, d_out) and (H, d_out) shapes
        key_matrix = torch.stack(key_weights, dim=0)
        key_bias = torch.stack(key_biases, dim=0)

        hat_weights[layer_idx]["key_matrix"].append(key_matrix)
        hat_weights[layer_idx]["key_bias"].append(key_bias)

        print(
            f"Key layer {layer_idx_mlp}: weight shape {key_matrix.shape}, "
            f"bias shape {key_bias.shape}"
        )

    # Extract query matrices and biases
    for layer_idx_mlp in range(len(q_linear_layers)):
        # Stack weights and biases across heads
        query_weights = []
        query_biases = []

        for head_idx in range(num_heads):
            # Get the actual linear layer from the sequential module
            actual_layer = None
            layer_count = 0
            for module in usa_module.learning_to_hash_transformation_q[
                head_idx
            ]:
                if isinstance(module, nn.Linear):
                    if layer_count == layer_idx_mlp:
                        actual_layer = module
                        break
                    layer_count += 1

            if actual_layer is not None:
                query_weights.append(
                    actual_layer.weight.detach().clone().T
                )  # Transpose for correct shape
                query_biases.append(actual_layer.bias.detach().clone())

        # Stack to get (H, d_in, d_out) and (H, d_out) shapes
        query_matrix = torch.stack(query_weights, dim=0)
        query_bias = torch.stack(query_biases, dim=0)

        hat_weights[layer_idx]["query_matrix"].append(query_matrix)
        hat_weights[layer_idx]["query_bias"].append(query_bias)

        print(
            f"Query layer {layer_idx_mlp}: weight shape {query_matrix.shape}, "
            f"bias shape {query_bias.shape}"
        )

    return hat_weights
