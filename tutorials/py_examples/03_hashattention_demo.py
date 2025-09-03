#!/usr/bin/env python3
"""
HashAttention Example

Minimal example demonstrating HashAttention implementation using LocalMasker, 
SinkMasker, and HashAttentionTopKMasker with weight loading from USA checkpoint.
"""

import os
import torch
import time
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
    HashAttentionTopKMaskerConfig,
)
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.adapters import Request


def convert_usa_weights_to_hash_attention(
    usa_checkpoint_path: str,
    num_layers: int = 32,
    num_heads: int = 32,
    device: str = "cpu",
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """Convert USA module weights to HashAttentionTopKMasker format."""

    print(f"Loading USA weights from {usa_checkpoint_path}")
    usa_state_dict = torch.load(usa_checkpoint_path, map_location=device)

    hat_weights = {}

    for layer_idx in range(num_layers):
        layer_weights = {
            "query_matrix": [],
            "query_bias": [],
            "key_matrix": [],
            "key_bias": [],
        }

        # Collect weights for all heads in this layer
        query_matrices_per_layer = [[] for _ in range(3)]  # 3 linear layers
        query_biases_per_layer = [[] for _ in range(3)]
        key_matrices_per_layer = [[] for _ in range(3)]
        key_biases_per_layer = [[] for _ in range(3)]

        for head_idx in range(num_heads):
            query_prefix = f"{layer_idx}.learning_to_hash_transformation_q.{head_idx}"
            key_prefix = f"{layer_idx}.learning_to_hash_transformation_k.{head_idx}"

            # Extract weights from 3-layer MLP (linear layers at indices 0, 2, 4)
            for i, linear_idx in enumerate([0, 2, 4]):
                # Query weights
                weight_key = f"{query_prefix}.{linear_idx}.weight"
                bias_key = f"{query_prefix}.{linear_idx}.bias"

                if weight_key in usa_state_dict:
                    weight = usa_state_dict[
                        weight_key
                    ].t()  # Transpose to (in_features, out_features)
                    query_matrices_per_layer[i].append(weight)

                    if bias_key in usa_state_dict:
                        query_biases_per_layer[i].append(usa_state_dict[bias_key])
                    else:
                        query_biases_per_layer[i].append(
                            torch.zeros(usa_state_dict[weight_key].shape[0])
                        )

                # Key weights
                weight_key = f"{key_prefix}.{linear_idx}.weight"
                bias_key = f"{key_prefix}.{linear_idx}.bias"

                if weight_key in usa_state_dict:
                    weight = usa_state_dict[
                        weight_key
                    ].t()  # Transpose to (in_features, out_features)
                    key_matrices_per_layer[i].append(weight)

                    if bias_key in usa_state_dict:
                        key_biases_per_layer[i].append(usa_state_dict[bias_key])
                    else:
                        key_biases_per_layer[i].append(
                            torch.zeros(usa_state_dict[weight_key].shape[0])
                        )

        # Stack all heads for each layer
        for i in range(3):
            if query_matrices_per_layer[i]:
                layer_weights["query_matrix"].append(
                    torch.stack(query_matrices_per_layer[i])
                )
                layer_weights["query_bias"].append(
                    torch.stack(query_biases_per_layer[i])
                )
                layer_weights["key_matrix"].append(
                    torch.stack(key_matrices_per_layer[i])
                )
                layer_weights["key_bias"].append(torch.stack(key_biases_per_layer[i]))

        hat_weights[layer_idx] = layer_weights

    print(f"✅ Converted weights for {num_layers} layers, {num_heads} heads")
    return hat_weights


def create_dummy_weights(
    num_layers: int = 32, num_heads: int = 32
) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """Create dummy weights for demonstration purposes."""

    hat_weights = {}
    for layer_idx in range(num_layers):
        hat_weights[layer_idx] = {
            "query_matrix": [
                torch.randn(num_heads, 128, 128),  # First linear layer
                torch.randn(num_heads, 128, 128),  # Second linear layer
                torch.randn(num_heads, 128, 32),  # Third linear layer
            ],
            "query_bias": [
                torch.randn(num_heads, 128),
                torch.randn(num_heads, 128),
                torch.randn(num_heads, 32),
            ],
            "key_matrix": [
                torch.randn(num_heads, 128, 128),
                torch.randn(num_heads, 128, 128),
                torch.randn(num_heads, 128, 32),
            ],
            "key_bias": [
                torch.randn(num_heads, 128),
                torch.randn(num_heads, 128),
                torch.randn(num_heads, 32),
            ],
        }
    return hat_weights


def main():
    """Main function demonstrating HashAttention."""

    print("🚀 HashAttention Example")
    print("=" * 50)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load weights
    usa_checkpoint_path = (
        "/home/apd10/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.pt"
    )

    if os.path.exists(usa_checkpoint_path):
        try:
            hat_weights = convert_usa_weights_to_hash_attention(
                usa_checkpoint_path, device=device
            )
            print("✅ Successfully loaded USA weights")
        except Exception as e:
            print(f"❌ Error loading USA weights: {e}")
            print("Creating dummy weights...")
            hat_weights = create_dummy_weights()
    else:
        print(f"❌ USA checkpoint not found at {usa_checkpoint_path}")
        print("Creating dummy weights for demonstration...")
        hat_weights = create_dummy_weights()

    # Configure HashAttention
    local_config = LocalMaskerConfig(window_size=4)
    sink_config = SinkMaskerConfig(sink_size=4)
    hash_config = HashAttentionTopKMaskerConfig(
        heavy_size=12,
        hat_bits=32,
        hat_mlp_layers=3,
        hat_mlp_hidden_size=128,
        hat_mlp_activation="silu",
        hat_weights=hat_weights,
    )

    research_config = ResearchAttentionConfig(
        masker_configs=[local_config, sink_config, hash_config]
    )

    print("✅ HashAttention config: Local(4) + Sink(4) + Hash(12 bits, 32 heavy)")

    # Load model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=research_config,
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "cuda"},
        device="cuda",
    )
    # Prepare test input
    test_context = """ 
            The concept of attention mechanisms has revolutionized natural language processing and machine learning.
            StreamingLLM addresses efficiency challenges by implementing sparse attention patterns that combine:
            1. Sink tokens: The first few tokens contain crucial global information
            2. Local attention: Recent tokens are most relevant for next token prediction
            This approach maintains performance while reducing computational costs for long sequences.
            """
    test_questions = [
        "Summarize the above in a single title with less than 10 words. Given only the title.",
        "What are other attention mechanisms that are used in the field of LLMs?",
    ]

    request = Request(
        context=test_context,
        questions=test_questions,
    )

    print("Running Hash Attention on Question")
    response = adapter.process_request(
        request, generation_kwargs={"max_new_tokens": 50}, request_kwargs={}
    )
    response_text = response.responses
    print("Hash Attention Response: ", response_text)


if __name__ == "__main__":
    main()
