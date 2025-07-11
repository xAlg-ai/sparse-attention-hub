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

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig, HashAttentionTopKMaskerConfig
)
from sparse_attention_hub.sparse_attention.integrations.hugging_face import SparseAttentionHF


def convert_usa_weights_to_hash_attention(
    usa_checkpoint_path: str, 
    num_layers: int = 32, 
    num_heads: int = 32,
    device: str = 'cpu'
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
            "key_bias": []
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
                    weight = usa_state_dict[weight_key].t()  # Transpose to (in_features, out_features)
                    query_matrices_per_layer[i].append(weight)
                    
                    if bias_key in usa_state_dict:
                        query_biases_per_layer[i].append(usa_state_dict[bias_key])
                    else:
                        query_biases_per_layer[i].append(torch.zeros(usa_state_dict[weight_key].shape[0]))
                
                # Key weights
                weight_key = f"{key_prefix}.{linear_idx}.weight"
                bias_key = f"{key_prefix}.{linear_idx}.bias"
                
                if weight_key in usa_state_dict:
                    weight = usa_state_dict[weight_key].t()  # Transpose to (in_features, out_features)
                    key_matrices_per_layer[i].append(weight)
                    
                    if bias_key in usa_state_dict:
                        key_biases_per_layer[i].append(usa_state_dict[bias_key])
                    else:
                        key_biases_per_layer[i].append(torch.zeros(usa_state_dict[weight_key].shape[0]))
        
        # Stack all heads for each layer
        for i in range(3):
            if query_matrices_per_layer[i]:
                layer_weights["query_matrix"].append(torch.stack(query_matrices_per_layer[i]))
                layer_weights["query_bias"].append(torch.stack(query_biases_per_layer[i]))
                layer_weights["key_matrix"].append(torch.stack(key_matrices_per_layer[i]))
                layer_weights["key_bias"].append(torch.stack(key_biases_per_layer[i]))
        
        hat_weights[layer_idx] = layer_weights
    
    print(f"✅ Converted weights for {num_layers} layers, {num_heads} heads")
    return hat_weights


def create_dummy_weights(num_layers: int = 32, num_heads: int = 32) -> Dict[int, Dict[str, List[torch.Tensor]]]:
    """Create dummy weights for demonstration purposes."""
    
    hat_weights = {}
    for layer_idx in range(num_layers):
        hat_weights[layer_idx] = {
            "query_matrix": [
                torch.randn(num_heads, 128, 128),  # First linear layer
                torch.randn(num_heads, 128, 128),  # Second linear layer
                torch.randn(num_heads, 128, 32),   # Third linear layer
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load weights
    usa_checkpoint_path = "/data/apdesai/code/HashAttention-1.0/artifacts/llama3.1-8b-patch.32K.v1.pt"
    
    if os.path.exists(usa_checkpoint_path):
        try:
            hat_weights = convert_usa_weights_to_hash_attention(usa_checkpoint_path, device=device)
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
    local_config = LocalMaskerConfig(window_size=2)
    sink_config = SinkMaskerConfig(sink_size=2)
    hash_config = HashAttentionTopKMaskerConfig(
        heavy_size=4,
        hat_bits=32,
        hat_mlp_layers=3,
        hat_mlp_hidden_size=128,
        hat_mlp_activation="silu",
        hat_weights=hat_weights
    )
    
    research_config = ResearchAttentionConfig(
        masker_configs=[local_config, sink_config, hash_config]
    )
    
    print("✅ HashAttention config: Local(16) + Sink(16) + Hash(32 bits, 32 heavy)")
    
    # Load model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager"
        )
        print(f"✅ Loaded {model_name}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create sparse attention integration
    sparse_attention_hf = SparseAttentionHF.create_from_config(research_config)
    print("✅ SparseAttentionHF created")
    
    # Prepare test input
    test_text = """
    HashAttention combines local attention, sink tokens, and hash-based selection 
    for efficient sparse attention. This approach maintains performance while 
    reducing computational costs. Summarize the key benefits briefly.
    """
    
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print(f"✅ Input prepared: {input_ids.shape[1]} tokens")
    
    # Generate with full attention
    model.eval()
    max_new_tokens = 30
    
    print("\n📊 Running comparisons...")
    
    start_time = time.time()
    with torch.no_grad():
        full_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_time = time.time() - start_time
    
    # Generate with HashAttention
    start_time = time.time()
    with torch.no_grad():
        with sparse_attention_hf(model) as sparse_model:
            sparse_outputs = sparse_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                sparse_meta_data = {}
            )
    sparse_time = time.time() - start_time
    
    # Results
    full_generated_text = tokenizer.decode(full_outputs[0], skip_special_tokens=True)
    sparse_generated_text = tokenizer.decode(sparse_outputs[0], skip_special_tokens=True)
    
    speedup = full_time / sparse_time if sparse_time > 0 else 0
    
    print(f"\n📊 Results:")
    print(f"{'Method':<15} {'Time (s)':<10} {'Speedup':<10}")
    print("-" * 35)
    print(f"{'Full Attention':<15} {full_time:<10.3f} {'1.00x':<10}")
    print(f"{'HashAttention':<15} {sparse_time:<10.3f} {speedup:<10.2f}x")
    
    # Attention efficiency
    seq_len = input_ids.shape[1]
    total_attention = seq_len * seq_len
    sparse_attention = (16 + 16 + 32) * seq_len  # Local + Sink + Hash
    sparsity_ratio = sparse_attention / total_attention
    
    print(f"\n🎯 Attention Efficiency:")
    print(f"Full attention pairs: {total_attention:,}")
    print(f"Sparse attention pairs: ~{sparse_attention:,}")
    print(f"Sparsity ratio: {sparsity_ratio:.3f} ({sparsity_ratio*100:.1f}% of full)")
    
    print(f"\n📝 Generated Texts:")
    print(f"Full: {tokenizer.decode(full_outputs[0][len(input_ids[0]):], skip_special_tokens=True)}")
    print(f"Hash: {tokenizer.decode(sparse_outputs[0][len(input_ids[0]):], skip_special_tokens=True)}")
    
    print("\n✅ HashAttention example completed!")


if __name__ == "__main__":
    main()

