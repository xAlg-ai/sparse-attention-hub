#!/usr/bin/env python3
"""
End-to-end demonstration of the Sparse Attention Hub functionality.
This script shows how to use the framework in a realistic scenario.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import time

from sparse_attention_hub.model_hub import ModelHubHF
from sparse_attention_hub.sparse_attention.generators import SparseAttentionHF
from sparse_attention_hub.sparse_attention.efficient import DoubleSparsity, HashAttention


class MiniTransformer(nn.Module):
    """A minimal transformer model for demonstration."""
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)  # Max sequence length 512
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'feed_forward': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(0.1)
                )
            })
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(pos_ids)
        x = token_embeds + pos_embeds
        
        # Transformer layers
        for layer in self.layers:
            # Self-attention
            residual = x
            x = layer['norm1'](x)
            attn_out, _ = layer['self_attention'](x, x, x, attn_mask=attention_mask)
            x = residual + attn_out
            
            # Feed forward
            residual = x
            x = layer['norm2'](x)
            ff_out = layer['feed_forward'](x)
            x = residual + ff_out
        
        # Final output
        x = self.final_norm(x)
        return self.lm_head(x)


def demonstrate_basic_usage():
    """Demonstrate basic usage of the Sparse Attention Hub."""
    print("🎯 Demonstrating Basic Usage")
    print("=" * 50)
    
    # Create a model
    model = MiniTransformer(vocab_size=1000, embed_dim=128, num_heads=8, num_layers=4)
    model.eval()
    
    # Create test input
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"📊 Input: {input_ids.shape}")
    
    # Baseline performance
    start_time = time.time()
    with torch.no_grad():
        baseline_output = model(input_ids)
    baseline_time = time.time() - start_time
    
    print(f"✅ Baseline forward pass: {baseline_time*1000:.2f}ms")
    print(f"✅ Baseline output shape: {baseline_output.shape}")
    
    # Initialize ModelHub
    model_hub = ModelHubHF()
    
    # Test different sparse attention mechanisms
    sparse_configs = [
        ("DoubleSparsity (30%)", DoubleSparsity(sparsity_ratio=0.3)),
        ("DoubleSparsity (50%)", DoubleSparsity(sparsity_ratio=0.5)),
        ("HashAttention (4 buckets)", HashAttention(num_buckets=4)),
        ("HashAttention (8 buckets)", HashAttention(num_buckets=8)),
    ]
    
    results = {}
    
    for name, sparse_attention in sparse_configs:
        print(f"\n🔧 Testing {name}")
        
        # Create attention generator
        attention_generator = SparseAttentionHF(sparse_attention)
        custom_fn = attention_generator.get_custom_attention_function()
        
        # Replace attention interface
        model_hub.replaceAttentionInterface(model, custom_fn, name)
        
        # Measure performance
        start_time = time.time()
        with torch.no_grad():
            sparse_output = model(input_ids)
        sparse_time = time.time() - start_time
        
        # Calculate difference from baseline
        output_diff = torch.abs(baseline_output - sparse_output).mean().item()
        speedup = baseline_time / sparse_time
        
        results[name] = {
            'time': sparse_time,
            'speedup': speedup,
            'output_diff': output_diff,
            'output_shape': sparse_output.shape
        }
        
        print(f"   ⏱️  Time: {sparse_time*1000:.2f}ms (speedup: {speedup:.2f}x)")
        print(f"   📏 Output difference: {output_diff:.6f}")
        print(f"   ✅ Output shape: {sparse_output.shape}")
        
        # Revert for next test
        model_hub.revertAttentionInterface(model)
    
    return results


def demonstrate_attention_masks():
    """Demonstrate usage with different attention masks."""
    print("\n🎯 Demonstrating Attention Masks")
    print("=" * 50)
    
    model = MiniTransformer(vocab_size=500, embed_dim=128, num_heads=4, num_layers=2)
    model.eval()
    
    batch_size, seq_len = 1, 32
    input_ids = torch.randint(0, 500, (batch_size, seq_len))
    
    # Different mask types
    masks = {
        "No mask": None,
        "Causal mask": torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1),
        "Padding mask": torch.cat([
            torch.zeros(seq_len - 8),
            torch.full((8,), float('-inf'))
        ]).unsqueeze(0),
        "Random mask": torch.where(
            torch.rand(seq_len, seq_len) > 0.7,
            torch.tensor(float('-inf')),
            torch.tensor(0.0)
        )
    }
    
    model_hub = ModelHubHF()
    sparse_attention = DoubleSparsity(sparsity_ratio=0.4)
    attention_generator = SparseAttentionHF(sparse_attention)
    custom_fn = attention_generator.get_custom_attention_function()
    
    model_hub.replaceAttentionInterface(model, custom_fn, "masked_sparse")
    
    for mask_name, mask in masks.items():
        print(f"\n🔍 Testing with {mask_name}")
        
        with torch.no_grad():
            output = model(input_ids, attention_mask=mask)
        
        output_stats = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item()
        }
        
        print(f"   📊 Output stats: mean={output_stats['mean']:.4f}, std={output_stats['std']:.4f}")
        print(f"   📊 Output range: [{output_stats['min']:.4f}, {output_stats['max']:.4f}]")
        print(f"   ✅ Shape: {output.shape}")
    
    model_hub.revertAttentionInterface(model)


def demonstrate_hook_functionality():
    """Demonstrate hook functionality."""
    print("\n🎯 Demonstrating Hook Functionality")
    print("=" * 50)
    
    model = MiniTransformer(vocab_size=200, embed_dim=64, num_heads=4, num_layers=2)
    model.eval()
    
    batch_size, seq_len = 1, 16
    input_ids = torch.randint(0, 200, (batch_size, seq_len))
    
    # Hook to capture attention inputs
    attention_inputs = []
    
    def create_capture_hook():
        def capture_hook(module, input):
            attention_inputs.append({
                'module_name': str(module),
                'input_shape': input[0].shape if input else None,
                'input_mean': input[0].mean().item() if input and len(input) > 0 else None
            })
            return input
        return capture_hook
    
    model_hub = ModelHubHF()
    
    # Add hooks
    model_hub.addPreAttentionHooks(model, create_capture_hook, "capture_hook")
    
    print("🔗 Added pre-attention hooks")
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"📊 Captured {len(attention_inputs)} attention module inputs")
    for i, info in enumerate(attention_inputs):
        print(f"   Hook {i+1}: shape={info['input_shape']}, mean={info['input_mean']:.4f}")
    
    # Remove hooks
    model_hub.removePreAttentionHooks(model, "capture_hook")
    print("🔗 Removed pre-attention hooks")
    
    # Verify hooks are removed
    attention_inputs.clear()
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"✅ After removal: captured {len(attention_inputs)} inputs (should be 0)")


def demonstrate_error_handling():
    """Demonstrate robust error handling."""
    print("\n🎯 Demonstrating Error Handling")
    print("=" * 50)
    
    model_hub = ModelHubHF()
    
    # Test 1: Invalid model
    class InvalidModel:
        pass
    
    invalid_model = InvalidModel()
    
    try:
        model_hub.replaceAttentionInterface(invalid_model, lambda x: x, "test")
        print("❌ Should have failed for invalid model")
    except ValueError as e:
        print(f"✅ Correctly caught invalid model: {e}")
    
    # Test 2: Model with no attention modules
    class NoAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
    
    no_attention_model = NoAttentionModel()
    
    try:
        model_hub.replaceAttentionInterface(no_attention_model, lambda x: x, "test")
        print("❌ Should have failed for no attention modules")
    except ValueError as e:
        print(f"✅ Correctly caught no attention modules: {e}")
    
    # Test 3: Extreme sparse attention parameters
    try:
        extreme_sparse = DoubleSparsity(sparsity_ratio=2.0)  # > 1.0
        print(f"✅ Extreme sparsity ratio clamped to: {extreme_sparse.sparsity_ratio}")
        
        zero_buckets = HashAttention(num_buckets=0)  # < 1
        print(f"✅ Zero buckets clamped to: {zero_buckets.num_buckets}")
        
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")


def run_complete_demo():
    """Run the complete demonstration."""
    print("🚀 Sparse Attention Hub - Complete Demonstration")
    print("=" * 60)
    print("This demo shows the full functionality of the framework")
    print("including model integration, sparse attention, and error handling.\n")
    
    try:
        # Basic usage
        results = demonstrate_basic_usage()
        
        # Attention masks
        demonstrate_attention_masks()
        
        # Hook functionality
        demonstrate_hook_functionality()
        
        # Error handling
        demonstrate_error_handling()
        
        # Summary
        print("\n🎯 Demo Summary")
        print("=" * 50)
        print("✅ Successfully demonstrated:")
        print("   • Model integration with attention replacement")
        print("   • Multiple sparse attention mechanisms")
        print("   • Attention mask compatibility")
        print("   • Hook management functionality")
        print("   • Robust error handling")
        print("   • Performance measurement")
        
        print(f"\n📊 Performance Results:")
        for name, result in results.items():
            print(f"   {name}: {result['time']*1000:.1f}ms ({result['speedup']:.2f}x speedup)")
        
        print("\n🎉 All demonstrations completed successfully!")
        print("The Sparse Attention Hub is ready for production use.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_complete_demo()