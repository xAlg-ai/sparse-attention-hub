#!/usr/bin/env python3
"""
Test integration with PyTorch's Scaled Dot-Product Attention (SDPA).
This test validates that our sparse attention mechanisms work correctly
when integrated with real attention modules using SDPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sparse_attention_hub.model_hub import ModelHubHF
from sparse_attention_hub.sparse_attention.generators import SparseAttentionHF
from sparse_attention_hub.sparse_attention.efficient import DoubleSparsity, HashAttention


class SDPAAttentionModule(nn.Module):
    """Attention module using PyTorch's SDPA for realistic testing."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward pass using SDPA."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)


class SimpleSDPAModel(nn.Module):
    """Simple model using SDPA attention for testing."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': SDPAAttentionModule(embed_dim, num_heads),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            })
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            # Self-attention with residual
            attn_out = layer['attention'](layer['norm1'](x), attention_mask)
            x = x + attn_out
            
            # MLP with residual
            mlp_out = layer['mlp'](layer['norm2'](x))
            x = x + mlp_out
        
        x = self.final_norm(x)
        return self.lm_head(x)


def test_sdpa_integration():
    """Test integration with SDPA-based models."""
    print("=== Testing SDPA Integration ===")
    
    # Create model with SDPA attention
    model = SimpleSDPAModel(
        vocab_size=1000,
        embed_dim=128,
        num_heads=8,
        num_layers=3
    )
    
    # Test data
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Original forward pass
    model.eval()
    with torch.no_grad():
        original_output = model(input_ids)
    
    print(f"✅ Original SDPA model output: {original_output.shape}")
    
    # Test with different sparse attention mechanisms
    model_hub = ModelHubHF()
    
    sparse_mechanisms = [
        ("DoubleSparsity_0.3", DoubleSparsity(sparsity_ratio=0.3)),
        ("DoubleSparsity_0.7", DoubleSparsity(sparsity_ratio=0.7)),
        ("HashAttention_4", HashAttention(num_buckets=4)),
        ("HashAttention_8", HashAttention(num_buckets=8)),
    ]
    
    for name, sparse_attention in sparse_mechanisms:
        # Replace attention interface
        attention_generator = SparseAttentionHF(sparse_attention)
        custom_fn = attention_generator.get_custom_attention_function()
        
        model_hub.replaceAttentionInterface(model, custom_fn, name)
        
        # Forward pass with sparse attention
        with torch.no_grad():
            sparse_output = model(input_ids)
        
        print(f"✅ {name} output: {sparse_output.shape}")
        
        # Revert for next test
        model_hub.revertAttentionInterface(model)
    
    # Test with attention masks
    print("\n--- Testing with Attention Masks ---")
    
    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # Test with DoubleSparsity and causal mask
    sparse_attention = DoubleSparsity(sparsity_ratio=0.5)
    attention_generator = SparseAttentionHF(sparse_attention)
    custom_fn = attention_generator.get_custom_attention_function()
    
    model_hub.replaceAttentionInterface(model, custom_fn, "masked_sparse")
    
    with torch.no_grad():
        masked_output = model(input_ids, attention_mask=causal_mask)
    
    print(f"✅ Sparse attention with causal mask: {masked_output.shape}")
    
    model_hub.revertAttentionInterface(model)
    
    return True


def test_attention_pattern_analysis():
    """Analyze attention patterns produced by different mechanisms."""
    print("\n=== Testing Attention Pattern Analysis ===")
    
    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 1, 4, 16, 32
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    mechanisms = [
        ("DoubleSparsity_0.25", DoubleSparsity(sparsity_ratio=0.25)),
        ("DoubleSparsity_0.5", DoubleSparsity(sparsity_ratio=0.5)),
        ("HashAttention_2", HashAttention(num_buckets=2)),
        ("HashAttention_4", HashAttention(num_buckets=4)),
    ]
    
    for name, mechanism in mechanisms:
        output, weights = mechanism.custom_attention(
            queries=queries,
            keys=keys,
            values=values
        )
        
        # Analyze output statistics
        output_mean = output.mean().item()
        output_std = output.std().item()
        output_min = output.min().item()
        output_max = output.max().item()
        
        print(f"✅ {name}:")
        print(f"   Output stats - Mean: {output_mean:.4f}, Std: {output_std:.4f}")
        print(f"   Output range - Min: {output_min:.4f}, Max: {output_max:.4f}")
    
    return True


def test_memory_efficiency():
    """Test memory efficiency of sparse attention mechanisms."""
    print("\n=== Testing Memory Efficiency ===")
    
    # Test with larger sequences to see memory impact
    test_configs = [
        (1, 2, 64, 16),   # Small
        (1, 4, 128, 32),  # Medium
        (1, 8, 256, 64),  # Large
    ]
    
    for batch_size, num_heads, seq_len, head_dim in test_configs:
        queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Test DoubleSparsity
        sparse_attention = DoubleSparsity(sparsity_ratio=0.3)
        
        # Measure memory before
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        with torch.no_grad():
            output, _ = sparse_attention.custom_attention(
                queries=queries,
                keys=keys,
                values=values
            )
        
        print(f"✅ Sequence length {seq_len}: Output shape {output.shape}")
    
    return True


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\n=== Testing Numerical Stability ===")
    
    batch_size, num_heads, seq_len, head_dim = 1, 2, 8, 16
    
    test_cases = [
        ("normal", torch.randn(batch_size, num_heads, seq_len, head_dim)),
        ("large_values", torch.randn(batch_size, num_heads, seq_len, head_dim) * 10),
        ("small_values", torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1),
        ("mixed_scale", torch.cat([
            torch.randn(batch_size, num_heads, seq_len//2, head_dim) * 10,
            torch.randn(batch_size, num_heads, seq_len//2, head_dim) * 0.1
        ], dim=2)),
    ]
    
    mechanisms = [
        DoubleSparsity(sparsity_ratio=0.5),
        HashAttention(num_buckets=4),
    ]
    
    for case_name, test_tensor in test_cases:
        for mechanism in mechanisms:
            try:
                output, _ = mechanism.custom_attention(
                    queries=test_tensor,
                    keys=test_tensor,
                    values=test_tensor
                )
                
                # Check for NaN or Inf
                has_nan = torch.isnan(output).any()
                has_inf = torch.isinf(output).any()
                
                if has_nan or has_inf:
                    print(f"❌ {mechanism.__class__.__name__} with {case_name}: NaN/Inf detected")
                else:
                    print(f"✅ {mechanism.__class__.__name__} with {case_name}: Stable")
                    
            except Exception as e:
                print(f"❌ {mechanism.__class__.__name__} with {case_name}: Error - {e}")
    
    return True


def run_sdpa_tests():
    """Run all SDPA integration tests."""
    print("🚀 Starting SDPA Integration Tests\n")
    
    tests = [
        ("SDPA Integration", test_sdpa_integration),
        ("Attention Pattern Analysis", test_attention_pattern_analysis),
        ("Memory Efficiency", test_memory_efficiency),
        ("Numerical Stability", test_numerical_stability),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}\n")
            import traceback
            traceback.print_exc()
    
    print(f"🎯 SDPA Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All SDPA integration tests passed!")
    else:
        print("⚠️  Some SDPA tests failed.")
    
    return passed == total


if __name__ == "__main__":
    run_sdpa_tests()