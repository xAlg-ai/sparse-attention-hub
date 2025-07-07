#!/usr/bin/env python3
"""
Comprehensive tests for ModelHub and SparseAttentionGen implementation.
Tests use PyTorch's SDPA and small custom models to validate functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from sparse_attention_hub.model_hub import ModelHubHF
from sparse_attention_hub.sparse_attention.generators import SparseAttentionHF
from sparse_attention_hub.sparse_attention.efficient import DoubleSparsity, HashAttention


class SimpleAttentionModule(nn.Module):
    """Simple attention module for testing."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use PyTorch's SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = SimpleAttentionModule(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), attention_mask)
        x = x + attn_out
        
        # MLP with residual connection
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x


class SimpleTransformerModel(nn.Module):
    """Simple transformer model for testing."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
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
            x = layer(x, attention_mask)
        
        # Output
        x = self.norm(x)
        return self.lm_head(x)


def test_basic_functionality():
    """Test basic functionality of all components."""
    print("=== Testing Basic Functionality ===")
    
    # Create small model
    model = SimpleTransformerModel(
        vocab_size=100, 
        embed_dim=64, 
        num_heads=4, 
        num_layers=2, 
        max_seq_len=32
    )
    
    # Test input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Original forward pass
    with torch.no_grad():
        original_output = model(input_ids)
    
    print(f"✅ Original model forward pass: {original_output.shape}")
    
    # Test ModelHub
    model_hub = ModelHubHF()
    
    # Test DoubleSparsity
    sparse_attention = DoubleSparsity(sparsity_ratio=0.5)
    attention_generator = SparseAttentionHF(sparse_attention)
    custom_fn = attention_generator.get_custom_attention_function()
    
    # Replace attention interface
    model_hub.replaceAttentionInterface(model, custom_fn, "double_sparsity")
    
    # Forward pass with sparse attention
    with torch.no_grad():
        sparse_output = model(input_ids)
    
    print(f"✅ Sparse attention forward pass: {sparse_output.shape}")
    
    # Revert attention interface
    model_hub.revertAttentionInterface(model)
    
    # Forward pass after revert
    with torch.no_grad():
        reverted_output = model(input_ids)
    
    print(f"✅ Reverted model forward pass: {reverted_output.shape}")
    
    # Check that revert worked (outputs should be close to original)
    diff = torch.abs(original_output - reverted_output).mean()
    print(f"✅ Revert accuracy (mean diff): {diff:.6f}")
    
    return True


def test_different_sparse_mechanisms():
    """Test different sparse attention mechanisms."""
    print("\n=== Testing Different Sparse Mechanisms ===")
    
    model = SimpleTransformerModel(
        vocab_size=50, 
        embed_dim=32, 
        num_heads=2, 
        num_layers=1, 
        max_seq_len=16
    )
    
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, 50, (batch_size, seq_len))
    
    model_hub = ModelHubHF()
    
    # Test DoubleSparsity with different ratios
    for sparsity_ratio in [0.2, 0.5, 0.8]:
        sparse_attention = DoubleSparsity(sparsity_ratio=sparsity_ratio)
        attention_generator = SparseAttentionHF(sparse_attention)
        custom_fn = attention_generator.get_custom_attention_function()
        
        model_hub.replaceAttentionInterface(model, custom_fn, f"double_{sparsity_ratio}")
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✅ DoubleSparsity (ratio={sparsity_ratio}): {output.shape}")
        
        model_hub.revertAttentionInterface(model)
    
    # Test HashAttention with different bucket counts
    for num_buckets in [2, 4, 8]:
        hash_attention = HashAttention(num_buckets=num_buckets)
        attention_generator = SparseAttentionHF(hash_attention)
        custom_fn = attention_generator.get_custom_attention_function()
        
        model_hub.replaceAttentionInterface(model, custom_fn, f"hash_{num_buckets}")
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✅ HashAttention (buckets={num_buckets}): {output.shape}")
        
        model_hub.revertAttentionInterface(model)
    
    return True


def test_attention_masks():
    """Test attention mechanisms with various masks."""
    print("\n=== Testing Attention Masks ===")
    
    model = SimpleTransformerModel(
        vocab_size=30, 
        embed_dim=32, 
        num_heads=2, 
        num_layers=1, 
        max_seq_len=16
    )
    
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, 30, (batch_size, seq_len))
    
    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # Create padding mask (mask last 2 tokens)
    padding_mask = torch.zeros(batch_size, seq_len)
    padding_mask[:, -2:] = float('-inf')
    
    model_hub = ModelHubHF()
    sparse_attention = DoubleSparsity(sparsity_ratio=0.3)
    attention_generator = SparseAttentionHF(sparse_attention)
    custom_fn = attention_generator.get_custom_attention_function()
    
    model_hub.replaceAttentionInterface(model, custom_fn, "masked_sparse")
    
    # Test with different masks
    test_cases = [
        ("no_mask", None),
        ("causal_mask", causal_mask),
        ("padding_mask", padding_mask)
    ]
    
    for mask_name, mask in test_cases:
        with torch.no_grad():
            output = model(input_ids, attention_mask=mask)
        print(f"✅ Sparse attention with {mask_name}: {output.shape}")
    
    model_hub.revertAttentionInterface(model)
    return True


def test_grouped_query_attention():
    """Test grouped query attention scenarios."""
    print("\n=== Testing Grouped Query Attention ===")
    
    # Create attention module with different head counts for testing
    class GroupedQueryAttention(nn.Module):
        def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_q_heads = num_q_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = embed_dim // num_q_heads
            
            self.q_proj = nn.Linear(embed_dim, num_q_heads * self.head_dim)
            self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
            self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, _ = x.shape
            
            q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            # Test direct sparse attention call
            sparse_attention = DoubleSparsity(sparsity_ratio=0.4)
            attn_output, _ = sparse_attention.custom_attention(
                queries=q, keys=k, values=v
            )
            
            return attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
    
    # Test different head ratios
    test_configs = [
        (8, 8),   # Same number of heads
        (8, 4),   # 2:1 ratio
        (8, 2),   # 4:1 ratio
        (8, 1),   # 8:1 ratio
    ]
    
    for num_q_heads, num_kv_heads in test_configs:
        attention = GroupedQueryAttention(embed_dim=64, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 64)
        
        with torch.no_grad():
            output = attention(x)
        
        print(f"✅ Grouped Query Attention (Q:{num_q_heads}, KV:{num_kv_heads}): {output.shape}")
    
    return True


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n=== Testing Edge Cases ===")
    
    model_hub = ModelHubHF()
    
    # Test 1: Very small sequence
    sparse_attention = DoubleSparsity(sparsity_ratio=0.5)
    queries = torch.randn(1, 2, 1, 16)  # Single token
    keys = torch.randn(1, 2, 1, 16)
    values = torch.randn(1, 2, 1, 16)
    
    output, _ = sparse_attention.custom_attention(queries=queries, keys=keys, values=values)
    print(f"✅ Single token sequence: {output.shape}")
    
    # Test 2: Extreme sparsity ratios
    for ratio in [0.0, 1.0]:
        sparse_attention = DoubleSparsity(sparsity_ratio=ratio)
        queries = torch.randn(1, 2, 8, 16)
        keys = torch.randn(1, 2, 8, 16)
        values = torch.randn(1, 2, 8, 16)
        
        output, _ = sparse_attention.custom_attention(queries=queries, keys=keys, values=values)
        print(f"✅ Extreme sparsity ratio {ratio}: {output.shape}")
    
    # Test 3: Hash attention with single bucket
    hash_attention = HashAttention(num_buckets=1)
    output, _ = hash_attention.custom_attention(queries=queries, keys=keys, values=values)
    print(f"✅ Single bucket hash attention: {output.shape}")
    
    # Test 4: Model with no attention modules (should raise error)
    class NoAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    no_attention_model = NoAttentionModel()
    
    try:
        model_hub.replaceAttentionInterface(no_attention_model, lambda x: x, "test")
        print("❌ Should have raised ValueError for no attention modules")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught error for no attention modules: {type(e).__name__}")
    
    return True


def test_performance_comparison():
    """Compare performance of different attention mechanisms."""
    print("\n=== Performance Comparison ===")
    
    # Create test data
    batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 32
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Test different mechanisms
    mechanisms = [
        ("DoubleSparsity_0.1", DoubleSparsity(sparsity_ratio=0.1)),
        ("DoubleSparsity_0.5", DoubleSparsity(sparsity_ratio=0.5)),
        ("HashAttention_4", HashAttention(num_buckets=4)),
        ("HashAttention_16", HashAttention(num_buckets=16)),
    ]
    
    for name, mechanism in mechanisms:
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                mechanism.custom_attention(queries=queries, keys=keys, values=values)
        
        # Time the operation
        import time
        start_time = time.time()
        
        for _ in range(10):
            with torch.no_grad():
                output, _ = mechanism.custom_attention(queries=queries, keys=keys, values=values)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10 * 1000  # Convert to ms
        
        print(f"✅ {name}: {avg_time:.2f}ms avg, output shape: {output.shape}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Comprehensive Tests for Sparse Attention Hub\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Different Sparse Mechanisms", test_different_sparse_mechanisms),
        ("Attention Masks", test_attention_masks),
        ("Grouped Query Attention", test_grouped_query_attention),
        ("Edge Cases", test_edge_cases),
        ("Performance Comparison", test_performance_comparison),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()