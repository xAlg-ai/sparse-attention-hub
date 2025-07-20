#!/usr/bin/env python3
"""
Local + Sink + Oracle-TopK + Adaptive Sampling Demo

A comprehensive example demonstrating the combination of multiple maskers:
- LocalMasker (window_size=4): Local attention within a 4-token window
- SinkMasker (sink_size=4): Sink attention for the first 4 tokens
- OracleTopKMasker (heavy_size=4): Oracle-based top-K attention with 4 tokens
- AdaptiveSamplingMasker (base_rate=0.1): Adaptive sampling with 10% base rate

This example shows how different attention patterns can be combined to create
sophisticated sparse attention mechanisms that balance efficiency and performance.

Usage:
    python 05_local_sink_oracle_adaptive_demo.py
"""

import os
import time
from pathlib import Path

import torch

# Ensure we're in the correct directory and add to Python path
import sys
os.chdir('/data/apdesai/code/sparse-attention-hub')
sys.path.insert(0, '/data/apdesai/code/sparse-attention-hub')

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig, OracleTopKConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig
)
from sparse_attention_hub.adapters.huggingface import ModelAdapterHF
from sparse_attention_hub.adapters import Request


def main():
    """Run a demo with combined maskers: Local + Sink + Oracle-TopK + Adaptive Sampling."""
    
    print("🎯 Local + Sink + Oracle-TopK + Adaptive Sampling Demo")
    print("=" * 60)
    
    # Configuration
    model_name = "microsoft/Phi-4-mini-instruct"  # Small model for quick testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    # Create combined masker configuration
    print("\n🔧 Creating combined masker configuration...")
    
    # 1. Local masker: 4-token window for local attention
    local_config = LocalMaskerConfig(window_size=4)
    print("  ✓ LocalMasker: window_size=4")
    
    # 2. Sink masker: 4 sink tokens for global information
    sink_config = SinkMaskerConfig(sink_size=4)
    print("  ✓ SinkMasker: sink_size=4")
    
    # 3. Oracle-TopK masker: 4 tokens based on oracle attention scores
    oracle_config = OracleTopKConfig(heavy_size=4)
    print("  ✓ OracleTopKMasker: heavy_size=4")
    
    # 4. Adaptive sampling masker: 10% base rate with statistical guarantees
    adaptive_config = AdaptiveSamplingMaskerConfig(
        base_rate_sampling=0.1,  # 10% base sampling rate
        epsilon=0.1,             # Error bound
        delta=0.05,              # Confidence bound
        init_offset=4,           # Start from beginning
        local_offset=4           # End at sequence end
    )
    print("  ✓ AdaptiveSamplingMasker: base_rate=0.1, epsilon=0.1, delta=0.05")
    
    # Combine all maskers in order of application
    combined_config = ResearchAttentionConfig(
        masker_configs=[local_config, 
        sink_config, 
        oracle_config, 
        adaptive_config,
        ]
    )
    
    print("\n📋 Combined Configuration:")
    print("  • Local(4) + Sink(4) + Oracle-TopK(4) + Adaptive(0.1)")
    print("  • Total maskers: 4")
    print("  • Expected sparsity: High (multiple sparse patterns combined)")
    
    # Common model arguments
    model_kwargs = {
        "model_kwargs": {"torch_dtype": torch.bfloat16},
        "device": str(device)
    }
    
    # Initialize adapter
    print("\n🔧 Loading model with combined maskers...")
    
    try:
        adapter = ModelAdapterHF(
            model_name=model_name,
            sparse_attention_config=combined_config,
            **model_kwargs
        )
        print("  ✅ Successfully loaded model with combined maskers")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return
    
    # Prepare test input
    print("\n📝 Preparing test input...")
    
    test_context = """
    The sparse attention mechanism combines multiple attention patterns to achieve 
    both computational efficiency and performance. This approach uses:
    
    1. Local attention: Captures immediate context within a small window
    2. Sink attention: Preserves global information from early tokens
    3. Oracle attention: Selects the most relevant tokens based on actual attention scores
    4. Adaptive sampling: Dynamically adjusts sampling based on statistical error bounds
    
    This combination allows the model to maintain high performance while significantly 
    reducing computational complexity for long sequences.
    """
    
    test_questions = [
        "What are the four attention patterns used in this sparse attention mechanism?",
        "How does adaptive sampling contribute to the overall efficiency?",
        "Explain the difference between local and sink attention patterns."
    ]
    
    request = Request(
        context=test_context,
        questions=test_questions,
    )
    
    print(f"  ✓ Context length: {len(test_context.split())} words")
    print(f"  ✓ Number of questions: {len(test_questions)}")
    
    # Run inference
    print("\n🧪 Running inference with combined maskers...")
    start_time = time.time()
    
    try:
        response = adapter.process_request(request, generation_kwargs={"max_new_tokens": 100}, request_kwargs={"max_context": 1024})
        response_text = response.responses
        
        inference_time = time.time() - start_time
        print(f"  ✅ Inference completed in {inference_time:.2f}s")
        
        # Display results
        print("\n📊 Results:")
        print("-" * 40)
        
        for i, (question, answer) in enumerate(zip(test_questions, response_text), 1):
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {answer}")
            
    except Exception as e:
        print(f"  ❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Performance analysis
    print("\n📈 Performance Analysis:")
    print("-" * 40)
    
    # Calculate expected sparsity
    # This is a rough estimate based on the masker configurations
    print("  • Local attention: ~25% sparsity (4-token window)")
    print("  • Sink attention: ~25% sparsity (4 sink tokens)")
    print("  • Oracle-TopK: ~25% sparsity (4 top tokens)")
    print("  • Adaptive sampling: ~10% base rate + adaptive budget")
    print("  • Combined effect: High sparsity with maintained performance")
    
    print("\n💡 Key Benefits:")
    print("  • Local attention captures immediate context")
    print("  • Sink attention preserves global information")
    print("  • Oracle attention selects most relevant tokens")
    print("  • Adaptive sampling provides statistical guarantees")
    print("  • Combined approach balances efficiency and performance")
    
    print(f"\n✅ Demo completed successfully!")
    print("   The combined masker approach demonstrates how different attention")
    print("   patterns can be layered to create sophisticated sparse attention mechanisms.")


if __name__ == "__main__":
    main() 
