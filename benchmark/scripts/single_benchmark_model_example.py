#!/usr/bin/env python3
"""
Simple Benchmark Example

A beginner-friendly example showing how to run a basic benchmark comparison
between dense and sparse attention using the sparse-attention-hub framework.

This example uses the MockBenchmark (5 simple samples) for quick demonstration:
- Easy-to-understand reading comprehension questions
- Short contexts (<250 words each)
- Fast execution for testing and learning

Usage:
    python 04_simple_benchmark_example.py
"""

import os
import time
from pathlib import Path

import torch

# Ensure we're in the correct directory and add to Python path
import sys

# Change to directory two levels below current location
os.chdir('/home/ubuntu/sparse-attention-hub')
sys.path.insert(0, '/home/ubuntu/sparse-attention-hub')

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig, OracleTopKConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig, MagicPigConfig
)

from benchmark.ruler32k import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = 0

    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
         SinkMaskerConfig(sink_size=128),
         LocalMaskerConfig(window_size=128),
         MagicPigConfig(
             lsh_l=75,  # Default value from search space
             lsh_k=8,   # Default value from search space
             ips_transformation="none"
         )
     ])
    
    print("  ✓ Loading model...")
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs= {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
        device=device
    )
    
    #for dataset in  [ 'niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_2', 'niah_multikey_3']:
    for dataset in  [  'niah_multikey_1', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1', 'qa_2']:
        benchmark = Ruler32K([dataset])
        result_dir = Path("./magicpig.cosine/"+dataset)
        result_dir.mkdir(exist_ok=True)
        benchmark.run_benchmark(adapter, result_dir, generation_kwargs={"max_new_tokens": 1000 }, 
                            request_kwargs={"max_requests": 50, "max_context_length": 32768, "dense_layers": [0, 16], "process_question_via_dense": True})
    
if __name__ == "__main__":
    main() 
