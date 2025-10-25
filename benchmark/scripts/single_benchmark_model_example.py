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
    python simple_benchmark_example.py
"""

import os
import time
from pathlib import Path

import torch

# Ensure we're in the correct directory and add to Python path
import sys

# Change to directory two levels below current location
os.chdir('/home/sj157/Experiments/sparse-attention-hub')
sys.path.insert(0, '/home/sj157/Experiments/sparse-attention-hub')

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig, OracleTopKConfig, QuestTopKMaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig, MagicPigConfig
)

from benchmark.ruler32k import Ruler32K
from benchmark.longbench.longbench import LongBench
from sparse_attention_hub.adapters import ModelAdapterHF

def main():
    model_name = "lmsys/longchat-7b-v1.5-32k"
    device = 0

    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128), 
        QuestTopKMaskerConfig(
            heavy_size=0.128,
            page_size=16
        )
        # OracleTopKConfig(heavy_size=0.1)
    ])
    
    print("  ✓ Loading model...")
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs= {"torch_dtype": torch.bfloat16},
        generate_kwargs={"max_new_tokens": 32},
        device=device
    )
    
    benchmark = LongBench(['multifieldqa_en'])

    result_dir = Path("./test_results.5cpt.topk.2/")
    result_dir.mkdir(exist_ok=True)

    benchmark.run_benchmark(adapter, result_dir, request_kwargs={"max_requests": 50, "max_context_length": 32000})
    
if __name__ == "__main__":
    main() 
