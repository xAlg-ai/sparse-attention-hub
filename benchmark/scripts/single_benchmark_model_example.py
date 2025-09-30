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
os.chdir('/home/ubuntu/TEMP/sparse-attention-hub')
sys.path.insert(0, '/home/ubuntu/TEMP/sparse-attention-hub')

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig, OracleTopKConfig, PQCacheConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig
)

from benchmark.ruler32k import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = 0

    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        #OracleTopKConfig(heavy_size=128),
        #AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.25, delta=0.25, init_offset=128, local_offset=128),
        PQCacheConfig(heavy_size=512,pq_sub_dim=128//2, pq_bits=6)
    ])

    
    print("  ✓ Loading model...")
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs= {"torch_dtype": torch.bfloat16},
        generate_kwargs={"max_new_tokens": 32},
        device=device
    )

    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(
             log_path="./test_results",
             enabled_metrics=["research_attention_density", "research_attention_output_error"],
    )

    
    benchmark = Ruler32K(['vt'])

    result_dir = Path("./test_results")
    result_dir.mkdir(exist_ok=True)

    benchmark.run_benchmark(adapter, result_dir, request_kwargs={"max_requests":10, "max_context_length": 1000000})
    
if __name__ == "__main__":
    main() 
