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
os.chdir('/root/prithvi/sparse-attention-hub')
sys.path.insert(0, '/root/prithvi/sparse-attention-hub')

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig, OracleTopKConfig, PQCacheConfig, XAttentionConfig,
    DoubleSparsityTopKMaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig
)

#from benchmark.longbench import LongBench
from benchmark.ruler32k import Ruler32K
from benchmark.longbench import LongBench
from sparse_attention_hub.adapters import ModelAdapterHF


def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = 0

    # sorted_channel_file is available in the author's repository
    # https://github.com/andy-yang-1/DoubleSparse/tree/main/config
    # TODO: is there a better way to use the paths in scripts?
    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),

        #OracleTopKConfig(heavy_size=5644),
        #AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.25, delta=0.25, init_offset=128, local_offset=128),
        #PQCacheConfig(heavy_size=1024,pq_sub_dim=64, pq_bits=7, kmeans_iters=25, sink_size = 4)
        XAttentionConfig(heavy_size=5644, importance_threshold=0.8, block_size=128, stride=8)
    ])

    
    print("  ✓ Loading model...")
     # use whichever is available
     # flash_attention_3 is for Hopper GPU
     # commonly flash_attention_2 is supported on other GPUs
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs= {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_3"},
        device=device
    )

    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(
             log_path="./test_results",
             enabled_metrics=["research_attention_density", "research_attention_output_error"],
    )

    
    benchmark = LongBench(['gov_report'])

    result_dir = Path("./test_results.vt.4096.2.2.q_proj/")
    result_dir.mkdir(exist_ok=True)

    benchmark.run_benchmark(adapter, result_dir, request_kwargs={"max_requests":50, "max_context_length": 1000000})
    
if __name__ == "__main__":
    main() 
