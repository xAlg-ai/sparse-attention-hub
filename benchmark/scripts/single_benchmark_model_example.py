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
os.chdir('/workspace/sparse-attention-hub')
sys.path.insert(0, '/workspace/sparse-attention-hub')

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig, OracleTopKConfig,HashAttentionTopKMaskerConfig
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig
)

from benchmark.ruler32k import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger


def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_file = "/workspace/HashAttention-1.0/artifacts/llama3.1-8b-patch.64K.v1.hat_weights.pkl"

    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
        SinkMaskerConfig(sink_size=128),
        LocalMaskerConfig(window_size=128),
        #OracleTopKConfig(heavy_size=0.085),
        HashAttentionTopKMaskerConfig(
            heavy_size=0.03125,
            hat_bits=32,
            hat_mlp_layers=3,
            hat_mlp_hidden_size=128,
            hat_mlp_activation="silu",
            hat_weight_file=weight_file,
            hat_weights=None,
            ),
        AdaptiveSamplingMaskerConfig(base_rate_sampling=0.03125, epsilon=0.025, delta=0.025, init_offset=128, local_offset=128)
    ])
    
    print("  ✓ Loading model...")
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs= {"torch_dtype": torch.bfloat16},
        generate_kwargs={"max_new_tokens": 32},
        device=device
    )
    
    benchmark = Ruler32K(['vt'])

    result_dir = Path("./test_results.5cpt.topk.2/")
    result_dir.mkdir(exist_ok=True)
    metric_logger = MicroMetricLogger()
    metric_logger.configure_logging(log_path=result_dir, enabled_metrics=["research_attention_density", "research_attention_output_error"])

    benchmark.run_benchmark(adapter, result_dir, request_kwargs={"max_requests": 1, "max_context_length": 1000000})
    metric_logger.flush()
    
if __name__ == "__main__":
    main() 
