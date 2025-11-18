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
os.chdir('/data/apdesai/code/sparse-attention-hub-rebuttal')
sys.path.insert(0, '/data/apdesai/code/sparse-attention-hub-rebuttal')

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SinkMaskerConfig,
    LocalMaskerConfig,
    OracleTopKConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)
#from benchmark.longbench import LongBench
from benchmark.ruler32k import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    device = 0
    delta_values = [0.01, 0.025, 0.05]
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    modes = ["denominator", "numerator"]
    
    # Loop through all combinations
    for mode in modes:
        for epsilon in epsilon_values:
            for delta in delta_values:
                print(f"\n{'='*80}")
                print(f"Running benchmark: mode={mode}, epsilon={epsilon}, delta={delta}")
                print(f"{'='*80}\n")
                
                # sorted_channel_file is available in the author's repository
                # https://github.com/andy-yang-1/DoubleSparse/tree/main/config
                # TODO: is there a better way to use the paths in scripts?
                sparse_attention_config = ResearchAttentionConfig(masker_configs=[
                    SinkMaskerConfig(sink_size=128),
                    LocalMaskerConfig(window_size=128),
                    OracleTopKConfig(heavy_size=0.05),
                    AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=epsilon, delta=delta, init_offset=128,
                    local_offset=128, mode=mode, use_exact_estimation=True)
                ])
                
                print("  ✓ Loading model...")
                # use whichever is available
                # flash_attention_3 is for Hopper GPU
                # commonly flash_attention_2 is supported on other GPUs
                adapter = ModelAdapterHF(
                    model_name=model_name,
                    sparse_attention_config=sparse_attention_config,
                    model_kwargs= {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
                    device=device
                )
                
                #benchmark = LongBench(['passage_retrieval_en'])
                benchmark = Ruler32K(['qa_1'])

                result_dir = Path(f"./ablations/{mode}.eps_{epsilon}.delta_{delta}/")
                result_dir.mkdir(exist_ok=True, parents=True)
                metric_logger = MicroMetricLogger()
                metric_logger.configure_logging(
                        log_path=result_dir,
                        enabled_metrics=[
                            "research_attention_density",
                            "research_attention_output_error",
                        ],
                    )
                metric_logger.flush()
                benchmark.run_benchmark(adapter, result_dir, request_kwargs={"max_requests": 10, "max_context_length": 1000000}, generation_kwargs={"max_new_tokens": 500})
                
                print(f"\n✓ Completed: mode={mode}, epsilon={epsilon}, delta={delta}")
                print(f"  Results saved to: {result_dir}\n")
                
                # Clean up to free GPU memory
                del adapter
                del benchmark
                torch.cuda.empty_cache()
                print("  ✓ Cleaned up GPU memory")
    
if __name__ == "__main__":
    main() 
