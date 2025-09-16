#!/usr/bin/env python3
"""
Minimalistic benchmark runner for sparse attention evaluation.

This script defines models, sparse attention configurations, and benchmarks,
then runs comprehensive experiments using BenchmarkExecutor.

Usage:
    python benchmark/benchmark.py
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.executor import BenchmarkExecutor
from benchmark.executor_config import BenchmarkConfig, AdapterConfig
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, SinkMaskerConfig
)
from sparse_attention_hub.sparse_attention import (
    ChannelConfig,
    HashAttentionTopKMaskerConfig,
)

from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig
)

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    OracleTopKConfig
)
# ============================================================================
# CONFIGURATION
# ============================================================================

# GPU Configuration
GPUS = [0]  # Use all available GPUs
MAX_CONCURRENT_RUNS = 1  # One per GPU

# Model List
MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
]

usa_weight_file = "/workspace/HashAttention-1.0/artifacts/DeepSeek-R1-Distill-Llama-8B-patch-layers2-dim64-max-context-24K.pt"
weight_file = "/workspace/HashAttention-1.0/artifacts/DeepSeek-R1-Distill-Llama-8B-patch-layers2-dim64-max-context-24K.hat_weights.pkl"

from sparse_attention_hub.sparse_attention.utils.hashattention_utils import create_hat_weights_file_from_usa
create_hat_weights_file_from_usa(usa_weight_file, weight_file, num_layers=32, num_heads=32, device="cpu")

# Sparse Attention Configurations
ALEX_SPARSE_CONFIGS = [
    # Dense baseline (no sparse attention)
    #("dense", None),
    # hat2_NO_recovery_heavy_0.05 - 4 iterations
    ("hat2_NO_recovery_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=False,
    )),

    ("hat2_NO_recovery_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=False,
    )),

    ("hat2_NO_recovery_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=False,
    )),

    ("hat2_NO_recovery_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=False,
    )),

    # hat2_recovery_10000_heavy_0.05 - 4 iterations
    ("hat2_recovery_10000_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=10000,
    )),

    ("hat2_recovery_10000_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=10000,
    )),

    ("hat2_recovery_10000_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=10000,
    )),

    ("hat2_recovery_10000_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=10000,
    )),

    # hat2_recovery_100_heavy_0.05 - 4 iterations
    ("hat2_recovery_100_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=100,
    )),

    ("hat2_recovery_100_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=100,
    )),

    ("hat2_recovery_100_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=100,
    )),

    ("hat2_recovery_100_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=100,
    )),

    # hat2_recovery_200_heavy_0.05 - 4 iterations
    ("hat2_recovery_200_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=200,
    )),

    ("hat2_recovery_200_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=200,
    )),

    ("hat2_recovery_200_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=200,
    )),

    ("hat2_recovery_200_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=200,
    )),

    # hat2_recovery_300_heavy_0.05 - 4 iterations
    ("hat2_recovery_300_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=300,
    )),

    ("hat2_recovery_300_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=300,
    )),

    ("hat2_recovery_300_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=300,
    )),

    ("hat2_recovery_300_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=300,
    )),

    # hat2_recovery_500_heavy_0.05 - 4 iterations
    ("hat2_recovery_500_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=500,
    )),

    ("hat2_recovery_500_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=500,
    )),

    ("hat2_recovery_500_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=500,
    )),

    ("hat2_recovery_500_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=500,
    )),

    # hat2_recovery_1000_heavy_0.05 - 4 iterations
    ("hat2_recovery_1000_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=1000,
    )),

    ("hat2_recovery_1000_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=1000,
    )),

    ("hat2_recovery_1000_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=1000,
    )),

    ("hat2_recovery_1000_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=1000,
    )),

    # hat2_recovery_2000_heavy_0.05 - 4 iterations
    ("hat2_recovery_2000_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=2000,
    )),

    ("hat2_recovery_2000_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=2000,
    )),

    ("hat2_recovery_2000_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=2000,
    )),

    ("hat2_recovery_2000_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=2000,
    )),

    # hat2_recovery_5000_heavy_0.05 - 4 iterations
    ("hat2_recovery_5000_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=5000,
    )),

    ("hat2_recovery_5000_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=5000,
    )),

    ("hat2_recovery_5000_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=5000,
    )),

    ("hat2_recovery_5000_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=5000,
    )),

    # hat2_recovery_10000_heavy_0.05 - 4 iterations (Note: This was duplicated earlier, so I'm placing it here in proper order)
    ("hat2_recovery_20000_heavy_0.05_1", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=20000,
    )),

    ("hat2_recovery_20000_heavy_0.05_2", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=20000,
    )),

    ("hat2_recovery_20000_heavy_0.05_3", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=20000,
    )),

    ("hat2_recovery_20000_heavy_0.05_4", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),           # Keep first 128 tokens (sink attention)
            LocalMaskerConfig(window_size=128),       # Local attention window
            HashAttentionTopKMaskerConfig(heavy_size=0.05, 
                                        hat_bits=64, 
                                        hat_mlp_layers=2, 
                                        hat_mlp_hidden_size=128, 
                                        hat_mlp_activation="silu",
                                        hat_weight_file=weight_file,
                                        hat_weights=None),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,              # 10% base sampling rate
                epsilon=0.05,                         # 20% error bound
                delta=0.05,                           # 20% confidence bound
                init_offset=0.01,                       # Start sampling after local window
                local_offset=0.01                      # Sample within local context
            )
        ],
        recovery_enabled=True,
        recovery_interval=20000,
    )),
]


SPARSE_CONFIGS = [
    #("dense", None),
    ("test_oracle_topk_adaptive_norecovery", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            OracleTopKConfig(heavy_size=0.025),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.025,
                epsilon=0.05,
                delta=0.05,
                init_offset=128,
                local_offset=128
            )
        ],
        recovery_enabled=False,
        recovery_interval=32000,
    )),
    ("test_oracle_topk_adaptive_recovery_100", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            OracleTopKConfig(heavy_size=0.025),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.025,
                epsilon=0.05,
                delta=0.05,
                init_offset=128,
                local_offset=128
            )
        ],
        recovery_enabled=True,
        recovery_interval=100,
    )),
    ("test_oracle_topk_adaptive_recovery_400", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            OracleTopKConfig(heavy_size=0.025),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.025,
                epsilon=0.05,
                delta=0.05,
                init_offset=128,
                local_offset=128
            )
        ],
        recovery_enabled=True,
        recovery_interval=400,
    )),
    ("test_oracle_topk_adaptive_recovery_800", ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=128),
            OracleTopKConfig(heavy_size=0.025),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.025,
                epsilon=0.05,
                delta=0.05,
                init_offset=128,
                local_offset=128
            )
        ],
        recovery_enabled=True,
        recovery_interval=800,
    )),
]




# Benchmark List
# 1. InfiniteBench - using passkey task
infinite_bench_config = BenchmarkConfig(
    benchmark_name="infinite_bench",
    subsets=["passkey"]
)

# 2. Ruler - using 4096 context length
ruler_config = BenchmarkConfig(
    benchmark_name="ruler",
    subsets=["4096"]
)

# 3. Loogle - using shortdep_qa task
loogle_config = BenchmarkConfig(
    benchmark_name="loogle",
    subsets=["shortdep_qa"]
)

# 4. ZeroScrolls - using gov_report task
zero_scrolls_config = BenchmarkConfig(
    benchmark_name="zero_scrolls",
    subsets=["default"]
)

# 5. LongBenchv2 - using 0shot task
longbenchv2_config = BenchmarkConfig(
    benchmark_name="longbenchv2",
    subsets=["0shot"]
)

# 6. AIME2024 - using single task
aime2024_config = BenchmarkConfig(
    benchmark_name="aime2024",
    subsets=["aime2024"]
)

# 7. AIME2025 - using single task
aime2025_config = BenchmarkConfig(
    benchmark_name="aime2025",
    subsets=["aime2025"]
)

# 8. LongBench (existing) - using narrativeqa task
longbench_config = BenchmarkConfig(
    benchmark_name="longbench",
    subsets=["narrativeqa"]
)

# 9. Mock Benchmark (existing) - using single task
mock_benchmark_config = BenchmarkConfig(
    benchmark_name="mock_benchmark",
    subsets=["reading_comprehension"]
)

# List of all sample configurations
BENCHMARKS = [
    aime2024_config
]


# Adapter Configuration
ADAPTER_CONFIG = AdapterConfig(
    adapter_name="huggingface",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
    tokenizer_kwargs={
        "padding_side": "left",
    }
)

# Generation Parameters
GENERATION_KWARGS = {
    "max_new_tokens": 32768,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "pad_token_id": None,
}

# Request Parameters
REQUEST_KWARGS = {
    "max_context_length": 32768,
    "max_requests": 2,  # Limit for testing
}

# Execution Settings
RESULT_DIR = "./benchmark_results_test.1"
ENABLE_RESUMABILITY = True
TIMEOUT_PER_BENCHMARK = 60 * 60 * 24  # 1 day

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Minimalistic Benchmark Suite")
    print("=" * 50)
    
    print(f"🔧 Configuration:")
    print(f"  - GPUs: {GPUS}")
    print(f"  - Models: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"    {i}. {model}")
    print(f"  - Sparse configs: {len(SPARSE_CONFIGS)}")
    for name, config in SPARSE_CONFIGS:
        if config is None:
            print(f"    - {name}: dense (no sparse attention)")
        else:
            sink_size = config.masker_configs[0].sink_size
            window_size = config.masker_configs[1].window_size
            print(f"    - {name}: Sink({sink_size}) + Local({window_size})")
    print(f"  - Benchmarks: {len(BENCHMARKS)}")
    for i, benchmark in enumerate(BENCHMARKS, 1):
        if benchmark.subsets:
            print(f"    {i}. {benchmark.benchmark_name}: {len(benchmark.subsets)} subsets")
        else:
            print(f"    {i}. {benchmark.benchmark_name}: all subsets")
    print(f"  - Max concurrent: {MAX_CONCURRENT_RUNS}")
    print(f"  - Result dir: {RESULT_DIR}")
    print(f"  - Resumability: {'enabled' if ENABLE_RESUMABILITY else 'disabled'}")
    
    # Calculate total combinations
    total_models = len(MODELS)
    total_configs = len(SPARSE_CONFIGS)
    total_benchmarks = sum(len(b.subsets) if b.subsets else 1 for b in BENCHMARKS)
    total_combinations = total_models * total_configs * total_benchmarks
    
    print(f"\n📊 Experiment Matrix: {total_combinations} total combinations")
    print(f"  - Models: {total_models}")
    print(f"  - Sparse configs: {total_configs}")
    print(f"  - Benchmark-subsets: {total_benchmarks}")
    print(f"  - Estimated time: {total_combinations * TIMEOUT_PER_BENCHMARK / 3600:.1f} hours (worst case)")
    
    # Create executor
    print(f"\n🔧 Initializing BenchmarkExecutor...")
    executor = BenchmarkExecutor(
        gpu_ids=GPUS,
        max_concurrent_runs=MAX_CONCURRENT_RUNS,
        base_result_dir=RESULT_DIR,
        enable_resumability=ENABLE_RESUMABILITY,
        required_result_files=["raw_results.csv"],
        timeout_per_benchmark=TIMEOUT_PER_BENCHMARK,
        verbose=True
    )
    
    # Run benchmarks
    print(f"\n🎯 Running Benchmark Matrix...")
    try:
        results = executor.run_benchmark_matrix(
            model_names=MODELS,
            sparse_attention_configs=SPARSE_CONFIGS,
            benchmark_configs=BENCHMARKS,
            adapter_config=ADAPTER_CONFIG,
            generation_kwargs=GENERATION_KWARGS,
            request_kwargs=REQUEST_KWARGS
        )
        
        # Print summary
        print(f"\n✅ Benchmark Execution Completed!")
        print(f"  - Total: {results.progress.total_stubs}")
        print(f"  - Completed: {results.progress.completed_stubs}")
        print(f"  - Failed: {results.progress.failed_stubs}")
        print(f"  - Skipped: {results.progress.skipped_stubs}")
        print(f"  - Results saved to: {RESULT_DIR}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        print(f"  Partial results in: {RESULT_DIR}")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        raise 
