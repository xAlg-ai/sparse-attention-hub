"""
Sparse Attention Hub - A comprehensive framework for sparse attention mechanisms.

This package provides implementations of various sparse attention algorithms,
benchmarking tools, and integration with popular model frameworks.
"""

__version__ = "0.1.0"
__author__ = "AlexCuadron"

from .benchmark import Benchmark, BenchmarkExecutor
from .metrics import MicroMetric, MicroMetricLogger
from .model_hub import ModelHub, ModelHubHF
from .pipeline import Pipeline, PipelineHF, SparseAttentionServer
from .plotting import Granularity, PlotGenerator
from .sparse_attention import (
    EfficientAttention,
    ResearchAttention,
    SparseAttention,
    SparseAttentionGen,
    SparseAttentionHF,
)

__all__ = [
    "SparseAttention",
    "EfficientAttention",
    "ResearchAttention",
    "SparseAttentionHF",
    "SparseAttentionGen",
    "ModelHub",
    "ModelHubHF",
    "Pipeline",
    "PipelineHF",
    "SparseAttentionServer",
    "Benchmark",
    "BenchmarkExecutor",
    "MicroMetricLogger",
    "MicroMetric",
    "PlotGenerator",
    "Granularity",
]
