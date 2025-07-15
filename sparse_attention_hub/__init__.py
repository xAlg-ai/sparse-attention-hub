"""
Sparse Attention Hub - A comprehensive framework for sparse attention mechanisms.

This package provides implementations of various sparse attention algorithms,
benchmarking tools, and integration with popular model frameworks.
"""

__version__ = "0.1.0"
__author__ = "AlexCuadron"

from .adapters import (
    ModelAdapter,
    ModelAdapterHF,
    ModelHubAdapterInterface,
    Request,
    RequestResponse,
    SparseAttentionAdapterInterface,
)
from .metrics import MicroMetric, MicroMetricLogger
from .plotting import Granularity, PlotGenerator
from .sparse_attention import EfficientAttention, ResearchAttention, SparseAttention

__all__ = [
    "SparseAttention",
    "EfficientAttention",
    "ResearchAttention",
    "MicroMetricLogger",
    "MicroMetric",
    "PlotGenerator",
    "Granularity",
    "Request",
    "RequestResponse",
    "ModelHubAdapterInterface",
    "SparseAttentionAdapterInterface",
    "ModelAdapter",
    "ModelAdapterHF",
]
