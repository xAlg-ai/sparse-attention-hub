# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark package for evaluating various language model capabilities.
"""

# Import the base benchmark class
from .base import Benchmark

# Import registry functions
from .benchmark_registry import (
    register_benchmark,
    get_registered_benchmarks,
    get_available_benchmark_names,
    create_benchmark_instance,
    ensure_benchmarks_loaded,
    validate_benchmark_config,
    get_benchmark_subsets
)

# Import concrete benchmark implementations
from .longbench import LongBench
from .mock_benchmark import MockBenchmark
from .loogle import Loogle

# Import all benchmark submodules to make them available for import
from . import AIME2024
from . import AIME2025
from . import infinite_bench
from . import longbench
from . import longbenchv2
from . import loogle
from . import mock_benchmark
from . import ruler
from . import zero_scrolls

__all__ = [
    "Benchmark",
    "register_benchmark",
    "get_registered_benchmarks", 
    "get_available_benchmark_names",
    "create_benchmark_instance",
    "ensure_benchmarks_loaded",
    "validate_benchmark_config",
    "get_benchmark_subsets",
    "LongBench",
    "MockBenchmark",
    "Loogle",
    "AIME2024",
    "AIME2025", 
    "infinite_bench",
    "longbench",
    "longbenchv2",
    "loogle",
    "mock_benchmark",
    "ruler",
    "zero_scrolls"
]
