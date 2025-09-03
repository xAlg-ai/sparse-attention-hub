# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIME2025 benchmark module for evaluating mathematical reasoning.
"""

from .calculate_metrics import calculate_metrics
from .aime2025 import AIME2025

__all__ = ["calculate_metrics", "AIME2025"]
