# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIME2025 benchmark module for evaluating mathematical reasoning capabilities.
"""

from .calculate_metrics import calculate_metrics
from .create_huggingface_dataset import create_aime2025_dataset

__all__ = ["calculate_metrics", "create_aime2025_dataset"]