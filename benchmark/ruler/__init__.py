# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ruler benchmark module for evaluating retrieval and understanding capabilities.
"""

from .calculate_metrics import calculate_metrics

__all__ = ["calculate_metrics"]
