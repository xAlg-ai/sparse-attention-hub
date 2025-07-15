# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Zero Scrolls benchmark module for evaluating long document understanding.
"""

from .calculate_metrics import calculate_metrics

__all__ = ["calculate_metrics"]
