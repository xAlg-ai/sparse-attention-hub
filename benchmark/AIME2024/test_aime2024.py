#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for AIME2024 benchmark implementation.

This script demonstrates how to use the AIME2024 benchmark and validates
that the implementation works correctly.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from create_huggingface_dataset import create_aime2024_dataset
from calculate_metrics import calculate_metrics, print_metrics_summary
import pandas as pd


def test_dataset_creation():
    """Test that the dataset can be created successfully."""
    print("Testing AIME2024 dataset creation...")

    dataset = create_aime2024_dataset()

    print(f"✓ Dataset created successfully with {len(dataset)} problems")

    # Verify dataset structure
    required_keys = [
        "context",
        "question",
        "answer_prefix",
        "answer",
        "id",
        "max_new_tokens",
    ]
    first_example = dataset[0]

    for key in required_keys:
        assert key in first_example, f"Missing required key: {key}"

    print("✓ Dataset structure is correct")

    # Show some examples
    print(f"\nSample problems:")
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        print(f"  ID: {example['id']}")
        print(f"  Answer: {example['answer']}")
        print(f"  Problem preview: {example['context'][:100]}...")
        print()

    return dataset


def test_metrics_calculation():
    """Test the metrics calculation with sample data."""
    print("Testing AIME2024 metrics calculation...")

    # Create test data with various scenarios
    test_data = {
        "answer": ["23", "33", "156", "902", "45"],
        "predicted_answer": [
            "The answer is \\boxed{23}.",  # Correct with boxed format
            "After solving, we get \\boxed{033}.",  # Correct with leading zeros
            "Therefore, the answer is \\boxed{156}.",  # Correct
            "The final answer is 902.",  # Correct without boxed format
            "I think the answer is \\boxed{44}.",  # Incorrect
        ],
        "id": ["2024-I-1", "2024-I-2", "2024-I-3", "2024-I-4", "2024-I-5"],
    }

    test_df = pd.DataFrame(test_data)
    metrics = calculate_metrics(test_df)

    print_metrics_summary(metrics)

    # Verify expected results
    assert (
        metrics["total_problems"] == 5
    ), f"Expected 5 problems, got {metrics['total_problems']}"
    assert (
        metrics["correct_answers"] == 4
    ), f"Expected 4 correct answers, got {metrics['correct_answers']}"
    assert (
        metrics["accuracy"] == 0.8
    ), f"Expected accuracy 0.8, got {metrics['accuracy']}"
    assert (
        metrics["extraction_failures"] == 0
    ), f"Expected 0 extraction failures, got {metrics['extraction_failures']}"

    print("✓ Metrics calculation is correct")

    return metrics


def test_integration():
    """Test integration with the main evaluation system."""
    print("Testing AIME2024 integration...")

    # Test that the dataset can be imported and used
    try:
        sys.path.append("..")
        from AIME2024.calculate_metrics import calculate_metrics
        from AIME2024.create_huggingface_dataset import create_aime2024_dataset

        # Test dataset creation through import
        dataset = create_aime2024_dataset()
        print(f"✓ Integration test passed - dataset has {len(dataset)} examples")

        # Verify the dataset can be converted to pandas
        df = dataset.to_pandas()
        print(f"✓ Dataset conversion to pandas successful - shape: {df.shape}")

        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests for AIME2024 benchmark."""
    print("AIME2024 Benchmark Test Suite")
    print("=" * 50)

    try:
        # Test dataset creation
        dataset = test_dataset_creation()
        print()

        # Test metrics calculation
        metrics = test_metrics_calculation()
        print()

        # Test integration
        integration_success = test_integration()
        print()

        if integration_success:
            print("🎉 All tests passed! AIME2024 benchmark is ready to use.")
            print()
            print("Usage examples:")
            print("  cd evaluation")
            print(
                "  python evaluate.py --dataset aime2024 --base_model meta-llama/Llama-3.1-8B-Instruct --device cpu"
            )
            print("  python evaluate.py --dataset aime2024 --device 1 --num_samples 10")
        else:
            print("❌ Some tests failed. Please check the implementation.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
