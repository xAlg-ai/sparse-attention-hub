# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import pandas as pd
from typing import List, Dict, Any


def extract_boxed_answer(text: str) -> str:
    """
    Extract the answer from \boxed{...} format in the text.

    Args:
        text: The model's response text

    Returns:
        The extracted answer as a string, or empty string if not found
    """
    # Look for \boxed{...} pattern
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(boxed_pattern, text)

    if matches:
        # Take the last boxed answer in case there are multiple
        answer = matches[-1].strip()

        # Extract just the number if there's additional formatting
        # Handle cases like "033", "23", "$23$", etc.
        number_match = re.search(r"\d+", answer)
        if number_match:
            return number_match.group()
        else:
            return answer

    # Fallback: look for numbers at the end of the text
    # This handles cases where the model doesn't use \boxed format
    lines = text.strip().split("\n")
    for line in reversed(lines):
        if line.strip():
            # Look for a number in the last non-empty line
            number_match = re.search(r"\b(\d{1,3})\b", line)
            if number_match:
                return number_match.group(1)

    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer to a standard format.

    Args:
        answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    # Remove leading zeros but keep at least one digit
    answer = answer.strip()
    if answer.isdigit():
        return str(int(answer))
    return answer


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for AIME2024 benchmark.

    Args:
        df: DataFrame with columns 'answer' (ground truth) and 'predicted_answer' (model output)

    Returns:
        Dictionary containing evaluation metrics
    """
    if "predicted_answer" not in df.columns:
        raise ValueError("DataFrame must contain 'predicted_answer' column")
    if "answer" not in df.columns:
        raise ValueError("DataFrame must contain 'answer' column")

    total_problems = len(df)
    correct_answers = 0
    extraction_failures = 0

    detailed_results = []

    for idx, row in df.iterrows():
        ground_truth = normalize_answer(str(row["answer"]))
        predicted_text = (
            str(row["predicted_answer"]) if pd.notna(row["predicted_answer"]) else ""
        )

        # Extract the predicted answer
        extracted_answer = extract_boxed_answer(predicted_text)

        if not extracted_answer:
            extraction_failures += 1
            is_correct = False
        else:
            extracted_answer = normalize_answer(extracted_answer)
            is_correct = extracted_answer == ground_truth
            if is_correct:
                correct_answers += 1

        detailed_results.append(
            {
                "id": row.get("id", f"problem_{idx}"),
                "ground_truth": ground_truth,
                "predicted_text": predicted_text,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "extraction_failed": not bool(extracted_answer),
            }
        )

    # Calculate metrics
    accuracy = correct_answers / total_problems if total_problems > 0 else 0.0
    extraction_success_rate = (
        (total_problems - extraction_failures) / total_problems
        if total_problems > 0
        else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "correct_answers": correct_answers,
        "total_problems": total_problems,
        "extraction_success_rate": extraction_success_rate,
        "extraction_failures": extraction_failures,
        "detailed_results": detailed_results,
    }

    return metrics


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the evaluation metrics.

    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("AIME2024 Evaluation Results")
    print("=" * 40)
    print(f"Total Problems: {metrics['total_problems']}")
    print(f"Correct Answers: {metrics['correct_answers']}")
    print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(
        f"Extraction Success Rate: {metrics['extraction_success_rate']:.3f} ({metrics['extraction_success_rate']*100:.1f}%)"
    )
    print(f"Extraction Failures: {metrics['extraction_failures']}")

    if metrics["extraction_failures"] > 0:
        print(
            f"\nNote: {metrics['extraction_failures']} problems had answer extraction failures."
        )
        print("These are counted as incorrect answers.")


if __name__ == "__main__":
    # Test the metrics calculation
    test_data = {
        "answer": ["23", "33", "156", "902"],
        "predicted_answer": [
            "The answer is \\boxed{23}.",
            "After solving, we get \\boxed{033}.",
            "Therefore, the answer is \\boxed{156}.",
            "The final answer is 902.",  # Test fallback extraction
        ],
        "id": ["2024-I-1", "2024-I-2", "2024-I-3", "2024-I-4"],
    }

    test_df = pd.DataFrame(test_data)
    metrics = calculate_metrics(test_df)
    print_metrics_summary(metrics)

    print("\nDetailed Results:")
    for result in metrics["detailed_results"]:
        print(f"ID: {result['id']}")
        print(f"  Ground Truth: {result['ground_truth']}")
        print(f"  Extracted: {result['extracted_answer']}")
        print(f"  Correct: {result['is_correct']}")
        print()
