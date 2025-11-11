# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re
import pandas as pd
from typing import List, Union
import ast

def extract_boxed_answer(text: str) -> str:
    """
    Extract the answer from \boxed{...} format in the text.
    
    Args:
        text: The model's response text
        
    Returns:
        The extracted answer as a string, or empty string if not found
    """
    if type(text) is not str:
        return ""
    # Look for \boxed{...} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, text)
    
    if matches:
        # Return the last boxed answer found (in case there are multiple)
        return matches[-1].strip()
    
    # Fallback: look for boxed without backslash (in case the model omits it)
    boxed_pattern_alt = r'boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern_alt, text)
    
    if matches:
        return matches[-1].strip()
    
    return ""


def extract_numerical_answer(text: str) -> str:
    """
    Extract a numerical answer from text, with fallback strategies.
    
    Args:
        text: The text to extract from
        
    Returns:
        The extracted numerical answer as a string
    """
    if type(text) is not str:
        return ""
    # First try to extract from boxed format
    boxed_answer = extract_boxed_answer(text)
    if boxed_answer:
        # Extract just the number from the boxed content
        numbers = re.findall(r'\d+', boxed_answer)
        if numbers:
            return numbers[-1]  # Take the last number found
    
    # Fallback 1: Look for "answer is X" or "answer: X" patterns
    answer_patterns = [
        r'(?:answer|solution)\s*(?:is|:)\s*(\d+)',
        r'(?:the\s+)?answer\s*(?:is|:)\s*(\d+)',
        r'(?:therefore|thus|so)\s*(?:the\s+)?(?:answer|solution)\s*(?:is|:)\s*(\d+)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Check if the number is in valid AIME range
            num = int(matches[-1])
            if 0 <= num <= 999:
                return matches[-1]
    
    # Fallback 2: Look for numbers at the end of the text
    # This catches cases where the model just states the number
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            numbers = re.findall(r'\b\d+\b', line)
            if numbers:
                # Check if the number is in valid AIME range (0-999)
                num = int(numbers[-1])
                if 0 <= num <= 999:
                    return str(num)
    
    # Fallback 3: Find any number in valid AIME range
    all_numbers = re.findall(r'\b\d+\b', text)
    for num_str in reversed(all_numbers):  # Check from end to beginning
        num = int(num_str)
        if 0 <= num <= 999:
            return str(num)
    
    return ""


def calculate_exact_match_score(predictions: List[str], references: List[List[str]]) -> float:
    """
    Calculate exact match accuracy between predictions and references.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers (each can have multiple valid answers)
        
    Returns:
        Exact match accuracy as a float between 0 and 1
    """
    correct = 0
    total = len(predictions)
    
    for pred, ref_list in zip(predictions, references):
        # Extract numerical answer from prediction
        pred_answer = extract_numerical_answer(pred)
        
        # Check if prediction matches any of the reference answers
        if pred_answer in ref_list:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate metrics for the AIME2025 benchmark.
    
    Args:
        df: DataFrame with columns 'predicted_answer' and 'answer'
        
    Returns:
        Dictionary containing the calculated metrics
    """
    predictions = df["predicted_answer"].tolist()
    references = df["answer"].tolist()
    
    references = [ast.literal_eval(ref) for ref in references]
    
    # Calculate exact match accuracy
    exact_match = calculate_exact_match_score(predictions, references)
    
    # Additional analysis: count how many answers were successfully extracted
    extracted_count = 0
    boxed_count = 0
    
    for pred in predictions:
        extracted = extract_numerical_answer(pred)
        if extracted:
            extracted_count += 1
        
        # Count how many used the boxed format
        if extract_boxed_answer(pred):
            boxed_count += 1
    
    extraction_rate = extracted_count / len(predictions) if predictions else 0.0
    boxed_format_rate = boxed_count / len(predictions) if predictions else 0.0
    
    return {
        "exact_match": exact_match,
        "extraction_rate": extraction_rate,
        "boxed_format_rate": boxed_format_rate,
        "total_problems": len(predictions)
    }


def analyze_errors(df: pd.DataFrame) -> dict:
    """
    Analyze common error patterns in the predictions.
    
    Args:
        df: DataFrame with predictions and references
        
    Returns:
        Dictionary with error analysis
    """
    predictions = df["predicted_answer"].tolist()
    references = df["answer"].tolist()
    
    if references and not isinstance(references[0], list):
        references = [[str(ref)] for ref in references]
    
    error_types = {
        "no_answer_extracted": 0,
        "wrong_answer": 0,
        "out_of_range": 0,
        "format_issues": 0
    }
    
    for pred, ref_list in zip(predictions, references):
        extracted = extract_numerical_answer(pred)
        
        if not extracted:
            error_types["no_answer_extracted"] += 1
        elif extracted not in ref_list:
            error_types["wrong_answer"] += 1
            try:
                num = int(extracted)
                if num < 0 or num > 999:
                    error_types["out_of_range"] += 1
            except ValueError:
                error_types["format_issues"] += 1
    
    return error_types
