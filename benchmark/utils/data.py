"""Data processing utilities for benchmark execution.

This module contains utility functions for data processing, formatting, and export.
"""

import pandas as pd
import csv
from typing import Any


def escape_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Escape special characters in DataFrame for safe CSV export.
    
    This function prepares a DataFrame for CSV export by escaping newlines,
    carriage returns, and other special characters that could cause parsing issues.
    
    Args:
        df: DataFrame to escape
        
    Returns:
        DataFrame with escaped string columns
        
    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['Line 1\nLine 2', 'Text with, comma']
        ... })
        >>> escaped_df = escape_dataframe_for_csv(df)
        >>> escaped_df.to_csv('output.csv', quoting=csv.QUOTE_ALL)
    """
    escaped_df = df.copy()
    
    for col in escaped_df.columns:
        if escaped_df[col].dtype == 'object':  # Only process string columns
            escaped_df[col] = escaped_df[col].astype(str).str.replace('\n', '\\n').str.replace('\r', '\\r')
    
    return escaped_df


def save_dataframe_to_csv(
    df: pd.DataFrame, 
    file_path: str, 
    index: bool = False,
    quoting: int = csv.QUOTE_ALL
) -> None:
    """Save DataFrame to CSV with proper escaping and quoting.
    
    This is a convenience function that combines escaping and CSV export
    with consistent settings for benchmark results.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the CSV file
        index: Whether to include DataFrame index in output
        quoting: CSV quoting mode (default: QUOTE_ALL)
        
    Example:
        >>> df = pd.DataFrame({'text': ['Line 1\nLine 2']})
        >>> save_dataframe_to_csv(df, 'results.csv')
    """
    escaped_df = escape_dataframe_for_csv(df)
    escaped_df.to_csv(file_path, index=index, quoting=quoting) 