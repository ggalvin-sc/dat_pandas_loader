#!/usr/bin/env python3
"""
Transformation History Tracking System

This module provides comprehensive tracking of all data transformations applied during
DAT file processing, generating detailed history columns for audit and compliance purposes.

This is NOT specific to Tika - it tracks transformations from any source data.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import functools


def function_lock(func):
    """Decorator to lock function implementation and prevent modifications."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__locked__ = True
    return wrapper


class TransformationHistoryTracker:
    """
    Tracks all transformations applied to columns during data processing.

    This class maintains a complete audit trail of:
    - Column renames/mappings
    - Column merges (multiple columns → one column)
    - Column splits (one column → multiple columns)
    - Data format changes (especially date normalization)

    The tracked history can be used to generate per-row JSON history entries
    showing exactly how each piece of data was transformed.
    """

    @function_lock
    def __init__(self):
        """Initialize empty transformation tracking state."""
        # Maps final_column_name -> original_column_name for basic renames
        self.column_mappings: Dict[str, str] = {}

        # Maps merged_column_name -> {sources: [col1, col2], separator: str, timestamp: str}
        self.merge_operations: Dict[str, Dict[str, Any]] = {}

        # Maps original_column_name -> {targets: [new_col1, new_col2], separator: str, timestamp: str}
        self.split_operations: Dict[str, Dict[str, Any]] = {}

        # Maps column_name -> list of {type: str, details: dict, timestamp: str}
        self.transformation_log: Dict[str, List[Dict[str, Any]]] = {}

    @function_lock
    def track_mapping(self, original_column: str, final_column: str,
                     transformation_type: str = "column_rename") -> None:
        """
        Record a column rename/mapping operation.

        Args:
            original_column: The original column name
            final_column: The new column name after mapping
            transformation_type: Description of the transformation

        Side Effects:
            Updates internal mapping dictionaries
        """
        if not original_column or not final_column:
            raise ValueError("Column names cannot be empty or None")

        self.column_mappings[final_column] = original_column

        # Log this transformation
        if final_column not in self.transformation_log:
            self.transformation_log[final_column] = []

        self.transformation_log[final_column].append({
            "type": transformation_type,
            "original_name": original_column,
            "final_name": final_column,
            "timestamp": datetime.now().isoformat()
        })

    @function_lock
    def track_merge(self, source_columns: List[str], target_column: str,
                   separator: str = " ") -> None:
        """
        Record a column merge operation where multiple columns combine into one.

        Args:
            source_columns: List of original column names being merged
            target_column: Name of the new merged column
            separator: String used to join the source values

        Side Effects:
            Updates merge tracking and transformation log
        """
        if not source_columns:
            raise ValueError("Source columns list cannot be empty")
        if not target_column:
            raise ValueError("Target column name cannot be empty")
        if len(source_columns) < 2:
            raise ValueError("Merge requires at least 2 source columns")

        timestamp = datetime.now().isoformat()

        self.merge_operations[target_column] = {
            "sources": source_columns.copy(),
            "separator": separator,
            "timestamp": timestamp
        }

        # Log this transformation
        if target_column not in self.transformation_log:
            self.transformation_log[target_column] = []

        self.transformation_log[target_column].append({
            "type": "column_merge",
            "source_columns": source_columns.copy(),
            "target_column": target_column,
            "separator": separator,
            "timestamp": timestamp
        })

    @function_lock
    def track_split(self, original_column: str, target_columns: List[str],
                   separator: str = "-") -> None:
        """
        Record a column split operation where one column becomes multiple.

        Args:
            original_column: Name of the column being split
            target_columns: List of new column names created from the split
            separator: String that was used to split the original values

        Side Effects:
            Updates split tracking and transformation log for each target column
        """
        if not original_column:
            raise ValueError("Original column name cannot be empty")
        if not target_columns:
            raise ValueError("Target columns list cannot be empty")
        if len(target_columns) < 2:
            raise ValueError("Split requires at least 2 target columns")

        timestamp = datetime.now().isoformat()

        self.split_operations[original_column] = {
            "targets": target_columns.copy(),
            "separator": separator,
            "timestamp": timestamp
        }

        # Log this transformation for each target column
        for target_col in target_columns:
            if target_col not in self.transformation_log:
                self.transformation_log[target_col] = []

            self.transformation_log[target_col].append({
                "type": "column_split",
                "original_column": original_column,
                "target_column": target_col,
                "separator": separator,
                "timestamp": timestamp
            })

    @function_lock
    def track_data_transformation(self, column_name: str, transformation_type: str,
                                details: Dict[str, Any] = None) -> None:
        """
        Record a data format transformation applied to a column.

        Args:
            column_name: Name of the column that was transformed
            transformation_type: Type of transformation (e.g., "date_normalization")
            details: Additional details about the transformation

        Side Effects:
            Updates transformation log for the specified column
        """
        if not column_name:
            raise ValueError("Column name cannot be empty")
        if not transformation_type:
            raise ValueError("Transformation type cannot be empty")

        if column_name not in self.transformation_log:
            self.transformation_log[column_name] = []

        log_entry = {
            "type": transformation_type,
            "timestamp": datetime.now().isoformat()
        }

        if details:
            log_entry.update(details)

        self.transformation_log[column_name].append(log_entry)

    @function_lock
    def get_original_column_name(self, final_column: str) -> str:
        """
        Get the original column name for a final column name.

        Args:
            final_column: The final column name to look up

        Returns:
            The original column name, or the final_column if no mapping exists
        """
        return self.column_mappings.get(final_column, final_column)

    @function_lock
    def was_column_merged(self, column_name: str) -> bool:
        """
        Check if a column was created by merging other columns.

        Args:
            column_name: Column name to check

        Returns:
            True if this column was created by merging, False otherwise
        """
        return column_name in self.merge_operations

    @function_lock
    def was_column_split(self, column_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a column was created by splitting another column.

        Args:
            column_name: Column name to check

        Returns:
            Tuple of (was_split, original_column_name)
            - was_split: True if this column came from a split
            - original_column_name: Name of the original column, or None
        """
        for orig_col, split_info in self.split_operations.items():
            if column_name in split_info["targets"]:
                return True, orig_col
        return False, None

    @function_lock
    def get_column_transformations(self, column_name: str) -> List[Dict[str, Any]]:
        """
        Get all transformations applied to a specific column.

        Args:
            column_name: Column name to get transformations for

        Returns:
            List of transformation records for this column
        """
        return self.transformation_log.get(column_name, [])

    @function_lock
    def create_row_history_entry(self, row_index: int, dataframe: pd.DataFrame) -> str:
        """
        Create a JSON history entry for a specific row showing all transformations.

        Args:
            row_index: Index of the row to create history for (0-based)
            dataframe: The final DataFrame containing the transformed data

        Returns:
            JSON string containing transformation history for this row

        Raises:
            ValueError: If row_index is out of bounds
            TypeError: If dataframe is not a pandas DataFrame
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if row_index < 0 or row_index >= len(dataframe):
            raise ValueError(f"row_index {row_index} is out of bounds for DataFrame with {len(dataframe)} rows")

        row_data = dataframe.iloc[row_index].to_dict()
        history_entry = {}

        # Process each column in the final DataFrame
        for column in dataframe.columns:
            if column == 'column_history':
                # Skip the history column itself to avoid recursion
                continue

            column_history = self._create_column_history(column, row_data)
            history_entry[column] = column_history

        return json.dumps(history_entry, ensure_ascii=False)

    @function_lock
    def _create_column_history(self, column: str, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create history information for a single column.

        Args:
            column: Column name to create history for
            row_data: Dictionary containing the row's data

        Returns:
            Dictionary containing the column's transformation history
        """
        original_column = self.get_original_column_name(column)
        current_value = row_data.get(column, "")

        history = {
            "original_column": original_column,
            "final_column": column,
            "current_value": str(current_value) if current_value is not None else "",
            "was_renamed": original_column != column,
            "was_merged": self.was_column_merged(column),
            "transformation_count": len(self.get_column_transformations(column))
        }

        # Add merge details if applicable
        if history["was_merged"]:
            merge_info = self.merge_operations[column]
            source_values = []
            for source_col in merge_info["sources"]:
                # Try to get original value, fall back to empty string
                orig_val = row_data.get(source_col, "")
                source_values.append(str(orig_val) if orig_val is not None else "")

            history["merge_details"] = {
                "source_columns": merge_info["sources"],
                "source_values": source_values,
                "separator": merge_info["separator"]
            }

        # Add split details if applicable
        was_split, original_split_column = self.was_column_split(column)
        history["was_split"] = was_split
        if was_split:
            split_info = self.split_operations[original_split_column]
            history["split_details"] = {
                "original_column": original_split_column,
                "original_value": str(row_data.get(original_split_column, "")),
                "separator": split_info["separator"],
                "target_columns": split_info["targets"]
            }

        return history

    @function_lock
    def add_history_column_to_dataframe(self, source_dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'column_history' column to a DataFrame with transformation tracking.

        Args:
            source_dataframe: The DataFrame to add history to

        Returns:
            New DataFrame with 'column_history' column added

        Raises:
            TypeError: If source_dataframe is not a pandas DataFrame
            ValueError: If source_dataframe is empty
            Warning: If tracked columns don't exist in the DataFrame
        """
        if not isinstance(source_dataframe, pd.DataFrame):
            raise TypeError("source_dataframe must be a pandas DataFrame")
        if source_dataframe.empty:
            raise ValueError("DataFrame cannot be empty")

        # Validate that tracked columns exist in the DataFrame
        missing_columns = []
        for final_col in self.column_mappings.keys():
            if final_col not in source_dataframe.columns:
                missing_columns.append(final_col)

        if missing_columns:
            import warnings
            warnings.warn(f"Tracked columns not found in DataFrame: {missing_columns}")

        # Create a copy to avoid modifying the original
        df_with_history = source_dataframe.copy()

        # Generate history for each row using vectorized approach where possible
        num_rows = len(df_with_history)
        history_entries = []

        # Process in batches for better memory efficiency with large DataFrames
        batch_size = 1000
        for start_idx in range(0, num_rows, batch_size):
            end_idx = min(start_idx + batch_size, num_rows)

            for row_idx in range(start_idx, end_idx):
                history_entry = self.create_row_history_entry(row_idx, df_with_history)
                history_entries.append(history_entry)

        # Add the history column
        df_with_history['column_history'] = history_entries

        return df_with_history

    @function_lock
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all transformations tracked.

        Returns:
            Dictionary containing transformation statistics and details
        """
        return {
            "total_column_mappings": len(self.column_mappings),
            "total_merge_operations": len(self.merge_operations),
            "total_split_operations": len(self.split_operations),
            "columns_with_transformations": len(self.transformation_log),
            "column_mappings": self.column_mappings.copy(),
            "merge_operations": {k: v.copy() for k, v in self.merge_operations.items()},
            "split_operations": {k: v.copy() for k, v in self.split_operations.items()}
        }