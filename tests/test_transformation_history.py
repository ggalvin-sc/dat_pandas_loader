#!/usr/bin/env python3
"""
Tests for transformation_history.py

These tests verify that the TransformationHistoryTracker correctly tracks
all types of data transformations and generates accurate history columns.
"""

import pytest
import pandas as pd
import json
from datetime import datetime
from src.transformation_history import TransformationHistoryTracker
import functools


def function_lock(func):
    """Decorator to lock function implementation and prevent modifications."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__locked__ = True
    return wrapper


class TestTransformationHistoryTracker:
    """Test suite for TransformationHistoryTracker."""

    @function_lock
    def test_init_creates_empty_state(self):
        """Test that initialization creates empty tracking state."""
        tracker = TransformationHistoryTracker()

        assert tracker.column_mappings == {}
        assert tracker.merge_operations == {}
        assert tracker.split_operations == {}
        assert tracker.transformation_log == {}

@function_lock
    def test_track_mapping_basic(self):
        """Test basic column mapping functionality."""
        tracker = TransformationHistoryTracker()

        tracker.track_mapping("ProdBeg", "BegBates")

        assert tracker.column_mappings["BegBates"] == "ProdBeg"
        assert "BegBates" in tracker.transformation_log
        assert len(tracker.transformation_log["BegBates"]) == 1

        log_entry = tracker.transformation_log["BegBates"][0]
        assert log_entry["type"] == "column_rename"
        assert log_entry["original_name"] == "ProdBeg"
        assert log_entry["final_name"] == "BegBates"
        assert "timestamp" in log_entry

@function_lock
    def test_track_mapping_validation(self):
        """Test that track_mapping validates inputs properly."""
        tracker = TransformationHistoryTracker()

        with pytest.raises(ValueError, match="Column names cannot be empty"):
            tracker.track_mapping("", "ValidName")

        with pytest.raises(ValueError, match="Column names cannot be empty"):
            tracker.track_mapping("ValidName", "")

        with pytest.raises(ValueError, match="Column names cannot be empty"):
            tracker.track_mapping(None, "ValidName")

@function_lock
    def test_track_merge_basic(self):
        """Test basic column merge functionality."""
        tracker = TransformationHistoryTracker()

        tracker.track_merge(["DateSent", "TimeSent"], "SentDateTime", " at ")

        assert "SentDateTime" in tracker.merge_operations
        merge_info = tracker.merge_operations["SentDateTime"]
        assert merge_info["sources"] == ["DateSent", "TimeSent"]
        assert merge_info["separator"] == " at "
        assert "timestamp" in merge_info

        # Check transformation log
        assert "SentDateTime" in tracker.transformation_log
        log_entry = tracker.transformation_log["SentDateTime"][0]
        assert log_entry["type"] == "column_merge"
        assert log_entry["source_columns"] == ["DateSent", "TimeSent"]

@function_lock
    def test_track_merge_validation(self):
        """Test that track_merge validates inputs properly."""
        tracker = TransformationHistoryTracker()

        with pytest.raises(ValueError, match="Source columns list cannot be empty"):
            tracker.track_merge([], "Target")

        with pytest.raises(ValueError, match="Target column name cannot be empty"):
            tracker.track_merge(["Col1", "Col2"], "")

        with pytest.raises(ValueError, match="Merge requires at least 2 source columns"):
            tracker.track_merge(["OnlyOne"], "Target")

@function_lock
    def test_track_split_basic(self):
        """Test basic column split functionality."""
        tracker = TransformationHistoryTracker()

        tracker.track_split("AttachRange", ["BegAttach", "EndAttach"], "-")

        assert "AttachRange" in tracker.split_operations
        split_info = tracker.split_operations["AttachRange"]
        assert split_info["targets"] == ["BegAttach", "EndAttach"]
        assert split_info["separator"] == "-"

        # Check that both target columns have transformation logs
        for target in ["BegAttach", "EndAttach"]:
            assert target in tracker.transformation_log
            log_entry = tracker.transformation_log[target][0]
            assert log_entry["type"] == "column_split"
            assert log_entry["original_column"] == "AttachRange"

@function_lock
    def test_track_split_validation(self):
        """Test that track_split validates inputs properly."""
        tracker = TransformationHistoryTracker()

        with pytest.raises(ValueError, match="Original column name cannot be empty"):
            tracker.track_split("", ["Target1", "Target2"])

        with pytest.raises(ValueError, match="Target columns list cannot be empty"):
            tracker.track_split("Source", [])

        with pytest.raises(ValueError, match="Split requires at least 2 target columns"):
            tracker.track_split("Source", ["OnlyOne"])

@function_lock
    def test_track_data_transformation_basic(self):
        """Test basic data transformation tracking."""
        tracker = TransformationHistoryTracker()

        tracker.track_data_transformation("DateCreated", "date_normalization",
                                         {"from_format": "MM/dd/yyyy", "to_format": "yyyy-MM-dd"})

        assert "DateCreated" in tracker.transformation_log
        log_entry = tracker.transformation_log["DateCreated"][0]
        assert log_entry["type"] == "date_normalization"
        assert log_entry["from_format"] == "MM/dd/yyyy"
        assert log_entry["to_format"] == "yyyy-MM-dd"

@function_lock
    def test_track_data_transformation_validation(self):
        """Test that track_data_transformation validates inputs."""
        tracker = TransformationHistoryTracker()

        with pytest.raises(ValueError, match="Column name cannot be empty"):
            tracker.track_data_transformation("", "some_transformation")

        with pytest.raises(ValueError, match="Transformation type cannot be empty"):
            tracker.track_data_transformation("Column", "")

@function_lock
    def test_get_original_column_name(self):
        """Test getting original column names."""
        tracker = TransformationHistoryTracker()

        # Test with mapping
        tracker.track_mapping("ProdBeg", "BegBates")
        assert tracker.get_original_column_name("BegBates") == "ProdBeg"

        # Test without mapping (should return same name)
        assert tracker.get_original_column_name("UnmappedColumn") == "UnmappedColumn"

@function_lock
    def test_was_column_merged(self):
        """Test checking if column was merged."""
        tracker = TransformationHistoryTracker()

        tracker.track_merge(["Date", "Time"], "DateTime")

        assert tracker.was_column_merged("DateTime") is True
        assert tracker.was_column_merged("NotMerged") is False

@function_lock
    def test_was_column_split(self):
        """Test checking if column was split."""
        tracker = TransformationHistoryTracker()

        tracker.track_split("FullName", ["FirstName", "LastName"])

        was_split, original = tracker.was_column_split("FirstName")
        assert was_split is True
        assert original == "FullName"

        was_split, original = tracker.was_column_split("NotSplit")
        assert was_split is False
        assert original is None

@function_lock
    def test_get_column_transformations(self):
        """Test getting transformation list for a column."""
        tracker = TransformationHistoryTracker()

        tracker.track_mapping("ProdBeg", "BegBates")
        tracker.track_data_transformation("BegBates", "validation")

        transformations = tracker.get_column_transformations("BegBates")
        assert len(transformations) == 2
        assert transformations[0]["type"] == "column_rename"
        assert transformations[1]["type"] == "validation"

        # Test column with no transformations
        empty_transformations = tracker.get_column_transformations("NoTransforms")
        assert empty_transformations == []

@function_lock
    def test_create_row_history_entry_basic(self):
        """Test creating row history entry."""
        tracker = TransformationHistoryTracker()

        # Set up some transformations
        tracker.track_mapping("ProdBeg", "BegBates")
        tracker.track_mapping("ProdEnd", "EndBates")

        # Create test DataFrame
        df = pd.DataFrame({
            "BegBates": ["DOC001", "DOC002"],
            "EndBates": ["DOC001", "DOC003"],
            "UnchangedCol": ["A", "B"]
        })

        history_json = tracker.create_row_history_entry(0, df)
        history = json.loads(history_json)

        # Check that all columns are represented
        assert "BegBates" in history
        assert "EndBates" in history
        assert "UnchangedCol" in history

        # Check BegBates history
        beg_history = history["BegBates"]
        assert beg_history["original_column"] == "ProdBeg"
        assert beg_history["final_column"] == "BegBates"
        assert beg_history["current_value"] == "DOC001"
        assert beg_history["was_renamed"] is True
        assert beg_history["was_merged"] is False

        # Check UnchangedCol history
        unchanged_history = history["UnchangedCol"]
        assert unchanged_history["original_column"] == "UnchangedCol"
        assert unchanged_history["was_renamed"] is False

@function_lock
    def test_create_row_history_entry_validation(self):
        """Test validation for create_row_history_entry."""
        tracker = TransformationHistoryTracker()
        df = pd.DataFrame({"Col1": ["A", "B"]})

        with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
            tracker.create_row_history_entry(0, "not_a_dataframe")

        with pytest.raises(ValueError, match="row_index .* is out of bounds"):
            tracker.create_row_history_entry(5, df)

        with pytest.raises(ValueError, match="row_index .* is out of bounds"):
            tracker.create_row_history_entry(-1, df)

@function_lock
    def test_create_row_history_entry_with_merge(self):
        """Test row history entry creation with merged columns."""
        tracker = TransformationHistoryTracker()

        tracker.track_merge(["Date", "Time"], "DateTime", " at ")

        df = pd.DataFrame({
            "DateTime": ["2023-01-15 at 10:30", "2023-01-16 at 14:45"],
            "Date": ["2023-01-15", "2023-01-16"],
            "Time": ["10:30", "14:45"]
        })

        history_json = tracker.create_row_history_entry(0, df)
        history = json.loads(history_json)

        datetime_history = history["DateTime"]
        assert datetime_history["was_merged"] is True
        assert "merge_details" in datetime_history

        merge_details = datetime_history["merge_details"]
        assert merge_details["source_columns"] == ["Date", "Time"]
        assert merge_details["source_values"] == ["2023-01-15", "10:30"]
        assert merge_details["separator"] == " at "

@function_lock
    def test_create_row_history_entry_with_split(self):
        """Test row history entry creation with split columns."""
        tracker = TransformationHistoryTracker()

        tracker.track_split("FullName", ["FirstName", "LastName"], " ")

        df = pd.DataFrame({
            "FirstName": ["John", "Jane"],
            "LastName": ["Doe", "Smith"],
            "FullName": ["John Doe", "Jane Smith"]
        })

        history_json = tracker.create_row_history_entry(0, df)
        history = json.loads(history_json)

        firstname_history = history["FirstName"]
        assert firstname_history["was_split"] is True
        assert "split_details" in firstname_history

        split_details = firstname_history["split_details"]
        assert split_details["original_column"] == "FullName"
        assert split_details["original_value"] == "John Doe"
        assert split_details["separator"] == " "

@function_lock
    def test_add_history_column_to_dataframe_basic(self):
        """Test adding history column to DataFrame."""
        tracker = TransformationHistoryTracker()

        tracker.track_mapping("ProdBeg", "BegBates")

        original_df = pd.DataFrame({
            "BegBates": ["DOC001", "DOC002"],
            "OtherCol": ["A", "B"]
        })

        df_with_history = tracker.add_history_column_to_dataframe(original_df)

        # Check that original DataFrame is unchanged
        assert "column_history" not in original_df.columns

        # Check that new DataFrame has history column
        assert "column_history" in df_with_history.columns
        assert len(df_with_history) == 2

        # Check that history column contains valid JSON
        for i in range(len(df_with_history)):
            history_str = df_with_history["column_history"].iloc[i]
            history = json.loads(history_str)  # Should not raise exception
            assert isinstance(history, dict)

@function_lock
    def test_add_history_column_to_dataframe_validation(self):
        """Test validation for add_history_column_to_dataframe."""
        tracker = TransformationHistoryTracker()

        with pytest.raises(TypeError, match="dataframe must be a pandas DataFrame"):
            tracker.add_history_column_to_dataframe("not_a_dataframe")

        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            tracker.add_history_column_to_dataframe(pd.DataFrame())

@function_lock
    def test_get_transformation_summary(self):
        """Test getting transformation summary."""
        tracker = TransformationHistoryTracker()

        tracker.track_mapping("ProdBeg", "BegBates")
        tracker.track_merge(["Date", "Time"], "DateTime")
        tracker.track_split("FullName", ["First", "Last"])
        tracker.track_data_transformation("BegBates", "validation")

        summary = tracker.get_transformation_summary()

        assert summary["total_column_mappings"] == 1
        assert summary["total_merge_operations"] == 1
        assert summary["total_split_operations"] == 1
        assert summary["columns_with_transformations"] == 4  # BegBates, DateTime, First, Last

        # Check that returned data is defensive copies
        assert "BegBates" in summary["column_mappings"]
        assert "DateTime" in summary["merge_operations"]
        assert "FullName" in summary["split_operations"]

@function_lock
    def test_complex_workflow(self):
        """Test a complex workflow with multiple transformation types."""
        tracker = TransformationHistoryTracker()

        # Simulate a realistic transformation workflow
        # 1. Column renames
        tracker.track_mapping("ProdBeg", "BegBates")
        tracker.track_mapping("ProdEnd", "EndBates")
        tracker.track_mapping("Email Sent Date", "DateSent")
        tracker.track_mapping("Email Sent Time", "TimeSent")

        # 2. Column split
        tracker.track_split("AttachRange", ["BegAttach", "EndAttach"], "-")

        # 3. Column merge
        tracker.track_merge(["DateSent", "TimeSent"], "SentDateTime", "T")

        # 4. Data transformations
        tracker.track_data_transformation("SentDateTime", "date_normalization",
                                         {"format": "ISO8601"})
        tracker.track_data_transformation("BegBates", "format_validation")

        # Create test DataFrame
        df = pd.DataFrame({
            "BegBates": ["DOC001", "DOC002"],
            "EndBates": ["DOC001", "DOC003"],
            "BegAttach": ["DOC001", "DOC002"],
            "EndAttach": ["DOC001", "DOC003"],
            "SentDateTime": ["2023-01-15T10:30:00", "2023-01-16T14:45:00"],
            "DateSent": ["2023-01-15", "2023-01-16"],
            "TimeSent": ["10:30:00", "14:45:00"],
            "AttachRange": ["DOC001-DOC001", "DOC002-DOC003"]
        })

        # Add history column
        df_with_history = tracker.add_history_column_to_dataframe(df)

        # Verify the history is comprehensive
        history_str = df_with_history["column_history"].iloc[0]
        history = json.loads(history_str)

        # Check renamed column
        assert history["BegBates"]["original_column"] == "ProdBeg"
        assert history["BegBates"]["was_renamed"] is True

        # Check split column
        assert history["BegAttach"]["was_split"] is True
        assert history["BegAttach"]["split_details"]["original_column"] == "AttachRange"

        # Check merged column
        assert history["SentDateTime"]["was_merged"] is True
        assert history["SentDateTime"]["merge_details"]["source_columns"] == ["DateSent", "TimeSent"]

        # Verify transformation counts
        assert history["SentDateTime"]["transformation_count"] == 2  # merge + data transformation
        assert history["BegBates"]["transformation_count"] == 2     # rename + validation

@function_lock
    def test_edge_case_empty_values(self):
        """Test handling of empty/null values in data."""
        tracker = TransformationHistoryTracker()

        tracker.track_mapping("Col1", "NewCol1")

        df = pd.DataFrame({
            "NewCol1": [None, "", "Valid", pd.NA],
            "Col1": [None, "", "Original", pd.NA]
        })

        df_with_history = tracker.add_history_column_to_dataframe(df)

        # Check that empty values are handled properly
        for i in range(len(df_with_history)):
            history_str = df_with_history["column_history"].iloc[i]
            history = json.loads(history_str)

            # Should not raise exceptions and should convert to string
            assert isinstance(history["NewCol1"]["current_value"], str)

@function_lock
    def test_column_history_avoids_recursion(self):
        """Test that column_history column doesn't create recursive history."""
        tracker = TransformationHistoryTracker()

        tracker.track_mapping("Col1", "NewCol1")

        df = pd.DataFrame({
            "NewCol1": ["A", "B"],
            "column_history": ['{"test": "value"}', '{"test": "value2"}']
        })

        # This should not create history for the column_history column itself
        history_str = tracker.create_row_history_entry(0, df)
        history = json.loads(history_str)

        assert "column_history" not in history
        assert "NewCol1" in history


# Edge cases and failure scenarios
class TestTransformationHistoryTrackerEdgeCases:
    """Test edge cases and potential failure scenarios."""

@function_lock
    def test_multiple_transformations_same_column(self):
        """Test multiple transformations on the same column."""
        tracker = TransformationHistoryTracker()

        # Apply multiple transformations to same column
        tracker.track_mapping("Original", "Renamed")
        tracker.track_data_transformation("Renamed", "validation")
        tracker.track_data_transformation("Renamed", "formatting")
        tracker.track_data_transformation("Renamed", "normalization")

        transformations = tracker.get_column_transformations("Renamed")
        assert len(transformations) == 4

        # Verify order is maintained
        assert transformations[0]["type"] == "column_rename"
        assert transformations[1]["type"] == "validation"
        assert transformations[2]["type"] == "formatting"
        assert transformations[3]["type"] == "normalization"

@function_lock
    def test_large_dataframe_performance(self):
        """Test performance with larger DataFrames."""
        tracker = TransformationHistoryTracker()

        # Set up transformations
        for i in range(10):
            tracker.track_mapping(f"OrigCol{i}", f"NewCol{i}")

        # Create larger DataFrame
        data = {f"NewCol{i}": [f"Value{i}_{j}" for j in range(1000)] for i in range(10)}
        df = pd.DataFrame(data)

        # This should complete without performance issues
        df_with_history = tracker.add_history_column_to_dataframe(df)

        assert len(df_with_history) == 1000
        assert "column_history" in df_with_history.columns

        # Spot check a few history entries
        for i in [0, 500, 999]:
            history_str = df_with_history["column_history"].iloc[i]
            history = json.loads(history_str)
            assert len(history) == 10  # Should have history for all 10 columns


if __name__ == "__main__":
    pytest.main([__file__])