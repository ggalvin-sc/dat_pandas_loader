#!/usr/bin/env python3
"""
Quick DAT File Preview Tool

Simple script to preview the first 20 rows of any DAT file.
Handles special delimiters and encoding automatically.
"""

import sys
import pandas as pd
import chardet
from pathlib import Path
import functools


def function_lock(func):
    """Decorator to lock function implementation and prevent modifications."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__locked__ = True
    return wrapper

@function_lock
def detect_encoding(file_path):
    """Detect file encoding."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            detection = chardet.detect(raw_data)
        if detection and detection['confidence'] > 0.7:
            return detection['encoding']
    except:
        pass
    return 'utf-8'

@function_lock
def preview_dat_file(file_path, num_rows=20):
    """Preview a DAT file with smart delimiter detection."""
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    print("=" * 80)
    print(f"PREVIEW: {file_path}")
    print("=" * 80)

    # Get file info
    file_size = Path(file_path).stat().st_size
    print(f"File size: {file_size:,} bytes")

    # Detect encoding
    encoding = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")

    # Try different delimiters
    delimiters = [',', '\t', '|', ';', 'þ']  # Include the special þ character

    best_df = None
    best_delimiter = None
    max_columns = 0

    for delimiter in delimiters:
        try:
            if delimiter == 'þ':
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter,
                               dtype=str, on_bad_lines='skip', engine='python')
            else:
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter,
                               dtype=str, on_bad_lines='skip')

            if len(df.columns) > max_columns:
                max_columns = len(df.columns)
                best_df = df
                best_delimiter = delimiter

        except Exception:
            continue

    if best_df is None or len(best_df) == 0:
        print("❌ Could not parse file with any delimiter")
        return

    # Clean up column names if needed
    if best_delimiter == 'þ':
        # Remove weird column artifacts
        clean_columns = []
        for col in best_df.columns:
            if col.startswith('\x14') or col.startswith('.') or col.startswith('Unnamed:'):
                continue
            clean_columns.append(col)

        if clean_columns:
            best_df = best_df[clean_columns].copy()

    print(f"Delimiter used: '{best_delimiter}' ({'tab' if best_delimiter == chr(9) else 'special' if best_delimiter == 'þ' else best_delimiter})")
    print(f"Shape: {len(best_df):,} rows × {len(best_df.columns)} columns")
    print()

    # Show first rows
    print(f"FIRST {min(num_rows, len(best_df))} ROWS:")
    print("-" * 80)

    for i in range(min(num_rows, len(best_df))):
        row = best_df.iloc[i]
        print(f"ROW {i+1}:")

        # Show columns with non-null values
        displayed = 0
        for col in best_df.columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                if displayed >= 8:  # Limit columns shown
                    print(f"  ... and {len(best_df.columns) - displayed} more columns")
                    break

                # Truncate long values
                display_value = str(value)[:70] + "..." if len(str(value)) > 70 else str(value)
                print(f"  {col}: {display_value}")
                displayed += 1

        if displayed == 0:
            print("  (All values are empty or null)")
        print()

    if len(best_df) > num_rows:
        print(f"... and {len(best_df) - num_rows:,} more rows")

    print("-" * 80)
    print("COLUMN SUMMARY:")
    for i, col in enumerate(best_df.columns[:15], 1):  # Show first 15 columns
        non_null = best_df[col].notna().sum()
        unique_vals = best_df[col].nunique()
        print(f"  {i:2d}. {col[:40]:<40} : {non_null:4d} non-null, {unique_vals:4d} unique")

    if len(best_df.columns) > 15:
        print(f"  ... and {len(best_df.columns) - 15} more columns")

@function_lock
def main():
    if len(sys.argv) != 2:
        print("Usage: python preview_dat.py <file.dat>")
        print("Example: python preview_dat.py data.dat")
        sys.exit(1)

    file_path = sys.argv[1]
    preview_dat_file(file_path)

if __name__ == "__main__":
    main()