#!/usr/bin/env python3
"""
DataFrame View - Demonstration of working with DAT files as pandas DataFrames

This script shows how to load DAT files into pandas DataFrames and perform
common data analysis tasks using the dat_loader module.
"""

import pandas as pd
from dat_loader import load_dat_file, get_dat_info, quick_load
import sys
from pathlib import Path

def demonstrate_basic_loading(file_path):
    """Show basic ways to load a DAT file."""
    print("=" * 80)
    print("1. BASIC LOADING METHODS")
    print("=" * 80)

    # Method 1: Standard load with verbose output
    print("\n1.1 Standard Load (with progress info):")
    df = load_dat_file(file_path, verbose=True)
    print(f"   Loaded DataFrame: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Method 2: Quick load (no output)
    print("\n1.2 Quick Load (silent):")
    df_quick = quick_load(file_path)
    print(f"   Loaded DataFrame: {df_quick.shape[0]:,} rows x {df_quick.shape[1]} columns")

    # Method 3: Load with cleaning
    print("\n1.3 Load with Data Cleaning:")
    df_clean = load_dat_file(file_path, clean=True, verbose=True)
    print(f"   Cleaned DataFrame: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")

    return df_clean  # Return the cleaned version for further analysis

def demonstrate_file_info(file_path):
    """Show how to get file information before loading."""
    print("\n" + "=" * 80)
    print("2. FILE INFORMATION (before loading)")
    print("=" * 80)

    info = get_dat_info(file_path)
    file_size_mb = info['file_size_bytes'] / 1024 / 1024
    print(f"File size: {info['file_size_bytes']:,} bytes ({file_size_mb:.2f} MB)")
    print(f"Estimated rows: {info['estimated_rows']:,}")
    print(f"Detected encoding: {info['encoding']}")
    print(f"First few lines preview:")
    for i, line in enumerate(info['sample_lines'][:3], 1):
        print(f"  Line {i}: {line[:100]}...")

def demonstrate_dataframe_analysis(df):
    """Show common DataFrame analysis techniques."""
    print("\n" + "=" * 80)
    print("3. DATAFRAME ANALYSIS")
    print("=" * 80)

    # Basic info
    print("\n3.1 Basic DataFrame Information:")
    print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"   Total missing values: {df.isnull().sum().sum():,}")

    # Column overview
    print(f"\n3.2 Column Overview:")
    print(f"   Column names: {list(df.columns[:5])}..." if len(df.columns) > 5 else f"   Column names: {list(df.columns)}")

    # Data types
    print(f"\n3.3 Data Types:")
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"   {dtype}: {count} columns")

    # Missing data analysis
    print(f"\n3.4 Missing Data by Column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    for col in df.columns[:10]:  # Show first 10 columns
        if missing[col] > 0:
            print(f"   {col}: {missing[col]:,} missing ({missing_pct[col]}%)")
        else:
            print(f"   {col}: No missing values")

    if len(df.columns) > 10:
        print(f"   ... and {len(df.columns) - 10} more columns")

def demonstrate_data_exploration(df):
    """Show data exploration techniques."""
    print("\n" + "=" * 80)
    print("4. DATA EXPLORATION")
    print("=" * 80)

    # First few rows
    print("\n4.1 First 5 Rows:")
    print(df.head())

    # Unique values in key columns
    print(f"\n4.2 Unique Values in Key Columns:")
    for col in df.columns[:5]:  # Check first 5 columns
        unique_count = df[col].nunique()
        print(f"   {col}: {unique_count:,} unique values")

        # Show sample values for columns with reasonable number of unique values
        if 2 <= unique_count <= 10:
            sample_values = df[col].value_counts().head(3)
            print(f"      Top values: {list(sample_values.index)}")

    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        print(f"\n4.3 Numeric Columns Summary:")
        print(df[numeric_cols].describe())
    else:
        print(f"\n4.3 No numeric columns found")

def demonstrate_filtering_and_selection(df):
    """Show filtering and data selection techniques."""
    print("\n" + "=" * 80)
    print("5. FILTERING AND SELECTION")
    print("=" * 80)

    # Column selection
    print("\n5.1 Column Selection:")
    if 'Custodian' in df.columns:
        custodian_data = df['Custodian']
        print(f"   Selected 'Custodian' column: {len(custodian_data):,} values")
        print(f"   Unique custodians: {custodian_data.nunique()}")
        print(f"   Top custodians: {list(custodian_data.value_counts().head(3).index)}")

    # Multiple column selection
    key_cols = [col for col in ['ProdBeg', 'ProdEnd', 'Custodian', 'FileName'] if col in df.columns]
    if key_cols:
        subset = df[key_cols]
        print(f"\n5.2 Multiple Column Selection ({len(key_cols)} columns):")
        print(f"   Selected columns: {key_cols}")
        print(f"   Subset shape: {subset.shape}")

    # Row filtering
    print(f"\n5.3 Row Filtering Examples:")

    # Filter for non-empty values in first column
    first_col = df.columns[0]
    non_empty = df[df[first_col].notna() & (df[first_col] != '')]
    print(f"   Rows with non-empty {first_col}: {len(non_empty):,} / {len(df):,}")

    # Filter for specific values if Custodian exists
    if 'Custodian' in df.columns:
        top_custodian = df['Custodian'].value_counts().index[0] if len(df['Custodian'].value_counts()) > 0 else None
        if top_custodian:
            filtered = df[df['Custodian'] == top_custodian]
            print(f"   Rows for custodian '{top_custodian}': {len(filtered):,}")

def demonstrate_export_options(df, original_file_path):
    """Show how to export the DataFrame to different formats."""
    print("\n" + "=" * 80)
    print("6. EXPORT OPTIONS")
    print("=" * 80)

    base_name = Path(original_file_path).stem

    # CSV export
    csv_file = f"{base_name}_exported.csv"
    df.to_csv(csv_file, index=False)
    print(f"   [OK] Exported to CSV: {csv_file}")

    # Excel export (if openpyxl available)
    try:
        excel_file = f"{base_name}_exported.xlsx"
        df.to_excel(excel_file, index=False, sheet_name='Data')
        print(f"   [OK] Exported to Excel: {excel_file}")
    except ImportError:
        print(f"   [SKIP] Excel export not available (install openpyxl)")

    # Parquet export (if pyarrow available)
    try:
        parquet_file = f"{base_name}_exported.parquet"
        df.to_parquet(parquet_file, index=False)
        print(f"   [OK] Exported to Parquet: {parquet_file}")
    except ImportError:
        print(f"   [SKIP] Parquet export not available (install pyarrow)")

    # JSON export (sample)
    json_file = f"{base_name}_sample.json"
    df.head(10).to_json(json_file, orient='records', indent=2)
    print(f"   [OK] Exported sample (10 rows) to JSON: {json_file}")

def demonstrate_advanced_operations(df):
    """Show advanced DataFrame operations."""
    print("\n" + "=" * 80)
    print("7. ADVANCED OPERATIONS")
    print("=" * 80)

    # Groupby operations
    if 'Custodian' in df.columns:
        print("\n7.1 Group By Operations:")
        grouped = df.groupby('Custodian').size().sort_values(ascending=False)
        print(f"   Records per custodian:")
        for custodian, count in grouped.head(5).items():
            print(f"     {custodian}: {count:,} records")

    # Value counts
    print(f"\n7.2 Value Counts for Key Columns:")
    for col in df.columns[:3]:  # First 3 columns
        if df[col].dtype == 'object':  # Text columns
            unique_count = df[col].nunique()
            if unique_count <= 20:  # Only show if reasonable number
                print(f"   {col} value counts:")
                counts = df[col].value_counts().head(3)
                for value, count in counts.items():
                    print(f"     '{value}': {count:,}")

    # Data quality checks
    print(f"\n7.3 Data Quality Checks:")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates:,}")

    # Check for completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    print(f"   Completely empty rows: {empty_rows:,}")

    # Check for columns with all same values
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    print(f"   Columns with constant values: {len(constant_cols)}")
    if constant_cols:
        print(f"     {constant_cols[:5]}..." if len(constant_cols) > 5 else f"     {constant_cols}")

def main():
    """Main demonstration function."""
    if len(sys.argv) != 2:
        print("Usage: python dataframe_view.py <dat_file_path>")
        print("\nExample:")
        print("   python dataframe_view.py data.dat")
        print("   python dataframe_view.py \"C:\\path\\to\\your\\file.dat\"")
        return

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return

    print("DAT FILE DATAFRAME ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print(f"File: {file_path}")

    try:
        # 1. Get file info first
        demonstrate_file_info(file_path)

        # 2. Load the file
        df = demonstrate_basic_loading(file_path)

        # 3. Analyze the DataFrame
        demonstrate_dataframe_analysis(df)

        # 4. Explore the data
        demonstrate_data_exploration(df)

        # 5. Show filtering techniques
        demonstrate_filtering_and_selection(df)

        # 6. Export options
        demonstrate_export_options(df, file_path)

        # 7. Advanced operations
        demonstrate_advanced_operations(df)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Your DataFrame is ready for further analysis:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns[:5])}..." if len(df.columns) > 5 else f"   Columns: {list(df.columns)}")
        print("\nNext steps:")
        print("   1. Use the exported files for sharing or further analysis")
        print("   2. Import dat_loader in your own scripts:")
        print("      from dat_loader import load_dat_file")
        print("      df = load_dat_file('your_file.dat')")
        print("   3. Apply your specific analysis requirements to the DataFrame")

    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()