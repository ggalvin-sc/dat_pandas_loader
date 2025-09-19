#!/usr/bin/env python3
"""
Basic usage examples for the DAT Pandas Loader toolkit.

This script demonstrates the most common use cases for loading and working
with DAT files using the dat_pandas_loader package.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dat_loader import load_dat_file, get_dat_info, quick_load

def example_1_basic_loading():
    """Example 1: Basic file loading"""
    print("=" * 60)
    print("EXAMPLE 1: Basic File Loading")
    print("=" * 60)

    # Create a sample DAT file for demonstration
    sample_file = "sample.dat"
    sample_content = """þProdBegþþProdEndþþCustodianþþTitleþ
þ3M_WFDPD_00000022þþ3M_WFDPD_00000023þþKeown_Johnþþ Material Safety Data Sheetþ
þ3M_WFDPD_00000024þþ3M_WFDPD_00000025þþSmith_Sarahþþ Environmental Impact Reportþ"""

    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)

    try:
        # Load the file
        print("Loading DAT file...")
        df = load_dat_file(sample_file, verbose=True)

        print(f"\nLoaded DataFrame:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df)

    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)

def example_2_file_info():
    """Example 2: Getting file information before loading"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: File Information")
    print("=" * 60)

    # Create sample file
    sample_file = "info_test.dat"
    sample_content = "Name,Age,City\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"

    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)

    try:
        # Get file info
        print("Getting file information...")
        info = get_dat_info(sample_file)

        print(f"File size: {info['file_size_bytes']:,} bytes")
        print(f"Estimated rows: {info['estimated_rows']}")
        print(f"Detected encoding: {info['encoding']}")
        print(f"Sample lines:")
        for i, line in enumerate(info['sample_lines'][:3], 1):
            print(f"  Line {i}: {line}")

    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)

def example_3_export_data():
    """Example 3: Loading and exporting data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Export to Different Formats")
    print("=" * 60)

    # Create sample file
    sample_file = "export_test.dat"
    sample_content = """ID,Name,Department,Salary
1,Alice,Engineering,75000
2,Bob,Marketing,65000
3,Carol,Sales,70000"""

    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)

    try:
        # Load and export
        print("Loading data...")
        df = quick_load(sample_file)  # Silent load

        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Export to different formats
        df.to_csv('output.csv', index=False)
        print("[OK] Exported to CSV: output.csv")

        try:
            df.to_excel('output.xlsx', index=False)
            print("[OK] Exported to Excel: output.xlsx")
        except ImportError:
            print("[SKIP] Excel export not available (install openpyxl)")

        try:
            df.to_parquet('output.parquet', index=False)
            print("[OK] Exported to Parquet: output.parquet")
        except ImportError:
            print("[SKIP] Parquet export not available (install pyarrow)")

    finally:
        # Clean up
        for file in [sample_file, 'output.csv', 'output.xlsx', 'output.parquet']:
            if os.path.exists(file):
                os.remove(file)

def example_4_data_analysis():
    """Example 4: Basic data analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Data Analysis")
    print("=" * 60)

    # Create sample file with more interesting data
    sample_file = "analysis_test.dat"
    sample_content = """Product,Category,Price,Stock
Laptop,Electronics,999.99,15
Mouse,Electronics,29.99,50
Chair,Furniture,199.99,8
Desk,Furniture,299.99,5
Monitor,Electronics,249.99,12"""

    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)

    try:
        # Load and analyze
        print("Loading and analyzing data...")
        df = load_dat_file(sample_file, clean=True)

        print(f"\nDataFrame Info:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        print(f"\nData Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")

        print(f"\nSample Data:")
        print(df.head())

        # Basic analysis
        if 'Category' in df.columns:
            print(f"\nCategory Counts:")
            print(df['Category'].value_counts())

        # Numeric analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric Summary:")
            print(df[numeric_cols].describe())

    finally:
        # Clean up
        if os.path.exists(sample_file):
            os.remove(sample_file)

def main():
    """Run all examples"""
    print("DAT PANDAS LOADER - BASIC USAGE EXAMPLES")
    print("=" * 60)

    try:
        example_1_basic_loading()
        example_2_file_info()
        example_3_export_data()
        example_4_data_analysis()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Try with your own DAT files")
        print("2. Use the command-line tools in src/")
        print("3. Check the docs/ folder for detailed documentation")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()