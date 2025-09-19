#!/usr/bin/env python3
"""
Example usage of the DAT loader module.

This script demonstrates different ways to use the dat_loader module
to load DAT files into pandas DataFrames.
"""

import pandas as pd
from dat_loader import load_dat_file, quick_load, load_and_clean, get_dat_info

def example_basic_usage():
    """Example of basic DAT file loading."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)

    # Replace 'sample.dat' with your actual file path
    file_path = 'sample.dat'

    try:
        # Simple load
        df = load_dat_file(file_path)
        print(f"✅ Loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Create a sample DAT file to test with this example.")
    except Exception as e:
        print(f"❌ Error loading file: {e}")

def example_verbose_loading():
    """Example of loading with verbose output."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Verbose Loading")
    print("=" * 60)

    file_path = 'sample.dat'

    try:
        # Load with verbose output to see what's happening
        df = load_dat_file(file_path, verbose=True)
        print(f"\n✅ Final result: {df.shape}")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")

def example_cleaning():
    """Example of loading with data cleaning."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Loading with Data Cleaning")
    print("=" * 60)

    file_path = 'sample.dat'

    try:
        # Load and clean the data
        df = load_and_clean(file_path)
        print(f"\n✅ Cleaned DataFrame shape: {df.shape}")

        # Show data quality info
        print("\nData Quality Summary:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Missing values: {df.isnull().sum().sum():,}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")

def example_file_info():
    """Example of getting file information without loading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: File Information")
    print("=" * 60)

    file_path = 'sample.dat'

    try:
        # Get file info without fully loading
        info = get_dat_info(file_path)

        print("File Information:")
        print(f"  File size: {info['file_size_bytes']:,} bytes")
        print(f"  Detected encoding: {info['encoding']}")
        print(f"  Estimated rows: {info['estimated_rows']:,}")
        print(f"\nSample lines:")
        for i, line in enumerate(info['sample_lines'], 1):
            print(f"  {i}: {line}")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error getting file info: {e}")

def example_different_formats():
    """Example of handling different file formats."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Different File Formats")
    print("=" * 60)

    # List of different file types to try
    test_files = [
        'comma_separated.dat',
        'tab_separated.dat',
        'pipe_separated.dat',
        'fixed_width.dat'
    ]

    for file_path in test_files:
        try:
            print(f"\nTrying to load: {file_path}")
            df = load_dat_file(file_path, verbose=True)
            print(f"  ✅ Success! Shape: {df.shape}")

        except FileNotFoundError:
            print(f"  ⚠️  File not found: {file_path}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def example_integration_workflow():
    """Example of a complete data processing workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Complete Workflow")
    print("=" * 60)

    file_path = 'sample.dat'

    try:
        # Step 1: Check file info first
        print("Step 1: Getting file information...")
        info = get_dat_info(file_path)
        print(f"  File size: {info['file_size_bytes']:,} bytes")
        print(f"  Estimated rows: {info['estimated_rows']:,}")

        # Step 2: Load and clean the data
        print("\nStep 2: Loading and cleaning data...")
        df = load_and_clean(file_path, verbose=False)

        # Step 3: Basic analysis
        print(f"\nStep 3: Basic analysis...")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        # Step 4: Data type analysis
        print(f"\nStep 4: Data types...")
        for col in df.columns:
            non_null = df[col].count()
            unique_vals = df[col].nunique()
            print(f"  {col}: {non_null:,} non-null, {unique_vals:,} unique")

        # Step 5: Save to different formats
        print(f"\nStep 5: Saving to different formats...")
        df.to_csv('output.csv', index=False)
        print("  ✅ Saved to output.csv")

        try:
            df.to_excel('output.xlsx', index=False)
            print("  ✅ Saved to output.xlsx")
        except ImportError:
            print("  ⚠️  Excel support not available (install openpyxl)")

        try:
            df.to_parquet('output.parquet', index=False)
            print("  ✅ Saved to output.parquet")
        except ImportError:
            print("  ⚠️  Parquet support not available (install pyarrow)")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error in workflow: {e}")

def create_sample_dat_file():
    """Create a sample DAT file for testing."""
    print("\n" + "=" * 60)
    print("CREATING SAMPLE DAT FILE")
    print("=" * 60)

    sample_data = """ID,Name,Age,City,Amount
001,John Smith,25,New York,100.50
002,Jane Doe,30,Los Angeles,250.00
003,Bob Johnson,35,Chicago,75.25
004,Alice Brown,28,Houston,180.75
005,Charlie Wilson,42,Phoenix,320.00
006,Diana Miller,33,Philadelphia,95.50
007,Eve Davis,29,San Antonio,210.25
008,Frank Garcia,38,San Diego,155.00
009,Grace Rodriguez,31,Dallas,275.50
010,Henry Martinez,45,San Jose,400.00"""

    with open('sample.dat', 'w', encoding='utf-8') as f:
        f.write(sample_data)

    print("✅ Created sample.dat file for testing")
    print("You can now run the examples above!")

if __name__ == "__main__":
    print("DAT Loader Module - Example Usage")
    print("This script demonstrates how to use the dat_loader module.")
    print("\nChoose an option:")
    print("1. Create sample DAT file")
    print("2. Run all examples")
    print("3. Run specific example")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        create_sample_dat_file()

    elif choice == "2":
        # Run all examples
        example_basic_usage()
        example_verbose_loading()
        example_cleaning()
        example_file_info()
        example_different_formats()
        example_integration_workflow()

    elif choice == "3":
        print("\nAvailable examples:")
        print("1. Basic usage")
        print("2. Verbose loading")
        print("3. Data cleaning")
        print("4. File information")
        print("5. Different formats")
        print("6. Complete workflow")

        example_choice = input("Choose example (1-6): ").strip()

        if example_choice == "1":
            example_basic_usage()
        elif example_choice == "2":
            example_verbose_loading()
        elif example_choice == "3":
            example_cleaning()
        elif example_choice == "4":
            example_file_info()
        elif example_choice == "5":
            example_different_formats()
        elif example_choice == "6":
            example_integration_workflow()
        else:
            print("Invalid choice!")

    else:
        print("Invalid choice!")

    print("\n" + "=" * 60)
    print("Example usage in your own code:")
    print("=" * 60)
    print("""
# Simple usage
from dat_loader import load_dat_file
df = load_dat_file('my_file.dat')

# With cleaning and verbose output
df = load_dat_file('my_file.dat', clean=True, verbose=True)

# Quick load without any output
from dat_loader import quick_load
df = quick_load('my_file.dat')

# Get file info first
from dat_loader import get_dat_info
info = get_dat_info('my_file.dat')
print(f"File has approximately {info['estimated_rows']} rows")
""")