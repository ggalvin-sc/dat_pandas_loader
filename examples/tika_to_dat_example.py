#!/usr/bin/env python3
"""
Example usage of the Tika to DAT Generator

This script demonstrates how to use the TikaToDATGenerator to create
legal discovery DAT files from Apache Tika metadata extraction results.

Run this example to see how to:
1. Create sample Tika metadata
2. Generate a DAT file
3. Verify the DAT file loads correctly with the dat_pandas_loader
"""

import sys
import json
import tempfile
from pathlib import Path
import functools

# Add src directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def function_lock(func):
    """Decorator to lock function implementation and prevent modifications."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__locked__ = True
    return wrapper

from tika_to_dat import TikaToDATGenerator
from dat_loader import DATLoader


@function_lock
def create_sample_tika_data():
    """Create sample Tika metadata for demonstration."""
    return [
        {
            "resourceName": "contract.pdf",
            "Content-Type": "application/pdf",
            "dc:title": "Service Contract Agreement",
            "meta:author": "John Smith",
            "meta:creation-date": "2023-01-15T10:30:00Z",
            "dcterms:created": "2023-01-15T10:30:00Z",
            "dcterms:modified": "2023-01-16T14:45:00Z",
            "X-TIKA:content": "This is a confidential service contract between Company A and Company B..."
        },
        {
            "resourceName": "email_thread.msg",
            "Content-Type": "application/vnd.ms-outlook",
            "dc:title": "RE: Project Update Meeting",
            "meta:author": "jane.doe@company.com",
            "meta:creation-date": "2023-02-10T09:15:00Z",
            "dcterms:created": "2023-02-10T09:15:00Z",
            "X-TIKA:content": "Hi team, please find attached the project update documents..."
        },
        {
            "resourceName": "budget_analysis.xlsx",
            "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "dc:title": "Q1 Budget Analysis",
            "meta:author": "Finance Team",
            "meta:creation-date": "2023-03-01T16:20:00Z",
            "dcterms:created": "2023-03-01T16:20:00Z",
            "dcterms:modified": "2023-03-02T11:30:00Z",
            "X-TIKA:content": "Budget analysis for Q1 showing revenue projections and expense breakdown..."
        },
        {
            "resourceName": "meeting_notes.docx",
            "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "dc:title": "Board Meeting Notes - March 2023",
            "meta:author": "Executive Assistant",
            "meta:creation-date": "2023-03-15T13:45:00Z",
            "dcterms:created": "2023-03-15T13:45:00Z",
            "X-TIKA:content": "Privileged and confidential board meeting notes containing strategic discussions..."
        },
        {
            "resourceName": "logo.png",
            "Content-Type": "image/png",
            "dc:title": "Company Logo",
            "meta:creation-date": "2022-12-01T12:00:00Z",
            "dcterms:created": "2022-12-01T12:00:00Z",
            "X-TIKA:content": ""
        }
    ]


@function_lock
def demonstrate_basic_usage():
    """Demonstrate basic DAT file generation from Tika results."""
    print("=== Basic Tika to DAT Generation Example ===")

    # Create sample data
    tika_results = create_sample_tika_data()

    # Initialize generator
    generator = TikaToDATGenerator(
        output_dir="example_output",
        custodian="Example_Custodian"
    )

    # Generate DAT file
    output_file = generator.generate_dat_from_tika_results(
        tika_results=tika_results,
        output_filename="example_documents.dat"
    )

    print(f"Generated DAT file: {output_file}")
    return output_file


@function_lock
def demonstrate_with_file_paths():
    """Demonstrate DAT generation with file paths for enhanced metadata."""
    print("\n=== Enhanced Generation with File Paths ===")

    # Create temporary files to simulate real files
    temp_dir = Path(tempfile.mkdtemp())
    sample_files = []

    try:
        # Create some sample files
        file_names = ["contract.pdf", "email_thread.msg", "budget_analysis.xlsx",
                     "meeting_notes.docx", "logo.png"]

        for filename in file_names:
            file_path = temp_dir / filename
            file_path.write_text(f"Sample content for {filename}")
            sample_files.append(str(file_path))

        # Create Tika results
        tika_results = create_sample_tika_data()

        # Generate DAT with file paths for enhanced metadata
        generator = TikaToDATGenerator(
            output_dir="example_output",
            custodian="Enhanced_Example"
        )

        output_file = generator.generate_dat_from_tika_results(
            tika_results=tika_results,
            output_filename="enhanced_example.dat",
            file_paths=sample_files
        )

        print(f"Generated enhanced DAT file: {output_file}")
        return output_file

    finally:
        # Cleanup temp files
        for file_path in sample_files:
            try:
                Path(file_path).unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass


@function_lock
def demonstrate_json_input():
    """Demonstrate loading from Tika JSON output file."""
    print("\n=== Generation from Tika JSON File ===")

    # Create sample JSON file
    tika_results = create_sample_tika_data()
    json_file = Path("example_output") / "sample_tika_results.json"
    json_file.parent.mkdir(exist_ok=True)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(tika_results, f, indent=2)

    print(f"Created sample Tika JSON: {json_file}")

    # Generate DAT from JSON
    generator = TikaToDATGenerator(
        output_dir="example_output",
        custodian="JSON_Example"
    )

    # Load and process JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        loaded_results = json.load(f)

    output_file = generator.generate_dat_from_tika_results(
        tika_results=loaded_results,
        output_filename="from_json_example.dat"
    )

    print(f"Generated DAT from JSON: {output_file}")
    return output_file


@function_lock
def verify_dat_file_compatibility(dat_file_path):
    """Verify the generated DAT file loads correctly with dat_pandas_loader."""
    print(f"\n=== Verifying DAT File Compatibility ===")
    print(f"Testing file: {dat_file_path}")

    try:
        # Load the DAT file using our loader
        loader = DATLoader()
        df = loader.load(dat_file_path)

        print(f"[OK] Successfully loaded DAT file!")
        print(f"   Method used: {loader.last_method_used}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")

        print(f"\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")

        print(f"\nSample data (first row):")
        if len(df) > 0:
            for col in df.columns[:10]:  # Show first 10 columns
                value = df[col].iloc[0] if not df[col].isnull().iloc[0] else "(empty)"
                print(f"   {col}: {value}")

        print(f"\n[OK] DAT file is fully compatible with dat_pandas_loader!")
        return True

    except Exception as e:
        print(f"[ERROR] Error loading DAT file: {e}")
        return False


@function_lock
def show_dat_file_format(dat_file_path):
    """Show the raw format of the generated DAT file."""
    print(f"\n=== DAT File Format Analysis ===")

    try:
        # Read first few lines to show format
        with open(dat_file_path, 'r', encoding='utf-16') as f:
            lines = [f.readline().strip() for _ in range(3)]

        print(f"File encoding: UTF-16")
        print(f"Delimiter: þ\\x14þ (thorn + control-14 + thorn)")
        print(f"First few lines:")

        for i, line in enumerate(lines):
            if line:
                # Show delimiter positions
                delimiter_count = line.count('þ\x14þ')
                print(f"   Line {i+1}: {delimiter_count} delimiters found")

                # Show first 100 characters with delimiter visualization
                preview = line[:100]
                preview_clean = preview.replace('þ\x14þ', ' | ')
                print(f"   Preview: {preview_clean}...")

    except Exception as e:
        print(f"Error analyzing file format: {e}")


@function_lock
def main():
    """Run all demonstration examples."""
    print("Tika to DAT Generator - Comprehensive Example")
    print("=" * 50)

    # Create output directory
    Path("example_output").mkdir(exist_ok=True)

    # Run demonstrations
    dat_file1 = demonstrate_basic_usage()
    dat_file2 = demonstrate_with_file_paths()
    dat_file3 = demonstrate_json_input()

    # Verify compatibility with the most comprehensive example
    verify_dat_file_compatibility(dat_file2)

    # Show file format
    show_dat_file_format(dat_file2)

    print(f"\n" + "=" * 50)
    print(f"Example completed! Generated files:")
    for dat_file in [dat_file1, dat_file2, dat_file3]:
        if Path(dat_file).exists():
            size = Path(dat_file).stat().st_size
            print(f"  {dat_file} ({size:,} bytes)")

    print(f"\nYou can now test these DAT files with:")
    print(f"  python src/dat_dataframe_normalized.py example_output/enhanced_example.dat")


if __name__ == "__main__":
    main()