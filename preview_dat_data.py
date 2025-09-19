#!/usr/bin/env python3
"""
Enhanced DAT File Data Preview Tool

Advanced script to preview DAT files with detailed data content analysis,
parsing problem detection, and comprehensive reporting capabilities.
"""

import sys
import pandas as pd
import chardet
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re

def detect_encoding(file_path):
    """Detect file encoding with detailed reporting."""
    encodings_tried = []

    # Try chardet first
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            detection = chardet.detect(raw_data)

        if detection and detection['confidence'] > 0.7:
            encodings_tried.append({
                'encoding': detection['encoding'],
                'confidence': detection['confidence'],
                'method': 'chardet'
            })
            return detection['encoding'], encodings_tried
    except Exception as e:
        encodings_tried.append({
            'encoding': 'chardet_failed',
            'error': str(e),
            'method': 'chardet'
        })

    # Fallback encodings
    fallback_encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in fallback_encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1000)
            encodings_tried.append({
                'encoding': encoding,
                'status': 'success',
                'method': 'fallback'
            })
            return encoding, encodings_tried
        except Exception as e:
            encodings_tried.append({
                'encoding': encoding,
                'status': 'failed',
                'error': str(e),
                'method': 'fallback'
            })

    return 'utf-8', encodings_tried

def detect_parsing_problems(df, delimiter, encoding, file_path):
    """Detect potential parsing problems."""
    problems = []

    # Check for too many or too few columns
    if len(df.columns) == 1 and delimiter != 'single_column':
        problems.append({
            'type': 'single_column_detected',
            'severity': 'high',
            'description': f'Data loaded as single column despite using delimiter "{delimiter}". May indicate wrong delimiter.',
            'suggestion': 'Try different delimiters or check if file is actually single-column format.'
        })

    # Check for unnamed columns
    unnamed_cols = [col for col in df.columns if 'Unnamed:' in str(col)]
    if unnamed_cols:
        problems.append({
            'type': 'unnamed_columns',
            'severity': 'medium',
            'description': f'Found {len(unnamed_cols)} unnamed columns: {unnamed_cols[:5]}',
            'suggestion': 'May indicate header row issues or delimiter problems.'
        })

    # Check for columns with special characters
    special_char_cols = [col for col in df.columns if any(ord(c) > 127 for c in str(col))]
    if special_char_cols:
        problems.append({
            'type': 'special_characters_in_columns',
            'severity': 'medium',
            'description': f'Columns with special characters: {special_char_cols[:3]}',
            'suggestion': 'May indicate encoding issues or special delimiter characters.'
        })

    # Check for rows with mostly empty data
    if len(df) > 0:
        empty_row_threshold = 0.8  # 80% empty considered problematic
        rows_mostly_empty = 0
        for i in range(min(100, len(df))):  # Check first 100 rows
            row = df.iloc[i]
            empty_ratio = row.isnull().sum() / len(row)
            if empty_ratio > empty_row_threshold:
                rows_mostly_empty += 1

        if rows_mostly_empty > len(df) * 0.3:  # If 30% of rows are mostly empty
            problems.append({
                'type': 'many_empty_rows',
                'severity': 'medium',
                'description': f'{rows_mostly_empty} out of first 100 rows are >80% empty',
                'suggestion': 'May indicate parsing issues or data quality problems.'
            })

    # Check for extremely long column names (may indicate parsing issue)
    long_col_names = [col for col in df.columns if len(str(col)) > 100]
    if long_col_names:
        problems.append({
            'type': 'extremely_long_column_names',
            'severity': 'high',
            'description': f'Found columns with >100 characters in name',
            'suggestion': 'Likely indicates header parsing problem or delimiter issue.'
        })

    # Check for duplicate column names
    col_counts = pd.Series(df.columns).value_counts()
    duplicate_cols = col_counts[col_counts > 1]
    if len(duplicate_cols) > 0:
        problems.append({
            'type': 'duplicate_column_names',
            'severity': 'medium',
            'description': f'Duplicate column names found: {duplicate_cols.to_dict()}',
            'suggestion': 'May indicate header row issues or data structure problems.'
        })

    return problems

def analyze_data_content(df, num_rows=20):
    """Analyze actual data content for insights and problems."""
    analysis = {
        'data_samples': {},
        'data_patterns': {},
        'potential_issues': []
    }

    # Sample data from each column
    for col in df.columns:
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            sample_values = non_null_values.head(5).tolist()
            analysis['data_samples'][col] = {
                'sample_values': sample_values,
                'total_non_null': len(non_null_values),
                'unique_count': df[col].nunique(),
                'data_types_detected': []
            }

            # Try to detect data patterns
            for val in sample_values[:3]:
                val_str = str(val)
                if re.match(r'^\d{4}-\d{2}-\d{2}', val_str):
                    analysis['data_samples'][col]['data_types_detected'].append('date_like')
                elif re.match(r'^\d+$', val_str):
                    analysis['data_samples'][col]['data_types_detected'].append('integer_like')
                elif re.match(r'^\d+\.\d+$', val_str):
                    analysis['data_samples'][col]['data_types_detected'].append('float_like')
                elif '@' in val_str:
                    analysis['data_samples'][col]['data_types_detected'].append('email_like')
                elif 'http' in val_str.lower():
                    analysis['data_samples'][col]['data_types_detected'].append('url_like')

            # Remove duplicates
            analysis['data_samples'][col]['data_types_detected'] = list(set(analysis['data_samples'][col]['data_types_detected']))

    return analysis

def try_different_delimiters(file_path, encoding):
    """Try different delimiters and return results for comparison."""
    delimiters = [
        (',', 'comma'),
        ('\t', 'tab'),
        ('|', 'pipe'),
        (';', 'semicolon'),
        ('þ', 'special_thorn'),
        ('\x14', 'special_0x14'),
        (':', 'colon'),
        (' ', 'space')
    ]

    results = []

    for delimiter, name in delimiters:
        try:
            if delimiter in ['þ', '\x14']:
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter,
                               dtype=str, on_bad_lines='skip', engine='python')
            else:
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter,
                               dtype=str, on_bad_lines='skip')

            # Clean up weird columns and data for special delimiters
            if delimiter in ['þ', '\x14']:
                # First, clean column names by removing delimiter characters
                cleaned_column_mapping = {}
                clean_columns = []

                for col in df.columns:
                    if col.startswith('\x14') or col.startswith('.') or col.startswith('Unnamed:'):
                        continue

                    # Remove þ characters from column names
                    clean_col_name = str(col).replace('þ', '').strip()
                    if clean_col_name:  # Only keep non-empty column names
                        cleaned_column_mapping[col] = clean_col_name
                        clean_columns.append(col)

                if clean_columns:
                    # Select the columns and rename them
                    df = df[clean_columns].copy()
                    df = df.rename(columns=cleaned_column_mapping)

                    # Clean the data values by removing þ characters
                    for col in df.columns:
                        if df[col].dtype == 'object':  # Only clean string columns
                            df[col] = df[col].astype(str).str.replace('þ', '', regex=False)
                            # Also clean up any empty strings that result
                            df[col] = df[col].replace('', None)

            results.append({
                'delimiter': delimiter,
                'name': name,
                'success': True,
                'rows': len(df),
                'columns': len(df.columns),
                'non_empty_cells': df.count().sum(),
                'sample_columns': df.columns[:5].tolist(),
                'dataframe': df
            })

        except Exception as e:
            results.append({
                'delimiter': delimiter,
                'name': name,
                'success': False,
                'error': str(e),
                'dataframe': None
            })

    # Sort by success and number of columns (more columns usually better)
    successful_results = [r for r in results if r['success']]
    if successful_results:
        successful_results.sort(key=lambda x: (x['columns'], x['non_empty_cells']), reverse=True)

    return results, successful_results

def generate_spreadsheet_view(df, filename_base, num_rows=50):
    """Generate a spreadsheet-like view of the data."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    spreadsheet_filename = f"{filename_base}_spreadsheet_{timestamp}.txt"

    with open(spreadsheet_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("SPREADSHEET VIEW - DAT FILE DATA\n")
        f.write("=" * 120 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        f.write(f"Showing first {min(num_rows, len(df))} rows\n")
        f.write("\n")

        # Calculate column widths
        col_widths = {}
        display_columns = df.columns[:20]  # Limit to first 20 columns for readability

        for col in display_columns:
            # Width is max of column name and sample data
            max_data_width = df[col].astype(str).str.len().max() if len(df) > 0 else 0
            col_widths[col] = min(max(len(str(col)), max_data_width, 8), 25)  # Min 8, max 25 chars

        # Header row
        header_line = "| "
        separator_line = "|-"
        for col in display_columns:
            header_line += f"{str(col):<{col_widths[col]}} | "
            separator_line += "-" * col_widths[col] + "-|-"

        f.write(header_line + "\n")
        f.write(separator_line + "\n")

        # Data rows
        for i in range(min(num_rows, len(df))):
            row = df.iloc[i]
            data_line = "| "
            for col in display_columns:
                value = str(row[col]) if pd.notna(row[col]) else ""
                # Truncate if too long
                if len(value) > col_widths[col]:
                    value = value[:col_widths[col]-3] + "..."
                data_line += f"{value:<{col_widths[col]}} | "
            f.write(data_line + "\n")

        if len(df.columns) > 20:
            f.write(f"\nNote: Only showing first 20 columns. Total columns: {len(df.columns)}\n")
            f.write("Additional columns: " + ", ".join(df.columns[20:30]) +
                   (f"... and {len(df.columns) - 30} more" if len(df.columns) > 30 else "") + "\n")

        if len(df) > num_rows:
            f.write(f"\nNote: Only showing first {num_rows} rows. Total rows: {len(df):,}\n")

        f.write("\n" + "=" * 120 + "\n")

    return spreadsheet_filename

def generate_detailed_report(file_path, encoding_info, delimiter_results, best_df, best_delimiter, problems, data_analysis):
    """Generate a comprehensive text report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"dat_analysis_report_{timestamp}.txt"

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("DAT FILE ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"File: {file_path}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"File Size: {Path(file_path).stat().st_size:,} bytes\n")
        f.write("\n")

        # Encoding Analysis
        f.write("ENCODING ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        for enc in encoding_info:
            if enc.get('confidence'):
                f.write(f"  {enc['method']}: {enc['encoding']} (confidence: {enc['confidence']:.2f})\n")
            elif enc.get('status') == 'success':
                f.write(f"  {enc['method']}: {enc['encoding']} - SUCCESS\n")
            else:
                f.write(f"  {enc['method']}: {enc['encoding']} - FAILED ({enc.get('error', 'unknown error')})\n")
        f.write("\n")

        # Delimiter Analysis
        f.write("DELIMITER ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Delimiter':<15} {'Name':<15} {'Success':<8} {'Rows':<8} {'Columns':<8} {'Non-Empty':<10}\n")
        f.write("-" * 70 + "\n")

        for result in delimiter_results:
            delimiter_display = result['delimiter'] if result['delimiter'] != '\t' else '\\t'
            if result['success']:
                f.write(f"{delimiter_display:<15} {result['name']:<15} {'YES':<8} {result['rows']:<8} "
                       f"{result['columns']:<8} {result['non_empty_cells']:<10}\n")
            else:
                f.write(f"{delimiter_display:<15} {result['name']:<15} {'NO':<8} {'N/A':<8} {'N/A':<8} {'ERROR':<10}\n")
        f.write("\n")

        # Best Parsing Result
        f.write("BEST PARSING RESULT:\n")
        f.write("-" * 50 + "\n")
        if best_df is not None:
            f.write(f"Delimiter: {best_delimiter}\n")
            f.write(f"Shape: {best_df.shape[0]:,} rows × {best_df.shape[1]} columns\n")
            f.write(f"Memory usage: {best_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
            f.write("\n")

        # Column Details
        f.write("COLUMN ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        if best_df is not None:
            f.write(f"{'#':<3} {'Column Name':<40} {'Non-Null':<10} {'Unique':<8} {'Sample Values':<30}\n")
            f.write("-" * 100 + "\n")

            for i, col in enumerate(best_df.columns, 1):
                non_null = best_df[col].notna().sum()
                unique_vals = best_df[col].nunique()

                # Get sample values
                sample_vals = best_df[col].dropna().head(3).tolist()
                sample_str = ', '.join([str(v)[:20] + '...' if len(str(v)) > 20 else str(v) for v in sample_vals])
                sample_str = sample_str[:30] + '...' if len(sample_str) > 30 else sample_str

                col_display = col[:40] + '...' if len(col) > 40 else col
                f.write(f"{i:<3} {col_display:<40} {non_null:<10} {unique_vals:<8} {sample_str:<30}\n")
        f.write("\n")

        # Data Content Analysis
        if data_analysis and data_analysis.get('data_samples'):
            f.write("DATA CONTENT ANALYSIS:\n")
            f.write("-" * 50 + "\n")

            for col, info in list(data_analysis['data_samples'].items())[:10]:  # First 10 columns
                f.write(f"\nColumn: {col}\n")
                f.write(f"  Non-null values: {info['total_non_null']}\n")
                f.write(f"  Unique values: {info['unique_count']}\n")
                f.write(f"  Sample data: {info['sample_values']}\n")
                if info['data_types_detected']:
                    f.write(f"  Detected patterns: {', '.join(info['data_types_detected'])}\n")
            f.write("\n")

        # Parsing Problems
        f.write("PARSING PROBLEMS DETECTED:\n")
        f.write("-" * 50 + "\n")
        if problems:
            for i, problem in enumerate(problems, 1):
                f.write(f"{i}. {problem['type'].upper()} (Severity: {problem['severity']})\n")
                f.write(f"   Description: {problem['description']}\n")
                f.write(f"   Suggestion: {problem['suggestion']}\n")
                f.write("\n")
        else:
            f.write("No significant parsing problems detected.\n")
        f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 50 + "\n")
        if best_df is not None:
            if len(best_df.columns) > 20:
                f.write("- File has many columns. Consider using specific column selection for analysis.\n")
            if best_df.isnull().sum().sum() > len(best_df) * len(best_df.columns) * 0.3:
                f.write("- High percentage of null values detected. Consider data cleaning.\n")
            if any('special' in result['name'] for result in delimiter_results if result['success']):
                f.write("- File uses special delimiters. Ensure proper handling in production code.\n")
            f.write("- Use the dat_loader module for consistent parsing in your applications.\n")

        f.write("\n" + "=" * 100 + "\n")

    return report_filename

def preview_dat_data(file_path, num_rows=20, generate_report=True):
    """Enhanced preview with detailed data analysis."""
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    print("=" * 100)
    print(f"ENHANCED DAT FILE ANALYSIS: {Path(file_path).name}")
    print("=" * 100)

    # Get file info
    file_size = Path(file_path).stat().st_size
    print(f"File size: {file_size:,} bytes")

    # Detect encoding
    encoding, encoding_info = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")

    # Try different delimiters
    print(f"\nTesting different delimiters...")
    delimiter_results, successful_results = try_different_delimiters(file_path, encoding)

    # Show delimiter test results
    print(f"\nDelimiter Test Results:")
    print(f"{'Delimiter':<12} {'Name':<15} {'Success':<8} {'Rows':<8} {'Columns':<8}")
    print("-" * 60)
    for result in delimiter_results:
        delimiter_display = result['delimiter'] if result['delimiter'] != '\t' else '\\t'
        if result['success']:
            print(f"{delimiter_display:<12} {result['name']:<15} {'YES':<8} {result['rows']:<8} {result['columns']:<8}")
        else:
            print(f"{delimiter_display:<12} {result['name']:<15} {'NO':<8} {'N/A':<8} {'N/A':<8}")

    if not successful_results:
        print("\nCould not parse file with any delimiter")
        return

    # Use the best result
    best_result = successful_results[0]
    best_df = best_result['dataframe']
    best_delimiter = best_result['name']

    print(f"\nBest parsing result: {best_delimiter} delimiter")
    print(f"Shape: {len(best_df):,} rows × {len(best_df.columns)} columns")

    # Detect parsing problems
    problems = detect_parsing_problems(best_df, best_delimiter, encoding, file_path)

    if problems:
        print(f"\nWARNING: Parsing problems detected ({len(problems)}):")
        for problem in problems:
            severity_mark = "HIGH" if problem['severity'] == 'high' else "MED"
            print(f"  [{severity_mark}] {problem['type']}: {problem['description']}")

    # Analyze data content
    data_analysis = analyze_data_content(best_df, num_rows)

    # Show data preview
    print(f"\nFIRST {min(num_rows, len(best_df))} ROWS WITH DATA:")
    print("=" * 100)

    for i in range(min(num_rows, len(best_df))):
        row = best_df.iloc[i]
        print(f"\nROW {i+1}:")

        # Show all columns with actual data
        for col in best_df.columns[:12]:  # Show up to 12 columns
            value = row[col]
            if pd.notna(value) and str(value).strip():
                # Show full value but truncate display if too long
                value_str = str(value)
                if len(value_str) > 80:
                    display_value = value_str[:80] + f"... ({len(value_str)} chars total)"
                else:
                    display_value = value_str
                print(f"  {col}: {display_value}")

        # Count empty columns
        empty_cols = sum(1 for col in best_df.columns if pd.isna(row[col]) or str(row[col]).strip() == '')
        if empty_cols > 0:
            print(f"  ({empty_cols} empty columns not shown)")

    if len(best_df) > num_rows:
        print(f"\n... and {len(best_df) - num_rows:,} more rows")

    # Column summary
    print(f"\nCOLUMN SUMMARY:")
    print("-" * 80)
    print(f"{'#':<3} {'Column Name':<35} {'Non-Null':<10} {'Unique':<8} {'Data Type Hints':<20}")
    print("-" * 80)

    for i, col in enumerate(best_df.columns, 1):
        non_null = best_df[col].notna().sum()
        unique_vals = best_df[col].nunique()

        # Get data type hints from analysis
        type_hints = ""
        if col in data_analysis['data_samples']:
            hints = data_analysis['data_samples'][col]['data_types_detected']
            type_hints = ', '.join(hints) if hints else "text"

        col_display = col[:35] + '...' if len(col) > 35 else col
        print(f"{i:<3} {col_display:<35} {non_null:<10} {unique_vals:<8} {type_hints:<20}")

        if i >= 20:  # Limit display
            remaining = len(best_df.columns) - 20
            if remaining > 0:
                print(f"... and {remaining} more columns")
            break

    # Generate detailed report and spreadsheet view
    if generate_report:
        print(f"\nGenerating detailed report...")
        report_filename = generate_detailed_report(
            file_path, encoding_info, delimiter_results, best_df,
            best_delimiter, problems, data_analysis
        )
        print(f"Report saved to: {report_filename}")

        # Generate spreadsheet view
        print(f"Generating spreadsheet view...")
        file_base = Path(file_path).stem
        spreadsheet_filename = generate_spreadsheet_view(best_df, file_base)
        print(f"Spreadsheet view saved to: {spreadsheet_filename}")

    print(f"\n" + "=" * 100)

    return best_df, problems, data_analysis

def main():
    if len(sys.argv) < 2:
        print("Usage: python preview_dat_data.py <file.dat> [num_rows]")
        print("Example: python preview_dat_data.py data.dat 30")
        print("\nThis script provides enhanced DAT file analysis with:")
        print("  • Smart delimiter detection")
        print("  • Parsing problem identification")
        print("  • Detailed data content analysis")
        print("  • Comprehensive text report generation")
        sys.exit(1)

    file_path = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    try:
        df, problems, analysis = preview_dat_data(file_path, num_rows, generate_report=True)

        print(f"\nTIP: Use the generated report file for detailed analysis!")
        print(f"TIP: Load in Python with: from dat_loader import load_dat_file; df = load_dat_file('{file_path}')")

    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()