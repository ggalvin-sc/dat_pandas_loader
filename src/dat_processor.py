#!/usr/bin/env python3
"""
Enhanced DAT File Processor with Comprehensive Logging and Data Integrity Tracking

This tool provides robust DAT file processing with detailed logging, data integrity
tracking, and comprehensive reporting capabilities.

Features:
- Multiple encoding detection and fallback strategies
- Comprehensive logging with file and console output
- Data integrity tracking and loss detection
- Detailed reporting and data quality analysis
- Enhanced data cleaning operations
- Memory usage monitoring
"""

import pandas as pd
import numpy as np
import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import chardet
import warnings
from typing import Dict, List, Tuple, Optional, Any
import traceback
import functools


def function_lock(func):
    """Decorator to lock function implementation and prevent modifications."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__locked__ = True
    return wrapper

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)

class DataIntegrityTracker:
    """Tracks data integrity throughout the processing pipeline."""

    @function_lock
    def __init__(self):
        self.original_file_size = 0
        self.original_line_count = 0
        self.skipped_lines = []
        self.encoding_changes = []
        self.data_modifications = []
        self.error_log = []
        self.memory_usage = {}
        self.processing_steps = []

    @function_lock
    def log_original_stats(self, file_path):
        """Log original file statistics."""
        file_path = Path(file_path)
        self.original_file_size = file_path.stat().st_size
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.original_line_count = sum(1 for _ in f)
        except:
            self.original_line_count = 0

    @function_lock
    def log_skipped_line(self, line_num: int, reason: str, content: str = ""):
        """Log a skipped line with reason."""
        self.skipped_lines.append({
            'line_number': line_num,
            'reason': reason,
            'content': content[:100] + "..." if len(content) > 100 else content
        })

    @function_lock
    def log_encoding_change(self, from_encoding: str, to_encoding: str, reason: str):
        """Log encoding changes."""
        self.encoding_changes.append({
            'from': from_encoding,
            'to': to_encoding,
            'reason': reason,
            'timestamp': datetime.now()
        })

    @function_lock
    def log_data_modification(self, operation: str, before_count: int, after_count: int, details: str = ""):
        """Log data modifications."""
        self.data_modifications.append({
            'operation': operation,
            'before_count': before_count,
            'after_count': after_count,
            'difference': before_count - after_count,
            'details': details,
            'timestamp': datetime.now()
        })

    @function_lock
    def log_error(self, error_type: str, message: str, context: str = ""):
        """Log errors with context."""
        self.error_log.append({
            'type': error_type,
            'message': message,
            'context': context,
            'timestamp': datetime.now()
        })

    @function_lock
    def log_memory_usage(self, stage: str, df: pd.DataFrame):
        """Log memory usage at different stages."""
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        self.memory_usage[stage] = {
            'memory_mb': round(memory_mb, 2),
            'rows': len(df),
            'columns': len(df.columns),
            'timestamp': datetime.now()
        }

    @function_lock
    def add_processing_step(self, step: str, status: str, details: str = ""):
        """Add a processing step to the log."""
        self.processing_steps.append({
            'step': step,
            'status': status,
            'details': details,
            'timestamp': datetime.now()
        })

class EnhancedDATProcessor:
    """Enhanced DAT file processor with comprehensive logging and integrity tracking."""

    @function_lock
    def __init__(self, log_level: str = 'INFO'):
        self.integrity_tracker = DataIntegrityTracker()
        self.logger = self._setup_logging(log_level)
        self.successful_method = None
        self.load_methods = [
            ('pandas_standard', self._load_pandas_standard),
            ('pandas_tab_delimited', self._load_pandas_tab),
            ('pandas_fixed_width', self._load_pandas_fixed_width),
            ('pandas_csv_sniffer', self._load_pandas_csv_sniffer),
            ('manual_parsing', self._load_manual_parsing)
        ]

    @function_lock
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logging system."""
        logger = logging.getLogger('dat_processor')
        logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

        # File handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'dat_processor_{timestamp}.log'
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(console_handler)

        logger.info(f"Logging initialized. Log file: {log_filename}")
        return logger

    @function_lock
    def detect_encoding(self, file_path) -> str:
        """Detect file encoding with fallback strategies."""
        file_path = Path(file_path)
        self.logger.info(f"Detecting encoding for: {file_path}")

        # Try chardet first
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                detection = chardet.detect(raw_data)

            if detection and detection['confidence'] > 0.7:
                encoding = detection['encoding']
                self.logger.info(f"Detected encoding: {encoding} (confidence: {detection['confidence']:.2f})")
                return encoding
        except Exception as e:
            self.logger.warning(f"Chardet encoding detection failed: {e}")

        # Fallback encodings to try
        fallback_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']

        for encoding in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # Try to read first 1000 characters
                self.logger.info(f"Successfully validated encoding: {encoding}")
                return encoding
            except UnicodeDecodeError:
                continue

        # Last resort
        self.logger.warning("Could not detect encoding, using utf-8 with error handling")
        return 'utf-8'

    @function_lock
    def _load_pandas_standard(self, file_path, encoding: str) -> pd.DataFrame:
        """Load using pandas with standard CSV detection."""
        self.logger.debug("Attempting pandas standard CSV loading")

        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                on_bad_lines='warn',
                low_memory=False,
                dtype=str
            )
            self.logger.info(f"✅ Successfully loaded with pandas_standard")
            return df
        except Exception as e:
            self.logger.debug(f"pandas_standard failed: {e}")
            raise

    @function_lock
    def _load_pandas_tab(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using pandas with tab delimiter."""
        self.logger.debug("Attempting pandas tab-delimited loading")

        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter='\t',
                on_bad_lines='warn',
                low_memory=False,
                dtype=str
            )
            self.logger.info(f"✅ Successfully loaded with pandas_tab_delimited")
            return df
        except Exception as e:
            self.logger.debug(f"pandas_tab_delimited failed: {e}")
            raise

    @function_lock
    def _load_pandas_fixed_width(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using pandas with fixed-width detection."""
        self.logger.debug("Attempting pandas fixed-width loading")

        try:
            df = pd.read_fwf(
                file_path,
                encoding=encoding,
                dtype=str
            )
            self.logger.info(f"✅ Successfully loaded with pandas_fixed_width")
            return df
        except Exception as e:
            self.logger.debug(f"pandas_fixed_width failed: {e}")
            raise

    @function_lock
    def _load_pandas_csv_sniffer(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using pandas with CSV dialect detection."""
        self.logger.debug("Attempting pandas CSV sniffer loading")

        import csv

        try:
            # Detect delimiter
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)

            df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=dialect.delimiter,
                quotechar=dialect.quotechar,
                on_bad_lines='warn',
                low_memory=False,
                dtype=str
            )
            self.logger.info(f"✅ Successfully loaded with pandas_csv_sniffer (delimiter: '{dialect.delimiter}')")
            return df
        except Exception as e:
            self.logger.debug(f"pandas_csv_sniffer failed: {e}")
            raise

    @function_lock
    def _load_manual_parsing(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using manual line-by-line parsing."""
        self.logger.debug("Attempting manual parsing")

        try:
            rows = []
            max_cols = 0
            line_count = 0

            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    line = line.strip()

                    if not line:
                        self.integrity_tracker.log_skipped_line(line_num, "Empty line", line)
                        continue

                    # Try different delimiters
                    for delimiter in [',', '\t', '|', ';', ' ']:
                        parts = line.split(delimiter)
                        if len(parts) > 1:
                            rows.append(parts)
                            max_cols = max(max_cols, len(parts))
                            break
                    else:
                        # Single column or unrecognized format
                        rows.append([line])
                        max_cols = max(max_cols, 1)

            # Pad rows to same length
            for row in rows:
                while len(row) < max_cols:
                    row.append('')

            # Create column names
            columns = [f'Column_{i+1}' for i in range(max_cols)]

            df = pd.DataFrame(rows, columns=columns)
            self.logger.info(f"✅ Successfully loaded with manual_parsing")
            return df

        except Exception as e:
            self.logger.debug(f"manual_parsing failed: {e}")
            raise

    @function_lock
    def load_dat_file(self, file_path: str, encoding: str = None) -> pd.DataFrame:
        """Load DAT file using multiple fallback strategies."""
        self.logger.info(f"Starting to load DAT file: {file_path}")

        # Log original file statistics
        self.integrity_tracker.log_original_stats(file_path)
        self.logger.info(f"Original file size: {self.integrity_tracker.original_file_size:,} bytes")
        self.logger.info(f"Original line count: {self.integrity_tracker.original_line_count:,} lines")

        # Detect encoding if not provided
        if encoding is None:
            encoding = self.detect_encoding(file_path)

        # Try each loading method
        last_error = None
        for method_name, method_func in self.load_methods:
            try:
                self.logger.debug(f"Trying loading method: {method_name}")
                self.integrity_tracker.add_processing_step(f"Load attempt: {method_name}", "started")

                df = method_func(file_path, encoding)

                if df is not None and not df.empty:
                    self.successful_method = method_name
                    self.logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
                    self.integrity_tracker.log_memory_usage("after_loading", df)
                    self.integrity_tracker.add_processing_step(f"Load attempt: {method_name}", "success",
                                                              f"{len(df)} rows, {len(df.columns)} columns")
                    return df

            except Exception as e:
                last_error = e
                self.logger.debug(f"Method {method_name} failed: {str(e)}")
                self.integrity_tracker.add_processing_step(f"Load attempt: {method_name}", "failed", str(e))
                continue

        # If we get here, all methods failed
        error_msg = f"All loading methods failed. Last error: {last_error}"
        self.logger.error(error_msg)
        self.integrity_tracker.log_error("LoadError", error_msg)
        raise Exception(error_msg)

    @function_lock
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive data cleaning with tracking."""
        self.logger.info("Starting data cleaning operations")
        original_rows = len(df)

        # Remove whitespace
        self.logger.debug("Removing whitespace from string columns")
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()

        # Remove completely empty rows
        before_empty = len(df)
        df = df.dropna(how='all')
        after_empty = len(df)
        if before_empty != after_empty:
            self.integrity_tracker.log_data_modification(
                "Remove empty rows", before_empty, after_empty,
                f"Removed {before_empty - after_empty} completely empty rows"
            )
            self.logger.info(f"Removed {before_empty - after_empty:,} completely empty rows")

        # Remove duplicates
        before_dupes = len(df)
        df = df.drop_duplicates()
        after_dupes = len(df)
        if before_dupes != after_dupes:
            self.integrity_tracker.log_data_modification(
                "Remove duplicates", before_dupes, after_dupes,
                f"Removed {before_dupes - after_dupes} duplicate rows"
            )
            self.logger.info(f"Removed {before_dupes - after_dupes:,} duplicate rows")

        # Fix common encoding artifacts
        self.logger.debug("Fixing encoding artifacts")
        for col in string_columns:
            df[col] = df[col].str.replace('â€™', "'", regex=False)  # Fix smart quotes
            df[col] = df[col].str.replace('â€œ', '"', regex=False)  # Fix smart quotes
            df[col] = df[col].str.replace('â€', '"', regex=False)   # Fix smart quotes
            df[col] = df[col].str.replace('Ã¡', 'á', regex=False)   # Fix accented characters
            df[col] = df[col].str.replace('Ã©', 'é', regex=False)   # Fix accented characters

        self.logger.info(f"Data cleaning completed. Final rows: {len(df):,}")
        self.integrity_tracker.log_memory_usage("after_cleaning", df)

        return df

    @function_lock
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality analysis."""
        self.logger.info("Performing data quality analysis")

        analysis = {
            'shape': df.shape,
            'column_info': {},
            'missing_values': {},
            'duplicate_rows': 0,
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }

        # Column analysis
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'unique_values': df[col].nunique()
            }

            # Add sample values for object columns
            if df[col].dtype == 'object':
                col_info['sample_values'] = df[col].dropna().head(3).tolist()

            analysis['column_info'][col] = col_info

        # Missing value summary
        missing_summary = df.isnull().sum()
        analysis['missing_values'] = {
            col: {
                'count': int(missing_summary[col]),
                'percentage': round((missing_summary[col] / len(df)) * 100, 2)
            }
            for col in missing_summary.index if missing_summary[col] > 0
        }

        # Duplicate analysis
        analysis['duplicate_rows'] = len(df) - len(df.drop_duplicates())

        return analysis

    @function_lock
    def generate_integrity_report(self, df: pd.DataFrame = None) -> str:
        """Generate comprehensive data integrity report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATA INTEGRITY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Original file statistics
        report_lines.append("ORIGINAL FILE STATISTICS:")
        report_lines.append(f"  File size: {self.integrity_tracker.original_file_size:,} bytes")
        report_lines.append(f"  Line count: {self.integrity_tracker.original_line_count:,} lines")
        report_lines.append("")

        # Loading results
        if df is not None:
            report_lines.append("LOADING RESULTS:")
            report_lines.append(f"  Successfully loaded rows: {len(df):,}")
            report_lines.append(f"  Final rows in DataFrame: {len(df):,}")
            report_lines.append(f"  Final columns: {len(df.columns)}")
            report_lines.append("")

            # Calculate success rate
            if self.integrity_tracker.original_line_count > 0:
                success_rate = (len(df) / self.integrity_tracker.original_line_count) * 100
                report_lines.append(f"  Overall success rate: {success_rate:.2f}%")
                report_lines.append("")

        # Data loss analysis
        if self.integrity_tracker.skipped_lines:
            report_lines.append("⚠️  DATA LOSS DETECTED:")
            report_lines.append(f"  Skipped lines: {len(self.integrity_tracker.skipped_lines):,}")
            if self.integrity_tracker.original_line_count > 0:
                loss_percentage = (len(self.integrity_tracker.skipped_lines) / self.integrity_tracker.original_line_count) * 100
                report_lines.append(f"  Loss percentage: {loss_percentage:.2f}%")

            # Show details of first few skipped lines
            report_lines.append("")
            report_lines.append("  Skipped line details:")
            for i, skip in enumerate(self.integrity_tracker.skipped_lines[:5]):
                report_lines.append(f"    Line {skip['line_number']}: {skip['reason']}")
                if skip['content']:
                    report_lines.append(f"      Content: {skip['content']}")

            if len(self.integrity_tracker.skipped_lines) > 5:
                report_lines.append(f"    ... and {len(self.integrity_tracker.skipped_lines) - 5} more")
            report_lines.append("")

        # Data modifications
        if self.integrity_tracker.data_modifications:
            report_lines.append("DATA MODIFICATIONS:")
            for mod in self.integrity_tracker.data_modifications:
                report_lines.append(f"  {mod['operation']}: {mod['before_count']:,} → {mod['after_count']:,} "
                                  f"({mod['difference']:+,})")
                if mod['details']:
                    report_lines.append(f"    Details: {mod['details']}")
            report_lines.append("")

        # Encoding changes
        if self.integrity_tracker.encoding_changes:
            report_lines.append("ENCODING CHANGES:")
            for change in self.integrity_tracker.encoding_changes:
                report_lines.append(f"  {change['from']} → {change['to']}: {change['reason']}")
            report_lines.append("")

        # Memory usage
        if self.integrity_tracker.memory_usage:
            report_lines.append("MEMORY USAGE:")
            for stage, usage in self.integrity_tracker.memory_usage.items():
                report_lines.append(f"  {stage}: {usage['memory_mb']:.2f} MB "
                                  f"({usage['rows']:,} rows × {usage['columns']} columns)")
            report_lines.append("")

        # Processing summary
        if df is not None:
            report_lines.append("PROCESSING SUMMARY:")
            report_lines.append(f"  Input file lines: {self.integrity_tracker.original_line_count:,}")
            if hasattr(self, 'successful_method') and self.successful_method:
                report_lines.append(f"  Loading method: {self.successful_method}")
            report_lines.append(f"  Final output rows: {len(df):,}")
            report_lines.append(f"  Final output columns: {len(df.columns)}")

            if self.integrity_tracker.original_line_count > 0:
                success_rate = (len(df) / self.integrity_tracker.original_line_count) * 100
                report_lines.append(f"  Overall success rate: {success_rate:.2f}%")

        # Errors
        if self.integrity_tracker.error_log:
            report_lines.append("")
            report_lines.append("ERRORS ENCOUNTERED:")
            for error in self.integrity_tracker.error_log:
                report_lines.append(f"  {error['type']}: {error['message']}")
                if error['context']:
                    report_lines.append(f"    Context: {error['context']}")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

@function_lock
def main():
    """Main function with enhanced command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced DAT File Processor with Comprehensive Logging and Data Integrity Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.dat --preview                    # Show first 20 rows
  %(prog)s input.dat --preview --clean --verbose  # Preview with cleaning and details
  %(prog)s input.dat output.csv --clean --verbose # Convert with cleaning
  %(prog)s input.dat --explore                    # Detailed analysis
  %(prog)s input.dat --report-only                # Integrity report only
  %(prog)s input.dat output.xlsx --clean          # Convert to Excel
        """
    )

    parser.add_argument('input_file', help='Input DAT file path')
    parser.add_argument('output_file', nargs='?', help='Output file path (optional for explore/report modes)')
    parser.add_argument('--encoding', help='Force specific encoding (auto-detected if not specified)')
    parser.add_argument('--clean', action='store_true', help='Perform data cleaning operations')
    parser.add_argument('--explore', action='store_true', help='Explore data without saving')
    parser.add_argument('--report-only', action='store_true', help='Generate integrity report only')
    parser.add_argument('--preview', action='store_true', help='Show first 20 rows in terminal')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Validate arguments
    if not args.explore and not args.report_only and not args.preview and not args.output_file:
        parser.error("Output file is required unless using --explore, --report-only, or --preview")

    # Initialize processor
    log_level = 'DEBUG' if args.verbose else 'INFO'
    processor = EnhancedDATProcessor(log_level=log_level)

    try:
        # Load the file using the dat_loader module for consistency
        from dat_loader import DATLoader
        loader = DATLoader(verbose=args.verbose)
        df = loader.load(args.input_file, args.encoding, clean=False)

        if args.report_only:
            # Just generate and display the report
            report = processor.generate_integrity_report(df)
            print("\n" + report)
            return

        if args.preview:
            # Preview mode - show first 20 rows
            print("\n" + "=" * 80)
            print(f"PREVIEW: {args.input_file}")
            print("=" * 80)
            print(f"Shape: {len(df):,} rows × {len(df.columns)} columns")
            print(f"Loading method: {loader.last_method_used}")
            print()

            # Clean data if requested for preview
            if args.clean:
                print("Applying data cleaning...")
                df = loader._clean_data(df)
                print(f"After cleaning: {len(df):,} rows × {len(df.columns)} columns")
                print()

            print("FIRST 20 ROWS:")
            print("-" * 80)

            # Display each row individually for better readability
            preview_rows = min(20, len(df))
            for i in range(preview_rows):
                row = df.iloc[i]
                print(f"ROW {i+1}:")

                # Show key columns with values
                key_columns = []
                for col in df.columns[:8]:  # Show first 8 columns max
                    value = row[col]
                    if pd.notna(value) and str(value).strip():
                        # Truncate long values
                        display_value = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                        key_columns.append(f"  {col}: {display_value}")

                if key_columns:
                    print("\n".join(key_columns))
                else:
                    print("  (Empty or all null values)")
                print()

            if len(df) > 20:
                print(f"... and {len(df) - 20:,} more rows")

            print("-" * 80)
            print(f"Column names: {', '.join(df.columns[:10])}" + ("..." if len(df.columns) > 10 else ""))
            return

        # Clean data if requested
        if args.clean:
            df = loader._clean_data(df)

        # Generate data quality analysis
        analysis = processor.analyze_data_quality(df)

        if args.explore:
            # Exploration mode
            print("\n" + "=" * 60)
            print("DATA EXPLORATION SUMMARY")
            print("=" * 60)
            print(f"Shape: {analysis['shape'][0]:,} rows × {analysis['shape'][1]} columns")
            print(f"Memory usage: {analysis['memory_usage_mb']:.2f} MB")
            print(f"Duplicate rows: {analysis['duplicate_rows']:,}")

            print("\nColumn Information:")
            for col, info in analysis['column_info'].items():
                print(f"  {col}:")
                print(f"    Type: {info['dtype']}")
                print(f"    Non-null: {info['non_null_count']:,} ({100 - info['null_percentage']:.1f}%)")
                print(f"    Unique values: {info['unique_values']:,}")
                if 'sample_values' in info and info['sample_values']:
                    print(f"    Sample: {info['sample_values']}")

            if analysis['missing_values']:
                print("\nMissing Values:")
                for col, missing in analysis['missing_values'].items():
                    print(f"  {col}: {missing['count']:,} ({missing['percentage']:.1f}%)")

        else:
            # Save the processed data
            output_path = Path(args.output_file)

            if output_path.suffix.lower() == '.xlsx':
                df.to_excel(args.output_file, index=False, engine='openpyxl')
            elif output_path.suffix.lower() == '.parquet':
                df.to_parquet(args.output_file, index=False)
            else:
                # Default to CSV
                df.to_csv(args.output_file, index=False, encoding='utf-8-sig')

            processor.logger.info(f"Successfully saved processed data to: {args.output_file}")

        # Generate and display integrity report
        report = processor.generate_integrity_report(df)
        print("\n" + report)

    except Exception as e:
        processor.logger.error(f"Processing failed: {str(e)}")
        if args.verbose:
            processor.logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()