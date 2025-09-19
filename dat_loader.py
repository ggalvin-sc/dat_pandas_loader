"""
DAT File Loader Module

A simple, importable module for loading DAT files into pandas DataFrames.
This module provides both simple and advanced loading capabilities with optional
logging and data integrity tracking.

Usage:
    from dat_loader import load_dat_file
    df = load_dat_file('my_file.dat')
"""

import pandas as pd
import numpy as np
import chardet
import warnings
from typing import Optional, Dict, Any, List, Tuple
import logging
import os

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)

class DATLoader:
    """A robust DAT file loader with multiple fallback strategies."""

    def __init__(self, verbose: bool = False):
        """
        Initialize the DAT loader.

        Args:
            verbose: If True, enables detailed logging output
        """
        self.verbose = verbose
        self.logger = self._setup_logger() if verbose else None
        self.last_method_used = None

    def _setup_logger(self) -> logging.Logger:
        """Setup simple logger for verbose output."""
        logger = logging.getLogger('dat_loader')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _log(self, message: str, level: str = 'info'):
        """Log message if verbose mode is enabled."""
        if self.verbose and self.logger:
            getattr(self.logger, level)(message)

    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding with fallback strategies.

        Args:
            file_path: Path to the file

        Returns:
            str: The detected or fallback encoding
        """
        self._log(f"Detecting encoding for: {file_path}")

        # Try chardet first
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                detection = chardet.detect(raw_data)

            if detection and detection['confidence'] > 0.7:
                encoding = detection['encoding']
                self._log(f"Detected encoding: {encoding} (confidence: {detection['confidence']:.2f})")
                return encoding
        except Exception as e:
            self._log(f"Chardet encoding detection failed: {e}", 'warning')

        # Fallback encodings to try
        fallback_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']

        for encoding in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # Try to read first 1000 characters
                self._log(f"Successfully validated encoding: {encoding}")
                return encoding
            except UnicodeDecodeError:
                continue

        # Last resort
        self._log("Could not detect encoding, using utf-8 with error handling", 'warning')
        return 'utf-8'

    def _load_pandas_standard(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using pandas with standard CSV detection."""
        return pd.read_csv(
            file_path,
            encoding=encoding,
            on_bad_lines='skip',
            low_memory=False,
            dtype=str
        )

    def _load_pandas_tab(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using pandas with tab delimiter."""
        return pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter='\t',
            on_bad_lines='skip',
            low_memory=False,
            dtype=str
        )

    def _load_pandas_special_delimiter(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using pandas with special delimiters (þ, \x14)."""
        special_delimiters = ['þ', '\x14']

        for delimiter in special_delimiters:
            try:
                self._log(f"Trying special delimiter: '{delimiter}'")
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    on_bad_lines='skip',
                    dtype=str,
                    engine='python'  # Required for multi-char delimiters
                )

                self._log(f"Special delimiter '{delimiter}' resulted in {len(df)} rows and {len(df.columns)} columns")

                if len(df.columns) > 1:  # Successfully parsed into multiple columns
                    # Clean column names by removing delimiter characters
                    cleaned_columns = {}
                    for col in df.columns:
                        # Clean up special characters and whitespace
                        clean_name = str(col).replace('þ', '').replace('\x14', '').strip()
                        # Also remove other artifacts
                        clean_name = clean_name.replace('\x00', '').strip()

                        if (clean_name and
                            not clean_name.startswith('Unnamed:') and
                            not clean_name.startswith('.') and  # Skip ".1", ".2" artifact columns
                            len(clean_name) > 0):
                            cleaned_columns[col] = clean_name

                    if len(cleaned_columns) > 1:  # We have multiple valid columns
                        # Select only the columns we can clean
                        df = df[[col for col in df.columns if col in cleaned_columns]].copy()
                        df = df.rename(columns=cleaned_columns)

                        # Clean data values
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                df[col] = df[col].astype(str).str.replace('þ', '', regex=False)
                                df[col] = df[col].str.replace('\x14', '', regex=False)
                                df[col] = df[col].str.replace('\x00', '', regex=False)
                                # Replace empty strings with None
                                df[col] = df[col].replace('', None)
                                df[col] = df[col].replace('nan', None)

                        self._log(f"Successfully loaded with pandas_special_delimiter (delimiter: '{delimiter}', {len(df.columns)} clean columns)")
                        return df

            except Exception as e:
                self._log(f"Special delimiter '{delimiter}' failed: {e}")
                continue

        raise Exception("No special delimiter worked")

    def _load_pandas_fixed_width(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using pandas with fixed-width detection."""
        return pd.read_fwf(
            file_path,
            encoding=encoding,
            dtype=str
        )

    def _load_pandas_csv_sniffer(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using pandas with CSV dialect detection."""
        import csv

        # Detect delimiter
        with open(file_path, 'r', encoding=encoding) as f:
            sample = f.read(1024)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)

        return pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=dialect.delimiter,
            quotechar=dialect.quotechar,
            on_bad_lines='skip',
            low_memory=False,
            dtype=str
        )

    def _load_manual_parsing(self, file_path: str, encoding: str) -> pd.DataFrame:
        """Load using manual line-by-line parsing."""
        rows = []
        max_cols = 0

        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
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

        return pd.DataFrame(rows, columns=columns)

    def load(self, file_path: str, encoding: Optional[str] = None, clean: bool = False) -> pd.DataFrame:
        """
        Load a DAT file into a pandas DataFrame using multiple fallback strategies.

        Args:
            file_path: Path to the DAT file
            encoding: Force specific encoding (auto-detected if None)
            clean: If True, perform basic data cleaning

        Returns:
            pd.DataFrame: The loaded data

        Raises:
            Exception: If all loading methods fail
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self._log(f"Loading DAT file: {file_path}")

        # Detect encoding if not provided
        if encoding is None:
            encoding = self.detect_encoding(file_path)

        # Define loading methods to try - put special delimiter first for files that need it
        load_methods = [
            ('pandas_special_delimiter', self._load_pandas_special_delimiter),
            ('pandas_standard', self._load_pandas_standard),
            ('pandas_tab_delimited', self._load_pandas_tab),
            ('pandas_fixed_width', self._load_pandas_fixed_width),
            ('pandas_csv_sniffer', self._load_pandas_csv_sniffer),
            ('manual_parsing', self._load_manual_parsing)
        ]

        # Try each loading method
        last_error = None
        for method_name, method_func in load_methods:
            try:
                self._log(f"Trying loading method: {method_name}")
                df = method_func(file_path, encoding)

                if df is not None and not df.empty:
                    self.last_method_used = method_name
                    self._log(f"✅ Successfully loaded with {method_name}")
                    self._log(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

                    if clean:
                        df = self._clean_data(df)

                    return df

            except Exception as e:
                last_error = e
                self._log(f"Method {method_name} failed: {str(e)}", 'debug')
                continue

        # If we get here, all methods failed
        error_msg = f"All loading methods failed. Last error: {last_error}"
        raise Exception(error_msg)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning operations."""
        self._log("Performing data cleaning")
        original_rows = len(df)

        # Remove whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip()

        # Remove completely empty rows
        df = df.dropna(how='all')

        # Remove duplicates
        df = df.drop_duplicates()

        cleaned_rows = len(df)
        if original_rows != cleaned_rows:
            self._log(f"Cleaning removed {original_rows - cleaned_rows:,} rows ({original_rows:,} → {cleaned_rows:,})")

        return df


# Convenience functions for easy importing
def load_dat_file(file_path: str, encoding: Optional[str] = None,
                  clean: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Load a DAT file into a pandas DataFrame.

    This is the main function you'll use to load DAT files from anywhere in your code.

    Args:
        file_path: Path to the DAT file
        encoding: Force specific encoding (auto-detected if None)
        clean: If True, perform basic data cleaning (remove empty rows, duplicates, whitespace)
        verbose: If True, print detailed loading information

    Returns:
        pd.DataFrame: The loaded data

    Example:
        >>> from dat_loader import load_dat_file
        >>> df = load_dat_file('my_data.dat')
        >>> print(df.shape)
        (1000, 25)

        >>> # With cleaning and verbose output
        >>> df = load_dat_file('my_data.dat', clean=True, verbose=True)
        INFO: Loading DAT file: my_data.dat
        INFO: Detected encoding: utf-8 (confidence: 0.95)
        INFO: ✅ Successfully loaded with pandas_standard
        INFO: Loaded 1,000 rows and 25 columns
    """
    loader = DATLoader(verbose=verbose)
    return loader.load(file_path, encoding=encoding, clean=clean)


def quick_load(file_path: str) -> pd.DataFrame:
    """
    Quick load a DAT file with minimal output.

    Args:
        file_path: Path to the DAT file

    Returns:
        pd.DataFrame: The loaded data

    Example:
        >>> from dat_loader import quick_load
        >>> df = quick_load('data.dat')
    """
    return load_dat_file(file_path, verbose=False)


def load_and_clean(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load a DAT file and automatically clean the data.

    Args:
        file_path: Path to the DAT file
        verbose: If True, show cleaning progress

    Returns:
        pd.DataFrame: The loaded and cleaned data

    Example:
        >>> from dat_loader import load_and_clean
        >>> df = load_and_clean('messy_data.dat')
        INFO: Loading DAT file: messy_data.dat
        INFO: ✅ Successfully loaded with pandas_standard
        INFO: Performing data cleaning
        INFO: Cleaning removed 15 rows (1,015 → 1,000)
    """
    return load_dat_file(file_path, clean=True, verbose=verbose)


def get_dat_info(file_path: str) -> Dict[str, Any]:
    """
    Get basic information about a DAT file without fully loading it.

    Args:
        file_path: Path to the DAT file

    Returns:
        Dict containing file information

    Example:
        >>> from dat_loader import get_dat_info
        >>> info = get_dat_info('data.dat')
        >>> print(info)
        {
            'file_size_bytes': 1234567,
            'encoding': 'utf-8',
            'estimated_rows': 5000,
            'sample_lines': ['col1,col2,col3', 'data1,data2,data3']
        }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = DATLoader(verbose=False)

    # Get file size
    file_size = os.path.getsize(file_path)

    # Detect encoding
    encoding = loader.detect_encoding(file_path)

    # Get sample lines and estimate row count
    sample_lines = []
    line_count = 0

    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            for i, line in enumerate(f):
                if i < 5:  # Get first 5 lines as sample
                    sample_lines.append(line.strip())
                line_count += 1

                # For large files, estimate based on first 1000 lines
                if i >= 1000:
                    avg_line_length = f.tell() / (i + 1)
                    estimated_total_lines = int(file_size / avg_line_length)
                    line_count = estimated_total_lines
                    break
    except:
        line_count = 0

    return {
        'file_size_bytes': file_size,
        'encoding': encoding,
        'estimated_rows': line_count,
        'sample_lines': sample_lines
    }