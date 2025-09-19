"""
DAT Pandas Loader - A comprehensive toolkit for loading and processing DAT files into pandas DataFrames.

This package provides tools for:
- Loading DAT files with various delimiters and encodings
- Special delimiter handling (þ character, \x14 control character)
- Data cleaning and integrity tracking
- Comprehensive analysis and reporting
- Multiple export formats

Main modules:
- dat_loader: Core module for loading DAT files into pandas DataFrames
- dat_processor: Command-line processor with comprehensive logging
- preview_dat_data: Enhanced data analysis with spreadsheet output
- dataframe_view: Demonstration script for working with DataFrames

Example usage:
    from src.dat_loader import load_dat_file
    df = load_dat_file('your_file.dat')
"""

__version__ = "1.0.0"
__author__ = "DAT Loader Project"

# Import main functions for easy access
from .dat_loader import load_dat_file, quick_load, load_and_clean, get_dat_info

__all__ = [
    'load_dat_file',
    'quick_load',
    'load_and_clean',
    'get_dat_info'
]