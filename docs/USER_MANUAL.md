# DAT File Loader - User Manual

## 📖 Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Command Line Usage](#command-line-usage)
4. [Python Module Usage](#python-module-usage)
5. [Enhanced Data Analysis](#enhanced-data-analysis)
6. [Spreadsheet Output](#spreadsheet-output)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

## 🚀 Quick Start

### Preview a DAT file (first 20 rows):
```bash
# Enhanced analysis with spreadsheet output
python preview_dat_data.py input.dat

# Quick terminal preview
python dat_processor.py input.dat --preview
```

### Load into Python:
```python
from dat_loader import load_dat_file
df = load_dat_file('input.dat')  # Clean column names and data automatically
print(df.head())
```

### Convert to CSV:
```bash
python dat_processor.py input.dat output.csv
```

## 📦 Installation

1. **Download the files:**
   - `dat_processor.py` - Command line tool
   - `dat_loader.py` - Python module
   - `requirements.txt` - Dependencies

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test installation:**
   ```bash
   python dat_processor.py --help
   ```

## 💻 Command Line Usage

### Basic Syntax
```bash
python dat_processor.py [input_file] [output_file] [options]
```

### Available Commands

#### 🔍 **Preview Mode** (Show first 20 rows)
```bash
# Quick preview
python dat_processor.py input.dat --preview

# Preview with verbose output
python dat_processor.py input.dat --preview --verbose

# Preview with cleaning
python dat_processor.py input.dat --preview --clean
```

#### 📊 **Exploration Mode** (Detailed analysis)
```bash
# Explore data structure
python dat_processor.py input.dat --explore

# Explore with verbose logging
python dat_processor.py input.dat --explore --verbose
```

#### 📋 **Report Only** (Data integrity analysis)
```bash
# Generate integrity report
python dat_processor.py input.dat --report-only

# Detailed report with debug info
python dat_processor.py input.dat --report-only --verbose
```

#### 💾 **Convert Files**
```bash
# Convert to CSV
python dat_processor.py input.dat output.csv

# Convert to Excel
python dat_processor.py input.dat output.xlsx

# Convert to Parquet
python dat_processor.py input.dat output.parquet

# Convert with data cleaning
python dat_processor.py input.dat output.csv --clean

# Convert with verbose logging
python dat_processor.py input.dat output.csv --clean --verbose
```

#### ⚙️ **Advanced Options**
```bash
# Force specific encoding
python dat_processor.py input.dat output.csv --encoding utf-8

# Complete processing pipeline
python dat_processor.py input.dat output.csv --clean --verbose
```

### Command Options Reference

| Option | Description | Example |
|--------|-------------|---------|
| `--preview` | Show first 20 rows in terminal | `--preview` |
| `--explore` | Detailed data analysis without saving | `--explore` |
| `--report-only` | Generate integrity report only | `--report-only` |
| `--clean` | Perform data cleaning operations | `--clean` |
| `--verbose` | Enable detailed logging | `--verbose` |
| `--encoding` | Force specific file encoding | `--encoding utf-8` |

## 🐍 Python Module Usage

### Import the Module
```python
from dat_loader import load_dat_file, quick_load, load_and_clean, get_dat_info
```

## 🔍 Enhanced Data Analysis

### Advanced Preview Script - `preview_dat_data.py`
This is the most powerful analysis tool that provides comprehensive data insights:

```bash
# Full analysis with spreadsheet output
python preview_dat_data.py input.dat

# Analyze first 30 rows
python preview_dat_data.py input.dat 30

# Generates two files:
# 1. dat_analysis_report_TIMESTAMP.txt - Complete analysis
# 2. FILENAME_spreadsheet_TIMESTAMP.txt - Spreadsheet view
```

**What You Get:**
- **Delimiter Testing**: Tests 8 different delimiters automatically
- **Encoding Detection**: Automatic encoding detection with fallbacks
- **Parsing Problem Detection**: Identifies issues and suggests solutions
- **Data Content Analysis**: Shows actual data values, not just structure
- **Spreadsheet Output**: Clean tabular view of your data
- **Column Analysis**: Data types, null counts, unique values
- **Memory Usage**: DataFrame memory consumption

### Sample Output:
```
====================================================================================================
ENHANCED DAT FILE ANALYSIS: your_file.dat
====================================================================================================
File size: 120,042 bytes
Detected encoding: UTF-16

Testing different delimiters...

Delimiter Test Results:
Delimiter    Name            Success  Rows     Columns
------------------------------------------------------------
,            comma           YES      36       1
þ            special_thorn   YES      83       26      ← BEST
|            pipe            YES      83       1

Best parsing result: special_thorn delimiter
Shape: 83 rows × 26 columns

FIRST 20 ROWS WITH DATA:
====================================================================================================

ROW 1:
  ProdBeg: 3M_WFDPD_00000022
  ProdEnd: 3M_WFDPD_00000023
  Custodian: Keown_John
  Title: Distribution Code: 91-220 Material...

COLUMN SUMMARY:
--------------------------------------------------------------------------------
#   Column Name                         Non-Null   Unique   Data Type Hints
--------------------------------------------------------------------------------
1   ProdBeg                             83         83       text
2   ProdEnd                             83         83       text
3   Custodian                           83         17       text

Generating detailed report...
Report saved to: dat_analysis_report_20250919_114556.txt
Generating spreadsheet view...
Spreadsheet view saved to: filename_spreadsheet_20250919_114556.txt
```

## 📊 Spreadsheet Output

The enhanced preview automatically generates a spreadsheet-like text file that looks like this:

```
========================================================================================================================
SPREADSHEET VIEW - DAT FILE DATA
========================================================================================================================
Generated: 2025-09-19 11:45:56
Shape: 83 rows × 26 columns
Showing first 50 rows

| ProdBeg           | ProdEnd           | Custodian         | Title                     |
|-------------------|-------------------|-------------------|---------------------------|
| 3M_WFDPD_00000022 | 3M_WFDPD_00000023 | Keown_John        | Distribution Code: 91-... |
| 3M_WFDPD_00000024 | 3M_WFDPD_00000025 | Keown_John        | Material: Pressure Se... |
| 3M_WFDPD_00000026 | 3M_WFDPD_00000026 | Keown_John        | Adhesive Label Class...  |
```

**Features:**
- **Table Format**: Clean rows and columns with proper alignment
- **Automatic Column Sizing**: Columns sized based on content
- **Truncation**: Long values truncated with "..." indicator
- **First 20 Columns**: Shows most important columns for readability
- **Complete Data**: Up to 50 rows by default (configurable)
- **Additional Column List**: Shows remaining column names if truncated

### Basic Loading
```python
# Simple load
df = load_dat_file('data.dat')
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

# Quick load (no output)
df = quick_load('data.dat')

# Load with cleaning and verbose output
df = load_dat_file('data.dat', clean=True, verbose=True)
```

### Get File Information
```python
# Check file before loading
info = get_dat_info('large_file.dat')
print(f"File size: {info['file_size_bytes']:,} bytes")
print(f"Estimated rows: {info['estimated_rows']:,}")
print(f"Encoding: {info['encoding']}")
```

### Load and Clean
```python
# Automatic cleaning
df = load_and_clean('messy_data.dat')

# Manual cleaning options
df = load_dat_file('data.dat', clean=True, verbose=True)
```

### Error Handling
```python
try:
    df = load_dat_file('problematic_file.dat', verbose=True)
    print(f"Successfully loaded {len(df)} rows")
except FileNotFoundError:
    print("File not found!")
except Exception as e:
    print(f"Loading failed: {e}")
```

## 🔧 Advanced Features

### Data Integrity Tracking
The tool automatically tracks:
- **Data Loss**: Records skipped lines and reasons
- **Encoding Changes**: Logs when file encoding is changed
- **Data Modifications**: Tracks cleaning operations
- **Memory Usage**: Monitors DataFrame memory consumption

### Multiple Loading Strategies
Automatically tries 6 different loading methods in optimized order:
1. **pandas_special_delimiter**: Special delimiters (þ, \x14) with automatic cleaning
2. **pandas_standard**: Standard CSV detection
3. **pandas_tab_delimited**: Tab-separated values
4. **pandas_fixed_width**: Fixed-width format detection
5. **pandas_csv_sniffer**: Automatic delimiter detection
6. **manual_parsing**: Line-by-line parsing with multiple delimiters

### Special Character Handling
The system now automatically handles complex delimiter characters:
- **þ (thorn) character**: Common in legal/FOIA document productions
- **\x14 character**: Control character used in some export formats
- **Automatic cleaning**: Removes delimiter artifacts from column names and data
- **Clean output**: `þProdBegþ` becomes `ProdBeg`, `þ3M_WFDPD_00000022þ` becomes `3M_WFDPD_00000022`

### Data Cleaning Operations
When using `--clean` or `clean=True`:
- **Whitespace removal**: Strips leading/trailing spaces
- **Empty row removal**: Removes completely blank rows
- **Duplicate removal**: Removes identical rows
- **Encoding fixes**: Repairs common encoding artifacts

### Comprehensive Logging
- **File logs**: Timestamped log files for each run
- **Console output**: Real-time progress updates
- **Debug mode**: Detailed processing information
- **Error tracking**: Complete error context and stack traces

## 🔧 Troubleshooting

### Common Issues

#### **1. File Not Found**
```
Error: File not found: input.dat
```
**Solution:** Check file path and ensure file exists
```bash
ls -la input.dat
python dat_processor.py "C:\full\path\to\input.dat" --preview
```

#### **2. Encoding Issues**
```
Error: 'utf-8' codec can't decode byte
```
**Solution:** Let the tool auto-detect encoding or force a specific one
```bash
python dat_processor.py input.dat --preview --verbose
python dat_processor.py input.dat output.csv --encoding latin-1
```

#### **3. Empty or Malformed File**
```
Error: All loading methods failed
```
**Solution:** Use verbose mode to see what's happening
```bash
python dat_processor.py input.dat --preview --verbose
```

#### **4. Memory Issues with Large Files**
```
Error: Memory error
```
**Solution:** Use preview mode first, then process in chunks
```bash
# Check file size first
python dat_processor.py large_file.dat --report-only

# Preview to understand structure
python dat_processor.py large_file.dat --preview
```

#### **5. Permission Denied**
```
Error: Permission denied
```
**Solution:** Check file permissions and run with appropriate rights
```bash
chmod 644 input.dat
# Or run as administrator on Windows
```

### Debug Mode
Use `--verbose` for detailed troubleshooting:
```bash
python dat_processor.py input.dat --preview --verbose
```

This shows:
- Encoding detection process
- Each loading method attempted
- Detailed error messages
- File statistics and processing steps

## 📚 Examples

### Example 1: Quick File Preview
```bash
# See what's in the file
python dat_processor.py "EPA_Data.dat" --preview
```

### Example 2: Data Quality Check
```bash
# Analyze data quality
python dat_processor.py "survey_data.dat" --explore --verbose
```

### Example 3: Clean and Convert
```bash
# Clean data and save as Excel
python dat_processor.py "raw_data.dat" "clean_data.xlsx" --clean --verbose
```

### Example 4: Python Integration
```python
# In your Python script
from dat_loader import load_dat_file, get_dat_info

# Check file first
info = get_dat_info('customer_data.dat')
print(f"File has ~{info['estimated_rows']:,} rows")

# Load and analyze
df = load_dat_file('customer_data.dat', clean=True)
print("Data summary:")
print(df.describe())

# Save processed data
df.to_csv('processed_customers.csv', index=False)
```

### Example 5: Batch Processing
```python
# Process multiple files
import glob
from dat_loader import load_dat_file

dat_files = glob.glob('data/*.dat')
for file_path in dat_files:
    try:
        df = load_dat_file(file_path, clean=True)
        output_name = file_path.replace('.dat', '_processed.csv')
        df.to_csv(output_name, index=False)
        print(f"✅ Processed {file_path} -> {output_name}")
    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")
```

### Example 6: Complex Analysis
```python
from dat_loader import load_dat_file
import pandas as pd

# Load and analyze
df = load_dat_file('transaction_data.dat', clean=True, verbose=True)

# Data analysis
print("Dataset Overview:")
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print(f"  Missing values: {df.isnull().sum().sum():,}")

# Column analysis
print("\nColumn Information:")
for col in df.columns:
    non_null = df[col].count()
    unique_vals = df[col].nunique()
    print(f"  {col}: {non_null:,} non-null, {unique_vals:,} unique")

# Save report
summary = {
    'total_rows': len(df),
    'total_columns': len(df.columns),
    'missing_values': df.isnull().sum().to_dict(),
    'column_types': df.dtypes.astype(str).to_dict()
}

import json
with open('data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

## 📋 Quick Reference Card

### Most Common Commands
```bash
# Preview file
python dat_processor.py file.dat --preview

# Convert to CSV
python dat_processor.py file.dat output.csv

# Clean and convert
python dat_processor.py file.dat output.csv --clean

# Detailed analysis
python dat_processor.py file.dat --explore

# Python usage
from dat_loader import load_dat_file
df = load_dat_file('file.dat')
```

### File Formats Supported
- **Input**: .dat files (any delimiter, encoding)
- **Output**: .csv, .xlsx, .parquet

### Key Features
- ✅ Automatic encoding detection
- ✅ Multiple loading strategies
- ✅ Data integrity tracking
- ✅ Comprehensive logging
- ✅ Data cleaning operations
- ✅ Memory usage monitoring
- ✅ Error recovery and reporting

---

## 🆘 Need Help?

1. **Check the verbose output:**
   ```bash
   python dat_processor.py your_file.dat --preview --verbose
   ```

2. **Review the log files:**
   - Look for `dat_processor_YYYYMMDD_HHMMSS.log` files

3. **Try the Python module:**
   ```python
   from dat_loader import get_dat_info
   info = get_dat_info('your_file.dat')
   print(info)
   ```

4. **Check file permissions and encoding:**
   ```bash
   file your_file.dat
   ls -la your_file.dat
   ```

For additional support, check the README.md file or review the example_usage.py script for more detailed examples.