# Enhanced DAT File Processor

A comprehensive DAT file processing tool with detailed logging, data integrity tracking, and robust error handling.

## 🔍 Features

### **Comprehensive Logging**
- **File logging**: Creates timestamped log files for each run
- **Console logging**: Real-time progress updates
- **Multiple log levels**: INFO (default) and DEBUG (--verbose)
- **Error tracking**: Detailed error messages with context

### **Data Integrity Tracking**
- **Loss Detection**: Tracks exactly what data was lost and why
- **Line-by-line tracking**: Records skipped lines with reasons
- **Encoding changes**: Logs when encodings are changed
- **Data modifications**: Tracks changes from cleaning operations

### **Detailed Reporting**
- **Before/After comparison**: Shows original vs final row counts
- **Success rate calculation**: Percentage of data successfully processed
- **Missing value analysis**: Detailed breakdown of null values
- **Duplicate detection**: Finds and reports duplicate rows
- **Memory usage**: Shows DataFrame memory consumption

### **Robust Loading**
- **Multiple fallback strategies**: 5 different loading methods
- **Automatic encoding detection**: Uses chardet with fallbacks
- **Error recovery**: Continues processing when possible
- **Format flexibility**: Handles CSV, TSV, fixed-width, and custom formats

## 📦 Installation

1. Clone or download the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🛠 Usage

### **As a Python Module (Recommended)**

Import and use the DAT loader directly in your code:

```python
# Simple usage - load any DAT file into a pandas DataFrame
from dat_loader import load_dat_file
df = load_dat_file('my_file.dat')
print(df.shape)  # (1000, 25)

# With cleaning and verbose output
df = load_dat_file('my_file.dat', clean=True, verbose=True)

# Quick load without any output
from dat_loader import quick_load
df = quick_load('my_file.dat')

# Load and automatically clean
from dat_loader import load_and_clean
df = load_and_clean('my_file.dat')

# Get file info without loading
from dat_loader import get_dat_info
info = get_dat_info('my_file.dat')
print(f"File has {info['estimated_rows']} rows")
```

### **Command Line Processing**
```bash
# Simple conversion
python dat_processor.py input.dat output.csv

# With data cleaning
python dat_processor.py input.dat output.csv --clean

# Verbose logging
python dat_processor.py input.dat output.csv --clean --verbose
```

### **Exploration Mode**
```bash
# Explore data without saving
python dat_processor.py input.dat --explore

# Explore with verbose logging
python dat_processor.py input.dat --explore --verbose
```

### **Report Generation**
```bash
# Generate integrity report only
python dat_processor.py input.dat --report-only

# Report with verbose details
python dat_processor.py input.dat --report-only --verbose
```

### **Output Formats**
```bash
# CSV output (default)
python dat_processor.py input.dat output.csv

# Excel output
python dat_processor.py input.dat output.xlsx

# Parquet output
python dat_processor.py input.dat output.parquet
```

### **Advanced Options**
```bash
# Force specific encoding
python dat_processor.py input.dat output.csv --encoding utf-8

# Complete processing pipeline
python dat_processor.py input.dat output.csv --clean --verbose
```

## 📊 Sample Output

### **During Processing:**
```
2024-09-19 10:15:30 - INFO - Starting to load DAT file: input.dat
2024-09-19 10:15:30 - INFO - Original file size: 1,234,567 bytes
2024-09-19 10:15:30 - INFO - Original line count: 5,000 lines
2024-09-19 10:15:30 - INFO - Detecting encoding for: input.dat
2024-09-19 10:15:30 - INFO - Detected encoding: utf-8 (confidence: 0.95)
2024-09-19 10:15:31 - WARNING - Skipped line 245: Parse error
2024-09-19 10:15:31 - INFO - ✅ Successfully loaded with pandas_standard
2024-09-19 10:15:31 - INFO - Loaded 4,998 rows and 25 columns
2024-09-19 10:15:31 - INFO - Starting data cleaning operations
2024-09-19 10:15:31 - INFO - Removed 3 completely empty rows
2024-09-19 10:15:31 - INFO - Removed 2 duplicate rows
2024-09-19 10:15:31 - INFO - Data cleaning completed. Final rows: 4,993
2024-09-19 10:15:32 - INFO - Successfully saved processed data to: output.csv
```

### **Final Integrity Report:**
```
================================================================================
DATA INTEGRITY REPORT
================================================================================

ORIGINAL FILE STATISTICS:
  File size: 1,234,567 bytes
  Line count: 5,000 lines

LOADING RESULTS:
  Successfully loaded rows: 4,998
  Final rows in DataFrame: 4,993
  Final columns: 25

⚠️  DATA LOSS DETECTED:
  Skipped lines: 2
  Loss percentage: 0.04%

  Skipped line details:
    Line 245: Parse error
      Content: ,,,,malformed data...
    Line 1847: Empty line

DATA MODIFICATIONS:
  Remove empty rows: 4,998 → 4,995 (-3)
    Details: Removed 3 completely empty rows
  Remove duplicates: 4,995 → 4,993 (-2)
    Details: Removed 2 duplicate rows

MEMORY USAGE:
  after_loading: 15.47 MB (4,998 rows × 25 columns)
  after_cleaning: 15.21 MB (4,993 rows × 25 columns)

PROCESSING SUMMARY:
  Input file lines: 5,000
  Loading method: pandas_standard
  Final output rows: 4,993
  Final output columns: 25
  Overall success rate: 99.86%
================================================================================
```

### **Exploration Mode Output:**
```
============================================================
DATA EXPLORATION SUMMARY
============================================================
Shape: 4,993 rows × 25 columns
Memory usage: 15.21 MB
Duplicate rows: 0

Column Information:
  ID:
    Type: object
    Non-null: 4,993 (100.0%)
    Unique values: 4,993
    Sample: ['001', '002', '003']

  Name:
    Type: object
    Non-null: 4,987 (99.9%)
    Unique values: 4,823
    Sample: ['John Smith', 'Jane Doe', 'Bob Johnson']

  Amount:
    Type: object
    Non-null: 4,956 (99.3%)
    Unique values: 892
    Sample: ['100.50', '250.00', '75.25']

Missing Values:
  Name: 6 (0.1%)
  Amount: 37 (0.7%)
  Address: 156 (3.1%)
```

## 🔧 Data Cleaning Operations

When using the `--clean` flag, the processor performs:

1. **Whitespace removal**: Strips leading/trailing spaces from all text fields
2. **Empty row removal**: Removes rows that are completely blank
3. **Duplicate removal**: Removes identical rows based on all columns
4. **Encoding fixes**: Repairs common encoding artifacts like smart quotes

## 📝 Log Files

Every run creates a timestamped log file (e.g., `dat_processor_20240919_101530.log`) containing:
- Complete processing history
- Debug information (when using --verbose)
- Error details and stack traces
- Performance metrics

## 🔍 Loading Strategies

The processor tries multiple loading methods in order:

1. **pandas_standard**: Standard CSV detection
2. **pandas_tab_delimited**: Tab-separated values
3. **pandas_fixed_width**: Fixed-width format detection
4. **pandas_csv_sniffer**: Automatic delimiter detection
5. **manual_parsing**: Line-by-line parsing with multiple delimiter attempts

## 🚨 Error Handling

- **Graceful degradation**: Falls back through multiple loading strategies
- **Detailed error logging**: Records why each method failed
- **Data preservation**: Attempts to salvage as much data as possible
- **User feedback**: Clear messages about what went wrong and what was recovered

## 📋 Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- chardet >= 4.0.0
- openpyxl >= 3.0.9 (for Excel output)
- pyarrow >= 10.0.0 (for Parquet output)

## 🐍 Python Module Usage Examples

### Basic Integration
```python
import pandas as pd
from dat_loader import load_dat_file

# Load your DAT file
df = load_dat_file('data.dat')

# Now you have a regular pandas DataFrame
print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

# Use all normal pandas operations
filtered_df = df[df['amount'] > 100]
summary = df.groupby('category').sum()
df.to_csv('processed_data.csv')
```

### Data Analysis Workflow
```python
from dat_loader import load_dat_file, get_dat_info

# Check file first
info = get_dat_info('large_file.dat')
print(f"File has ~{info['estimated_rows']:,} rows")

# Load and analyze
df = load_dat_file('large_file.dat', clean=True, verbose=True)

# Standard pandas analysis
print("Data summary:")
print(df.describe())
print(f"Missing values: {df.isnull().sum().sum()}")

# Your analysis here...
result = df.groupby('category')['amount'].agg(['sum', 'mean', 'count'])
```

### Error Handling
```python
from dat_loader import load_dat_file

try:
    df = load_dat_file('problematic_file.dat', verbose=True)
    print(f"Successfully loaded {len(df)} rows")
except FileNotFoundError:
    print("File not found!")
except Exception as e:
    print(f"Loading failed: {e}")
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.