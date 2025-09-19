# DAT Pandas Loader

A comprehensive toolkit for loading and processing DAT files into pandas DataFrames with special delimiter support and comprehensive analysis features.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
# Import the main loading function
from src.dat_loader import load_dat_file

# Load your DAT file (automatically handles special delimiters like þ)
df = load_dat_file('your_file.dat')
print(df.head())
```

### Command Line Usage
```bash
# Quick preview of your data
python src/preview_dat_data.py your_file.dat

# Convert to CSV
python src/dat_processor.py your_file.dat output.csv

# See all data analysis features
python src/dataframe_view.py your_file.dat
```

## 📁 Project Structure

```
dat_pandas_loader/
├── src/                        # Source code
│   ├── __init__.py             # Package initialization
│   ├── dat_loader.py           # Core loading module
│   ├── dat_processor.py        # Command-line processor
│   ├── preview_dat_data.py     # Enhanced analysis tool
│   ├── dataframe_view.py       # DataFrame demo script
│   ├── preview_dat.py          # Simple preview tool
│   └── example_usage.py        # Usage examples
├── docs/                       # Documentation
│   ├── USER_MANUAL.md          # Complete user manual
│   ├── QUICK_START.md          # Quick reference
│   ├── FINAL_SUMMARY.md        # System overview
│   └── ENHANCED_PREVIEW_GUIDE.md
├── examples/                   # Example files (empty)
├── data/                       # Sample data (if any)
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## ✨ Key Features

- **📊 Special Delimiter Support**: Handles þ (thorn) and \\x14 control characters
- **🔍 Smart Detection**: Automatic encoding and delimiter detection
- **📈 Comprehensive Analysis**: 8 different delimiter tests with detailed reporting
- **📋 Spreadsheet Output**: Clean tabular text files for data review
- **🧹 Data Cleaning**: Automatic removal of delimiter artifacts
- **📁 Multiple Formats**: Export to CSV, Excel, Parquet, JSON
- **🔧 Robust Processing**: Multiple loading strategies with fallbacks
- **📝 Detailed Logging**: Complete integrity tracking and error reporting

## 🎯 Perfect for FOIA/Legal Documents

Specifically designed to handle documents with special delimiter characters commonly found in:
- EPA FOIA productions
- Legal discovery documents
- Government data exports
- Legacy system outputs

**Before**: `þProdBegþ` columns with `þ3M_WFDPD_00000022þ` data
**After**: `ProdBeg` columns with `3M_WFDPD_00000022` data

## 📖 Documentation

- **[User Manual](docs/USER_MANUAL.md)** - Complete guide with examples
- **[Quick Start](docs/QUICK_START.md)** - Fast reference
- **[System Overview](docs/FINAL_SUMMARY.md)** - Complete feature summary

## 🛠️ Tools Included

| Tool | Purpose | Usage |
|------|---------|-------|
| `dat_loader.py` | Core module for Python integration | `from src.dat_loader import load_dat_file` |
| `preview_dat_data.py` | Enhanced analysis with spreadsheet output | `python src/preview_dat_data.py file.dat` |
| `dat_processor.py` | Full-featured command-line processor | `python src/dat_processor.py file.dat output.csv` |
| `dataframe_view.py` | Complete DataFrame demonstration | `python src/dataframe_view.py file.dat` |
| `preview_dat.py` | Quick preview tool | `python src/preview_dat.py file.dat` |

## 📋 Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- chardet >= 4.0.0
- openpyxl >= 3.0.9 (for Excel export)
- pyarrow >= 10.0.0 (for Parquet export)

## 🤝 Contributing

This project was developed to solve real-world challenges with DAT file processing. Feel free to submit issues or improvements!

## 📄 License

Open source - feel free to use and modify for your needs.