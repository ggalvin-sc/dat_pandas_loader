# Tika to DAT Generator Usage Guide

This guide explains how to use the `tika_to_dat.py` module to convert Apache Tika metadata extraction results into legal discovery DAT files that are compatible with the `dat_pandas_loader` system.

## Overview

The `TikaToDATGenerator` creates DAT files using the exact format specification found in legal e-discovery software:
- **Delimiter**: `þ\x14þ` (thorn + ASCII control-14 + thorn)
- **Encoding**: UTF-16 with BOM
- **Format**: Standard legal discovery columns

## Quick Start

### Basic Usage

```python
from src.tika_to_dat import TikaToDATGenerator

# Sample Tika metadata
tika_results = [
    {
        "resourceName": "document.pdf",
        "Content-Type": "application/pdf",
        "dc:title": "Sample Document",
        "meta:author": "John Doe",
        "meta:creation-date": "2023-01-15T10:30:00Z"
    }
]

# Generate DAT file
generator = TikaToDATGenerator(custodian="MyProject")
output_file = generator.generate_dat_from_tika_results(
    tika_results=tika_results,
    output_filename="my_documents.dat"
)

print(f"Generated: {output_file}")
```

### Command Line Usage

```bash
# From Tika JSON results
python src/tika_to_dat.py --input-json tika_results.json --output my_docs.dat --custodian "ProjectX"

# From directory of files
python src/tika_to_dat.py --input-dir /path/to/files --output my_docs.dat --custodian "ProjectX"

# With both directory and Tika JSON
python src/tika_to_dat.py --input-dir /path/to/files --input-json tika_results.json --output my_docs.dat
```

## Generated DAT File Format

### Standard Columns

The generator creates DAT files with these standard legal discovery columns:

| Column | Description | Source |
|--------|-------------|---------|
| ProdBeg | Beginning Bates number | Auto-generated |
| ProdEnd | Ending Bates number | Auto-generated |
| ProdBegAttach | Beginning attachment Bates | Auto-generated |
| ProdEndAttach | Ending attachment Bates | Auto-generated |
| Custodian | Data custodian | Constructor parameter |
| Deduped Custodians | Deduplicated custodians | Same as Custodian |
| From | Email sender | Tika `meta:author` |
| To | Email recipient | Not available in Tika |
| CC | Carbon copy recipients | Not available in Tika |
| BCC | Blind carbon copy | Not available in Tika |
| Email Subject | Email/document subject | Tika `dc:title` or `title` |
| Email Sent Date | Date sent | Tika `meta:creation-date` |
| FileName | Original filename | File system or Tika `resourceName` |
| File Type | MIME type | File system or Tika `Content-Type` |
| FileExtension | File extension | Extracted from filename |
| ESI Type | Document classification | Auto-determined from content type |
| Original File Path | Full file path | File system path |
| Deduped Path | Deduplicated path | Same as Original File Path |
| Date Created | File creation date | File system `st_ctime` |
| Date Modified | File modification date | File system `st_mtime` |
| Title | Document title | Tika `dc:title` or `title` |
| Author | Document author | Tika `meta:author` or `dc:creator` |
| Confidentiality | Confidentiality level | Auto-determined from content |
| Hash | MD5 file hash | Calculated from file |
| NativeLink | Link to native file | File path |
| TextLink | Link to extracted text | File path + .txt |

### ESI Type Classification

The generator automatically classifies documents:

- **Email**: `.eml`, `.msg`, `.pst` files or email content types
- **PDF**: PDF files and content types
- **Document**: Word documents (`.doc`, `.docx`)
- **Spreadsheet**: Excel files (`.xls`, `.xlsx`)
- **Presentation**: PowerPoint files (`.ppt`, `.pptx`)
- **Image**: Image files (`.jpg`, `.png`, `.gif`, etc.)
- **Text**: Plain text files
- **Other**: All other file types

### Confidentiality Detection

The generator attempts to determine confidentiality from content:

- **Confidential**: Contains "confidential", "privileged", "attorney-client"
- **Internal**: Contains "internal", "restricted"
- **Public**: Default classification

## Advanced Usage

### Enhanced Metadata with File Paths

```python
# Provide file paths for enhanced metadata extraction
generator = TikaToDATGenerator(custodian="ProjectX")

output_file = generator.generate_dat_from_tika_results(
    tika_results=tika_results,
    file_paths=["/path/to/doc1.pdf", "/path/to/doc2.docx"],
    output_filename="enhanced.dat"
)
```

### Directory Processing

```python
# Process entire directory with optional Tika JSON
output_file = generator.generate_dat_from_directory(
    source_dir="/path/to/documents",
    tika_json_file="tika_results.json",  # Optional
    output_filename="directory_scan.dat"
)
```

### Custom Configuration

```python
# Custom output directory and custodian
generator = TikaToDATGenerator(
    output_dir="custom_output",
    custodian="Custom_Custodian_Name"
)

# The generator will create Bates numbers starting from 1
# Format: DOC000001, DOC000002, etc.
```

## Tika Integration Workflow

### 1. Extract Metadata with Tika

```bash
# Extract metadata from files
java -jar tika-app.jar --metadata --json /path/to/files/ > tika_results.json
```

### 2. Generate DAT File

```python
import json
from src.tika_to_dat import TikaToDATGenerator

# Load Tika results
with open('tika_results.json', 'r') as f:
    tika_data = json.load(f)

# Generate DAT
generator = TikaToDATGenerator(custodian="MyProject")
dat_file = generator.generate_dat_from_tika_results(tika_data)
```

### 3. Verify Compatibility

```python
from src.dat_loader import DATLoader

# Test loading with dat_pandas_loader
loader = DATLoader()
df = loader.load(dat_file)
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
```

## File Format Specifications

### Delimiter Format
```
Column1þ[CTRL-14]þColumn2þ[CTRL-14]þColumn3
Value1þ[CTRL-14]þValue2þ[CTRL-14]þValue3
```

Where:
- `þ` = Unicode U+00FE (thorn character)
- `[CTRL-14]` = ASCII 20 (\x14, Device Control 4)

### Encoding
- **Primary**: UTF-16 with Byte Order Mark (BOM)
- **Line endings**: `\n` (LF)
- **Character cleaning**: Delimiter characters removed from data values

### Data Cleaning
The generator automatically:
- Removes delimiter characters (`þ`, `\x14`) from values
- Removes null characters (`\x00`)
- Replaces line breaks with spaces
- Collapses multiple spaces
- Trims whitespace

## Testing and Validation

### Run Example
```bash
python examples/tika_to_dat_example.py
```

This will:
1. Create sample Tika metadata
2. Generate multiple DAT files
3. Verify compatibility with `dat_pandas_loader`
4. Show file format analysis

### Verify Generated Files
```bash
# Test with the column mapper
python src/dat_dataframe_normalized.py example_output/enhanced_example.dat
```

## Troubleshooting

### Common Issues

1. **Encoding Problems**: Ensure UTF-16 encoding is maintained
2. **Delimiter Issues**: Verify `þ\x14þ` sequence is correct
3. **Missing Metadata**: Check Tika extraction completeness
4. **File Path Errors**: Use absolute paths for reliability

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

generator = TikaToDATGenerator(custodian="Debug")
# Detailed logging will show processing steps
```

## Legal Discovery Compatibility

The generated DAT files are compatible with:
- Relativity
- Concordance
- Summation
- Other legal discovery platforms that accept standard DAT format

The format follows legal industry standards for:
- Bates numbering
- Metadata fields
- File encoding
- Delimiter specification

## Integration with dat_pandas_loader

Generated DAT files work seamlessly with the existing `dat_pandas_loader` system:

1. **Automatic Detection**: The loader's `pandas_special_delimiter` method will detect the format
2. **Column Mapping**: Generated columns map to the schema system
3. **Date Normalization**: Date fields are processed correctly
4. **Interactive Mapping**: Unmapped fields can be handled interactively

This ensures a complete workflow from Tika extraction to normalized pandas DataFrames.