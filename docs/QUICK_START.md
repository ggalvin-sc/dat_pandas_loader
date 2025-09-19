# DAT File Loader - Quick Start Guide

## 🚀 **Instantly Preview Any DAT File**

### **Simple Preview Command:**
```bash
python preview_dat.py your_file.dat
```

This shows you the first 20 rows with smart delimiter detection!

---

## 📋 **All Available Commands**

### **1. Quick Preview (Recommended)**
```bash
# Fast preview with automatic delimiter detection
python preview_dat.py data.dat
```

### **2. Advanced Preview**
```bash
# Preview through main processor (more detailed logging)
python dat_processor.py data.dat --preview
python dat_processor.py data.dat --preview --clean --verbose
```

### **3. Python Module Usage**
```python
# Load into pandas DataFrame from anywhere
from dat_loader import load_dat_file
df = load_dat_file('data.dat')
print(df.head(20))
```

### **4. Convert Files**
```bash
# Convert to CSV
python dat_processor.py input.dat output.csv

# Convert with cleaning
python dat_processor.py input.dat output.csv --clean

# Convert to Excel
python dat_processor.py input.dat output.xlsx --clean
```

### **5. Data Analysis**
```bash
# Detailed exploration
python dat_processor.py input.dat --explore

# Integrity report only
python dat_processor.py input.dat --report-only
```

---

## 🔧 **Example Output**

When you run `python preview_dat.py your_file.dat`, you'll see:

```
================================================================================
PREVIEW: your_file.dat
================================================================================
File size: 120,042 bytes
Detected encoding: UTF-16
Delimiter used: 'þ' (special)
Shape: 83 rows × 26 columns

FIRST 20 ROWS:
--------------------------------------------------------------------------------
ROW 1:
  ProdBeg: 3M_WFDPD_00000022
  ProdEnd: 3M_WFDPD_00000023
  Custodian: Keown_John
  Title: Distribution Code: 91-220 Material: Pressure Sensitive...
  TextLink: TEXT\3M_WFDPD_00000022\3M_WFDPD_00000022.txt

ROW 2:
  ProdBeg: 3M_WFDPD_00000024
  ProdEnd: 3M_WFDPD_00000025
  Custodian: Keown_John
  Title: Distribution Code: 91-220 Material: Pressure Sensitive...
  TextLink: TEXT\3M_WFDPD_00000022\3M_WFDPD_00000024.txt

... and 18 more rows shown
... and 63 more rows total

COLUMN SUMMARY:
   1. ProdBeg                : 83 non-null, 83 unique
   2. ProdEnd                : 83 non-null, 83 unique
   3. Custodian              : 83 non-null, 17 unique
   ... and 23 more columns
```

---

## 📂 **File Structure**

Your toolkit includes:

- **`preview_dat.py`** - ⚡ Fast preview script (use this first!)
- **`dat_processor.py`** - 🔧 Full-featured processor with logging
- **`dat_loader.py`** - 🐍 Python module for your own scripts
- **`USER_MANUAL.md`** - 📖 Complete documentation
- **`requirements.txt`** - 📦 Dependencies to install

---

## 🏃‍♂️ **Quick Installation**

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test it works:**
   ```bash
   python preview_dat.py --help
   ```

3. **Preview your first file:**
   ```bash
   python preview_dat.py your_data.dat
   ```

---

## 💡 **Pro Tips**

- **Start with preview:** Always use `preview_dat.py` first to understand your file
- **Use Python module:** Import `dat_loader` for integration with your analysis scripts
- **Handle errors gracefully:** The tools automatically try multiple loading strategies
- **Check encoding:** Tool auto-detects encoding but you can override if needed
- **Clean messy data:** Use `--clean` flag to remove duplicates and fix formatting

---

## 🆘 **Need Help?**

1. **Quick preview:** `python preview_dat.py your_file.dat`
2. **Full manual:** Read `USER_MANUAL.md`
3. **Python examples:** Check `example_usage.py`
4. **Command help:** `python dat_processor.py --help`

**Most common issue:** File not found - use full path: `"C:\full\path\to\file.dat"`