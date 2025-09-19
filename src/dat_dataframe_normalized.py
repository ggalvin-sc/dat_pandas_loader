#!/usr/bin/env python3
"""
DAT DataFrame Normalizer - Standardize column names and data formats

This tool applies column mappings from schema_mapping/column_mappings.json to normalize
DataFrame column names and data formats. It includes:
- Column name standardization based on mapping rules
- Date normalization to ISO format
- Column splitting and merging transformations
- Schema fingerprinting for automatic format detection
- Conflict resolution with user prompts
"""

import sys
import os
import json
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple, Any, Optional
import difflib
import copy

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dat_loader import load_dat_file

class SchemaFingerprinter:
    """Handles schema fingerprinting for automatic format detection."""

    def __init__(self, schema_dir: str):
        self.schema_dir = Path(schema_dir)
        self.known_schemas_file = self.schema_dir / "known_schemas.json"
        self.known_schemas = self._load_known_schemas()

    def _load_known_schemas(self) -> Dict:
        """Load known schema fingerprints."""
        if self.known_schemas_file.exists():
            try:
                with open(self.known_schemas_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def create_fingerprint(self, columns: List[str]) -> str:
        """Create a fingerprint hash from column names."""
        # Normalize column names: lowercase, remove spaces/underscores, sort
        normalized = sorted([
            col.lower().replace(' ', '').replace('_', '').replace('-', '')
            for col in columns
        ])

        # Create hash
        fingerprint = hashlib.md5('|'.join(normalized).encode()).hexdigest()
        return fingerprint

    def check_known_schema(self, columns: List[str]) -> Optional[Dict]:
        """Check if this column structure is known."""
        fingerprint = self.create_fingerprint(columns)
        return self.known_schemas.get(fingerprint)

    def save_schema(self, columns: List[str], mappings: Dict, transformations: Dict):
        """Save a confirmed schema mapping."""
        fingerprint = self.create_fingerprint(columns)

        schema_data = {
            'fingerprint': fingerprint,
            'original_columns': sorted(columns),
            'normalized_columns': sorted([col.lower().replace(' ', '').replace('_', '') for col in columns]),
            'mappings': mappings,
            'transformations': transformations,
            'created_date': datetime.now().isoformat(),
            'column_count': len(columns)
        }

        self.known_schemas[fingerprint] = schema_data

        # Save to file
        self.schema_dir.mkdir(exist_ok=True)
        with open(self.known_schemas_file, 'w', encoding='utf-8') as f:
            json.dump(self.known_schemas, f, indent=2)

        print(f"   [SCHEMA] Saved fingerprint: {fingerprint}")
        return fingerprint

class DateNormalizer:
    """Handles date normalization to ISO format."""

    def __init__(self):
        # Common date patterns to try
        self.date_patterns = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
        ]

    def normalize_date(self, date_str: Any) -> Optional[str]:
        """Convert various date formats to ISO format."""
        if pd.isna(date_str) or date_str == '' or date_str is None:
            return None

        date_str = str(date_str).strip()
        if not date_str:
            return None

        # Try to parse with pandas first
        try:
            parsed = pd.to_datetime(date_str, errors='coerce')
            if not pd.isna(parsed):
                return parsed.strftime('%Y-%m-%d')
        except:
            pass

        # Try specific patterns
        for pattern in self.date_patterns:
            try:
                parsed = datetime.strptime(date_str, pattern)
                return parsed.strftime('%Y-%m-%d')
            except:
                continue

        return date_str  # Return original if can't parse

    def normalize_datetime(self, date_str: Any, time_str: Any = None, timezone_str: Any = None) -> Optional[str]:
        """Combine date, time, and timezone into ISO datetime."""
        if pd.isna(date_str) or date_str == '' or date_str is None:
            return None

        date_str = str(date_str).strip()
        time_part = ""
        tz_part = ""

        # Handle time component
        if time_str and not pd.isna(time_str) and str(time_str).strip():
            time_part = f" {str(time_str).strip()}"

        # Handle timezone component
        if timezone_str and not pd.isna(timezone_str) and str(timezone_str).strip():
            tz_part = f" {str(timezone_str).strip()}"

        full_datetime = f"{date_str}{time_part}{tz_part}".strip()

        # Try to parse with pandas
        try:
            parsed = pd.to_datetime(full_datetime, errors='coerce')
            if not pd.isna(parsed):
                return parsed.strftime('%Y-%m-%dT%H:%M:%S')
        except:
            pass

        # Fallback to date only
        return self.normalize_date(date_str)

class DateFieldIdentifier:
    """Identifies and tracks which fields should be treated as date/datetime fields."""

    def __init__(self, schema_dir: str):
        self.schema_dir = Path(schema_dir)
        self.date_fields_file = self.schema_dir / "date_field_definitions.json"
        self.date_fields = self._load_date_fields()

    def _load_date_fields(self) -> Dict:
        """Load existing date field definitions."""
        if self.date_fields_file.exists():
            try:
                with open(self.date_fields_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self._create_empty_date_fields()
        return self._create_empty_date_fields()

    def _create_empty_date_fields(self) -> Dict:
        """Create empty date field definitions."""
        return {
            "_metadata": {
                "description": "User-defined date field identification",
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "confirmed_date_fields": {},
            "confirmed_datetime_fields": {},
            "confirmed_text_fields": {},
            "pending_review": {}
        }

    def ask_about_field_types(self, fields_to_check: List[str], df: pd.DataFrame = None) -> Dict[str, str]:
        """Ask user to identify date fields from a list of fields."""
        if not fields_to_check:
            return {}

        print(f"\n=== DATE FIELD IDENTIFICATION ===")
        print(f"Please identify which of these fields contain date/time data:")
        print(f"This helps ensure proper ISO date formatting.\n")

        field_types = {}

        for i, field in enumerate(fields_to_check, 1):
            # Show sample data if available
            if df is not None and field in df.columns:
                sample_data = df[field].dropna().head(3).astype(str).tolist()
                print(f"{i:2d}. {field}")
                print(f"    Sample data: {sample_data}")
            else:
                print(f"{i:2d}. {field}")

            print(f"    Options:")
            print(f"       1. Date field (YYYY-MM-DD)")
            print(f"       2. DateTime field (YYYY-MM-DDTHH:MM:SS)")
            print(f"       3. Text field (not a date)")
            print(f"       4. Skip/review later")

            while True:
                choice = input(f"    Choose type for '{field}' (1-4): ").strip()
                if choice == "1":
                    field_types[field] = "DateField"
                    self.date_fields["confirmed_date_fields"][field] = {
                        "type": "DateField",
                        "confirmed_date": datetime.now().isoformat(),
                        "sample_data": sample_data if df is not None and field in df.columns else []
                    }
                    print(f"    -> Marked as Date field")
                    break
                elif choice == "2":
                    field_types[field] = "DateTimeField"
                    self.date_fields["confirmed_datetime_fields"][field] = {
                        "type": "DateTimeField",
                        "confirmed_date": datetime.now().isoformat(),
                        "sample_data": sample_data if df is not None and field in df.columns else []
                    }
                    print(f"    -> Marked as DateTime field")
                    break
                elif choice == "3":
                    field_types[field] = "TextField"
                    self.date_fields["confirmed_text_fields"][field] = {
                        "type": "TextField",
                        "confirmed_date": datetime.now().isoformat(),
                        "sample_data": sample_data if df is not None and field in df.columns else []
                    }
                    print(f"    -> Marked as Text field")
                    break
                elif choice == "4":
                    self.date_fields["pending_review"][field] = {
                        "added_date": datetime.now().isoformat(),
                        "sample_data": sample_data if df is not None and field in df.columns else []
                    }
                    print(f"    -> Added to pending review")
                    break
                else:
                    print(f"    Invalid choice. Enter 1, 2, 3, or 4")

            print()  # Empty line for readability

        self._save_date_fields()
        return field_types

    def is_confirmed_date_field(self, field_name: str) -> bool:
        """Check if a field is confirmed as a date field."""
        return (field_name in self.date_fields["confirmed_date_fields"] or
                field_name in self.date_fields["confirmed_datetime_fields"])

    def get_confirmed_field_type(self, field_name: str) -> Optional[str]:
        """Get the confirmed field type for a field."""
        if field_name in self.date_fields["confirmed_date_fields"]:
            return "DateField"
        elif field_name in self.date_fields["confirmed_datetime_fields"]:
            return "DateTimeField"
        elif field_name in self.date_fields["confirmed_text_fields"]:
            return "TextField"
        return None

    def _save_date_fields(self):
        """Save date field definitions."""
        try:
            self.schema_dir.mkdir(exist_ok=True)
            self.date_fields["_metadata"]["last_updated"] = datetime.now().isoformat()

            with open(self.date_fields_file, 'w', encoding='utf-8') as f:
                json.dump(self.date_fields, f, indent=2)
        except Exception as e:
            print(f"   [ERROR] Failed to save date field definitions: {e}")

    def get_pending_review_fields(self) -> List[str]:
        """Get list of fields pending review."""
        return list(self.date_fields["pending_review"].keys())

class FieldMetadataTracker:
    """Tracks metadata for normalized fields including data types, transformations, and lineage."""

    def __init__(self, schema_dir: str):
        self.schema_dir = Path(schema_dir)
        self.metadata_file = self.schema_dir / "field_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load existing field metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self._create_empty_metadata()
        return self._create_empty_metadata()

    def _create_empty_metadata(self) -> Dict:
        """Create empty metadata structure."""
        return {
            "_metadata": {
                "description": "Metadata tracking for normalized fields",
                "created": datetime.now().isoformat(),
                "version": "1.0.0",
                "last_updated": datetime.now().isoformat()
            },
            "fields": {},
            "transformations": [],
            "statistics": {
                "total_normalizations": 0,
                "date_fields_processed": 0,
                "datetime_fields_processed": 0,
                "text_fields_processed": 0
            }
        }

    def track_field_normalization(self, field_name: str, original_name: str,
                                 field_type: str, transformation_type: str,
                                 sample_values: List[str] = None):
        """Track when a field is normalized."""
        if field_name not in self.metadata["fields"]:
            self.metadata["fields"][field_name] = {
                "standard_name": field_name,
                "field_type": field_type,
                "normalizations": [],
                "source_mappings": [],  # Start as list since JSON doesn't support sets
                "transformation_history": [],
                "data_samples": []
            }

        field_data = self.metadata["fields"][field_name]

        # Ensure source_mappings is a set for processing
        source_mappings = set(field_data["source_mappings"])

        # Track this normalization
        normalization_record = {
            "original_name": original_name,
            "transformation_type": transformation_type,
            "timestamp": datetime.now().isoformat(),
            "sample_values": sample_values[:3] if sample_values else []
        }

        field_data["normalizations"].append(normalization_record)
        source_mappings.add(original_name)
        field_data["transformation_history"].append(transformation_type)

        # Update statistics
        self.metadata["statistics"]["total_normalizations"] += 1
        if "date" in transformation_type.lower():
            if "datetime" in transformation_type.lower():
                self.metadata["statistics"]["datetime_fields_processed"] += 1
            else:
                self.metadata["statistics"]["date_fields_processed"] += 1
        else:
            self.metadata["statistics"]["text_fields_processed"] += 1

        # Convert set back to list for JSON storage
        field_data["source_mappings"] = list(source_mappings)

        self._save_metadata()

    def _save_metadata(self):
        """Save metadata to file."""
        try:
            self.schema_dir.mkdir(exist_ok=True)
            self.metadata["_metadata"]["last_updated"] = datetime.now().isoformat()

            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"   [ERROR] Failed to save field metadata: {e}")

    def get_field_lineage(self, field_name: str) -> Dict:
        """Get the transformation lineage for a field."""
        return self.metadata["fields"].get(field_name, {})

    def generate_metadata_report(self) -> str:
        """Generate a metadata report."""
        stats = self.metadata["statistics"]
        total_fields = len(self.metadata["fields"])

        report = f"""
=== FIELD METADATA REPORT ===
Total normalized fields: {total_fields}
Total normalizations performed: {stats['total_normalizations']}
Date fields processed: {stats['date_fields_processed']}
DateTime fields processed: {stats['datetime_fields_processed']}
Text fields processed: {stats['text_fields_processed']}

TOP FIELDS BY NORMALIZATION COUNT:
"""
        # Sort fields by number of normalizations
        sorted_fields = sorted(
            self.metadata["fields"].items(),
            key=lambda x: len(x[1]["normalizations"]),
            reverse=True
        )[:10]

        for i, (field_name, field_data) in enumerate(sorted_fields, 1):
            norm_count = len(field_data["normalizations"])
            field_type = field_data.get("field_type", "unknown")
            report += f"   {i:2d}. {field_name} ({field_type}): {norm_count} normalizations\n"

        return report

class ColumnTracker:
    """Tracks all encountered columns across files for analysis."""

    def __init__(self, schema_dir: str):
        self.schema_dir = Path(schema_dir)
        self.all_columns_file = self.schema_dir / "all_encountered_columns.json"
        self.column_data = self._load_column_data()

    def _load_column_data(self) -> Dict:
        """Load existing column tracking data."""
        if self.all_columns_file.exists():
            try:
                with open(self.all_columns_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return self._create_empty_structure()
        return self._create_empty_structure()

    def _create_empty_structure(self) -> Dict:
        """Create the initial structure for column tracking."""
        return {
            "_metadata": {
                "description": "Comprehensive tracking of all column names encountered across DAT files",
                "created": datetime.now().isoformat(),
                "total_unique_columns": 0,
                "total_files_processed": 0,
                "last_updated": datetime.now().isoformat()
            },
            "columns": {},
            "patterns": {
                "common_prefixes": {},
                "common_suffixes": {},
                "common_words": {}
            }
        }

    def track_columns(self, file_path: str, columns: List[str]):
        """Track columns from a file."""
        file_name = Path(file_path).name
        timestamp = datetime.now().isoformat()

        # Ensure metadata structure exists
        if "_metadata" not in self.column_data:
            self.column_data.update(self._create_empty_structure())

        # Update metadata
        self.column_data["_metadata"]["last_updated"] = timestamp
        self.column_data["_metadata"]["total_files_processed"] = self.column_data["_metadata"].get("total_files_processed", 0) + 1

        # Track each column
        for col in columns:
            normalized_col = col.lower().strip()

            if normalized_col not in self.column_data["columns"]:
                self.column_data["columns"][normalized_col] = {
                    "original_variations": [],
                    "first_seen": timestamp,
                    "first_seen_file": file_name,
                    "occurrence_count": 0,
                    "files_seen_in": [],
                    "common_patterns": []
                }

            # Track this occurrence
            col_data = self.column_data["columns"][normalized_col]

            # Add original variation if not already tracked
            if col not in col_data["original_variations"]:
                col_data["original_variations"].append(col)

            # Update counts and files
            col_data["occurrence_count"] += 1
            if file_name not in col_data["files_seen_in"]:
                col_data["files_seen_in"].append(file_name)

            # Analyze patterns
            self._analyze_column_patterns(normalized_col)

        # Update total unique count
        self.column_data["_metadata"]["total_unique_columns"] = len(self.column_data["columns"])

        # Save the updated data
        self._save_column_data()

        print(f"   [TRACKING] Logged {len(columns)} columns to all_encountered_columns.json")

    def _analyze_column_patterns(self, column: str):
        """Analyze patterns in column names."""
        # Common prefixes (first 3-5 characters)
        for length in [3, 4, 5]:
            if len(column) > length:
                prefix = column[:length]
                if prefix not in self.column_data["patterns"]["common_prefixes"]:
                    self.column_data["patterns"]["common_prefixes"][prefix] = 0
                self.column_data["patterns"]["common_prefixes"][prefix] += 1

        # Common suffixes (last 3-5 characters)
        for length in [3, 4, 5]:
            if len(column) > length:
                suffix = column[-length:]
                if suffix not in self.column_data["patterns"]["common_suffixes"]:
                    self.column_data["patterns"]["common_suffixes"][suffix] = 0
                self.column_data["patterns"]["common_suffixes"][suffix] += 1

        # Common words (split on common separators)
        separators = ['_', '-', ' ', 'date', 'time', 'beg', 'end', 'start', 'finish']
        words = []
        temp_word = column
        for sep in separators:
            temp_word = temp_word.replace(sep, '|')
        words = [w.strip() for w in temp_word.split('|') if w.strip()]

        for word in words:
            if len(word) > 2:  # Only track words longer than 2 characters
                if word not in self.column_data["patterns"]["common_words"]:
                    self.column_data["patterns"]["common_words"][word] = 0
                self.column_data["patterns"]["common_words"][word] += 1

    def _save_column_data(self):
        """Save column tracking data to file."""
        try:
            self.schema_dir.mkdir(exist_ok=True)
            with open(self.all_columns_file, 'w', encoding='utf-8') as f:
                json.dump(self.column_data, f, indent=2)
        except Exception as e:
            print(f"   [ERROR] Failed to save column tracking data: {e}")

    def get_column_suggestions(self, partial_name: str, limit: int = 5) -> List[str]:
        """Get suggestions for column names based on tracked data."""
        partial_lower = partial_name.lower()
        suggestions = []

        for col_name in self.column_data["columns"].keys():
            if partial_lower in col_name or col_name in partial_lower:
                suggestions.append(col_name)

        # Sort by occurrence count
        suggestions.sort(key=lambda x: self.column_data["columns"][x]["occurrence_count"], reverse=True)
        return suggestions[:limit]

    def generate_analysis_report(self) -> str:
        """Generate a summary report of tracked columns."""
        total_cols = len(self.column_data["columns"])
        total_files = self.column_data["_metadata"]["total_files_processed"]

        # Most common columns
        common_cols = sorted(
            self.column_data["columns"].items(),
            key=lambda x: x[1]["occurrence_count"],
            reverse=True
        )[:10]

        # Most common patterns
        common_prefixes = sorted(
            self.column_data["patterns"]["common_prefixes"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        common_suffixes = sorted(
            self.column_data["patterns"]["common_suffixes"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        report = f"""
=== COLUMN TRACKING ANALYSIS ===
Total unique columns tracked: {total_cols:,}
Total files processed: {total_files:,}

TOP 10 MOST COMMON COLUMNS:
""" + "\n".join([f"   {i+1:2d}. {col} (seen {data['occurrence_count']} times)" for i, (col, data) in enumerate(common_cols)]) + f"""

TOP 5 COMMON PREFIXES:
""" + "\n".join([f"   {prefix}: {count} occurrences" for prefix, count in common_prefixes]) + f"""

TOP 5 COMMON SUFFIXES:
""" + "\n".join([f"   {suffix}: {count} occurrences" for suffix, count in common_suffixes])

        return report

class ColumnMapper:
    """Main column mapping and normalization engine."""

    def __init__(self, mapping_file: str, schema_dir: str, interactive: bool = True, force_interactive: bool = False):
        self.mapping_file = Path(mapping_file)
        self.schema_dir = Path(schema_dir)
        self.mappings = self._load_mappings()
        self.fingerprinter = SchemaFingerprinter(schema_dir)
        self.date_normalizer = DateNormalizer()
        self.interactive = interactive
        self.force_interactive = force_interactive
        self.column_tracker = ColumnTracker(schema_dir)
        self.metadata_tracker = FieldMetadataTracker(schema_dir)
        self.date_field_identifier = DateFieldIdentifier(schema_dir)

        # Build reverse lookup for faster matching
        self._build_reverse_lookup()

    def _load_mappings(self) -> Dict:
        """Load column mappings from JSON file."""
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_file}")

        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_reverse_lookup(self):
        """Build reverse lookup from alternate names to standard names."""
        self.reverse_lookup = {}

        for standard_name, field_info in self.mappings['fields'].items():
            # Add the standard name itself
            normalized_standard = self._normalize_column_name(standard_name)
            self.reverse_lookup[normalized_standard] = standard_name

            # Add all alternates
            for alternate in field_info.get('alternates', []):
                normalized_alt = self._normalize_column_name(alternate)
                self.reverse_lookup[normalized_alt] = standard_name

    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name for matching."""
        return name.lower().replace(' ', '').replace('_', '').replace('-', '')

    def find_column_mapping(self, column_name: str) -> Optional[str]:
        """Find the standard name for a column."""
        normalized = self._normalize_column_name(column_name)
        return self.reverse_lookup.get(normalized)

    def apply_mappings(self, df: pd.DataFrame, file_path: str = "") -> Tuple[pd.DataFrame, Dict, List, List]:
        """Apply column mappings to DataFrame."""
        print(f"\n=== COLUMN MAPPING ANALYSIS ===")
        print(f"Original columns: {len(df.columns)}")

        # Track all columns encountered
        self.column_tracker.track_columns(file_path, df.columns.tolist())

        # Check for known schema first
        known_schema = self.fingerprinter.check_known_schema(df.columns.tolist())
        if known_schema and not self.force_interactive:
            print(f"[SCHEMA] Known format detected!")
            print(f"[SCHEMA] Fingerprint: {known_schema['fingerprint']}")
            print(f"[SCHEMA] Created: {known_schema.get('created_date', 'Unknown')}")
            print(f"[SCHEMA] Columns: {known_schema.get('column_count', 0)}")

            if self.interactive:
                print(f"\nOptions:")
                print(f"   1. Use saved mappings (automatic)")
                print(f"   2. Override and run in interactive mode")

                while True:
                    choice = input(f"   Choose option (1-2): ").strip()
                    if choice == "1":
                        print(f"   -> Using saved mappings")
                        return self._apply_known_mappings(df, known_schema)
                    elif choice == "2":
                        print(f"   -> Running in interactive mode (ignoring saved mappings)")
                        break
                    else:
                        print(f"   Invalid choice. Enter 1 or 2")
            else:
                # In automatic mode, always use saved mappings
                print(f"[SCHEMA] Using saved mappings (automatic mode)")
                return self._apply_known_mappings(df, known_schema)
        elif known_schema and self.force_interactive:
            print(f"[SCHEMA] Known format detected but forcing interactive mode")
            print(f"[SCHEMA] Fingerprint: {known_schema['fingerprint']}")
            print(f"[SCHEMA] Columns: {known_schema.get('column_count', 0)}")
            print(f"[SCHEMA] Ignoring saved mappings as requested")

        mapped_columns = {}
        conflicts = {}
        unmapped = []

        # Find mappings for each column
        for col in df.columns:
            standard_name = self.find_column_mapping(col)
            if standard_name:
                if standard_name in mapped_columns:
                    # Conflict detected
                    if standard_name not in conflicts:
                        conflicts[standard_name] = []
                    conflicts[standard_name].append(col)
                else:
                    mapped_columns[standard_name] = col
                    print(f"   {col} -> {standard_name}")
            else:
                unmapped.append(col)

        # Handle conflicts
        if conflicts:
            print(f"\n=== COLUMN CONFLICTS DETECTED ===")
            resolved_conflicts = self._resolve_conflicts(df, conflicts)
            mapped_columns.update(resolved_conflicts)

        # Handle unmapped columns
        if unmapped:
            print(f"\n=== UNMAPPED COLUMNS ===")

            # First, try to automatically map columns that have matches in existing standards
            auto_mapped, still_unmapped = self._auto_map_to_standards(unmapped)
            if auto_mapped:
                print(f"\nAuto-mapped {len(auto_mapped)} columns to standard fields:")
                mapped_columns.update(auto_mapped)

            # Handle remaining unmapped columns
            if still_unmapped:
                print(f"\nRemaining unmapped columns: {len(still_unmapped)}")
                if self.interactive:
                    new_mappings = self._handle_unmapped(still_unmapped, df, mapped_columns)
                else:
                    new_mappings = self._handle_unmapped_automatic(still_unmapped)
                mapped_columns.update(new_mappings)

            # Update the unmapped list for reporting
            unmapped = still_unmapped

        # Apply the mappings
        result_df = self._rename_columns(df, mapped_columns)

        # Ask about column merging opportunities
        if self.interactive:
            result_df = self._ask_about_column_merging(result_df)

        # Ask about date field identification for newly mapped fields
        if self.interactive and mapped_columns:
            self._ask_about_date_fields(result_df, mapped_columns)

        # Apply transformations (splits/merges)
        result_df = self._apply_transformations(result_df)

        # Normalize dates
        result_df = self._normalize_dates(result_df)

        # Find unused mappings
        used_standards = set(mapped_columns.keys())
        all_standards = set(self.mappings['fields'].keys())
        unused = list(all_standards - used_standards)

        print(f"\n=== MAPPING SUMMARY ===")
        print(f"Mapped columns: {len(mapped_columns)}")
        print(f"Conflicts resolved: {len(conflicts)}")
        print(f"Unmapped columns: {len(unmapped)}")
        print(f"Unused mappings: {len(unused)}")

        # Ask user if they want to save the schema mapping
        if mapped_columns and self.interactive:
            self._ask_to_save_schema(df.columns.tolist(), mapped_columns,
                                   {'conflicts': conflicts, 'unmapped': unmapped, 'unused': unused})
        elif mapped_columns and not self.interactive:
            # In automatic mode, save without asking
            self.fingerprinter.save_schema(
                df.columns.tolist(),
                mapped_columns,
                {'conflicts': conflicts, 'unmapped': unmapped, 'unused': unused}
            )

        return result_df, conflicts, unmapped, unused

    def _ask_about_date_fields(self, df: pd.DataFrame, mapped_columns: Dict):
        """Ask user to identify date fields from the normalized columns."""
        # Find fields that haven't been confirmed yet and might be dates
        fields_to_check = []

        for standard_field in mapped_columns.keys():
            # Skip if already confirmed
            if self.date_field_identifier.get_confirmed_field_type(standard_field):
                continue

            # Skip if already defined in schema with proper date type
            schema_type = self.mappings['fields'].get(standard_field, {}).get('type')
            if schema_type in ['DateField', 'DateTimeField', 'DatePointField']:
                continue

            # Check if the field name suggests it might be a date
            field_lower = standard_field.lower()
            date_indicators = ['date', 'time', 'created', 'modified', 'sent', 'received', 'start', 'end']

            if any(indicator in field_lower for indicator in date_indicators):
                fields_to_check.append(standard_field)

        if fields_to_check:
            user_field_types = self.date_field_identifier.ask_about_field_types(fields_to_check, df)

            # Update the schema mappings if needed
            for field_name, field_type in user_field_types.items():
                if field_name in self.mappings['fields'] and field_type in ['DateField', 'DateTimeField']:
                    self.mappings['fields'][field_name]['type'] = field_type
                    print(f"   Updated schema: {field_name} -> {field_type}")

            # Save updated mappings
            if user_field_types:
                self._save_mappings()

    def _ask_to_save_schema(self, columns: List[str], mappings: Dict, transformations: Dict):
        """Ask user if they want to save the schema mapping for future use."""
        # Check if a schema already exists for this column structure
        fingerprint = self.fingerprinter.create_fingerprint(columns)
        existing_schema = self.fingerprinter.known_schemas.get(fingerprint)

        print(f"\n=== SAVE SCHEMA MAPPING ===")

        if existing_schema:
            print(f"An existing schema mapping was found for this file structure:")
            print(f"   Fingerprint: {fingerprint}")
            print(f"   Created: {existing_schema.get('created_date', 'Unknown')}")
            print(f"   Columns: {existing_schema.get('column_count', 0)}")
            print(f"\nOptions:")
            print(f"   1. Overwrite existing schema with current mappings")
            print(f"   2. Keep existing schema (don't save current mappings)")
            print(f"   3. Skip saving")

            while True:
                choice = input(f"   Choose option (1-3): ").strip()
                if choice == "1":
                    self.fingerprinter.save_schema(columns, mappings, transformations)
                    print(f"   -> Overwritten existing schema mapping")
                    break
                elif choice == "2":
                    print(f"   -> Kept existing schema (current mappings not saved)")
                    break
                elif choice == "3":
                    print(f"   -> Skipped saving schema")
                    break
                else:
                    print(f"   Invalid choice. Enter 1, 2, or 3")
        else:
            print(f"No existing schema found for this file structure.")
            print(f"   Fingerprint: {fingerprint}")
            print(f"\nOptions:")
            print(f"   1. Save current mappings as new schema (recommended)")
            print(f"   2. Don't save schema")

            while True:
                choice = input(f"   Choose option (1-2): ").strip()
                if choice == "1":
                    self.fingerprinter.save_schema(columns, mappings, transformations)
                    print(f"   -> Saved new schema mapping")
                    break
                elif choice == "2":
                    print(f"   -> Schema not saved")
                    break
                else:
                    print(f"   Invalid choice. Enter 1 or 2")

    def _apply_known_mappings(self, df: pd.DataFrame, known_schema: Dict) -> Tuple[pd.DataFrame, Dict, List, List]:
        """Apply previously saved mappings."""
        mappings = known_schema.get('mappings', {})
        transformations = known_schema.get('transformations', {})

        # Apply column renames
        result_df = self._rename_columns(df, mappings)

        # Apply transformations
        result_df = self._apply_transformations(result_df)

        # Normalize dates
        result_df = self._normalize_dates(result_df)

        print(f"Applied {len(mappings)} known column mappings")

        return result_df, transformations.get('conflicts', {}), transformations.get('unmapped', []), transformations.get('unused', [])

    def _resolve_conflicts(self, df: pd.DataFrame, conflicts: Dict) -> Dict:
        """Resolve column naming conflicts by choosing one column."""
        resolved = {}

        for standard_name, conflicting_columns in conflicts.items():
            print(f"\nConflict for '{standard_name}' between: {conflicting_columns}")

            # Show sample data from each column
            for i, col in enumerate(conflicting_columns):
                sample_data = df[col].dropna().head(3).tolist()
                print(f"   {i+1}. {col}: {sample_data}")

            # Automatically choose the first non-empty column
            chosen_col = None
            for col in conflicting_columns:
                if not df[col].dropna().empty:
                    chosen_col = col
                    break

            if not chosen_col:
                chosen_col = conflicting_columns[0]  # Fallback to first

            resolved[standard_name] = chosen_col
            print(f"   -> Automatically chose: {chosen_col}")

        return resolved

    def _auto_map_to_standards(self, unmapped: List[str]) -> Tuple[Dict, List[str]]:
        """Automatically map unmapped columns to existing standard fields where possible."""
        auto_mapped = {}
        still_unmapped = []
        used_standards = set()

        all_standards = list(self.mappings['fields'].keys())

        for col in unmapped:
            # Try to find the best matching standard field
            normalized_col = self._normalize_column_name(col)

            best_match = None
            best_score = 0

            for standard_field in all_standards:
                if standard_field in used_standards:
                    continue

                # Check if this column matches the standard field name
                normalized_standard = self._normalize_column_name(standard_field)

                # Priority 1: Exact normalized match with standard field name
                if normalized_col == normalized_standard:
                    best_match = standard_field
                    best_score = 1.0
                    break

                # Priority 2: Check if column matches any alternates for this standard
                field_info = self.mappings['fields'][standard_field]
                for alternate in field_info.get('alternates', []):
                    normalized_alt = self._normalize_column_name(alternate)
                    if normalized_col == normalized_alt:
                        best_match = standard_field
                        best_score = 1.0
                        break

                if best_score == 1.0:
                    break

                # Priority 3: High similarity match with standard field name (>= 0.8)
                similarity = difflib.SequenceMatcher(None, normalized_col, normalized_standard).ratio()
                if similarity >= 0.8 and similarity > best_score:
                    best_match = standard_field
                    best_score = similarity

                # Priority 4: High similarity with alternates
                for alternate in field_info.get('alternates', []):
                    normalized_alt = self._normalize_column_name(alternate)
                    similarity = difflib.SequenceMatcher(None, normalized_col, normalized_alt).ratio()
                    if similarity >= 0.8 and similarity > best_score:
                        best_match = standard_field
                        best_score = similarity

            # If we found a good match, map to the STANDARD field name
            if best_match and best_score >= 0.8:
                auto_mapped[best_match] = col  # Map TO the standard field name
                used_standards.add(best_match)

                # Track the field mapping
                field_type = self.mappings['fields'][best_match].get('type', 'unknown')
                self.metadata_tracker.track_field_normalization(
                    field_name=best_match,
                    original_name=col,
                    field_type=field_type,
                    transformation_type="auto_column_mapping"
                )

                # Update the mapping file with this new alternate
                self._update_mapping_file(col, best_match)
                print(f"   Auto-mapped '{col}' -> '{best_match}' (score: {best_score:.2f})")
            else:
                still_unmapped.append(col)

        return auto_mapped, still_unmapped

    def _handle_unmapped(self, unmapped: List[str], df: pd.DataFrame, already_mapped: Dict = None) -> Dict:
        """Handle unmapped columns with interactive prompts."""
        new_mappings = {}
        all_standards = list(self.mappings['fields'].keys())

        # Track which standards are already used (including any auto-mapped ones)
        used_standards = set()
        if already_mapped:
            used_standards.update(already_mapped.keys())

        for col in unmapped:
            print(f"\nUnmapped column: '{col}'")

            # Show sample data
            sample_data = df[col].dropna().head(3).tolist()
            print(f"   Sample data: {sample_data}")

            # Find similar standard names (excluding already used ones)
            available_standards = [s for s in all_standards if s not in used_standards]
            similarities = difflib.get_close_matches(col, available_standards, n=5, cutoff=0.4)

            # Get suggestions from column tracking
            tracking_suggestions = self.column_tracker.get_column_suggestions(col, 3)

            print(f"   Options:")
            print(f"   1. Keep original name: '{col}'")

            option_num = 2
            if similarities:
                print(f"   2. Map to existing standard field:")
                for i, sim in enumerate(similarities):
                    print(f"      {i+1}. {sim}")
                option_num = 3

            if tracking_suggestions:
                print(f"   {option_num}. Similar columns seen before:")
                for i, suggestion in enumerate(tracking_suggestions):
                    seen_count = self.column_tracker.column_data["columns"][suggestion]["occurrence_count"]
                    print(f"      {i+1}. {suggestion} (seen {seen_count} times)")
                option_num += 1

            print(f"   {option_num}. Create new standard field")
            print(f"   {option_num + 1}. Skip this column (exclude from output)")

            # Get user choice
            while True:
                try:
                    choice = input(f"   Choose option (1-{option_num + 1}): ").strip()

                    if choice == "1":
                        # Keep original
                        new_mappings[col] = col
                        print(f"   -> Kept original name: {col}")
                        break
                    elif choice == "2" and similarities:
                        # Map to existing standard field
                        print(f"   Available standard fields:")
                        for i, sim in enumerate(similarities):
                            # Show category info to help user choose
                            category = self.mappings['fields'][sim].get('category', 'unknown')
                            print(f"      {i+1}. {sim} ({category})")

                        while True:
                            field_choice = input(f"   Select standard field (1-{len(similarities)}): ").strip()
                            try:
                                field_idx = int(field_choice) - 1
                                if 0 <= field_idx < len(similarities):
                                    chosen_field = similarities[field_idx]

                                    # Check if already used
                                    if chosen_field in used_standards or chosen_field in new_mappings:
                                        print(f"   ERROR: Standard field '{chosen_field}' is already mapped! Choose another.")
                                        continue

                                    new_mappings[chosen_field] = col
                                    used_standards.add(chosen_field)

                                    # Track the field mapping
                                    field_type = self.mappings['fields'][chosen_field].get('type', 'unknown')
                                    sample_values = df[col].dropna().head(3).astype(str).tolist()
                                    self.metadata_tracker.track_field_normalization(
                                        field_name=chosen_field,
                                        original_name=col,
                                        field_type=field_type,
                                        transformation_type="interactive_column_mapping",
                                        sample_values=sample_values
                                    )

                                    self._update_mapping_file(col, chosen_field)
                                    print(f"   -> Mapped '{col}' to standard field '{chosen_field}'")
                                    break
                                else:
                                    print(f"   Invalid choice. Enter 1-{len(similarities)}")
                            except ValueError:
                                print(f"   Invalid input. Enter a number 1-{len(similarities)}")
                        break
                    elif choice == str(option_num):
                        # Create new standard
                        new_field_name = input(f"   Enter new standard field name: ").strip()
                        if new_field_name:
                            # Add to mappings
                            new_mappings[new_field_name] = col
                            used_standards.add(new_field_name)
                            self._add_new_standard_field(new_field_name, col)
                            print(f"   -> Created new standard field '{new_field_name}' for '{col}'")
                        else:
                            print(f"   Invalid name. Keeping original: {col}")
                            new_mappings[col] = col
                        break
                    elif choice == str(option_num + 1):
                        # Skip column
                        print(f"   -> Skipped column '{col}' (excluded from output)")
                        break
                    else:
                        print(f"   Invalid choice. Enter 1-{option_num + 1}")
                except KeyboardInterrupt:
                    print(f"\n   -> Keeping original name: {col}")
                    new_mappings[col] = col
                    break

        return new_mappings

    def _handle_unmapped_automatic(self, unmapped: List[str]) -> Dict:
        """Handle unmapped columns automatically (non-interactive mode)."""
        new_mappings = {}
        for col in unmapped:
            print(f"   -> Keeping original name: {col}")
            new_mappings[col] = col
        return new_mappings

    def _update_mapping_file(self, original_name: str, standard_name: str):
        """Add an alternate name to an existing standard field."""
        if standard_name in self.mappings['fields']:
            if original_name not in self.mappings['fields'][standard_name]['alternates']:
                self.mappings['fields'][standard_name]['alternates'].append(original_name)
                self._save_mappings()
                print(f"   [MAPPING] Added '{original_name}' as alternate for '{standard_name}'")

    def _add_new_standard_field(self, standard_name: str, original_name: str):
        """Add a completely new standard field to the mappings."""
        self.mappings['fields'][standard_name] = {
            "alternates": [original_name] if original_name != standard_name else [],
            "type": "TextField",  # Default type
            "category": "custom"  # Default category
        }
        self._save_mappings()
        print(f"   [MAPPING] Created new standard field '{standard_name}'")

    def _save_mappings(self):
        """Save updated mappings back to the JSON file."""
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.mappings, f, indent=2)
            print(f"   [MAPPING] Updated mapping file: {self.mapping_file}")
        except Exception as e:
            print(f"   [ERROR] Failed to save mappings: {e}")

    def _ask_about_column_merging(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ask user about potential column merging opportunities."""
        print(f"\n=== COLUMN MERGING OPPORTUNITIES ===")

        # Look for potential date/time combinations
        date_time_pairs = self._find_date_time_pairs(df)
        if date_time_pairs:
            print(f"\nFound potential date/time combinations:")
            for i, (date_col, time_col) in enumerate(date_time_pairs, 1):
                print(f"   {i}. {date_col} + {time_col}")

            print(f"\nOptions:")
            print(f"   1. Merge all date/time pairs")
            print(f"   2. Select specific pairs to merge")
            print(f"   3. Skip merging")

            while True:
                choice = input(f"   Choose option (1-3): ").strip()
                if choice == "1":
                    # Merge all
                    for date_col, time_col in date_time_pairs:
                        df = self._merge_date_time_columns(df, date_col, time_col)
                    break
                elif choice == "2":
                    # Select specific
                    selected = input(f"   Enter pair numbers (comma-separated, e.g., 1,3): ").strip()
                    try:
                        indices = [int(x.strip()) - 1 for x in selected.split(",")]
                        for idx in indices:
                            if 0 <= idx < len(date_time_pairs):
                                date_col, time_col = date_time_pairs[idx]
                                df = self._merge_date_time_columns(df, date_col, time_col)
                    except:
                        print(f"   Invalid selection. Skipping merging.")
                    break
                elif choice == "3":
                    print(f"   Skipped column merging")
                    break
                else:
                    print(f"   Invalid choice. Enter 1-3")

        return df

    def _find_date_time_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find potential date/time column pairs."""
        pairs = []
        columns = df.columns.tolist()

        # Look for common patterns
        date_patterns = ['date', 'dt', 'day']
        time_patterns = ['time', 'tm', 'hour']

        for date_pattern in date_patterns:
            for time_pattern in time_patterns:
                # Find date columns
                date_cols = [col for col in columns if date_pattern.lower() in col.lower()]
                time_cols = [col for col in columns if time_pattern.lower() in col.lower()]

                # Match them up
                for date_col in date_cols:
                    for time_col in time_cols:
                        # Check if they're related (similar base name)
                        date_base = date_col.lower().replace(date_pattern.lower(), '').strip('_ -')
                        time_base = time_col.lower().replace(time_pattern.lower(), '').strip('_ -')

                        if date_base == time_base and (date_col, time_col) not in pairs:
                            pairs.append((date_col, time_col))

        return pairs

    def _merge_date_time_columns(self, df: pd.DataFrame, date_col: str, time_col: str) -> pd.DataFrame:
        """Merge date and time columns into datetime."""
        merged_col_name = f"{date_col.replace('Date', '').replace('date', '')}DateTime".strip()

        print(f"   [MERGE] {date_col} + {time_col} -> {merged_col_name}")

        merged_values = []
        for _, row in df.iterrows():
            merged = self.date_normalizer.normalize_datetime(
                row.get(date_col),
                row.get(time_col)
            )
            merged_values.append(merged)

        df[merged_col_name] = merged_values
        return df

    def _rename_columns(self, df: pd.DataFrame, mappings: Dict) -> pd.DataFrame:
        """Rename DataFrame columns based on mappings."""
        rename_dict = {original: standard for standard, original in mappings.items()}
        return df.rename(columns=rename_dict)

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply splits and merges from transformation rules."""
        result_df = df.copy()

        # Apply splits
        for split_rule in self.mappings.get('transformations', {}).get('splits', []):
            result_df = self._apply_split(result_df, split_rule)

        # Apply merges
        for merge_rule in self.mappings.get('transformations', {}).get('merges', []):
            result_df = self._apply_merge(result_df, merge_rule)

        return result_df

    def _apply_split(self, df: pd.DataFrame, split_rule: Dict) -> pd.DataFrame:
        """Apply a column split transformation."""
        source_field = split_rule['source_field']
        separator = split_rule['separator']
        target_fields = split_rule['target_fields']

        if source_field not in df.columns:
            return df

        print(f"   [SPLIT] {source_field} -> {list(target_fields.values())}")

        # Split the column
        split_data = df[source_field].astype(str).str.split(separator, n=1, expand=True)

        if len(split_data.columns) >= 1 and 'left' in target_fields:
            df[target_fields['left']] = split_data[0]

        if len(split_data.columns) >= 2 and 'right' in target_fields:
            df[target_fields['right']] = split_data[1]

        return df

    def _apply_merge(self, df: pd.DataFrame, merge_rule: Dict) -> pd.DataFrame:
        """Apply a column merge transformation."""
        source_fields = merge_rule['source_fields']
        target_field = merge_rule['target_field']

        # Check if source fields exist
        existing_sources = {key: field for key, field in source_fields.items() if field in df.columns}

        if not existing_sources:
            return df

        print(f"   [MERGE] {list(existing_sources.values())} -> {target_field}")

        # Merge date and time fields
        if 'date' in existing_sources and 'time' in existing_sources:
            date_col = existing_sources['date']
            time_col = existing_sources['time']

            merged_values = []
            for _, row in df.iterrows():
                merged = self.date_normalizer.normalize_datetime(
                    row.get(date_col),
                    row.get(time_col)
                )
                merged_values.append(merged)

            df[target_field] = merged_values

        return df

    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize date columns to ISO format."""
        date_types = ['DateField', 'DateTimeField', 'DatePointField']
        date_fields = []

        # Find date/datetime fields from schema definitions
        for field_name, field_info in self.mappings['fields'].items():
            if field_info.get('type') in date_types and field_name in df.columns:
                date_fields.append(field_name)

        # Also include user-confirmed date fields
        for field_name in df.columns:
            if (self.date_field_identifier.is_confirmed_date_field(field_name) and
                field_name not in date_fields):
                date_fields.append(field_name)

        if date_fields:
            print(f"\n=== DATE NORMALIZATION ===")
            for field in date_fields:
                # Get field type from schema or user confirmation
                schema_type = self.mappings['fields'].get(field, {}).get('type')
                user_confirmed_type = self.date_field_identifier.get_confirmed_field_type(field)

                # User confirmation takes precedence
                field_type = user_confirmed_type if user_confirmed_type else schema_type

                print(f"   Normalizing {field} ({field_type})...")

                # Get sample values for metadata tracking
                sample_values = df[field].dropna().head(3).astype(str).tolist()

                # Use appropriate normalization based on field type
                if field_type == 'DateTimeField':
                    # For datetime fields, try to preserve time if present
                    df[field] = df[field].apply(self._normalize_datetime_field)
                    transformation_type = "datetime_iso_normalization"
                else:
                    # For date fields, normalize to date only
                    df[field] = df[field].apply(self.date_normalizer.normalize_date)
                    transformation_type = "date_iso_normalization"

                # Track the field normalization
                self.metadata_tracker.track_field_normalization(
                    field_name=field,
                    original_name=field,  # Already mapped at this point
                    field_type=field_type,
                    transformation_type=transformation_type,
                    sample_values=sample_values
                )

        return df

    def _normalize_datetime_field(self, value) -> Optional[str]:
        """Normalize a datetime field value."""
        if pd.isna(value) or value == '' or value is None:
            return None

        value_str = str(value).strip()
        if not value_str:
            return None

        # Try to parse as datetime first
        try:
            parsed = pd.to_datetime(value_str, errors='coerce')
            if not pd.isna(parsed):
                # If it has time component, return as ISO datetime
                if parsed.time() != pd.Timestamp('1900-01-01').time():
                    return parsed.strftime('%Y-%m-%dT%H:%M:%S')
                else:
                    # If it's just a date, return as ISO date
                    return parsed.strftime('%Y-%m-%d')
        except:
            pass

        # Fallback to regular date normalization
        return self.date_normalizer.normalize_date(value)

def main():
    """Main function for the DataFrame normalizer."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python dat_dataframe_normalized.py <dat_file_path> [--interactive|--auto|--force-interactive]")
        print("\nExample:")
        print("   python dat_dataframe_normalized.py data.dat")
        print("   python dat_dataframe_normalized.py data.dat --interactive")
        print("   python dat_dataframe_normalized.py data.dat --auto")
        print("   python dat_dataframe_normalized.py data.dat --force-interactive")
        print("   python dat_dataframe_normalized.py \"C:\\path\\to\\file.dat\"")
        print("\nModes:")
        print("   --interactive      : Ask questions about unmapped columns and merging (default)")
        print("   --auto            : Keep unmapped columns as-is, no merging prompts")
        print("   --force-interactive: Always run in interactive mode, ignore saved schema mappings")
        return

    input_file = sys.argv[1]
    interactive = True
    force_interactive = False

    if len(sys.argv) == 3:
        mode = sys.argv[2].lower()
        if mode == "--auto":
            interactive = False
        elif mode == "--interactive":
            interactive = True
        elif mode == "--force-interactive":
            interactive = True
            force_interactive = True
        else:
            print(f"Unknown mode: {mode}. Using interactive mode.")
            interactive = True
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return

    # Setup paths
    project_root = Path(__file__).parent.parent
    mapping_file = project_root / "schema_mapping" / "column_mappings.json"
    schema_dir = project_root / "schema_mapping"

    if not mapping_file.exists():
        print(f"Error: Mapping file not found: {mapping_file}")
        return

    print("DAT DATAFRAME NORMALIZER")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Mapping file: {mapping_file}")
    if force_interactive:
        print(f"Mode: Force Interactive (ignoring saved schemas)")
    else:
        print(f"Mode: {'Interactive' if interactive else 'Automatic'}")

    try:
        # Load the DAT file
        print(f"\n=== LOADING DAT FILE ===")
        df = load_dat_file(input_file, verbose=True)
        print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Initialize mapper and apply transformations
        mapper = ColumnMapper(mapping_file, schema_dir, interactive, force_interactive)
        normalized_df, conflicts, unmapped, unused = mapper.apply_mappings(df, input_file)

        # Generate output filenames
        base_name = input_path.stem
        output_dat = input_path.parent / f"{base_name}_normalized.dat"
        output_csv = input_path.parent / f"{base_name}_normalized.csv"

        # Save outputs
        print(f"\n=== SAVING NORMALIZED DATA ===")

        # Save as DAT (using the same delimiter as detected)
        normalized_df.to_csv(output_dat, index=False, sep='þ')  # Use thorn delimiter
        print(f"   [OK] Saved normalized DAT: {output_dat}")

        # Save as CSV
        normalized_df.to_csv(output_csv, index=False)
        print(f"   [OK] Saved normalized CSV: {output_csv}")

        # Final summary
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Original shape: {df.shape}")
        print(f"Normalized shape: {normalized_df.shape}")
        print(f"Column changes: {df.shape[1]} -> {normalized_df.shape[1]}")

        if unmapped:
            print(f"\nUnmapped columns kept as-is: {len(unmapped)}")
            for col in unmapped[:5]:  # Show first 5
                print(f"   - {col}")
            if len(unmapped) > 5:
                print(f"   ... and {len(unmapped) - 5} more")

        if unused and len(unused) > 0:
            print(f"\nUnused mappings (columns not found in data): {len(unused)}")
            for col in unused[:10]:  # Show first 10
                print(f"   - {col}")
            if len(unused) > 10:
                print(f"   ... and {len(unused) - 10} more")

        # Generate column tracking analysis
        if mapper.column_tracker.column_data["_metadata"]["total_files_processed"] > 1:
            analysis = mapper.column_tracker.generate_analysis_report()
            print(analysis)

        # Generate field metadata report
        metadata_report = mapper.metadata_tracker.generate_metadata_report()
        print(metadata_report)

        print(f"\n[SUCCESS] Normalization complete!")
        print(f"[FILES] Output files:")
        print(f"   {output_dat}")
        print(f"   {output_csv}")
        print(f"   {schema_dir / 'all_encountered_columns.json'} (column tracking)")
        print(f"   {schema_dir / 'known_schemas.json'} (schema fingerprints)")
        print(f"   {schema_dir / 'field_metadata.json'} (field normalization metadata)")
        print(f"   {schema_dir / 'date_field_definitions.json'} (user-defined date field types)")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()