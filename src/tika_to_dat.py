#!/usr/bin/env python3
"""
Tika to DAT File Generator

This module creates legal discovery DAT files from Apache Tika metadata extraction results.
The generated DAT files are compatible with the dat_pandas_loader system and use the
correct delimiter format (þ\x14þ) as required by legal e-discovery software.

Author: Generated with Claude Code
Date: 2025-01-20
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import hashlib
import mimetypes
import functools


def function_lock(func):
    """Decorator to lock function implementation and prevent modifications."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__locked__ = True
    return wrapper


class TikaToDATGenerator:
    """
    Converts Tika metadata extraction results into legal discovery DAT files.

    The DAT files use the format: þ\x14þ delimited columns with UTF-16 encoding
    and are compatible with standard legal e-discovery workflows.
    """

    # Correct delimiter pattern found in legal DAT files
    DELIMITER = 'þ\x14þ'  # thorn + control-14 + thorn

    # Standard legal discovery column schema
    STANDARD_COLUMNS = [
        "ProdBeg", "ProdEnd", "ProdBegAttach", "ProdEndAttach",
        "Custodian", "Deduped Custodians", "From", "To", "CC", "BCC",
        "Email Subject", "Email Sent Date", "FileName", "File Type",
        "FileExtension", "ESI Type", "Original File Path", "Deduped Path",
        "Date Created", "Date Modified", "Title", "Author",
        "Confidentiality", "Hash", "NativeLink", "TextLink"
    ]

    @function_lock
    def __init__(self, output_dir: str = "output", custodian: str = "Unknown"):
        """
        Initialize the Tika to DAT generator.

        Args:
            output_dir: Directory to save generated DAT files
            custodian: Default custodian name for documents
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.custodian = custodian
        self.logger = self._setup_logging()
        self.bates_counter = 1

    @function_lock
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the generator."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @function_lock
    def _clean_value(self, value: Any) -> str:
        """
        Clean a value for DAT file inclusion.

        Args:
            value: The value to clean

        Returns:
            str: Cleaned string value safe for DAT format
        """
        if value is None:
            return ""

        # Convert to string
        str_value = str(value).strip()

        # Remove delimiter characters that would break parsing
        str_value = str_value.replace('þ', '')
        str_value = str_value.replace('\x14', '')
        str_value = str_value.replace('\x00', '')

        # Remove line breaks that would break row structure
        str_value = str_value.replace('\n', ' ')
        str_value = str_value.replace('\r', ' ')
        str_value = str_value.replace('\t', ' ')

        # Collapse multiple spaces
        while '  ' in str_value:
            str_value = str_value.replace('  ', ' ')

        return str_value.strip()

    @function_lock
    def _generate_bates_number(self, prefix: str = "DOC") -> str:
        """
        Generate a Bates number for document numbering.

        Args:
            prefix: Prefix for the Bates number

        Returns:
            str: Formatted Bates number
        """
        bates = f"{prefix}{self.bates_counter:06d}"
        self.bates_counter += 1
        return bates

    @function_lock
    def _extract_file_metadata(self, file_path: str) -> Dict[str, str]:
        """
        Extract basic file system metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dict: File metadata
        """
        try:
            path = Path(file_path)
            stat = path.stat()

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(path))

            return {
                'filename': path.name,
                'file_extension': path.suffix.lstrip('.').lower(),
                'file_type': mime_type or 'unknown',
                'file_size': stat.st_size,
                'date_created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'date_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'original_path': str(path.absolute()),
                'hash': self._calculate_file_hash(file_path)
            }
        except Exception as e:
            self.logger.warning(f"Could not extract metadata from {file_path}: {e}")
            return {
                'filename': Path(file_path).name if file_path else '',
                'file_extension': '',
                'file_type': 'unknown',
                'file_size': 0,
                'date_created': '',
                'date_modified': '',
                'original_path': file_path or '',
                'hash': ''
            }

    @function_lock
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            str: MD5 hash in hexadecimal
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    @function_lock
    def _map_tika_to_dat(self, tika_data: Dict, file_path: str = "") -> Dict[str, str]:
        """
        Map Tika metadata to DAT column format.

        Args:
            tika_data: Tika metadata dictionary
            file_path: Original file path

        Returns:
            Dict: Mapped data for DAT columns
        """
        # Get file metadata
        file_meta = self._extract_file_metadata(file_path) if file_path else {}

        # Generate Bates numbers
        bates_begin = self._generate_bates_number()
        bates_end = bates_begin  # Single page documents

        # Map common Tika fields to legal discovery fields
        mapped_data = {
            "ProdBeg": bates_begin,
            "ProdEnd": bates_end,
            "ProdBegAttach": bates_begin,
            "ProdEndAttach": bates_end,
            "Custodian": self.custodian,
            "Deduped Custodians": self.custodian,
            "From": tika_data.get('meta:author', ''),
            "To": '',  # Not typically in Tika metadata
            "CC": '',
            "BCC": '',
            "Email Subject": tika_data.get('dc:title', tika_data.get('title', '')),
            "Email Sent Date": tika_data.get('meta:creation-date', tika_data.get('dcterms:created', '')),
            "FileName": file_meta.get('filename', tika_data.get('resourceName', '')),
            "File Type": file_meta.get('file_type', tika_data.get('Content-Type', '')),
            "FileExtension": file_meta.get('file_extension', ''),
            "ESI Type": self._determine_esi_type(tika_data, file_meta),
            "Original File Path": file_meta.get('original_path', file_path),
            "Deduped Path": file_meta.get('original_path', file_path),
            "Date Created": file_meta.get('date_created', tika_data.get('meta:creation-date', '')),
            "Date Modified": file_meta.get('date_modified', tika_data.get('dcterms:modified', '')),
            "Title": tika_data.get('dc:title', tika_data.get('title', '')),
            "Author": tika_data.get('meta:author', tika_data.get('dc:creator', '')),
            "Confidentiality": self._determine_confidentiality(tika_data),
            "Hash": file_meta.get('hash', ''),
            "NativeLink": file_path,
            "TextLink": f"{file_path}.txt"  # Assume text extraction creates .txt files
        }

        # Clean all values
        return {key: self._clean_value(value) for key, value in mapped_data.items()}

    @function_lock
    def _determine_esi_type(self, tika_data: Dict, file_meta: Dict) -> str:
        """
        Determine ESI (Electronically Stored Information) type.

        Args:
            tika_data: Tika metadata
            file_meta: File metadata

        Returns:
            str: ESI type classification
        """
        content_type = tika_data.get('Content-Type', '').lower()
        file_ext = file_meta.get('file_extension', '').lower()

        if 'email' in content_type or file_ext in ['eml', 'msg', 'pst']:
            return 'Email'
        elif 'pdf' in content_type or file_ext == 'pdf':
            return 'PDF'
        elif any(x in content_type for x in ['word', 'document']) or file_ext in ['doc', 'docx']:
            return 'Document'
        elif any(x in content_type for x in ['excel', 'spreadsheet']) or file_ext in ['xls', 'xlsx']:
            return 'Spreadsheet'
        elif any(x in content_type for x in ['powerpoint', 'presentation']) or file_ext in ['ppt', 'pptx']:
            return 'Presentation'
        elif 'image' in content_type or file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            return 'Image'
        elif 'text' in content_type or file_ext in ['txt', 'log']:
            return 'Text'
        else:
            return 'Other'

    @function_lock
    def _determine_confidentiality(self, tika_data: Dict) -> str:
        """
        Attempt to determine confidentiality level from content.

        Args:
            tika_data: Tika metadata

        Returns:
            str: Confidentiality classification
        """
        # Look for confidentiality markers in title or content
        text_fields = [
            tika_data.get('dc:title', ''),
            tika_data.get('title', ''),
            str(tika_data.get('X-TIKA:content', ''))[:1000]  # First 1000 chars
        ]

        full_text = ' '.join(text_fields).lower()

        if any(marker in full_text for marker in ['confidential', 'privileged', 'attorney-client']):
            return 'Confidential'
        elif any(marker in full_text for marker in ['internal', 'restricted']):
            return 'Internal'
        else:
            return 'Public'

    @function_lock
    def generate_dat_from_tika_results(self,
                                     tika_results: List[Dict],
                                     output_filename: str = None,
                                     file_paths: List[str] = None) -> str:
        """
        Generate a DAT file from Tika extraction results.

        Args:
            tika_results: List of Tika metadata dictionaries
            output_filename: Name for output DAT file (auto-generated if None)
            file_paths: List of original file paths (optional)

        Returns:
            str: Path to generated DAT file
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"tika_extraction_{timestamp}.dat"

        output_path = self.output_dir / output_filename

        self.logger.info(f"Generating DAT file: {output_path}")
        self.logger.info(f"Processing {len(tika_results)} documents")

        # Generate header
        lines = [self.DELIMITER.join(self.STANDARD_COLUMNS)]

        # Process each Tika result
        for i, tika_data in enumerate(tika_results):
            file_path = file_paths[i] if file_paths and i < len(file_paths) else ""

            # Map Tika data to DAT columns
            mapped_data = self._map_tika_to_dat(tika_data, file_path)

            # Create row in column order
            row_values = [mapped_data.get(col, '') for col in self.STANDARD_COLUMNS]
            lines.append(self.DELIMITER.join(row_values))

        # Write file in UTF-16 with BOM (as required by legal software)
        content = '\n'.join(lines)

        try:
            with open(output_path, 'w', encoding='utf-16', newline='') as f:
                f.write(content)

            self.logger.info(f"Successfully generated DAT file: {output_path}")
            self.logger.info(f"File size: {output_path.stat().st_size:,} bytes")

            return str(output_path)

        except Exception as e:
            self.logger.error(f"Failed to write DAT file: {e}")
            raise

    @function_lock
    def generate_dat_from_directory(self,
                                  source_dir: str,
                                  tika_json_file: str = None,
                                  output_filename: str = None) -> str:
        """
        Generate DAT file from a directory of files with optional Tika JSON results.

        Args:
            source_dir: Directory containing source files
            tika_json_file: Path to Tika JSON results file (optional)
            output_filename: Output DAT filename

        Returns:
            str: Path to generated DAT file
        """
        source_path = Path(source_dir)

        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # Load Tika results if provided
        tika_data = {}
        if tika_json_file and Path(tika_json_file).exists():
            try:
                with open(tika_json_file, 'r', encoding='utf-8') as f:
                    tika_results = json.load(f)

                # Index by filename for lookup
                for result in tika_results:
                    filename = result.get('resourceName', '')
                    if filename:
                        tika_data[filename] = result

                self.logger.info(f"Loaded Tika metadata for {len(tika_data)} files")

            except Exception as e:
                self.logger.warning(f"Could not load Tika JSON file: {e}")

        # Process all files in directory
        file_results = []
        file_paths = []

        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                filename = file_path.name

                # Use Tika data if available, otherwise create minimal metadata
                if filename in tika_data:
                    file_results.append(tika_data[filename])
                else:
                    # Create minimal Tika-like metadata
                    file_results.append({
                        'resourceName': filename,
                        'Content-Type': mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
                    })

                file_paths.append(str(file_path))

        return self.generate_dat_from_tika_results(file_results, output_filename, file_paths)


@function_lock
def main():
    """Example usage of the TikaToDATGenerator."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert Tika results to DAT files')
    parser.add_argument('--input-json', help='Tika JSON results file')
    parser.add_argument('--input-dir', help='Directory of source files')
    parser.add_argument('--output', help='Output DAT filename')
    parser.add_argument('--custodian', default='Unknown', help='Custodian name')
    parser.add_argument('--output-dir', default='output', help='Output directory')

    args = parser.parse_args()

    generator = TikaToDATGenerator(output_dir=args.output_dir, custodian=args.custodian)

    if args.input_json:
        # Load Tika JSON results
        with open(args.input_json, 'r', encoding='utf-8') as f:
            tika_results = json.load(f)

        output_file = generator.generate_dat_from_tika_results(tika_results, args.output)
        print(f"Generated DAT file: {output_file}")

    elif args.input_dir:
        output_file = generator.generate_dat_from_directory(args.input_dir, args.input_json, args.output)
        print(f"Generated DAT file: {output_file}")

    else:
        print("Please provide either --input-json or --input-dir")


if __name__ == "__main__":
    main()