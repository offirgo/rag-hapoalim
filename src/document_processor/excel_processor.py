"""
Excel document processor
Converts Excel files to ProcessedDocument objects with text chunks
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.models.document import (
    DocumentType,
    ProcessingStatus,
    DocumentChunk,
    DocumentMetadata,
    ProcessedDocument,
    ProcessingConfig
)

logger = logging.getLogger(__name__)


class ExcelProcessingError(Exception):
    """Raised when Excel processing fails"""
    pass


class ExcelProcessor:
    """
    Processes Excel (.xlsx, .xls) files into structured document chunks

    Converts tabular data into natural language text that can be embedded
    and retrieved by RAG systems.
    """

    def __init__(self, config: ProcessingConfig = None):
        """
        Initialize Excel processor with configuration

        Args:
            config: Processing configuration, defaults to ProcessingConfig()
        """
        self.config = config or ProcessingConfig()
        self.config.validate_config()

        logger.info(f"ExcelProcessor initialized with max_chunk_size={self.config.max_chunk_size}")

    def process(self, file_path: str | Path) -> ProcessedDocument:
        """
        Process an Excel file into a ProcessedDocument

        Args:
            file_path: Path to the Excel file

        Returns:
            ProcessedDocument with chunks and metadata

        Raises:
            ExcelProcessingError: If processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)

        try:
            logger.info(f"Starting Excel processing: {file_path}")

            # Validate file
            self._validate_file(file_path)

            # Read Excel file (supports multiple sheets)
            sheets_data = self._read_excel_file(file_path)

            # Create metadata
            metadata = self._create_document_metadata(file_path, sheets_data)
            metadata.processing_status = ProcessingStatus.PROCESSING

            # Process all sheets into chunks
            all_chunks = []
            for sheet_name, df in sheets_data.items():
                logger.debug(f"Processing sheet '{sheet_name}' with {len(df)} rows")
                sheet_chunks = self._create_chunks_from_dataframe(
                    df, sheet_name, file_path.name
                )
                all_chunks.extend(sheet_chunks)

            # Update metadata with results
            processing_time = time.time() - start_time
            metadata.total_chunks = len(all_chunks)
            metadata.processing_time_seconds = processing_time
            metadata.processing_status = ProcessingStatus.COMPLETED

            logger.info(f"Excel processing completed: {len(all_chunks)} chunks in {processing_time:.2f}s")

            return ProcessedDocument(metadata=metadata, chunks=all_chunks)

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Excel processing failed for {file_path}: {str(e)}"
            logger.error(error_msg)

            # Create failed metadata if possible
            try:
                metadata = DocumentMetadata(
                    filename=file_path.name,
                    document_type=DocumentType.EXCEL,
                    file_size=file_path.stat().st_size if file_path.exists() else 0,
                    processing_status=ProcessingStatus.FAILED,
                    processing_time_seconds=processing_time,
                    error_message=str(e)
                )
                return ProcessedDocument(metadata=metadata, chunks=[])
            except:
                # If even metadata creation fails, raise the original error
                raise ExcelProcessingError(error_msg) from e

    def _validate_file(self, file_path: Path) -> None:
        """
        Validate that the file exists and is processable

        Args:
            file_path: Path to validate

        Raises:
            ExcelProcessingError: If validation fails
        """
        if not file_path.exists():
            raise ExcelProcessingError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ExcelProcessingError(f"Path is not a file: {file_path}")

        if file_path.stat().st_size == 0:
            raise ExcelProcessingError(f"File is empty: {file_path}")

        # Check file extension
        allowed_extensions = {'.xlsx', '.xls'}
        if file_path.suffix.lower() not in allowed_extensions:
            raise ExcelProcessingError(
                f"Unsupported file extension '{file_path.suffix}'. "
                f"Supported: {allowed_extensions}"
            )

        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except PermissionError:
            raise ExcelProcessingError(f"Permission denied reading file: {file_path}")
        except Exception as e:
            raise ExcelProcessingError(f"Cannot read file {file_path}: {e}")

    def _read_excel_file(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Read Excel file and return all sheets as DataFrames

        Args:
            file_path: Path to Excel file

        Returns:
            Dictionary mapping sheet names to DataFrames

        Raises:
            ExcelProcessingError: If reading fails
        """
        try:
            # Read all sheets
            #TODO change to stream
            excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

            if not excel_data:
                raise ExcelProcessingError(f"No sheets found in Excel file: {file_path}")

            # Filter out empty sheets
            non_empty_sheets = {}
            for sheet_name, df in excel_data.items():
                if not df.empty:
                    # Clean column names (remove extra whitespace)
                    df.columns = df.columns.astype(str).str.strip()
                    non_empty_sheets[sheet_name] = df
                else:
                    logger.warning(f"Skipping empty sheet: {sheet_name}")

            if not non_empty_sheets:
                raise ExcelProcessingError(f"All sheets are empty in file: {file_path}")

            logger.debug(f"Successfully read {len(non_empty_sheets)} non-empty sheets")
            return non_empty_sheets

        except Exception as e:
            if isinstance(e, ExcelProcessingError):
                raise

            # Handle specific pandas/openpyxl errors
            if "not supported" in str(e).lower() or "corrupt" in str(e).lower():
                raise ExcelProcessingError(f"File appears to be corrupted or in unsupported format: {e}")

            raise ExcelProcessingError(f"Failed to read Excel file: {e}")

    def _convert_row_to_text(self, row: pd.Series, sheet_name: str) -> str:
        """
        Convert a DataFrame row to natural language text

        DESIGN DECISION: We use a naive natural language approach as a first step.
        Alternative approaches to consider with more data and performance analysis:
        - Structured JSON representation for exact data preservation
        - Template-based conversion with domain-specific knowledge
        - Column importance weighting based on data analysis
        - Hierarchical representation for nested/related data

        Args:
            row: pandas Series representing one row
            sheet_name: Name of the sheet for context

        Returns:
            Natural language description of the row

        Example:
    Input (pandas Series):
        Employee_ID: "EMP001"
        Name: "John Smith"
        Department: "Engineering"
        Role: "Senior Software Developer"
        Salary: 95000
        Location: "New York"
        Skills: "Python, React, AWS"
        Years_Experience: 8
        Manager: "Sarah Johnson"

    Output (natural language):
        "In Employees: John Smith with Department Engineering, Role Senior Software Developer, Salary 95000, Location New York,
        Skills Python React AWS, Years_Experience 8, Manager Sarah Johnson"
        """
        # Handle missing/null values
        clean_row = {}
        for col, val in row.items():
            if pd.isna(val) or val == '' or str(val).lower() in ['none', 'null', 'nan']:
                continue
            clean_row[col] = str(val).strip()

        if not clean_row:
            return f"Empty row in sheet {sheet_name}"

        # Create natural language description
        # Format: "In [sheet]: [key info] with [additional details]"

        # Try to identify key columns (name, title, id, etc.)
        key_columns = []
        detail_columns = []

        for col, val in clean_row.items():
            col_lower = col.lower()
            if any(key in col_lower for key in ['name', 'title', 'id', 'employee']):
                key_columns.append(f"{val}")
            else:
                detail_columns.append(f"{col} {val}")

        # Build the sentence
        if key_columns:
            main_part = " ".join(key_columns)
        else:
            # If no key columns identified, use first available value
            main_part = list(clean_row.values())[0]

        if detail_columns:
            details_part = ", ".join(detail_columns)
            text = f"In {sheet_name}: {main_part} with {details_part}"
        else:
            text = f"In {sheet_name}: {main_part}"

        # Clean up the text
        text = " ".join(text.split())  # Normalize whitespace
        return text

    def _create_chunks_from_dataframe(
            self,
            df: pd.DataFrame,
            sheet_name: str,
            source: str
    ) -> List[DocumentChunk]:
        """
        Create document chunks from a DataFrame

        Args:
            df: DataFrame to process
            sheet_name: Name of the Excel sheet
            source: Source filename

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Convert each row to text
        row_texts = []
        for idx, row in df.iterrows():
            try:
                text = self._convert_row_to_text(row, sheet_name)
                if text.strip():  # Only include non-empty texts
                    row_texts.append((idx, text))
            except Exception as e:
                logger.warning(f"Failed to convert row {idx} in {sheet_name}: {e}")
                continue

        if not row_texts:
            logger.warning(f"No valid text extracted from sheet {sheet_name}")
            return chunks

        # Group texts into chunks based on max_chunk_size
        # NOTE: No overlapping needed for tabular data - each row is independent
        # and doesn't require context from adjacent rows, unlike continuous text documents
        current_chunk_text = ""
        current_chunk_rows = []
        chunk_index = 0

        for row_idx, text in row_texts:
            # Check if adding this text would exceed chunk size
            potential_chunk = current_chunk_text + "\n" + text if current_chunk_text else text

            if len(potential_chunk) <= self.config.max_chunk_size:
                # Add to current chunk
                current_chunk_text = potential_chunk
                current_chunk_rows.append(row_idx)
            else:
                # Current chunk is full, finalize it
                if current_chunk_text:
                    chunk = self._create_chunk(
                        content=current_chunk_text,
                        source=source,
                        chunk_index=chunk_index,
                        sheet_name=sheet_name,
                        row_indices=current_chunk_rows
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with current text
                current_chunk_text = text
                current_chunk_rows = [row_idx]

        # Add final chunk if there's remaining content
        if current_chunk_text:
            chunk = self._create_chunk(
                content=current_chunk_text,
                source=source,
                chunk_index=chunk_index,
                sheet_name=sheet_name,
                row_indices=current_chunk_rows
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
            self,
            content: str,
            source: str,
            chunk_index: int,
            sheet_name: str,
            row_indices: List[int]
    ) -> DocumentChunk:
        """
        Create a single DocumentChunk with proper metadata

        Args:
            content: Text content of the chunk
            source: Source filename
            chunk_index: Index of this chunk in the document
            sheet_name: Excel sheet name
            row_indices: List of original DataFrame row indices

        Returns:
            DocumentChunk object
        """
        chunk_id = f"{Path(source).stem}_{sheet_name}_{chunk_index:03d}"

        metadata = {
            "sheet_name": sheet_name,
            "row_indices": row_indices,
            "row_count": len(row_indices),
            "character_count": len(content)
        }

        return DocumentChunk(
            chunk_id=chunk_id,
            content=content.strip(),
            source=source,
            chunk_index=chunk_index,
            metadata=metadata
        )

    def _create_document_metadata(
            self,
            file_path: Path,
            sheets_data: Dict[str, pd.DataFrame]
    ) -> DocumentMetadata:
        """
        Create document metadata from file and sheet information

        Args:
            file_path: Path to the Excel file
            sheets_data: Dictionary of sheet DataFrames

        Returns:
            DocumentMetadata object
        """
        total_rows = sum(len(df) for df in sheets_data.values())
        sheet_names = list(sheets_data.keys())

        return DocumentMetadata(
            filename=file_path.name,
            document_type=DocumentType.EXCEL,
            file_size=file_path.stat().st_size,
            sheet_names=sheet_names,
            total_rows=total_rows,
            processing_status=ProcessingStatus.PENDING
        )