"""
Tests for ExcelProcessor
Comprehensive test suite covering functionality, error handling, and edge cases
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from src.models.document import (
    ProcessingConfig,
    ProcessingStatus,
    DocumentType
)
from src.document_processor import ExcelProcessor
from src.document_processor.excel_processor import ExcelProcessingError


class TestExcelProcessorInitialization:
    """Test ExcelProcessor initialization and configuration"""

    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        processor = ExcelProcessor()

        assert processor.config is not None
        assert isinstance(processor.config, ProcessingConfig)
        assert processor.config.max_chunk_size == 800  # Default value
        assert processor.config.chunk_overlap == 100    # Default value

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration"""
        custom_config = ProcessingConfig(
            max_chunk_size=2000,
            chunk_overlap=200
        )
        processor = ExcelProcessor(custom_config)

        assert processor.config.max_chunk_size == 2000
        assert processor.config.chunk_overlap == 200

    def test_init_validates_config(self):
        """Test that invalid config raises error during initialization"""
        invalid_config = ProcessingConfig(
            max_chunk_size=500,
            chunk_overlap=600  # Overlap > max_chunk_size
        )

        with pytest.raises(ValueError, match="chunk_overlap must be less than max_chunk_size"):
            ExcelProcessor(invalid_config)


class TestExcelProcessorValidation:
    """Test file validation functionality"""

    def setup_method(self):
        """Set up test processor for each test"""
        self.processor = ExcelProcessor()

    def test_validate_nonexistent_file(self):
        """Test validation fails for nonexistent file"""
        document = self.processor.process("nonexistent.xlsx")

        assert document.metadata.processing_status == ProcessingStatus.FAILED
        assert "File does not exist" in document.metadata.error_message
        assert len(document.chunks) == 0

    def test_validate_directory_instead_of_file(self, tmp_path):
        """Test validation fails when path is a directory"""
        directory = tmp_path / "test_dir"
        directory.mkdir()

        document = self.processor.process(directory)
        assert document.metadata.processing_status == ProcessingStatus.FAILED
        assert "Path is not a file" in document.metadata.error_message
        assert len(document.chunks) == 0

    def test_validate_empty_file(self, tmp_path):
        """Test validation fails for empty file"""
        empty_file = tmp_path / "empty.xlsx"
        empty_file.touch()  # Create empty file

        document = self.processor.process(empty_file)
        assert document.metadata.processing_status == ProcessingStatus.FAILED
        assert "File is empty" in document.metadata.error_message
        assert len(document.chunks) == 0

    def test_validate_unsupported_extension(self, tmp_path):
        """Test validation fails for unsupported file extensions"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not an excel file")

        document = self.processor.process(txt_file)
        assert document.metadata.processing_status == ProcessingStatus.FAILED
        assert "Unsupported file extension" in document.metadata.error_message
        assert len(document.chunks) == 0

    def test_validate_supported_extensions(self, tmp_path):
        """Test that both .xlsx and .xls extensions are supported"""
        # Create mock Excel files with supported extensions
        xlsx_file = tmp_path / "test.xlsx"
        xls_file = tmp_path / "test.xls"

        # Create simple Excel files
        df = pd.DataFrame({"Name": ["Test"], "Value": [1]})
        df.to_excel(xlsx_file)
        df.to_excel(xls_file)

        # Both should process successfully (not fail validation)
        xlsx_doc = self.processor.process(xlsx_file)
        xls_doc = self.processor.process(xls_file)

        # Should complete successfully, not fail with validation error
        assert xlsx_doc.metadata.processing_status == ProcessingStatus.COMPLETED
        assert xls_doc.metadata.processing_status == ProcessingStatus.COMPLETED


class TestExcelProcessorWithRealData:
    """Test ExcelProcessor with real sample data"""

    def setup_method(self):
        """Set up test processor and check for sample data"""
        self.processor = ExcelProcessor()

        # Find sample file relative to project root (not current working directory)
        project_root = Path(__file__).parent.parent.parent
        self.sample_file = project_root / "data" / "sample_employees.xlsx"

        if not self.sample_file.exists():
            pytest.skip(f"Sample data file not found at {self.sample_file}. Run 'python create_sample_data.py' first.")

    def test_process_sample_employees_file(self):
        """Test processing the sample employees Excel file"""
        document = self.processor.process(self.sample_file)

        # Check metadata
        assert document.metadata.filename == "sample_employees.xlsx"
        assert document.metadata.document_type == DocumentType.EXCEL
        assert document.metadata.processing_status == ProcessingStatus.COMPLETED
        assert document.metadata.file_size > 0
        assert document.metadata.processing_time_seconds is not None
        assert document.metadata.processing_time_seconds > 0

        # Check sheets
        assert document.metadata.sheet_names is not None
        assert len(document.metadata.sheet_names) >= 1
        assert "Employees" in document.metadata.sheet_names

        # Check rows and chunks
        assert document.metadata.total_rows > 0
        assert document.metadata.total_chunks > 0
        assert len(document.chunks) == document.metadata.total_chunks

        # Check document properties
        assert document.chunk_count == len(document.chunks)
        assert document.is_processed is True
        assert len(document.get_chunks_with_embeddings()) == 0  # No embeddings yet

    def test_chunk_content_format(self):
        """Test that chunk content follows expected natural language format"""
        document = self.processor.process(self.sample_file)

        # Should have at least one chunk
        assert len(document.chunks) > 0

        # Check first chunk content
        first_chunk = document.chunks[0]
        assert first_chunk.content
        assert isinstance(first_chunk.content, str)
        assert len(first_chunk.content.strip()) > 0

        # Should contain natural language patterns
        content = first_chunk.content.lower()
        assert "in " in content  # "In [sheet_name]: ..."

        # Check chunk metadata
        assert first_chunk.chunk_id
        assert first_chunk.source == "sample_employees.xlsx"
        assert first_chunk.chunk_index >= 0
        assert isinstance(first_chunk.metadata, dict)
        assert "sheet_name" in first_chunk.metadata
        assert "row_indices" in first_chunk.metadata

    def test_multiple_sheets_processed(self):
        """Test that all non-empty sheets are processed"""
        document = self.processor.process(self.sample_file)

        # Should have multiple sheets based on our sample data
        sheet_names = document.metadata.sheet_names
        assert len(sheet_names) >= 2  # At least Employees and Department_Summary

        # Check that chunks exist for different sheets
        chunk_sheet_names = set()
        for chunk in document.chunks:
            sheet_name = chunk.metadata.get("sheet_name")
            if sheet_name:
                chunk_sheet_names.add(sheet_name)

        assert len(chunk_sheet_names) >= 1  # At least one sheet should have chunks

    def test_chunking_respects_max_size(self):
        """Test that chunks respect the maximum chunk size configuration"""
        config = ProcessingConfig(max_chunk_size=500)  # Small chunks
        processor = ExcelProcessor(config)

        document = processor.process(self.sample_file)

        # All chunks should be within the size limit
        for chunk in document.chunks:
            assert len(chunk.content) <= config.max_chunk_size
            assert len(chunk.content) > 0  # Should not be empty


class TestExcelProcessorWithMockData:
    """Test ExcelProcessor with controlled mock data"""

    def setup_method(self):
        """Set up test processor"""
        self.processor = ExcelProcessor()

    def create_test_excel_file(self, tmp_path, data_dict, sheet_names=None):
        """Helper to create test Excel files"""
        excel_file = tmp_path / "test.xlsx"

        if sheet_names is None:
            sheet_names = ["Sheet1"]

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for i, sheet_name in enumerate(sheet_names):
                if i < len(data_dict):
                    df = pd.DataFrame(data_dict[i])
                else:
                    df = pd.DataFrame({"col1": [f"data_{i}"]})
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        return excel_file

    def test_process_simple_dataframe(self, tmp_path):
        """Test processing simple structured data"""
        data = {
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35],
            "Department": ["Engineering", "Sales", "Marketing"]
        }

        excel_file = self.create_test_excel_file(tmp_path, [data])
        document = self.processor.process(excel_file)

        assert document.metadata.processing_status == ProcessingStatus.COMPLETED
        assert document.metadata.total_rows == 3
        assert len(document.chunks) > 0

        # Check content contains expected data
        all_content = " ".join(chunk.content for chunk in document.chunks)
        assert "alice" in all_content.lower() or "Alice" in all_content
        assert "engineering" in all_content.lower() or "Engineering" in all_content

    def test_process_mixed_data_types(self, tmp_path):
        """Test processing data with mixed types (strings, numbers, dates)"""
        data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", None],  # Include None value
            "Salary": [50000.5, 60000, 70000],
            "Active": [True, False, True],
            "Notes": ["", "Good performer", None]  # Empty and None values
        }

        excel_file = self.create_test_excel_file(tmp_path, [data])
        document = self.processor.process(excel_file)

        assert document.metadata.processing_status == ProcessingStatus.COMPLETED
        assert len(document.chunks) > 0

        # Should handle mixed data types gracefully
        for chunk in document.chunks:
            assert isinstance(chunk.content, str)
            assert len(chunk.content.strip()) > 0

    def test_process_empty_sheet(self, tmp_path):
        """Test processing Excel file with empty sheet"""
        # Create file with one empty sheet and one with data
        data_sheet = {"Name": ["Alice"], "Value": [1]}
        empty_data = {}

        excel_file = self.create_test_excel_file(
            tmp_path,
            [data_sheet, empty_data],
            ["DataSheet", "EmptySheet"]
        )

        document = self.processor.process(excel_file)

        # Should process successfully, ignoring empty sheet
        assert document.metadata.processing_status == ProcessingStatus.COMPLETED

        # Should only include non-empty sheets
        processed_sheets = {chunk.metadata["sheet_name"] for chunk in document.chunks}
        assert "DataSheet" in processed_sheets
        # EmptySheet might not appear in chunks if it's truly empty

    def test_chunking_with_custom_config(self, tmp_path):
        """Test chunking behavior with custom configuration"""
        # Create data that will definitely create multiple chunks
        large_data = {
            "ID": list(range(100)),
            "Description": [f"This is a long description for item {i} with lots of text to make chunks larger" for i in range(100)]
        }

        excel_file = self.create_test_excel_file(tmp_path, [large_data])

        # Test with small chunk size
        config = ProcessingConfig(max_chunk_size=201)
        processor = ExcelProcessor(config)

        document = processor.process(excel_file)

        # Should create multiple small chunks
        assert len(document.chunks) > 1
        for chunk in document.chunks:
            assert len(chunk.content) <= 200


class TestExcelProcessorErrorHandling:
    """Test error handling and edge cases"""

    def setup_method(self):
        """Set up test processor"""
        self.processor = ExcelProcessor()

    def test_corrupted_excel_file(self, tmp_path):
        """Test handling of corrupted Excel file"""
        fake_excel = tmp_path / "corrupted.xlsx"
        fake_excel.write_text("This is not a real Excel file")

        document = self.processor.process(fake_excel)

        # Should return failed document, not crash
        assert document.metadata.processing_status == ProcessingStatus.FAILED
        assert document.metadata.error_message is not None
        assert len(document.chunks) == 0

    @patch('pandas.read_excel')
    def test_pandas_read_error(self, mock_read_excel, tmp_path):
        """Test handling of pandas read errors"""
        mock_read_excel.side_effect = Exception("Simulated pandas error")

        # Create a real file to pass validation
        excel_file = tmp_path / "test.xlsx"
        pd.DataFrame({"col": [1]}).to_excel(excel_file)

        document = self.processor.process(excel_file)

        assert document.metadata.processing_status == ProcessingStatus.FAILED
        assert "Simulated pandas error" in document.metadata.error_message

    def test_permission_denied(self):
        """Test handling of permission denied errors"""
        # This test might not work on all systems, so we'll mock it
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_file', return_value=True):
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 1024
                    with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                        document = self.processor.process("protected.xlsx")
                        assert document.metadata.processing_status == ProcessingStatus.FAILED
                        assert "Permission denied" in document.metadata.error_message
                        assert len(document.chunks) == 0


class TestExcelProcessorRowToTextConversion:
    """Test the row-to-text conversion logic specifically"""

    def setup_method(self):
        """Set up test processor"""
        self.processor = ExcelProcessor()

    def test_convert_simple_row(self):
        """Test converting a simple row to text"""
        row_data = pd.Series({
            "Name": "John Doe",
            "Age": 30,
            "Department": "Engineering"
        })

        text = self.processor._convert_row_to_text(row_data, "TestSheet")

        assert "John Doe" in text
        assert "TestSheet" in text
        assert "Engineering" in text
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_convert_row_with_nulls(self):
        """Test converting row with null/empty values"""
        row_data = pd.Series({
            "Name": "Jane Smith",
            "Age": None,
            "Department": "",
            "Notes": "Active employee"
        })

        text = self.processor._convert_row_to_text(row_data, "TestSheet")

        # Should include non-null values
        assert "Jane Smith" in text
        assert "Active employee" in text

        # Should handle nulls gracefully (not crash)
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_convert_empty_row(self):
        """Test converting completely empty row"""
        row_data = pd.Series({
            "Col1": None,
            "Col2": "",
            "Col3": "nan"
        })

        text = self.processor._convert_row_to_text(row_data, "TestSheet")

        # Should return something meaningful even for empty row
        assert isinstance(text, str)
        assert len(text.strip()) > 0
        assert "TestSheet" in text


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_excel_processor.py -v
    pytest.main([__file__, "-v"])