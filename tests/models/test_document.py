"""
Tests for document models
Tests our Pydantic models to ensure they work correctly
"""
from unittest.mock import patch, MagicMock

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models.document import (
    DocumentType,
    ProcessingStatus,
    DocumentChunk,
    DocumentMetadata,
    ProcessedDocument,
    ProcessingConfig
)


class TestDocumentType:
    """Test DocumentType enum"""

    def test_document_type_values(self):
        """Test that DocumentType has expected values"""
        assert DocumentType.EXCEL == "excel"
        assert DocumentType.WORD == "word"
        assert DocumentType.UNKNOWN == "unknown"


class TestDocumentChunk:
    """Test DocumentChunk model"""

    def test_create_valid_chunk(self):
        """Test creating a valid document chunk"""
        chunk = DocumentChunk(
            chunk_id="chunk_001",
            content="This is some test content for the chunk.",
            source="test_document.xlsx",
            chunk_index=0,
            metadata={"sheet_name": "Sheet1", "row_range": "1-5"}
        )

        assert chunk.chunk_id == "chunk_001"
        assert chunk.content == "This is some test content for the chunk."
        assert chunk.source == "test_document.xlsx"
        assert chunk.chunk_index == 0
        assert chunk.metadata["sheet_name"] == "Sheet1"
        assert chunk.embedding is None  # Should be None by default

    def test_chunk_with_embedding(self):
        """Test chunk with embedding vector"""
        chunk = DocumentChunk(
            chunk_id="chunk_002",
            content="Another test chunk",
            source="test.docx",
            chunk_index=1,
            embedding=[0.1, 0.2, 0.3, 0.4]  # Mock embedding
        )

        assert chunk.embedding == [0.1, 0.2, 0.3, 0.4]

    def test_invalid_chunk_empty_content(self):
        """Test that empty content raises validation error"""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                chunk_id="chunk_003",
                content="",  # Empty content should fail
                source="test.xlsx",
                chunk_index=0
            )

        assert "at least 1 character" in str(exc_info.value)

    def test_invalid_chunk_negative_index(self):
        """Test that negative chunk_index raises validation error"""
        with pytest.raises(ValidationError):
            DocumentChunk(
                chunk_id="chunk_004",
                content="Valid content",
                source="test.xlsx",
                chunk_index=-1  # Negative index should fail
            )

    def test_chunk_immutability(self):
        """Test that chunks are immutable (frozen=True)"""
        chunk = DocumentChunk(
            chunk_id="chunk_005",
            content="Test content",
            source="test.xlsx",
            chunk_index=0
        )

        # Should not be able to modify after creation
        with pytest.raises(ValidationError):
            chunk.content = "Modified content"


class TestDocumentMetadata:
    """Test DocumentMetadata model"""

    def test_create_basic_metadata(self):
        """Test creating basic document metadata"""
        metadata = DocumentMetadata(
            filename="employees.xlsx",
            document_type=DocumentType.EXCEL,
            file_size=1024
        )

        assert metadata.filename == "employees.xlsx"
        assert metadata.document_type == DocumentType.EXCEL
        assert metadata.file_size == 1024
        assert metadata.processing_status == ProcessingStatus.PENDING
        assert isinstance(metadata.processed_at, datetime)

    def test_excel_specific_metadata(self):
        """Test Excel-specific metadata fields"""
        metadata = DocumentMetadata(
            filename="data.xlsx",
            document_type=DocumentType.EXCEL,
            file_size=2048,
            sheet_names=["Sheet1", "Summary"],
            total_rows=150,
            total_chunks=15
        )

        assert metadata.sheet_names == ["Sheet1", "Summary"]
        assert metadata.total_rows == 150
        assert metadata.total_chunks == 15
        assert metadata.word_count is None  # Should be None for Excel

    def test_word_specific_metadata(self):
        """Test Word document specific metadata"""
        metadata = DocumentMetadata(
            filename="handbook.docx",
            document_type=DocumentType.WORD,
            file_size=5120,
            word_count=2500,
            total_chunks=25
        )

        assert metadata.word_count == 2500
        assert metadata.sheet_names is None  # Should be None for Word docs
        assert metadata.total_rows is None


class TestProcessedDocument:
    """Test ProcessedDocument model"""

    def test_empty_processed_document(self):
        """Test creating document with no chunks"""
        metadata = DocumentMetadata(
            filename="test.xlsx",
            document_type=DocumentType.EXCEL,
            file_size=512
        )

        doc = ProcessedDocument(metadata=metadata)

        assert doc.chunk_count == 0
        assert len(doc.chunks) == 0
        assert not doc.is_processed  # Should be False since status is PENDING

    def test_processed_document_with_chunks(self):
        """Test document with multiple chunks"""
        # Create metadata
        metadata = DocumentMetadata(
            filename="test.xlsx",
            document_type=DocumentType.EXCEL,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED,
            total_chunks=2
        )

        # Create chunks
        chunks = [
            DocumentChunk(
                chunk_id="chunk_001",
                content="First chunk content",
                source="test.xlsx",
                chunk_index=0
            ),
            DocumentChunk(
                chunk_id="chunk_002",
                content="Second chunk content",
                source="test.xlsx",
                chunk_index=1,
                embedding=[0.1, 0.2, 0.3]  # One chunk has embedding
            )
        ]

        doc = ProcessedDocument(metadata=metadata, chunks=chunks)

        assert doc.chunk_count == 2
        assert doc.is_processed  # Should be True since status is COMPLETED

        # Test getting chunks with embeddings
        embedded_chunks = doc.get_chunks_with_embeddings()
        assert len(embedded_chunks) == 1
        assert embedded_chunks[0].chunk_id == "chunk_002"


class TestProcessingConfig:
    """Test ProcessingConfig model"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ProcessingConfig()

        assert config.max_chunk_size == 1000
        assert config.chunk_overlap == 100
        assert config.include_header_context is True
        assert config.skip_empty_paragraphs is True

    def test_custom_config(self):
        """Test creating custom configuration"""
        config = ProcessingConfig(
            max_chunk_size=2000,
            chunk_overlap=200,
            excel_row_to_text_template="Employee {row_num}: {content}",
            preserve_formatting=True
        )

        assert config.max_chunk_size == 2000
        assert config.chunk_overlap == 200
        assert "Employee" in config.excel_row_to_text_template
        assert config.preserve_formatting is True

    def test_invalid_config_overlap_too_large(self):
        """Test that overlap >= max_chunk_size raises error"""
        config = ProcessingConfig(
            max_chunk_size=500,
            chunk_overlap=500  # Equal to max_chunk_size
        )

        with pytest.raises(ValueError, match="chunk_overlap must be less than max_chunk_size"):
            config.validate_config()

    def test_config_validation_boundaries(self):
        """Test configuration boundary validations"""
        # Test minimum chunk size
        with pytest.raises(ValidationError):
            ProcessingConfig(max_chunk_size=50)  # Below minimum of 100

        # Test maximum chunk size
        with pytest.raises(ValidationError):
            ProcessingConfig(max_chunk_size=20000)  # Above maximum of 10000

        # Test negative overlap
        with pytest.raises(ValidationError):
            ProcessingConfig(chunk_overlap=-10)


class TestProcessedDocumentEmbeddingIntegration:
    """Test ProcessedDocument integration with embedding system"""

    def setup_method(self):
        """Set up test documents with and without embeddings"""
        # Create sample chunks
        self.chunks_no_embeddings = [
            DocumentChunk(
                chunk_id="chunk_001",
                content="First chunk content",
                source="test.docx",
                chunk_index=0,
                metadata={"section": "intro"}
            ),
            DocumentChunk(
                chunk_id="chunk_002",
                content="Second chunk content",
                source="test.docx",
                chunk_index=1,
                metadata={"section": "body"}
            )
        ]

        self.chunks_with_embeddings = [
            DocumentChunk(
                chunk_id="chunk_001",
                content="First chunk content",
                source="test.docx",
                chunk_index=0,
                metadata={"section": "intro"},
                embedding=[0.1, 0.2, 0.3]
            ),
            DocumentChunk(
                chunk_id="chunk_002",
                content="Second chunk content",
                source="test.docx",
                chunk_index=1,
                metadata={"section": "body"},
                embedding=None  # Partial embeddings
            )
        ]

    def test_has_embeddings_property_false(self):
        """Test has_embeddings property when no embeddings exist"""
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_no_embeddings)

        assert doc.has_embeddings is False

    def test_has_embeddings_property_true(self):
        """Test has_embeddings property when some embeddings exist"""
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_with_embeddings)

        assert doc.has_embeddings is True  # At least one chunk has embedding

    def test_embedding_completion_rate_zero(self):
        """Test completion rate when no embeddings exist"""
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_no_embeddings)

        assert doc.embedding_completion_rate == 0.0

    def test_embedding_completion_rate_partial(self):
        """Test completion rate when some embeddings exist"""
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_with_embeddings)

        assert doc.embedding_completion_rate == 0.5  # 1 out of 2 chunks has embedding

    def test_embedding_completion_rate_empty_document(self):
        """Test completion rate for document with no chunks"""
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=[])

        assert doc.embedding_completion_rate == 0.0

    def test_get_embedding_stats(self):
        """Test embedding statistics method"""
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_with_embeddings)

        stats = doc.get_embedding_stats()

        assert stats["total_chunks"] == 2
        assert stats["embedded_chunks"] == 1
        assert stats["missing_embeddings"] == 1
        assert stats["completion_rate"] == 0.5
        assert stats["has_embeddings"] is True

    def test_generate_embeddings_success(self, mock_embedder):
        """Test successful embedding generation"""
        # Mock the embed_chunks method to return chunks with embeddings
        mock_embedded_chunks = [
            DocumentChunk(
                chunk_id="chunk_001",
                content="First chunk content",
                source="test.docx",
                chunk_index=0,
                metadata={"section": "intro"},
                embedding=[0.1, 0.2, 0.3]
            ),
            DocumentChunk(
                chunk_id="chunk_002",
                content="Second chunk content",
                source="test.docx",
                chunk_index=1,
                metadata={"section": "body"},
                embedding=[0.4, 0.5, 0.6]
            )
        ]

        mock_embedder.embed_chunks.return_value = mock_embedded_chunks

        # Create document and generate embeddings
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_no_embeddings)

        # Generate embeddings
        embedded_doc = doc.generate_embeddings(mock_embedder)

        # Verify results
        assert embedded_doc is not doc  # Should return new document
        assert len(embedded_doc.chunks) == 2
        assert embedded_doc.chunks[0].embedding == [0.1, 0.2, 0.3]
        assert embedded_doc.chunks[1].embedding == [0.4, 0.5, 0.6]
        assert embedded_doc.embedding_completion_rate == 1.0

        # Verify embedder was called correctly
        mock_embedder.embed_chunks.assert_called_once_with(self.chunks_no_embeddings)

    def test_generate_embeddings_invalid_embedder(self):
        """Test error handling with invalid embedder type"""
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_no_embeddings)

        # Try with invalid embedder
        with pytest.raises(TypeError, match="embedder must be an instance of BaseEmbedder"):
            doc.generate_embeddings("not_an_embedder")

    def test_generate_embeddings_failed_document(self, mock_embedder):
        """Test error handling with failed document processing"""
        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.FAILED  # Failed processing
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_no_embeddings)


        with pytest.raises(ValueError, match="Cannot generate embeddings for failed document"):
            doc.generate_embeddings(mock_embedder)

    def test_with_embeddings_fluent_api(self, mock_embedder):
        """Test fluent API for embedding generation"""
        mock_embedder.embed_chunks.return_value = self.chunks_with_embeddings

        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=self.chunks_no_embeddings)

        # Test fluent API
        embedded_doc = doc.with_embeddings(mock_embedder)

        assert embedded_doc is not doc
        assert embedded_doc.has_embeddings is True
        mock_embedder.embed_chunks.assert_called_once()

    def test_generate_embeddings_empty_document(self, mock_embedder):
        """Test embedding generation with empty document"""


        metadata = DocumentMetadata(
            filename="test.docx",
            document_type=DocumentType.WORD,
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )

        doc = ProcessedDocument(metadata=metadata, chunks=[])

        # Generate embeddings for empty document
        embedded_doc = doc.generate_embeddings(mock_embedder)

        assert len(embedded_doc.chunks) == 0
        assert embedded_doc.embedding_completion_rate == 0.0
        # Should not call embedder for empty document
        mock_embedder.embed_chunks.assert_not_called()


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that properly inherits from BaseEmbedder"""
    from src.embeddings.base import BaseEmbedder

    class MockEmbedder(BaseEmbedder):
        def embed_text(self, text: str):
            return [0.1, 0.2, 0.3]

        def get_embedding_dimension(self):
            return 3

        def get_model_info(self):
            return {"provider": "mock", "model_name": "test"}

    mock = MockEmbedder()
    mock.embed_chunks = MagicMock()  # Mock the method we actually care about
    return mock

# Example of how to use these models (this won't run as a test)
def example_usage():
    """Example showing how to use the models together"""
    # This is just for documentation - not a real test
    config = ProcessingConfig(max_chunk_size=800, chunk_overlap=50)

    metadata = DocumentMetadata(
        filename="employees.xlsx",
        document_type=DocumentType.EXCEL,
        file_size=2048
    )

    chunk = DocumentChunk(
        chunk_id="emp_001",
        content="John Smith works in Engineering as Senior Developer",
        source="employees.xlsx",
        chunk_index=0,
        metadata={"department": "Engineering", "row": 1}
    )

    document = ProcessedDocument(
        metadata=metadata,
        chunks=[chunk]
    )

    return document