"""
Pydantic models for document processing
Defines the data structures used throughout the RAG system
"""

from typing import List, Dict, Optional, Union, Any
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class DocumentType(str, Enum):
    """Supported document types"""
    EXCEL = "excel"
    WORD = "word"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentChunk(BaseModel):
    """A processed chunk of document content"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True  # Immutable after creation
    )

    chunk_id: str = Field(description="Unique identifier for this chunk")
    content: str = Field(min_length=1, description="The actual text content")
    source: str = Field(description="Source document filename")
    chunk_index: int = Field(ge=0, description="Position in the document (0-based)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")

    # Optional embedding (will be populated by embedding service)
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the chunk")


class DocumentMetadata(BaseModel):
    """Metadata about a processed document"""
    model_config = ConfigDict(validate_assignment=True)

    filename: str = Field(description="Original filename")
    document_type: DocumentType = Field(description="Type of document")
    file_size: int = Field(ge=0, description="File size in bytes")
    processed_at: datetime = Field(default_factory=datetime.now, description="When document was processed")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)

    # Document-specific metadata
    sheet_names: Optional[List[str]] = Field(default=None, description="Excel sheet names")
    total_rows: Optional[int] = Field(default=None, ge=0, description="Total rows in Excel")
    word_count: Optional[int] = Field(default=None, ge=0, description="Word count for Word docs")

    # Processing results
    total_chunks: int = Field(default=0, ge=0, description="Number of chunks created")
    processing_time_seconds: Optional[float] = Field(default=None, ge=0)
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")


class ProcessedDocument(BaseModel):
    """Complete processed document with chunks and metadata"""
    model_config = ConfigDict(validate_assignment=True)

    metadata: DocumentMetadata = Field(description="Document metadata")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Processed text chunks")

    @property
    def chunk_count(self) -> int:
        """Get number of chunks"""
        return len(self.chunks)

    @property
    def is_processed(self) -> bool:
        """Check if document processing completed successfully"""
        return self.metadata.processing_status == ProcessingStatus.COMPLETED

    def get_chunks_with_embeddings(self) -> List[DocumentChunk]:
        """Get only chunks that have embeddings"""
        return [chunk for chunk in self.chunks if chunk.embedding is not None]


class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    model_config = ConfigDict(validate_assignment=True)

    # Chunking parameters
    max_chunk_size: int = Field(default=1000, ge=100, le=10000, description="Maximum characters per chunk")
    chunk_overlap: int = Field(default=100, ge=0, description="Character overlap between chunks")

    # Excel-specific settings
    excel_row_to_text_template: str = Field(
        default="Row {row_num}: {content}",
        description="Template for converting Excel rows to text"
    )
    include_header_context: bool = Field(default=True, description="Include column headers in chunks")

    # Word-specific settings
    preserve_formatting: bool = Field(default=False, description="Attempt to preserve text formatting")
    skip_empty_paragraphs: bool = Field(default=True, description="Skip empty paragraphs")

    def validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("chunk_overlap must be less than max_chunk_size")
