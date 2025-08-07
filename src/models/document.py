"""
Pydantic models for document processing
Defines the data structures used throughout the RAG system
"""

from typing import List, Dict, Optional, Any
from enum import Enum
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

    @property
    def has_embeddings(self) -> bool:
        """Check if document has any embeddings generated"""
        return len(self.get_chunks_with_embeddings()) > 0

    @property
    def embedding_completion_rate(self) -> float:
        """Get percentage of chunks that have embeddings (0.0 to 1.0)"""
        if not self.chunks:
            return 0.0
        return len(self.get_chunks_with_embeddings()) / len(self.chunks)

    def generate_embeddings(self, embedder) -> "ProcessedDocument":
        """
        Generate embeddings for all chunks using the provided embedder

        Args:
            embedder: An instance of BaseEmbedder (e.g., SentenceTransformersEmbedder)

        Returns:
            New ProcessedDocument with embeddings populated
        """
        # Import here to avoid circular imports
        from src.embeddings.base import BaseEmbedder

        if not isinstance(embedder, BaseEmbedder):
            raise TypeError("embedder must be an instance of BaseEmbedder")

        if not self.is_processed:
            raise ValueError("Cannot generate embeddings for failed document processing")

        if not self.chunks:
            return ProcessedDocument(
                metadata=self.metadata.model_copy(),
                chunks=[]
            )

        # Generate embeddings for all chunks
        embedded_chunks = embedder.embed_chunks(self.chunks)

        # Create new document with embedded chunks
        return ProcessedDocument(
            metadata=self.metadata.model_copy(),
            chunks=embedded_chunks
        )

    def with_embeddings(self, embedder) -> "ProcessedDocument":
        """
        Alias for generate_embeddings() with more fluent API

        Example:
            document = processor.process("file.docx").with_embeddings(embedder)
        """
        return self.generate_embeddings(embedder)

    def get_embedding_stats(self) -> dict:
        """Get statistics about embeddings in this document"""
        total_chunks = len(self.chunks)
        embedded_chunks = len(self.get_chunks_with_embeddings())

        return {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "missing_embeddings": total_chunks - embedded_chunks,
            "completion_rate": self.embedding_completion_rate,
            "has_embeddings": self.has_embeddings
        }

class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    model_config = ConfigDict(validate_assignment=True)

    # Chunking parameters - optimized for RAG retrieval
    max_chunk_size: int = Field(default=800, ge=200, le=10000, description="Maximum characters per chunk")
    min_chunk_size: int = Field(default=200, ge=50, le=5000, description="Minimum characters per chunk")
    chunk_overlap: int = Field(default=100, ge=0, description="Character overlap between chunks")

    # Embedding model compatibility - common model limits
    max_embedding_tokens: int = Field(default=512, ge=100, le=8192, description="Max tokens for embedding model")
    chars_per_token_estimate: float = Field(default=4.0, ge=1.0, le=10.0, description="Rough chars per token estimate")

    # Advanced chunking settings - optimized for context preservation
    target_sentences_per_chunk: int = Field(default=3, ge=1, le=10, description="Target sentences per chunk for better context")
    overlap_sentences: int = Field(default=1, ge=0, le=3, description="Number of sentences to overlap when splitting")

    # Excel-specific settings
    excel_row_to_text_template: str = Field(
        default="Row {row_num}: {content}",
        description="Template for converting Excel rows to text"
    )
    include_header_context: bool = Field(default=True, description="Include column headers in chunks")

    # Word-specific settings
    preserve_formatting: bool = Field(default=False, description="Attempt to preserve text formatting")
    skip_empty_paragraphs: bool = Field(default=True, description="Skip empty paragraphs")
    enforce_embedding_limits: bool = Field(default=True, description="Split chunks that exceed embedding token limits")

    @property
    def max_embedding_chars(self) -> int:
        """Calculate approximate max characters based on token limit"""
        return int(self.max_embedding_tokens * self.chars_per_token_estimate)

    def validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError("chunk_overlap must be less than max_chunk_size")

        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")

        if self.enforce_embedding_limits and self.max_chunk_size > self.max_embedding_chars:
            import logging
            logging.warning(
                f"max_chunk_size ({self.max_chunk_size}) exceeds estimated embedding limit "
                f"({self.max_embedding_chars} chars for {self.max_embedding_tokens} tokens). "
                f"Consider reducing max_chunk_size or increasing max_embedding_tokens."
            )

