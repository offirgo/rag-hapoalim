"""
Base vector store interface
Defines the contract that all vector storage systems must implement
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from src.models.document import ProcessedDocument, DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail"""
    pass


class SearchResult:
    """
    Represents a search result from vector similarity search

    Contains the found chunk plus similarity score and metadata
    """

    def __init__(
            self,
            chunk: DocumentChunk,
            score: float,
            rank: int,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize search result

        Args:
            chunk: The document chunk that matched the query
            score: Similarity score (higher = more similar, typically 0.0 to 1.0)
            rank: Position in search results (0 = best match)
            metadata: Additional search metadata
        """
        self.chunk = chunk
        self.score = score
        self.rank = rank
        self.metadata = metadata or {}

    @property
    def chunk_id(self) -> str:
        """Get the chunk ID for convenience"""
        return self.chunk.chunk_id

    @property
    def content(self) -> str:
        """Get the chunk content for convenience"""
        return self.chunk.content

    @property
    def source(self) -> str:
        """Get the source document for convenience"""
        return self.chunk.source

    def __repr__(self) -> str:
        return f"SearchResult(chunk_id='{self.chunk_id}', score={self.score:.3f}, rank={self.rank})"

    def __str__(self) -> str:
        return f"#{self.rank + 1} ({self.score:.3f}): {self.content[:100]}..."


class VectorQuery:
    """
    Represents a search query to the vector store

    Encapsulates the query text, parameters, and filtering options
    """

    def __init__(
            self,
            text: str,
            top_k: int = 5,
            filters: Optional[Dict[str, Any]] = None,
            min_score: Optional[float] = None
    ):
        """
        Initialize vector query

        Args:
            text: The query text to search for
            top_k: Maximum number of results to return
            filters: Optional filters to apply (e.g., {"source": "handbook.docx"})
            min_score: Minimum similarity score threshold
        """
        self.text = text
        self.top_k = top_k
        self.filters = filters or {}
        self.min_score = min_score

        if not text.strip():
            raise ValueError("Query text cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

    def __repr__(self) -> str:
        return f"VectorQuery(text='{self.text[:50]}...', top_k={self.top_k})"


class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage systems

    This defines the interface that any vector store must implement,
    whether it's FAISS, Qdrant, Pinecone, or others.
    """

    def __init__(self, embedding_dimension: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize vector store

        Args:
            embedding_dimension: Expected dimension of embedding vectors
            config: Provider-specific configuration
        """
        self.embedding_dimension = embedding_dimension
        self.config = config or {}

        # Statistics tracking
        self.stats = {
            "documents_added": 0,
            "chunks_indexed": 0,
            "total_searches": 0,
            "last_search_time": None
        }

        logger.info(f"Initializing {self.__class__.__name__}")

    @abstractmethod
    def add_document(self, document: ProcessedDocument) -> None:
        """
        Add a processed document to the vector store

        Args:
            document: ProcessedDocument with embedded chunks

        Raises:
            VectorStoreError: If document cannot be added
            ValueError: If document has no embeddings
        """
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add individual chunks to the vector store

        Args:
            chunks: List of chunks with embeddings

        Raises:
            VectorStoreError: If chunks cannot be added
            ValueError: If chunks have no embeddings
        """
        pass

    @abstractmethod
    def search(self, query: Union[str, VectorQuery]) -> List[SearchResult]:
        """
        Search for similar chunks using text query

        Args:
            query: Query string or VectorQuery object

        Returns:
            List of SearchResult objects ranked by similarity

        Raises:
            VectorStoreError: If search fails
        """
        pass

    @abstractmethod
    def search_by_embedding(self, embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """
        Search for similar chunks using embedding vector directly

        Args:
            embedding: Query embedding vector
            top_k: Maximum number of results

        Returns:
            List of SearchResult objects ranked by similarity

        Raises:
            VectorStoreError: If search fails
        """
        pass

    @abstractmethod
    def get_chunk_count(self) -> int:
        """
        Get total number of chunks in the vector store

        Returns:
            Number of indexed chunks
        """
        pass

    @abstractmethod
    def remove_document(self, source: str) -> int:
        """
        Remove all chunks from a specific document

        Args:
            source: Source filename to remove

        Returns:
            Number of chunks removed

        Raises:
            VectorStoreError: If removal fails
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all documents and chunks from the vector store

        Raises:
            VectorStoreError: If clearing fails
        """
        pass

    @abstractmethod
    def save_index(self, path: Union[str, Path]) -> None:
        """
        Save the vector index to disk for persistence

        Args:
            path: Path where to save the index

        Raises:
            VectorStoreError: If saving fails
        """
        pass

    @abstractmethod
    def load_index(self, path: Union[str, Path]) -> None:
        """
        Load a vector index from disk

        Args:
            path: Path to the saved index

        Raises:
            VectorStoreError: If loading fails
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics

        Returns:
            Dictionary with usage statistics
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset vector store statistics"""
        self.stats = {
            "documents_added": 0,
            "chunks_indexed": 0,
            "total_searches": 0,
            "last_search_time": None
        }
        logger.debug("Vector store statistics reset")

    def _validate_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """
        Validate that chunks have embeddings and correct dimensions

        Args:
            chunks: List of chunks to validate

        Raises:
            ValueError: If chunks are invalid
        """
        if not chunks:
            raise ValueError("No chunks provided")

        missing_embeddings = [chunk.chunk_id for chunk in chunks if chunk.embedding is None]
        if missing_embeddings:
            raise ValueError(f"Chunks missing embeddings: {missing_embeddings[:5]}")

        # Validate embedding dimensions if we know what to expect
        if self.embedding_dimension:
            wrong_dimensions = [
                (chunk.chunk_id, len(chunk.embedding))
                for chunk in chunks
                if len(chunk.embedding) != self.embedding_dimension
            ]
            if wrong_dimensions:
                examples = wrong_dimensions[:3]
                raise ValueError(f"Wrong embedding dimensions: {examples}")

    def _validate_document_embeddings(self, document: ProcessedDocument) -> None:
        """
        Validate that document has embeddings

        Args:
            document: Document to validate

        Raises:
            ValueError: If document is invalid
        """
        if not document.is_processed:
            raise ValueError("Cannot index failed document processing")

        if not document.has_embeddings:
            raise ValueError("Document has no embeddings. Call document.generate_embeddings() first.")

        embedded_chunks = document.get_chunks_with_embeddings()
        if not embedded_chunks:
            raise ValueError("No chunks with embeddings found in document")

        # Validate the embedded chunks
        self._validate_embeddings(embedded_chunks)

    def __repr__(self) -> str:
        chunk_count = self.get_chunk_count() if hasattr(self, '_initialized') else 'unknown'
        return f"{self.__class__.__name__}(chunks={chunk_count}, dim={self.embedding_dimension})"