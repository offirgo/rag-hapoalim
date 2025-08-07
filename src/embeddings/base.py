"""
Base embedding interface
Defines the contract that all embedding providers must implement
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from src.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails"""
    pass


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding providers

    This defines the interface that any embedding service must implement,
    whether it's local (sentence-transformers) or API-based (OpenAI, Cohere, etc.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize embedder with configuration

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config or {}
        self.embedding_dim: Optional[int] = None  # Set by implementations
        self.model_name: Optional[str] = None  # Set by implementations

        # Statistics tracking
        self.stats = {
            "total_chunks_processed": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_characters_embedded": 0
        }

        logger.info(f"Initializing {self.__class__.__name__}")

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model

        Returns:
            Integer dimension (e.g., 384, 768, 1536)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model

        Returns:
            Dictionary with model metadata (name, version, etc.)
        """
        pass

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for a list of document chunks

        This is the main method used by the document processing pipeline.
        It processes chunks one by one and handles errors gracefully.

        Args:
            chunks: List of DocumentChunk objects to embed

        Returns:
            List of DocumentChunk objects with embeddings populated
            Chunks that failed to embed will have embedding=None
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return chunks

        logger.info(f"Starting embedding generation for {len(chunks)} chunks")

        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding for this chunk
                embedding = self.embed_text(chunk.content)

                # Create new chunk with embedding
                embedded_chunk = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                    embedding=embedding
                )

                embedded_chunks.append(embedded_chunk)

                # Update statistics
                self.stats["successful_embeddings"] += 1
                self.stats["total_characters_embedded"] += len(chunk.content)

                logger.debug(f"Successfully embedded chunk {i + 1}/{len(chunks)}: {chunk.chunk_id}")

            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Failed to embed chunk {chunk.chunk_id}: {e}")

                # Create chunk without embedding (embedding=None)
                failed_chunk = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                    embedding=None  # Explicitly None for failed embeddings
                )

                embedded_chunks.append(failed_chunk)
                self.stats["failed_embeddings"] += 1

        # Update total statistics
        self.stats["total_chunks_processed"] += len(chunks)

        # Log summary
        success_rate = (self.stats["successful_embeddings"] / len(chunks)) * 100
        logger.info(
            f"Embedding completed: {self.stats['successful_embeddings']}/{len(chunks)} "
            f"chunks successful ({success_rate:.1f}%)"
        )

        if self.stats["failed_embeddings"] > 0:
            logger.warning(
                f"{self.stats['failed_embeddings']} chunks failed to embed. "
                f"Vector search accuracy may be reduced for this document."
            )

        return embedded_chunks

    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding statistics

        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset embedding statistics"""
        self.stats = {
            "total_chunks_processed": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_characters_embedded": 0
        }
        logger.debug("Embedding statistics reset")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dim={self.embedding_dim})"