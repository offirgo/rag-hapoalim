"""
Sentence-Transformers local embedding implementation
Uses local transformer models to generate embeddings without API calls
"""

import logging
from typing import List, Dict, Any, Optional

from .base import BaseEmbedder, EmbeddingError
from ..models.document import DocumentChunk

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Local embedding provider using sentence-transformers library

    This implementation runs entirely offline and doesn't require API keys.
    It's suitable for development, testing, and production environments
    where data privacy or API costs are concerns.

    TODO: Add support for OpenAI embeddings for higher quality results
    TODO: Implement batch processing for better performance with large documents
    TODO: Add model caching and lazy loading for faster startup
    TODO: Support for different models based on use case (multilingual, domain-specific)
    TODO: Add embedding normalization options
    """

    # Popular sentence-transformer models and their dimensions
    DEFAULT_MODELS = {
        "all-MiniLM-L6-v2": 384,  # Fast, good balance of speed/quality
        "all-mpnet-base-v2": 768,  # Higher quality, slower
        "all-MiniLM-L12-v2": 384,  # Good for production
        "paraphrase-MiniLM-L6-v2": 384,  # Good for semantic similarity
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sentence-transformers embedder

        Args:
            model_name: Name of the sentence-transformers model to use
            config: Additional configuration options
        """
        super().__init__(config)

        self.model_name = model_name
        self.embedding_dim = self.DEFAULT_MODELS.get(model_name)
        self._model = None  # Lazy-loaded when first used

        logger.info(f"Initializing SentenceTransformersEmbedder with model: {model_name}")

        # Validate model name
        if model_name not in self.DEFAULT_MODELS:
            logger.warning(
                f"Model '{model_name}' not in known models list. "
                f"Known models: {list(self.DEFAULT_MODELS.keys())}"
            )

    def _load_model(self):
        """
        Lazy-load the sentence-transformers model

        This delays the actual model loading until first use, which speeds up
        initialization and allows for better error handling.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            logger.info("This may download the model on first use (one-time setup)")

            self._model = SentenceTransformer(self.model_name)

            # Update embedding dimension if it wasn't known
            if self.embedding_dim is None:
                # Get dimension from a test encoding
                test_embedding = self._model.encode("test", convert_to_tensor=False)
                self.embedding_dim = len(test_embedding)
                logger.info(f"Detected embedding dimension: {self.embedding_dim}")

            logger.info(f"Model loaded successfully: {self.model_name}")
            return self._model

        except ImportError as e:
            error_msg = (
                "sentence-transformers library not found. Please install it:\n"
                "pip install sentence-transformers"
            )
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to load model '{self.model_name}': {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

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
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty or whitespace-only text")

        try:
            # Load model if not already loaded
            model = self._load_model()

            # Generate embedding
            # TODO: Consider text preprocessing (truncation, cleaning)
            # TODO: Add batch processing when we have multiple texts
            embedding = model.encode(
                text.strip(),
                convert_to_tensor=False,  # Return numpy array, not tensor
                show_progress_bar=False  # Don't show progress for single items
            )

            # Convert to regular Python list for JSON serialization
            embedding_list = embedding.tolist()

            logger.debug(f"Generated embedding of dimension {len(embedding_list)} for text: {text[:50]}...")

            return embedding_list

        except Exception as e:
            error_msg = f"Failed to generate embedding for text '{text[:50]}...': {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model

        Returns:
            Integer dimension (e.g., 384, 768)
        """
        if self.embedding_dim is None:
            # Force model loading to determine dimension
            self._load_model()

        return self.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model

        Returns:
            Dictionary with model metadata
        """
        return {
            "provider": "sentence-transformers",
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "local": True,
            "requires_api_key": False,
            "supported_languages": "primarily English (varies by model)",
            "max_sequence_length": "512 tokens (typical)",
            "quality_note": "Good for most use cases. For highest quality, consider OpenAI embeddings.",
            "performance_note": "Processes one chunk at a time. Batch processing would be faster."
        }

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks with local model

        Overrides base method to add specific warnings and optimizations
        for sentence-transformers.

        Args:
            chunks: List of DocumentChunk objects to embed

        Returns:
            List of DocumentChunk objects with embeddings populated
        """
        if not chunks:
            return chunks

        # Show informational messages for local embedding
        logger.info("🤖 Using local sentence-transformers for embedding generation")
        logger.info("💡 Note: Local embeddings are good quality but API-based models may perform better")
        logger.info("⚡ Processing chunks one-by-one. Batch processing would be faster for large documents")

        # Use base class implementation
        result = super().embed_chunks(chunks)

        # Add specific warnings if some embeddings failed
        failed_count = sum(1 for chunk in result if chunk.embedding is None)
        if failed_count > 0:
            logger.warning(
                f"{failed_count} chunks failed to embed with local model. "
                f"This may affect search accuracy. Consider checking chunk content or trying a different model."
            )

        return result

    @classmethod
    def list_available_models(cls) -> Dict[str, int]:
        """
        Get list of recommended sentence-transformer models

        Returns:
            Dictionary mapping model names to their embedding dimensions
        """
        return cls.DEFAULT_MODELS.copy()

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "not loaded"
        return f"SentenceTransformersEmbedder(model={self.model_name}, status={status}, dim={self.embedding_dim})"


# Helper function for easy instantiation
def create_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformersEmbedder:
    """
    Create a sentence-transformers embedder with sensible defaults

    Args:
        model_name: Name of the model to use

    Returns:
        Configured SentenceTransformersEmbedder instance
    """
    return SentenceTransformersEmbedder(model_name=model_name)