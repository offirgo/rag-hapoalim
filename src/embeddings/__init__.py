"""
Embedding generation package
Converts text chunks into vector embeddings for semantic search
"""

from .base import BaseEmbedder, EmbeddingError
from .sentence_transformers_embedder import SentenceTransformersEmbedder, create_embedder

# TODO: Add when implemented
# from .openai_embedder import OpenAIEmbedder
# from .embedding_config import EmbeddingConfig

__all__ = [
    # Base classes
    "BaseEmbedder",
    "EmbeddingError",

    # Local embedding implementations
    "SentenceTransformersEmbedder",
    "create_embedder",

    # TODO: Cloud embedding implementations
    # "OpenAIEmbedder",

    # TODO: Configuration
    # "EmbeddingConfig",
]

# Version info
__version__ = "0.1.0"

# Default embedder for easy imports
default_embedder = create_embedder


def get_available_embedders():
    """
    Get list of available embedding providers

    Returns:
        Dict mapping provider names to their classes
    """
    return {
        "sentence-transformers": SentenceTransformersEmbedder,
        # TODO: Add when implemented
        # "openai": OpenAIEmbedder,
    }


def create_default_embedder():
    """
    Create embedder with recommended default settings

    Returns:
        SentenceTransformersEmbedder with balanced speed/quality model
    """
    return create_embedder(model_name="all-MiniLM-L6-v2")