"""
FAISS vector store implementation
Uses Facebook AI Similarity Search for efficient vector storage and retrieval
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import numpy as np

from .base import BaseVectorStore, VectorStoreError, SearchResult, VectorQuery
from src.models.document import ProcessedDocument, DocumentChunk

logger = logging.getLogger(__name__)


class FaissVectorStore(BaseVectorStore):
    """
    FAISS-based vector storage implementation

    Uses FAISS (Facebook AI Similarity Search) for efficient similarity search
    and separate JSON storage for chunk metadata.

    TODO: Add support for different FAISS index types (IVF, HNSW) for large datasets
    TODO: Implement batch operations for better performance with many chunks
    TODO: Add index optimization and compression options
    TODO: Support for filtering during search (not just post-filtering)
    TODO: Add support for updating individual chunks without full rebuild
    """

    def __init__(
            self,
            embedding_dimension: int = 384,
            embedder=None,
            index_type: str = "flat",
            config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FAISS vector store

        Args:
            embedding_dimension: Dimension of embedding vectors (e.g., 384, 768)
            embedder: Embedder instance for converting queries to vectors
            index_type: Type of FAISS index to use ("flat", "ivf" - flat is best for small datasets)
            config: Additional configuration options
        """
        super().__init__(embedding_dimension, config)

        self.embedder = embedder
        self.index_type = index_type

        # FAISS components
        self._index = None
        self._chunk_metadata = {}  # Maps FAISS index position → chunk metadata
        self._chunk_id_to_position = {}  # Maps chunk_id → FAISS index position
        self._next_position = 0

        # Initialize FAISS index
        self._initialize_index()

        logger.info(f"FaissVectorStore initialized with {index_type} index, dimension {embedding_dimension}")

    def _initialize_index(self) -> None:
        """Initialize the FAISS index based on configuration"""
        try:
            import faiss

            if self.index_type == "flat":
                # Simple brute-force search - best for small datasets (< 10k vectors)
                self._index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner Product (cosine similarity)

            elif self.index_type == "ivf":
                # TODO: Implement IVF index for larger datasets
                # Inverted File index - faster for large datasets but requires training
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                nlist = 100  # Number of clusters
                self._index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
                logger.info("IVF index created but needs training before use")

            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

            logger.debug(f"FAISS index initialized: {self._index}")
            self._initialized = True

        except ImportError as e:
            error_msg = (
                "FAISS library not found. Please install it:\n"
                "pip install faiss-cpu  # For CPU-only version\n"
                "pip install faiss-gpu  # For GPU version"
            )
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to initialize FAISS index: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def add_document(self, document: ProcessedDocument) -> None:
        """
        Add a processed document to the vector store

        Args:
            document: ProcessedDocument with embedded chunks

        Raises:
            VectorStoreError: If document cannot be added
            ValueError: If document has no embeddings
        """
        start_time = time.time()

        # Validate document
        self._validate_document_embeddings(document)

        # Get chunks with embeddings
        embedded_chunks = document.get_chunks_with_embeddings()

        logger.info(f"Adding document '{document.metadata.filename}' with {len(embedded_chunks)} chunks")

        # Add chunks to index
        self.add_chunks(embedded_chunks)

        # Update statistics
        self.stats["documents_added"] += 1
        processing_time = time.time() - start_time

        logger.info(
            f"Document '{document.metadata.filename}' added successfully in {processing_time:.2f}s "
            f"({len(embedded_chunks)} chunks)"
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add individual chunks to the vector store

        Args:
            chunks: List of chunks with embeddings

        Raises:
            VectorStoreError: If chunks cannot be added
            ValueError: If chunks have no embeddings
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return

        # Validate chunks
        self._validate_embeddings(chunks)

        try:
            # Prepare embeddings for FAISS
            embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

            # Normalize embeddings for cosine similarity
            # TODO: Make normalization configurable
            faiss = self._get_faiss()
            faiss.normalize_L2(embeddings)

            # Add to FAISS index
            start_position = self._next_position
            self._index.add(embeddings)

            # Store metadata for each chunk
            for i, chunk in enumerate(chunks):
                position = start_position + i

                # Store chunk metadata
                self._chunk_metadata[position] = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "embedding_dimension": len(chunk.embedding)
                }

                # Update position mapping
                self._chunk_id_to_position[chunk.chunk_id] = position

            # Update next position
            self._next_position += len(chunks)

            # Update statistics
            self.stats["chunks_indexed"] += len(chunks)

            logger.debug(
                f"Added {len(chunks)} chunks to FAISS index (positions {start_position}-{self._next_position - 1})")

        except Exception as e:
            error_msg = f"Failed to add chunks to FAISS index: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

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
        start_time = time.time()

        # Convert string to VectorQuery if needed
        if isinstance(query, str):
            query = VectorQuery(text=query)

        logger.debug(f"Searching for: '{query.text}' (top_k={query.top_k})")

        # Convert query text to embedding
        if not self.embedder:
            raise VectorStoreError("No embedder provided. Cannot convert query text to embedding.")

        try:
            query_embedding = self.embedder.embed_text(query.text)
        except Exception as e:
            error_msg = f"Failed to embed query text: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

        # Search by embedding
        results = self.search_by_embedding(query_embedding, query.top_k)

        # Apply filters if specified
        if query.filters:
            results = self._apply_filters(results, query.filters)

        # Apply minimum score threshold
        if query.min_score is not None:
            results = [r for r in results if r.score >= query.min_score]

        # Update statistics
        self.stats["total_searches"] += 1
        self.stats["last_search_time"] = time.time()
        search_time = time.time() - start_time

        logger.info(f"Search completed: {len(results)} results in {search_time:.3f}s")

        return results

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
        if self.get_chunk_count() == 0:
            logger.warning("Vector store is empty")
            return []

        try:
            # Prepare query embedding for FAISS
            query_vector = np.array([embedding], dtype=np.float32)

            # Normalize for cosine similarity
            faiss = self._get_faiss()
            faiss.normalize_L2(query_vector)

            # Search FAISS index
            actual_k = min(top_k, self.get_chunk_count())
            scores, indices = self._index.search(query_vector, actual_k)

            # Convert to SearchResult objects
            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue

                # Get chunk metadata
                chunk_meta = self._chunk_metadata.get(idx)
                if not chunk_meta:
                    logger.warning(f"No metadata found for index {idx}")
                    continue

                # Reconstruct DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=chunk_meta["chunk_id"],
                    content=chunk_meta["content"],
                    source=chunk_meta["source"],
                    chunk_index=chunk_meta["chunk_index"],
                    metadata=chunk_meta["metadata"],
                    embedding=embedding  # We could store original embedding if needed
                )

                # Create SearchResult
                result = SearchResult(
                    chunk=chunk,
                    score=float(score),  # Convert numpy float to Python float
                    rank=rank,
                    metadata={"faiss_index": int(idx)}
                )

                results.append(result)

            logger.debug(f"Found {len(results)} similar chunks")
            return results

        except Exception as e:
            error_msg = f"FAISS search failed: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the vector store"""
        return self._index.ntotal if self._index else 0

    def remove_document(self, source: str) -> int:
        """
        Remove all chunks from a specific document

        Args:
            source: Source filename to remove

        Returns:
            Number of chunks removed
        """
        # Find positions of chunks to remove
        positions_to_remove = []
        chunk_ids_to_remove = []

        # Identify chunks to remove
        for pos, meta in list(self._chunk_metadata.items()):
            if meta.get("source") == source or meta.get("source", "").endswith(f"/{source}") or meta.get("source",
                                                                                                         "").endswith(
                    f"\\{source}"):
                positions_to_remove.append(pos)
                chunk_ids_to_remove.append(meta.get("chunk_id"))

        if not positions_to_remove:
            logger.warning(f"No chunks found for document '{source}'")
            return 0

        logger.info(f"Removing {len(positions_to_remove)} chunks for document '{source}'")

        # Since FAISS doesn't support direct removal, we need to rebuild the index
        # 1. Get all embeddings except those to remove
        remaining_positions = [i for i in range(self._index.ntotal) if i not in positions_to_remove]

        if not remaining_positions:
            # If removing all chunks, just clear the index
            self.clear()
            return len(positions_to_remove)

        # 2. Create new index
        import faiss
        new_index = faiss.IndexFlatIP(self.embedding_dimension)

        # 3. Add remaining vectors to the new index
        if remaining_positions:
            vectors = self._index.reconstruct_batch(remaining_positions)
            new_index.add(vectors)

        # 4. Update metadata
        new_metadata = {}
        new_id_to_position = {}
        position_map = {}  # Maps old positions to new positions

        new_pos = 0
        for old_pos in remaining_positions:
            position_map[old_pos] = new_pos
            new_metadata[new_pos] = self._chunk_metadata[old_pos]

            # Update chunk_id to position mapping
            chunk_id = new_metadata[new_pos]["chunk_id"]
            new_id_to_position[chunk_id] = new_pos

            new_pos += 1

        # 5. Replace the index and metadata
        self._index = new_index
        self._chunk_metadata = new_metadata
        self._chunk_id_to_position = new_id_to_position
        self._next_position = new_pos

        return len(positions_to_remove)
    def clear(self) -> None:
        """Remove all documents and chunks from the vector store"""
        logger.info("Clearing vector store")

        # Reinitialize FAISS index
        self._initialize_index()

        # Clear metadata
        self._chunk_metadata.clear()
        self._chunk_id_to_position.clear()
        self._next_position = 0

        # Reset statistics
        self.reset_stats()

        logger.info("Vector store cleared")

    def save_index(self, path: Union[str, Path]) -> None:
        """
        Save the vector index to disk for persistence

        Args:
            path: Directory path where to save the index

        Raises:
            VectorStoreError: If saving fails
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        try:
            faiss = self._get_faiss()

            # Save FAISS index
            faiss_path = path / "faiss_index.bin"
            faiss.write_index(self._index, str(faiss_path))

            # Save metadata
            metadata_path = path / "chunk_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self._chunk_metadata, f, indent=2, ensure_ascii=False)

            # Save index mappings and configuration
            config_path = path / "index_config.json"
            config_data = {
                "embedding_dimension": self.embedding_dimension,
                "index_type": self.index_type,
                "chunk_id_to_position": self._chunk_id_to_position,
                "next_position": self._next_position,
                "stats": self.stats,
                "created_at": time.time()
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Vector store saved to {path}")

        except Exception as e:
            error_msg = f"Failed to save vector store: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def load_index(self, path: Union[str, Path]) -> None:
        """
        Load a vector index from disk

        Args:
            path: Directory path to the saved index

        Raises:
            VectorStoreError: If loading fails
        """
        path = Path(path)

        if not path.exists():
            raise VectorStoreError(f"Index path does not exist: {path}")

        try:
            faiss = self._get_faiss()

            # Load FAISS index
            faiss_path = path / "faiss_index.bin"
            if not faiss_path.exists():
                raise VectorStoreError(f"FAISS index file not found: {faiss_path}")

            self._index = faiss.read_index(str(faiss_path))

            # Load metadata
            metadata_path = path / "chunk_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    # Convert string keys back to integers
                    loaded_metadata = json.load(f)
                    self._chunk_metadata = {int(k): v for k, v in loaded_metadata.items()}

            # Load configuration
            config_path = path / "index_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                self.embedding_dimension = config_data.get("embedding_dimension", self.embedding_dimension)
                self.index_type = config_data.get("index_type", self.index_type)
                self._chunk_id_to_position = config_data.get("chunk_id_to_position", {})
                self._next_position = config_data.get("next_position", 0)
                self.stats = config_data.get("stats", self.stats)

            logger.info(f"Vector store loaded from {path} ({self.get_chunk_count()} chunks)")

        except Exception as e:
            error_msg = f"Failed to load vector store: {e}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """
        Apply filters to search results

        Args:
            results: List of search results to filter
            filters: Dictionary of filters to apply

        Returns:
            Filtered list of search results
        """
        filtered_results = []

        for result in results:
            match = True

            # Check each filter
            for key, value in filters.items():
                if key == "source":
                    if result.chunk.source != value:
                        match = False
                        break
                elif key == "chunk_index":
                    if result.chunk.chunk_index != value:
                        match = False
                        break
                # TODO: Add more filter types as needed

            if match:
                filtered_results.append(result)

        return filtered_results

    def _get_faiss(self):
        """Get FAISS module with proper error handling"""
        try:
            import faiss
            return faiss
        except ImportError:
            raise VectorStoreError("FAISS library not available")

    def get_embedder_info(self) -> Dict[str, Any]:
        """Get information about the embedder being used"""
        if not self.embedder:
            return {"embedder": None}

        if hasattr(self.embedder, 'get_model_info'):
            return {"embedder": self.embedder.get_model_info()}
        else:
            return {"embedder": str(type(self.embedder))}

    def __repr__(self) -> str:
        chunk_count = self.get_chunk_count()
        return f"FaissVectorStore(chunks={chunk_count}, dim={self.embedding_dimension}, type={self.index_type})"

    # Add this method to src/vector_stores/faiss_store.py
    def get_indexed_sources(self) -> set:
        """Get a set of all source files that have been indexed"""
        sources = set()
        for meta in self._chunk_metadata.values():
            if 'source' in meta:
                sources.add(meta['source'])
        return sources

# Helper function for easy instantiation
def create_faiss_store(embedder=None, dimension: int = 384) -> FaissVectorStore:
    """
    Create a FAISS vector store with sensible defaults

    Args:
        embedder: Embedder instance for query processing
        dimension: Embedding dimension

    Returns:
        Configured FaissVectorStore instance
    """
    return FaissVectorStore(
        embedding_dimension=dimension,
        embedder=embedder,
        index_type="flat"  # Best for small-medium datasets
    )

