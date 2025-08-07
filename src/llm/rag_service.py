import logging
import time
from typing import Dict, Any, List, Optional

from src.vector_stores.base import BaseVectorStore, VectorQuery
from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation (RAG) service

    Coordinates retrieval of relevant chunks and generation of responses
    """

    def __init__(
            self,
            vector_store: BaseVectorStore,
            llm: BaseLLM,
            config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RAG service

        Args:
            vector_store: Vector store for retrieving chunks
            llm: LLM client for generating responses
            config: Optional configuration
        """
        self.vector_store = vector_store
        self.llm = llm
        self.config = config or {}

        # Default parameters
        self.default_top_k = self.config.get("default_top_k", 5)
        self.default_min_score = self.config.get("default_min_score", 0.6)

        # Prompt templates
        self.prompt_template = self.config.get("prompt_template", self._default_prompt_template)

        logger.info(f"RAGService initialized with {vector_store} and {llm.__class__.__name__}")

    def _default_prompt_template(self, query: str, contexts: List[str]) -> str:
        """
        Default prompt template for RAG

        Args:
            query: User query
            contexts: List of context passages

        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join([f"PASSAGE {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        return f"""You are a helpful AI assistant. Answer the question based only on the provided passages. 
If the information needed is not present in the passages, say "I don't have enough information to answer this question."

PASSAGES:
{context_text}

QUESTION: {query}

ANSWER:"""

    def answer(
            self,
            query: str,
            top_k: Optional[int] = None,
            min_score: Optional[float] = None,
            filters: Optional[Dict[str, Any]] = None,
            **llm_params
    ) -> Dict[str, Any]:
        """
        Generate answer to query using RAG

        Args:
            query: User question
            top_k: Number of chunks to retrieve (default: from config)
            min_score: Minimum similarity score (default: from config)
            filters: Optional filters for retrieval
            **llm_params: Additional parameters for LLM

        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()

        # Use defaults if not specified
        top_k = top_k or self.default_top_k
        min_score = min_score or self.default_min_score

        # 1. Retrieve relevant chunks
        logger.debug(f"Retrieving chunks for query: '{query}'")
        vector_query = VectorQuery(
            text=query,
            top_k=top_k or self.default_top_k,
            min_score=min_score or self.default_min_score,
            filters=filters
        )
        results = self.vector_store.search(vector_query)

        # Check if we have any results
        if not results:
            logger.warning(f"No relevant chunks found for query: '{query}'")
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "retrieval_time": 0,
                "generation_time": 0,
                "total_time": time.time() - start_time
            }

        # 2. Format prompt with retrieved chunks
        chunks = [result.chunk.content for result in results]
        prompt = self.prompt_template(query, chunks) if callable(self.prompt_template) else self.prompt_template.format(
            query=query,
            contexts="\n\n".join([f"PASSAGE {i + 1}:\n{ctx}" for i, ctx in enumerate(chunks)])
        )

        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieved {len(results)} chunks in {retrieval_time:.2f}s")

        # 3. Generate answer with LLM
        generation_start = time.time()
        answer = self.llm.generate(prompt, **llm_params)
        generation_time = time.time() - generation_start

        # 4. Format and return response
        sources = []
        for result in results:
            chunk = result.chunk
            sources.append({
                "source": chunk.source,
                "score": result.score,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata
            })

        total_time = time.time() - start_time
        logger.info(
            f"Generated answer in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system"""
        return {
            "vector_store": {
                "type": self.vector_store.__class__.__name__,
                "chunks": self.vector_store.get_chunk_count(),
                "embedder": self.vector_store.get_embedder_info() if hasattr(self.vector_store,
                                                                             'get_embedder_info') else None
            },
            "llm": self.llm.get_model_info(),
            "config": {
                "default_top_k": self.default_top_k,
                "default_min_score": self.default_min_score
            }
        }