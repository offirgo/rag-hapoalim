# src/test_rag.py
import logging
import argparse
from pathlib import Path

from src.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.vector_stores.faiss_store import FaissVectorStore
from src.llm.echo import EchoLLM
from src.llm.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description="Test RAG system")
    parser.add_argument("--index", type=str, required=True, help="Path to vector index")
    parser.add_argument("--query", type=str, help="Test query (if not provided, will prompt)")
    args = parser.parse_args()

    # Initialize components
    print("Initializing embedder...")
    embedder = SentenceTransformersEmbedder()

    print("Initializing vector store...")
    vector_store = FaissVectorStore(
        embedding_dimension=embedder.embedding_dim,
        embedder=embedder
    )

    # Load vector index
    index_path = Path(args.index)
    if index_path.exists():
        print(f"Loading vector index from {index_path}...")
        vector_store.load_index(index_path)
        print(f"Loaded {vector_store.get_chunk_count()} chunks")
    else:
        print(f"Error: Vector index not found at {index_path}")
        return

    # Initialize LLM (using EchoLLM for testing)
    print("Initializing test LLM...")
    llm = EchoLLM()

    # Initialize RAG service
    rag = RAGService(
        vector_store=vector_store,
        llm=llm,
        config={
            "default_top_k": 3,
            "default_min_score": 0.5
        }
    )

    # Print system info
    print("\nRAG System Info:")
    info = rag.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Process query
    query = args.query
    while True:
        if not query:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() in ('quit', 'exit', 'q'):
                break

        print(f"\nQuestion: {query}")
        print("Generating answer...")

        result = rag.answer(query)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for i, source in enumerate(result["sources"]):
            print(f"  {i + 1}. {source['source']} (score: {source['score']:.3f})")

        print(
            f"\nTime: {result['total_time']:.2f}s (retrieval: {result['retrieval_time']:.2f}s, generation: {result['generation_time']:.2f}s)")

        # Reset query to enable loop
        query = None

# python test_rag.py [--index INDEX_PATH] [--query "Your test question"]
### Parameters

# - `--index`: Path to the vector index directory (default: "vector_index")
# - `--query`: (Optional) Test query. If not provided, the script will prompt for input interactively

# python fast_test_rag.py --index vector_index --query  "What is the company policy on remote work?"

if __name__ == "__main__":
    main()