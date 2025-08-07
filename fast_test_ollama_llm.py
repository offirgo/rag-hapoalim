# test_rag_with_ollama.py
import logging
import argparse
from pathlib import Path
import subprocess
import sys
import platform

from src.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.vector_stores.faiss_store import FaissVectorStore
from src.llm.echo import EchoLLM
from src.llm.ollama_llm import OllamaLLM
from src.llm.rag_service import RAGService
from src.llm.base import LLMError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_ollama_installed():
    """Check if Ollama is installed and accessible"""
    try:
        # Different command based on OS
        if platform.system() == "Windows":
            result = subprocess.run(["where", "ollama"], capture_output=True, text=True)
        else:  # macOS or Linux
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)

        return result.returncode == 0
    except Exception:
        return False


def display_ollama_instructions():
    """Display instructions for installing Ollama"""
    print("\n" + "=" * 80)
    print("OLLAMA NOT FOUND")
    print("=" * 80)
    print("To use real LLM capabilities, please install Ollama:")
    print("1. Visit https://ollama.ai/ and download the installer for your OS")
    print("2. Run the installer and follow the instructions")
    print("3. Start Ollama: ollama serve")
    print("4. Pull a model: ollama pull llama3")
    print("\nFor now, using EchoLLM (test-only implementation)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test RAG system with Ollama")
    parser.add_argument("--index", type=str, default="vector_index",
                        help="Path to vector index (default: vector_index)")
    parser.add_argument("--model", type=str, default="llama3",
                        help="Ollama model to use (default: llama3)")
    parser.add_argument("--query", type=str, help="Test query (if not provided, will prompt)")
    args = parser.parse_args()

    # Initialize embedder
    print("Initializing embedder...")
    embedder = SentenceTransformersEmbedder()

    # Initialize vector store
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

    # Initialize LLM
    if check_ollama_installed():
        try:
            print(f"Initializing Ollama LLM with model '{args.model}'...")
            llm = OllamaLLM(model=args.model)
            model_info = llm.get_model_info()
            print(f"Using model: {model_info.get('model')} ({model_info.get('parameter_size', 'unknown')})")
        except LLMError as e:
            print(f"Error initializing Ollama: {e}")
            print("Falling back to EchoLLM")
            llm = EchoLLM()
    else:
        display_ollama_instructions()
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


if __name__ == "__main__":
    main()