# src/create_index.py
import logging
import argparse
from pathlib import Path
import os

from src.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.vector_stores.faiss_store import FaissVectorStore
from src.document_processor.word_processor import WordProcessor
from src.document_processor.excel_processor import ExcelProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description="Create vector index from documents")
    parser.add_argument("--data", type=str, required=True, help="Directory with documents")
    parser.add_argument("--output", type=str, required=True, help="Output directory for index")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)

    # Initialize embedder
    print("Initializing embedder...")
    embedder = SentenceTransformersEmbedder()

    # Initialize processors
    word_processor = WordProcessor()
    excel_processor = ExcelProcessor()

    # Initialize vector store
    print("Initializing vector store...")
    vector_store = FaissVectorStore(
        embedding_dimension=embedder.embedding_dim,
        embedder=embedder
    )

    # Process documents
    files_processed = 0
    for file_path in data_dir.glob("**/*"):
        if not file_path.is_file():
            continue

        print(f"Processing {file_path.name}...")

        try:
            # Choose processor based on file extension
            if file_path.suffix.lower() == ".docx":
                document = word_processor.process(file_path)
            elif file_path.suffix.lower() in (".xlsx", ".xls"):
                document = excel_processor.process(file_path)
            else:
                print(f"Skipping unsupported file: {file_path.name}")
                continue

            # Generate embeddings
            document_with_embeddings = document.with_embeddings(embedder)

            # Add to vector store
            vector_store.add_document(document_with_embeddings)
            files_processed += 1

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Save index
    if files_processed > 0:
        print(f"Saving index to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        vector_store.save_index(output_dir)
        print(f"Index saved with {vector_store.get_chunk_count()} chunks from {files_processed} files")
    else:
        print("No files were processed successfully. Index not saved.")


if __name__ == "__main__":
    main()

# how to run - from root folder run
    # python  create_index.py - -data # path / to / your / documents --output  # path / to / save / index
    # python src/create_index.py --data data --output vector_index