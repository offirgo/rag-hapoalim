import unittest
import tempfile
import numpy as np
from pathlib import Path
import shutil

from src.models.document import (
    ProcessedDocument, DocumentMetadata, DocumentChunk,
    DocumentType, ProcessingStatus
)
from src.vector_stores.faiss_store import FaissVectorStore


class MockEmbedder:
    """Mock embedder for testing"""

    def embed_text(self, text):
        """Return a fake embedding of the correct dimension"""
        # Generate a reproducible embedding based on the hash of the text
        np.random.seed(hash(text) % 2 ** 32)
        embedding = np.random.rand(384).astype(np.float32)
        # Normalize the embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def get_model_info(self):
        return {"name": "MockEmbedder", "dimension": 384}


class TestFaissVectorStore(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test"""
        self.embedder = MockEmbedder()
        self.store = FaissVectorStore(embedding_dimension=384, embedder=self.embedder)
        self.temp_dir = tempfile.mkdtemp()

        # Create test chunks and documents
        self.test_chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                content="This is the first test chunk about AI.",
                source="test_doc.docx",
                chunk_index=0,
                metadata={"page": 1},
                embedding=self.embedder.embed_text("This is the first test chunk about AI.")
            ),
            DocumentChunk(
                chunk_id="chunk2",
                content="This is the second test chunk about machine learning.",
                source="test_doc.docx",
                chunk_index=1,
                metadata={"page": 1},
                embedding=self.embedder.embed_text("This is the second test chunk about machine learning.")
            ),
            DocumentChunk(
                chunk_id="chunk3",
                content="This chunk is about databases and storage systems.",
                source="test_doc2.docx",
                chunk_index=0,
                metadata={"page": 1},
                embedding=self.embedder.embed_text("This chunk is about databases and storage systems.")
            )
        ]

        # Create a test document with proper enum values
        self.test_doc = ProcessedDocument(
            metadata=DocumentMetadata(
                filename="test_doc.docx",
                document_type=DocumentType.WORD,  # Use enum value
                file_size=1024,
                processing_status=ProcessingStatus.COMPLETED
                # processed_at has a default factory, so we don't need to provide it
                # processing_status has a default, so we don't need to provide it
            ),
            chunks=self.test_chunks[:2]  # Use first two chunks
        )

    def tearDown(self):
        """Clean up after each test"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that the store initializes correctly"""
        self.assertEqual(self.store.embedding_dimension, 384)
        self.assertEqual(self.store.index_type, "flat")
        self.assertEqual(self.store.get_chunk_count(), 0)

    def test_add_chunks(self):
        """Test adding chunks to the store"""
        self.store.add_chunks(self.test_chunks)
        self.assertEqual(self.store.get_chunk_count(), 3)

    def test_add_document(self):
        """Test adding a document to the store"""
        self.store.add_document(self.test_doc)
        self.assertEqual(self.store.get_chunk_count(), 2)
        self.assertEqual(self.store.stats["documents_added"], 1)

    def test_search(self):
        """Test searching the store"""
        # Add chunks first
        self.store.add_chunks(self.test_chunks)

        # Search for something related to AI
        results = self.store.search("artificial intelligence")

        # We should get results
        self.assertGreater(len(results), 0)

        # The first chunk should be more relevant than the database chunk
        ai_chunks = [r for r in results if "AI" in r.chunk.content]
        db_chunks = [r for r in results if "databases" in r.chunk.content]

        if ai_chunks and db_chunks:
            self.assertLess(ai_chunks[0].rank, db_chunks[0].rank)

    def test_persistence(self):
        """Test saving and loading the index"""
        # Add chunks
        self.store.add_chunks(self.test_chunks)

        # Save the index
        save_path = Path(self.temp_dir) / "test_index"
        self.store.save_index(save_path)

        # Verify files were created
        self.assertTrue((save_path / "faiss_index.bin").exists())
        self.assertTrue((save_path / "chunk_metadata.json").exists())
        self.assertTrue((save_path / "index_config.json").exists())

        # Create a new store and load the index
        new_store = FaissVectorStore(embedding_dimension=384, embedder=self.embedder)
        new_store.load_index(save_path)

        # Verify chunk count matches
        self.assertEqual(new_store.get_chunk_count(), 3)

        # Verify search works on loaded index
        results = new_store.search("artificial intelligence")
        self.assertGreater(len(results), 0)

    def test_clear(self):
        """Test clearing the store"""
        # Add chunks
        self.store.add_chunks(self.test_chunks)
        self.assertEqual(self.store.get_chunk_count(), 3)

        # Clear the store
        self.store.clear()

        # Verify it's empty
        self.assertEqual(self.store.get_chunk_count(), 0)


if __name__ == "__main__":
    unittest.main()