import unittest
from unittest.mock import MagicMock
from src.llm.rag_service import RAGService


class TestRAGService(unittest.TestCase):
    def setUp(self):
        # Create mock vector store and LLM
        self.mock_vector_store = MagicMock()
        self.mock_llm = MagicMock()

        # Mock the SearchResult objects
        mock_chunks = []
        for i in range(3):
            mock_chunk = MagicMock()
            mock_chunk.content = f"Test content {i}"
            mock_chunk.source = "test.docx"
            mock_chunk.chunk_index = i
            mock_chunk.metadata = {"page": 1}

            mock_result = MagicMock()
            mock_result.chunk = mock_chunk
            mock_result.score = 0.8 - (i * 0.1)  # 0.8, 0.7, 0.6
            mock_result.rank = i

            mock_chunks.append(mock_result)

        # Configure mock vector store to return these results
        self.mock_vector_store.search.return_value = mock_chunks

        # Configure mock LLM
        self.mock_llm.generate.return_value = "Generated answer"

        # Create RAG service
        self.rag = RAGService(
            vector_store=self.mock_vector_store,
            llm=self.mock_llm
        )

    def test_answer(self):
        # Test the answer method
        result = self.rag.answer("Test question")

        # Check that vector store was called
        self.mock_vector_store.search.assert_called_once()

        # Check that LLM was called with a prompt containing context
        llm_call_args = self.mock_llm.generate.call_args[0][0]
        self.assertIn("PASSAGE 1", llm_call_args)
        self.assertIn("Test content 0", llm_call_args)
        self.assertIn("Test question", llm_call_args)

        # Check the result structure
        self.assertEqual(result["answer"], "Generated answer")
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(result["sources"][0]["source"], "test.docx")
        self.assertIn("retrieval_time", result)
        self.assertIn("generation_time", result)
        self.assertIn("total_time", result)

    def test_answer_no_results(self):
        # Set vector store to return empty results
        self.mock_vector_store.search.return_value = []

        # Test the answer method
        result = self.rag.answer("Test question")

        # LLM should not be called
        self.mock_llm.generate.assert_not_called()

        # Check the result structure
        self.assertIn("I don't have enough information", result["answer"])
        self.assertEqual(len(result["sources"]), 0)

    def test_get_system_info(self):
        # Configure mock components
        self.mock_vector_store.get_chunk_count.return_value = 10
        self.mock_vector_store.get_embedder_info.return_value = {"model": "test-embedder"}
        self.mock_llm.get_model_info.return_value = {"model": "test-llm"}

        # Test the get_system_info method
        info = self.rag.get_system_info()

        # Check the result structure
        self.assertEqual(info["vector_store"]["chunks"], 10)
        self.assertEqual(info["vector_store"]["embedder"]["model"], "test-embedder")
        self.assertEqual(info["llm"]["model"], "test-llm")
        self.assertIn("config", info)