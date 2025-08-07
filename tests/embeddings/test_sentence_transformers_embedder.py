"""
Tests for SentenceTransformersEmbedder
Comprehensive test suite with mocked dependencies to avoid requiring actual models
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.models.document import DocumentChunk
from src.embeddings import SentenceTransformersEmbedder, create_embedder, EmbeddingError


class TestSentenceTransformersEmbedderInitialization:
    """Test embedder initialization and configuration"""

    def test_init_with_default_model(self):
        """Test initialization with default model"""
        embedder = SentenceTransformersEmbedder()

        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.embedding_dim == 384  # Known dimension for this model
        assert embedder._model is None  # Lazy loading - not loaded yet
        assert embedder.stats["total_chunks_processed"] == 0

    def test_init_with_custom_model(self):
        """Test initialization with custom model"""
        embedder = SentenceTransformersEmbedder(model_name="all-mpnet-base-v2")

        assert embedder.model_name == "all-mpnet-base-v2"
        assert embedder.embedding_dim == 768  # Known dimension for this model
        assert embedder._model is None

    def test_init_with_unknown_model(self):
        """Test initialization with unknown model name"""
        with patch('src.embeddings.sentence_transformers_embedder.logger') as mock_logger:
            embedder = SentenceTransformersEmbedder(model_name="unknown-model")

            assert embedder.model_name == "unknown-model"
            assert embedder.embedding_dim is None  # Unknown model dimension
            mock_logger.warning.assert_called_once()

    def test_init_with_config(self):
        """Test initialization with custom configuration"""
        config = {"some_option": "value"}
        embedder = SentenceTransformersEmbedder(config=config)

        assert embedder.config == config

    def test_list_available_models(self):
        """Test getting list of available models"""
        models = SentenceTransformersEmbedder.list_available_models()

        assert isinstance(models, dict)
        assert "all-MiniLM-L6-v2" in models
        assert models["all-MiniLM-L6-v2"] == 384
        assert "all-mpnet-base-v2" in models
        assert models["all-mpnet-base-v2"] == 768


class TestSentenceTransformersEmbedderModelLoading:
    """Test model loading functionality"""

    def setup_method(self):
        """Set up test embedder"""
        self.embedder = SentenceTransformersEmbedder()

    @patch('sentence_transformers.SentenceTransformer')
    def test_successful_model_loading(self, mock_sentence_transformer):
        """Test successful model loading"""
        # Mock the SentenceTransformer class
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Load model
        model = self.embedder._load_model()

        assert model == mock_model
        assert self.embedder._model == mock_model
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    @patch('sentence_transformers.SentenceTransformer')
    def test_model_loading_with_dimension_detection(self, mock_sentence_transformer):
        """Test model loading with automatic dimension detection"""
        # Create embedder with unknown model
        embedder = SentenceTransformersEmbedder(model_name="unknown-model")

        # Mock model and its encode method
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])  # 3-dimensional test embedding
        mock_sentence_transformer.return_value = mock_model

        # Load model
        model = embedder._load_model()

        assert embedder.embedding_dim == 3  # Should detect dimension
        mock_model.encode.assert_called_once_with("test", convert_to_tensor=False)

    def test_model_loading_import_error(self):
        """Test model loading when sentence-transformers is not installed"""
        # Mock import error
        with patch('sentence_transformers.SentenceTransformer',
                   side_effect=ImportError("No module named 'sentence_transformers'")):
            with pytest.raises(EmbeddingError, match="sentence-transformers library not found"):
                self.embedder._load_model()

    @patch('sentence_transformers.SentenceTransformer')
    def test_model_loading_general_error(self, mock_sentence_transformer):
        """Test model loading with general error"""
        mock_sentence_transformer.side_effect = Exception("Model download failed")

        with pytest.raises(EmbeddingError, match="Failed to load model"):
            self.embedder._load_model()

    @patch('sentence_transformers.SentenceTransformer')
    def test_model_cached_after_first_load(self, mock_sentence_transformer):
        """Test that model is cached after first successful load"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        # Load model twice
        model1 = self.embedder._load_model()
        model2 = self.embedder._load_model()

        # Should be the same instance
        assert model1 == model2
        # SentenceTransformer should only be called once
        mock_sentence_transformer.assert_called_once()


class TestSentenceTransformersEmbedderTextEmbedding:
    """Test text embedding functionality"""

    def setup_method(self):
        """Set up test embedder with mocked model"""
        self.embedder = SentenceTransformersEmbedder()

        # Mock the model
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])

        # Patch the model loading to return our mock
        self.model_patcher = patch.object(self.embedder, '_load_model', return_value=self.mock_model)
        self.model_patcher.start()

    def teardown_method(self):
        """Clean up patches"""
        self.model_patcher.stop()

    def test_embed_text_success(self):
        """Test successful text embedding"""
        text = "This is a test sentence."

        embedding = self.embedder.embed_text(text)

        assert embedding == [0.1, 0.2, 0.3, 0.4]
        self.mock_model.encode.assert_called_once_with(
            text,
            convert_to_tensor=False,
            show_progress_bar=False
        )

    def test_embed_text_with_whitespace(self):
        """Test embedding text with leading/trailing whitespace"""
        text = "  This is a test sentence.  "

        embedding = self.embedder.embed_text(text)

        # Should strip whitespace before encoding
        self.mock_model.encode.assert_called_once_with(
            "This is a test sentence.",
            convert_to_tensor=False,
            show_progress_bar=False
        )

    def test_embed_empty_text(self):
        """Test embedding empty text"""
        with pytest.raises(EmbeddingError, match="Cannot embed empty or whitespace-only text"):
            self.embedder.embed_text("")

    def test_embed_whitespace_only_text(self):
        """Test embedding whitespace-only text"""
        with pytest.raises(EmbeddingError, match="Cannot embed empty or whitespace-only text"):
            self.embedder.embed_text("   \n\t   ")

    def test_embed_text_model_error(self):
        """Test embedding when model throws error"""
        self.mock_model.encode.side_effect = Exception("CUDA out of memory")

        with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
            self.embedder.embed_text("test text")


class TestSentenceTransformersEmbedderChunkEmbedding:
    """Test chunk embedding functionality"""

    def setup_method(self):
        """Set up test embedder and chunks"""
        self.embedder = SentenceTransformersEmbedder()

        # Create test chunks
        self.chunks = [
            DocumentChunk(
                chunk_id="chunk_001",
                content="First chunk content",
                source="test.docx",
                chunk_index=0,
                metadata={"section": "intro"}
            ),
            DocumentChunk(
                chunk_id="chunk_002",
                content="Second chunk content",
                source="test.docx",
                chunk_index=1,
                metadata={"section": "body"}
            )
        ]

        # Mock successful embedding generation
        self.embed_text_patcher = patch.object(
            self.embedder,
            'embed_text',
            side_effect=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        self.embed_text_patcher.start()

    def teardown_method(self):
        """Clean up patches"""
        self.embed_text_patcher.stop()

    def test_embed_chunks_success(self):
        """Test successful chunk embedding"""
        with patch('src.embeddings.base.logger') as mock_logger:
            result = self.embedder.embed_chunks(self.chunks)

        # Should return same number of chunks
        assert len(result) == 2

        # All chunks should have embeddings
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[1].embedding == [0.4, 0.5, 0.6]

        # Original data should be preserved
        assert result[0].chunk_id == "chunk_001"
        assert result[0].content == "First chunk content"
        assert result[1].chunk_id == "chunk_002"

        # Statistics should be updated
        assert self.embedder.stats["successful_embeddings"] == 2
        assert self.embedder.stats["failed_embeddings"] == 0
        assert self.embedder.stats["total_chunks_processed"] == 2

    def test_embed_empty_chunks_list(self):
        """Test embedding empty chunks list"""
        result = self.embedder.embed_chunks([])

        assert result == []
        assert self.embedder.stats["total_chunks_processed"] == 0

    def test_embed_chunks_with_failures(self):
        """Test chunk embedding with some failures"""
        # Mock embed_text to fail on second chunk
        self.embed_text_patcher.stop()
        with patch.object(self.embedder, 'embed_text') as mock_embed:
            mock_embed.side_effect = [
                [0.1, 0.2, 0.3],  # First chunk succeeds
                Exception("Embedding failed")  # Second chunk fails
            ]

            with patch('src.embeddings.base.logger') as mock_logger:
                result = self.embedder.embed_chunks(self.chunks)

        # Should return all chunks
        assert len(result) == 2

        # First chunk should have embedding
        assert result[0].embedding == [0.1, 0.2, 0.3]

        # Second chunk should have None embedding
        assert result[1].embedding is None
        assert result[1].content == "Second chunk content"  # Other data preserved

        # Statistics should reflect failure
        assert self.embedder.stats["successful_embeddings"] == 1
        assert self.embedder.stats["failed_embeddings"] == 1

        # Should log error
        mock_logger.error.assert_called()
        mock_logger.warning.assert_called()



class TestSentenceTransformersEmbedderUtilityMethods:
    """Test utility methods"""

    def setup_method(self):
        """Set up test embedder"""
        self.embedder = SentenceTransformersEmbedder()

    def test_get_embedding_dimension_known_model(self):
        """Test getting dimension for known model"""
        dimension = self.embedder.get_embedding_dimension()
        assert dimension == 384  # Known dimension for all-MiniLM-L6-v2

    def test_get_embedding_dimension_unknown_model(self):
        """Test getting dimension for unknown model (forces model loading)"""
        embedder = SentenceTransformersEmbedder(model_name="unknown-model")

        def mock_load_model():
            embedder.embedding_dim = 512  # Set dimension inside the mock
            return MagicMock()  # Return a mock model

        # Mock model loading to set dimension
        with patch.object(embedder, '_load_model', side_effect=mock_load_model) as mock_load:
            dimension = embedder.get_embedding_dimension()
            assert dimension == 512
            mock_load.assert_called_once()

    def test_get_model_info(self):
        """Test getting model information"""
        info = self.embedder.get_model_info()

        assert info["provider"] == "sentence-transformers"
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["embedding_dimension"] == 384
        assert info["local"] is True
        assert info["requires_api_key"] is False
        assert "quality_note" in info
        assert "performance_note" in info

    def test_get_stats(self):
        """Test getting statistics"""
        # Modify some stats
        self.embedder.stats["successful_embeddings"] = 5
        self.embedder.stats["failed_embeddings"] = 1

        stats = self.embedder.get_stats()

        assert stats["successful_embeddings"] == 5
        assert stats["failed_embeddings"] == 1
        assert isinstance(stats, dict)

        # Should be a copy, not reference
        stats["successful_embeddings"] = 10
        assert self.embedder.stats["successful_embeddings"] == 5

    def test_reset_stats(self):
        """Test resetting statistics"""
        # Set some stats
        self.embedder.stats["successful_embeddings"] = 5
        self.embedder.stats["total_chunks_processed"] = 10

        self.embedder.reset_stats()

        assert self.embedder.stats["successful_embeddings"] == 0
        assert self.embedder.stats["total_chunks_processed"] == 0

    def test_repr(self):
        """Test string representation"""
        repr_str = repr(self.embedder)

        assert "SentenceTransformersEmbedder" in repr_str
        assert "all-MiniLM-L6-v2" in repr_str
        assert "384" in repr_str or "not loaded" in repr_str


class TestCreateEmbedderHelper:
    """Test the helper function for creating embedders"""

    def test_create_embedder_default(self):
        """Test creating embedder with defaults"""
        embedder = create_embedder()

        assert isinstance(embedder, SentenceTransformersEmbedder)
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_create_embedder_custom_model(self):
        """Test creating embedder with custom model"""
        embedder = create_embedder(model_name="all-mpnet-base-v2")

        assert isinstance(embedder, SentenceTransformersEmbedder)
        assert embedder.model_name == "all-mpnet-base-v2"


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/embedding/test_sentence_transformers_embedder.py -v
    pytest.main([__file__, "-v"])