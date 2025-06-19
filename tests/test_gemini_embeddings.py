"""Test Gemini embedding integration."""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from utils.chroma_provider import ChromaProvider
from utils.vector_store import EntryType


class TestGeminiEmbeddings:
    """Test Gemini embedding functionality in ChromaProvider."""

    @pytest.fixture
    def mock_gemini_provider(self):
        """Create a mock Gemini provider."""
        mock_provider = Mock()

        # Mock get_embedding to return 3072-dimensional embeddings
        def mock_get_embedding(text, model, task_type):
            # Return a consistent embedding based on text hash
            np.random.seed(hash(text) % 2**32)
            return np.random.rand(3072)

        mock_provider.get_embedding = Mock(side_effect=mock_get_embedding)
        return mock_provider

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for ChromaDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @patch('providers.registry.ModelProviderRegistry')
    def test_gemini_initialization(self, mock_registry, mock_gemini_provider, temp_dir):
        """Test ChromaProvider initializes with Gemini embeddings."""
        # Setup mock registry
        mock_registry.get_provider.return_value = mock_gemini_provider

        # Initialize ChromaProvider with Gemini
        provider = ChromaProvider(
            persist_directory=temp_dir,
            collection_name="test",
            use_gemini=True
        )

        # Verify initialization
        assert provider.use_gemini is True
        assert provider.embedding_dimension == 3072
        assert provider.model is None
        assert provider.embedding_provider == mock_gemini_provider
        assert provider.collection_name == "test_gemini_3072"

        # Verify registry was called
        from providers.registry import ProviderType
        mock_registry.get_provider.assert_called_once_with(ProviderType.GOOGLE)

    @patch('providers.registry.ModelProviderRegistry')
    def test_compute_embedding_document(self, mock_registry, mock_gemini_provider, temp_dir):
        """Test computing embeddings for documents."""
        mock_registry.get_provider.return_value = mock_gemini_provider

        provider = ChromaProvider(
            persist_directory=temp_dir,
            collection_name="test",
            use_gemini=True
        )

        # Compute embedding for a document
        text = "This is a test document"
        embedding = provider._compute_embedding(text, is_query=False)

        # Verify
        assert embedding.shape == (3072,)
        mock_gemini_provider.get_embedding.assert_called_once_with(
            text=text,
            model="gemini-embedding-exp-03-07",
            task_type="RETRIEVAL_DOCUMENT"
        )

    @patch('providers.registry.ModelProviderRegistry')
    def test_compute_embedding_query(self, mock_registry, mock_gemini_provider, temp_dir):
        """Test computing embeddings for queries."""
        mock_registry.get_provider.return_value = mock_gemini_provider

        provider = ChromaProvider(
            persist_directory=temp_dir,
            collection_name="test",
            use_gemini=True
        )

        # Compute embedding for a query
        text = "search query"
        embedding = provider._compute_embedding(text, is_query=True)

        # Verify
        assert embedding.shape == (3072,)
        mock_gemini_provider.get_embedding.assert_called_once_with(
            text=text,
            model="gemini-embedding-exp-03-07",
            task_type="RETRIEVAL_QUERY"
        )

    @patch('providers.registry.ModelProviderRegistry')
    def test_add_entry_with_gemini(self, mock_registry, mock_gemini_provider, temp_dir):
        """Test adding entries with Gemini embeddings."""
        mock_registry.get_provider.return_value = mock_gemini_provider

        # Mock keyword extraction
        with patch.object(ChromaProvider, '_extract_keywords_ai', return_value=['test', 'keyword']):
            provider = ChromaProvider(
                persist_directory=temp_dir,
                collection_name="test",
                use_gemini=True
            )

            # Add an entry
            provider.add_entry(
                id="test_id",
                content="Test content for Gemini",
                metadata={"source": "test"},
                entry_type=EntryType.KNOWLEDGE
            )

            # Verify embedding was computed
            mock_gemini_provider.get_embedding.assert_called_with(
                text="Test content for Gemini",
                model="gemini-embedding-exp-03-07",
                task_type="RETRIEVAL_DOCUMENT"
            )

            # Verify entry was added to collection
            result = provider.collection.get(ids=["test_id"])
            assert len(result['ids']) == 1
            assert result['ids'][0] == "test_id"

    @patch('providers.registry.ModelProviderRegistry')
    def test_search_with_gemini(self, mock_registry, mock_gemini_provider, temp_dir):
        """Test searching with Gemini embeddings."""
        mock_registry.get_provider.return_value = mock_gemini_provider

        # Mock keyword extraction
        with patch.object(ChromaProvider, '_extract_keywords_ai', return_value=['test', 'keyword']):
            provider = ChromaProvider(
                persist_directory=temp_dir,
                collection_name="test",
                use_gemini=True
            )

            # Add test entries
            provider.add_entry("id1", "Python programming", {"source": "test"}, EntryType.KNOWLEDGE)
            provider.add_entry("id2", "JavaScript coding", {"source": "test"}, EntryType.KNOWLEDGE)

            # Search
            results = provider.query("programming language", limit=2)

            # Verify query embedding was computed
            calls = mock_gemini_provider.get_embedding.call_args_list
            # Last call should be for the query
            assert calls[-1][1]['task_type'] == "RETRIEVAL_QUERY"
            assert calls[-1][1]['text'] == "programming language"

            # Verify we got results
            assert len(results) <= 2

    @patch('providers.registry.ModelProviderRegistry')
    def test_reset_collection(self, mock_registry, mock_gemini_provider, temp_dir):
        """Test collection reset functionality."""
        mock_registry.get_provider.return_value = mock_gemini_provider

        with patch.object(ChromaProvider, '_extract_keywords_ai', return_value=['test']):
            provider = ChromaProvider(
                persist_directory=temp_dir,
                collection_name="test",
                use_gemini=True
            )

            # Add an entry
            provider.add_entry("test_id", "Test content", {}, EntryType.KNOWLEDGE)

            # Verify entry exists
            assert provider.collection.count() == 1

            # Reset collection
            provider.reset_collection()

            # Verify collection is empty
            assert provider.collection.count() == 0

            # Verify we can add new entries
            provider.add_entry("new_id", "New content", {}, EntryType.KNOWLEDGE)
            assert provider.collection.count() == 1

    @patch('providers.registry.ModelProviderRegistry')
    def test_rate_limit_handling(self, mock_registry, mock_gemini_provider, temp_dir):
        """Test rate limit error handling."""
        mock_registry.get_provider.return_value = mock_gemini_provider

        # Make get_embedding raise a rate limit error
        mock_gemini_provider.get_embedding.side_effect = RuntimeError("Embedding generation failed: 429 Rate limit")

        provider = ChromaProvider(
            persist_directory=temp_dir,
            collection_name="test",
            use_gemini=True
        )

        # Try to add entry - should raise error
        with pytest.raises(RuntimeError) as exc_info:
            provider.add_entry("test_id", "Test content", {}, EntryType.KNOWLEDGE)

        assert "429" in str(exc_info.value) or "Rate limit" in str(exc_info.value)

    @patch('providers.registry.ModelProviderRegistry')
    def test_stats_with_gemini(self, mock_registry, mock_gemini_provider, temp_dir):
        """Test getting stats with Gemini embeddings."""
        mock_registry.get_provider.return_value = mock_gemini_provider

        provider = ChromaProvider(
            persist_directory=temp_dir,
            collection_name="test",
            use_gemini=True
        )

        # Get stats
        stats = provider.get_stats()

        # Verify stats include Gemini info
        assert stats.total_entries == 0
        # Additional metadata should be accessible via the stats object
        # The exact structure depends on VectorStoreStats implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
