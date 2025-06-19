"""Unit tests for ChromaProvider implementation."""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

from utils.chroma_provider import ChromaProvider
from utils.vector_store import EntryType, QueryResult, VectorStoreStats


class TestChromaProvider:
    """Test cases for ChromaProvider class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_model(self):
        """Create a mock sentence transformer model."""
        model = Mock()
        model.encode.return_value = np.random.rand(1024)  # Mock embedding
        model.get_sentence_embedding_dimension.return_value = 1024
        return model

    @pytest.fixture
    def provider(self, temp_dir, mock_model):
        """Create a ChromaProvider instance with mocked model and ChromaDB."""
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_collection.side_effect = ValueError("Collection not found")
        mock_client.create_collection.return_value = mock_collection

        with (
            patch("utils.chroma_provider.SentenceTransformer", return_value=mock_model),
            patch("utils.chroma_provider.chromadb.PersistentClient", return_value=mock_client),
        ):
            provider = ChromaProvider(
                persist_directory=temp_dir, collection_name="test_collection", model_name="test-model"
            )
            # Make sure collection is set to the mock
            provider.collection = mock_collection
            provider.client = mock_client
            yield provider

    def test_init_creates_directory(self, temp_dir, mock_model):
        """Test that initialization creates the persist directory."""
        persist_dir = os.path.join(temp_dir, "chroma_test")
        assert not os.path.exists(persist_dir)

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_collection.side_effect = ValueError("Collection not found")
        mock_client.create_collection.return_value = mock_collection

        with (
            patch("utils.chroma_provider.SentenceTransformer", return_value=mock_model),
            patch("utils.chroma_provider.chromadb.PersistentClient", return_value=mock_client),
        ):
            ChromaProvider(persist_directory=persist_dir)

        assert os.path.exists(persist_dir)

    def test_format_instruction_query(self, provider):
        """Test instruction formatting for queries."""
        query = "find similar code"
        formatted = provider._format_instruction_query(query)

        assert formatted.startswith("Instruct: Retrieve semantically similar technical content")
        assert "Query: find similar code" in formatted

    def test_compute_embedding(self, provider):
        """Test embedding computation."""
        text = "test content"

        # Test document embedding
        embedding = provider._compute_embedding(text, is_query=False)
        provider.model.encode.assert_called_with(text, normalize_embeddings=True)
        assert isinstance(embedding, np.ndarray)

        # Test query embedding with instruction formatting
        provider._compute_embedding(text, is_query=True)
        expected_query = provider._format_instruction_query(text)
        provider.model.encode.assert_called_with(expected_query, normalize_embeddings=True)

    def test_add_entry_success(self, provider):
        """Test successful entry addition."""
        # Mock collection.get to return empty (no existing entry)
        provider.collection.get = Mock(return_value={"ids": []})

        entry = provider.add_entry(
            id="test-1", content="Test content", metadata={"key": "value"}, entry_type=EntryType.CODE
        )

        assert entry.id == "test-1"
        assert entry.content == "Test content"
        # The add_entry method adds entry_type and timestamp to metadata internally
        # but returns only the original metadata to the user
        assert entry.metadata == {"key": "value"}
        assert entry.entry_type == EntryType.CODE
        assert isinstance(entry.timestamp, datetime)
        assert entry.vector is not None

        # Verify add was called
        provider.collection.add.assert_called_once()

    def test_add_entry_empty_content(self, provider):
        """Test adding entry with empty content raises ValueError."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            provider.add_entry("test-1", "")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            provider.add_entry("test-1", "   ")

    def test_add_entry_duplicate_id(self, provider):
        """Test adding entry with duplicate ID raises ValueError."""
        # First call returns empty, second returns existing
        provider.collection.get = Mock(
            side_effect=[{"ids": []}, {"ids": ["test-1"]}]  # First add succeeds  # Second add fails
        )

        provider.add_entry("test-1", "Content 1")

        with pytest.raises(ValueError, match="already exists"):
            provider.add_entry("test-1", "Content 2")

    def test_add_entry_with_custom_vector(self, provider):
        """Test adding entry with pre-computed vector."""
        provider.collection.get = Mock(return_value={"ids": []})
        custom_vector = np.random.rand(1024)

        # Reset the model encode call count
        provider.model.encode.reset_mock()

        entry = provider.add_entry(id="test-1", content="Test content", vector=custom_vector)

        # Should not call encode when vector is provided
        provider.model.encode.assert_not_called()
        assert np.array_equal(entry.vector, custom_vector)

    def test_add_entries_batch_success(self, provider):
        """Test successful batch entry addition."""
        provider.collection.get = Mock(return_value={"ids": []})

        entries = [
            ("id1", "Content 1", {"key1": "value1"}, EntryType.CODE),
            ("id2", "Content 2", {"key2": "value2"}, EntryType.DOCUMENTATION),
            ("id3", "Content 3", None, EntryType.CONTEXT),
        ]

        # Mock batch encoding
        provider.model.encode.return_value = np.random.rand(3, 1024)

        results = provider.add_entries_batch(entries)

        assert len(results) == 3
        assert results[0].id == "id1"
        assert results[1].content == "Content 2"
        assert results[2].entry_type == EntryType.CONTEXT

        # Check batch encoding was called
        provider.model.encode.assert_called_once()
        call_args = provider.model.encode.call_args[0][0]
        assert len(call_args) == 3
        assert call_args == ["Content 1", "Content 2", "Content 3"]

    def test_add_entries_batch_empty(self, provider):
        """Test adding empty batch returns empty list."""
        results = provider.add_entries_batch([])
        assert results == []

    def test_add_entries_batch_invalid_format(self, provider):
        """Test batch with invalid entry format raises ValueError."""
        with pytest.raises(ValueError, match="must be a tuple"):
            provider.add_entries_batch([("id1", "content")])  # Missing fields

    def test_query_success(self, provider):
        """Test successful query operation."""
        # Mock query results
        provider.collection.query = Mock(
            return_value={
                "ids": [["id1", "id3"]],
                "distances": [[0.1, 0.3]],
                "documents": [["Python function implementation", "Python class definition"]],
                "metadatas": [
                    [
                        {"entry_type": "other", "timestamp": datetime.now().isoformat()},
                        {"entry_type": "other", "timestamp": datetime.now().isoformat()},
                    ]
                ],
            }
        )

        results = provider.query("Python code", limit=2)

        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)
        assert results[0].score > results[1].score  # Higher score = more similar
        assert results[0].distance < results[1].distance

    def test_query_with_filters(self, provider):
        """Test query with type and metadata filters."""
        provider.collection.query = Mock(
            return_value={
                "ids": [["id1"]],
                "distances": [[0.2]],
                "documents": [["Code content"]],
                "metadatas": [[{"entry_type": "code", "timestamp": datetime.now().isoformat()}]],
            }
        )

        provider.query("test query", entry_types=[EntryType.CODE], metadata_filter={"project": "test"}, threshold=0.7)

        # Check where clause was built correctly
        call_args = provider.collection.query.call_args[1]
        where_clause = call_args.get("where")
        # ChromaProvider uses $and when multiple conditions exist
        assert "$and" in where_clause
        assert {"entry_type": {"$in": ["code"]}} in where_clause["$and"]
        assert {"project": "test"} in where_clause["$and"]

    def test_query_empty_text(self, provider):
        """Test query with empty text raises ValueError."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            provider.query("")

    def test_query_invalid_limit(self, provider):
        """Test query with invalid limit raises ValueError."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            provider.query("test", limit=0)

    def test_delete_entry_success(self, provider):
        """Test successful entry deletion."""
        # Mock get to return existing entry
        provider.collection.get = Mock(return_value={"ids": ["test-1"]})

        result = provider.delete_entry("test-1")
        assert result is True
        provider.collection.delete.assert_called_with(ids=["test-1"])

    def test_delete_entry_not_found(self, provider):
        """Test deleting non-existent entry returns False."""
        provider.collection.get = Mock(return_value={"ids": []})

        result = provider.delete_entry("nonexistent")
        assert result is False
        provider.collection.delete.assert_not_called()

    def test_get_stats(self, provider):
        """Test getting vector store statistics."""
        # Mock collection methods
        provider.collection.count = Mock(return_value=10)
        provider.collection.get = Mock(
            return_value={
                "metadatas": [
                    {"entry_type": "code"},
                    {"entry_type": "code"},
                    {"entry_type": "documentation"},
                    {"entry_type": "context"},
                    {"entry_type": "code"},
                    {"entry_type": "documentation"},
                    {"entry_type": "knowledge"},
                    {"entry_type": "other"},
                    {"entry_type": "other"},
                    {"entry_type": "code"},
                ]
            }
        )

        stats = provider.get_stats()

        assert isinstance(stats, VectorStoreStats)
        assert stats.total_entries == 10
        assert stats.entries_by_type["code"] == 4
        assert stats.entries_by_type["documentation"] == 2
        assert stats.entries_by_type["context"] == 1
        assert stats.entries_by_type["knowledge"] == 1
        assert stats.entries_by_type["other"] == 2
        assert stats.metadata["model_name"] == "test-model"
        assert stats.metadata["embedding_dimension"] == 1024

    def test_clear_all_entries(self, provider):
        """Test clearing all entries."""
        provider.collection.count = Mock(return_value=5)

        count = provider.clear()

        assert count == 5
        provider.client.delete_collection.assert_called_with(name="test_collection")

    def test_clear_by_type(self, provider):
        """Test clearing entries by type."""
        provider.collection.get = Mock(return_value={"ids": ["id1", "id2", "id3"]})

        count = provider.clear(entry_types=[EntryType.CODE, EntryType.DOCUMENTATION])

        assert count == 3
        provider.collection.delete.assert_called_with(ids=["id1", "id2", "id3"])

        # Check where clause
        call_args = provider.collection.get.call_args[1]
        where_clause = call_args["where"]
        assert where_clause["entry_type"]["$in"] == ["code", "documentation"]

    def test_update_entry_success(self, provider):
        """Test successful entry update."""
        # Mock existing entry
        provider.collection.get = Mock(
            return_value={
                "ids": ["test-1"],
                "documents": ["Old content"],
                "metadatas": [{"entry_type": "code", "timestamp": "2024-01-01T00:00:00"}],
            }
        )

        updated = provider.update_entry("test-1", content="New content", metadata={"updated": True})

        assert updated is not None
        assert updated.content == "New content"
        assert updated.metadata["updated"] is True
        provider.collection.update.assert_called_once()

    def test_update_entry_not_found(self, provider):
        """Test updating non-existent entry returns None."""
        provider.collection.get = Mock(return_value={"ids": []})

        result = provider.update_entry("nonexistent", content="New")
        assert result is None

    def test_get_entry_success(self, provider):
        """Test retrieving single entry."""
        provider.collection.get = Mock(
            return_value={
                "ids": ["test-1"],
                "documents": ["Test content"],
                "metadatas": [{"entry_type": "code", "timestamp": datetime.now().isoformat()}],
            }
        )

        entry = provider.get_entry("test-1")

        assert entry is not None
        assert entry.id == "test-1"
        assert entry.content == "Test content"
        assert entry.entry_type == EntryType.CODE

    def test_get_entry_not_found(self, provider):
        """Test getting non-existent entry returns None."""
        provider.collection.get = Mock(return_value={"ids": []})

        entry = provider.get_entry("nonexistent")
        assert entry is None

    def test_list_entries(self, provider):
        """Test listing entries with pagination."""
        # Mock data
        mock_ids = [f"id{i}" for i in range(10)]
        mock_docs = [f"Content {i}" for i in range(10)]
        mock_metas = [{"entry_type": "code", "timestamp": datetime.now().isoformat()} for _ in range(10)]

        provider.collection.get = Mock(return_value={"ids": mock_ids, "documents": mock_docs, "metadatas": mock_metas})

        # Test pagination
        entries = provider.list_entries(offset=2, limit=3)

        assert len(entries) == 3
        # Note: Results are sorted by timestamp (newest first) in the implementation

    def test_list_entries_invalid_params(self, provider):
        """Test list entries with invalid parameters."""
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            provider.list_entries(offset=-1)

        with pytest.raises(ValueError, match="Limit must be positive"):
            provider.list_entries(limit=0)

    def test_export_data(self, provider, temp_dir):
        """Test exporting data to file."""
        # Mock data
        provider.collection.get = Mock(
            return_value={
                "ids": ["id1", "id2"],
                "documents": ["Content 1", "Content 2"],
                "metadatas": [
                    {"entry_type": "code", "timestamp": "2024-01-01T00:00:00"},
                    {"entry_type": "documentation", "timestamp": "2024-01-02T00:00:00"},
                ],
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            }
        )

        export_path = os.path.join(temp_dir, "export.json")
        provider.export_data(export_path)

        assert os.path.exists(export_path)

        with open(export_path) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert data["model"] == "test-model"
        assert len(data["entries"]) == 2
        assert data["entries"][0]["id"] == "id1"

    def test_export_data_empty(self, provider, temp_dir):
        """Test exporting empty vector store."""
        provider.collection.get = Mock(return_value={"ids": []})

        export_path = os.path.join(temp_dir, "export_empty.json")
        provider.export_data(export_path)

        with open(export_path) as f:
            data = json.load(f)

        assert data["entries"] == []

    def test_import_data_success(self, provider, temp_dir):
        """Test importing data from file."""
        # Create import file
        import_data = {
            "version": "1.0",
            "model": "test-model",
            "entries": [
                {
                    "id": "import-1",
                    "content": "Imported content 1",
                    "metadata": {"type": "test"},
                    "embedding": [0.1] * 1024,
                },
                {
                    "id": "import-2",
                    "content": "Imported content 2",
                    "metadata": {"type": "test"},
                    # No embedding - should be computed
                },
            ],
        }

        import_path = os.path.join(temp_dir, "import.json")
        with open(import_path, "w") as f:
            json.dump(import_data, f)

        # Mock clear for non-merge import
        provider.clear = Mock()

        count = provider.import_data(import_path, merge=False)

        assert count == 2
        provider.clear.assert_called_once()
        provider.collection.add.assert_called_once()

    def test_import_data_invalid_format(self, provider, temp_dir):
        """Test importing invalid data raises ValueError."""
        # Invalid format - missing entries
        invalid_path = os.path.join(temp_dir, "invalid.json")
        with open(invalid_path, "w") as f:
            json.dump({"version": "1.0"}, f)

        with pytest.raises(RuntimeError, match="Failed to import data"):
            provider.import_data(invalid_path)

    def test_import_data_file_not_found(self, provider):
        """Test importing from non-existent file raises IOError."""
        with pytest.raises(IOError):
            provider.import_data("/nonexistent/file.json")

    def test_error_handling(self, provider):
        """Test error handling for ChromaDB failures."""
        # Mock collection.get to return empty (no existing entry)
        provider.collection.get = Mock(return_value={"ids": []})

        # Simulate ChromaDB failure on add
        provider.collection.add = Mock(side_effect=Exception("ChromaDB error"))

        with pytest.raises(RuntimeError, match="Failed to add entry"):
            provider.add_entry("test-1", "Content")

        # Test query error
        provider.collection.query = Mock(side_effect=Exception("Query failed"))

        with pytest.raises(RuntimeError, match="Failed to query"):
            provider.query("test query")

    @patch("providers.registry.ProviderType")
    @patch("providers.registry.ModelProviderRegistry")
    def test_extract_keywords_ai_success(self, mock_registry, mock_provider_type, provider):
        """Test successful keyword extraction using AI."""
        # Mock the provider and response
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='["python", "function", "class", "api", "error"]'))]
        mock_provider.generate.return_value = mock_response
        mock_registry.get_provider.return_value = mock_provider

        text = "This is a Python function that handles API errors in a class-based approach."
        keywords = provider._extract_keywords_ai(text)

        assert keywords == ["python", "function", "class", "api", "error"]
        assert len(keywords) <= 20  # Should limit to 20 keywords
        mock_provider.generate.assert_called_once()

        # Test caching - second call should not invoke generate again
        mock_provider.generate.reset_mock()
        keywords2 = provider._extract_keywords_ai(text)
        assert keywords2 == keywords
        mock_provider.generate.assert_not_called()

    @patch("providers.registry.ProviderType")
    @patch("providers.registry.ModelProviderRegistry")
    def test_extract_keywords_ai_invalid_response(self, mock_registry, mock_provider_type, provider):
        """Test handling of invalid AI response for keyword extraction."""
        # Mock invalid response (not JSON)
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="This is not JSON"))]
        mock_provider.generate.return_value = mock_response
        mock_registry.get_provider.return_value = mock_provider

        with pytest.raises(ValueError, match="Invalid keyword extraction response"):
            provider._extract_keywords_ai("Some text")

    @patch("providers.registry.ProviderType")
    @patch("providers.registry.ModelProviderRegistry")
    def test_extract_keywords_ai_no_provider(self, mock_registry, mock_provider_type, provider):
        """Test error when provider is not available."""
        mock_registry.get_provider.return_value = None

        with pytest.raises(ValueError, match="Flash provider not available"):
            provider._extract_keywords_ai("Some text")

    @patch("providers.registry.ProviderType")
    @patch("providers.registry.ModelProviderRegistry")
    def test_add_entry_with_keyword_extraction(self, mock_registry, mock_provider_type, provider):
        """Test adding entry with keyword extraction."""
        # Mock successful keyword extraction
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='["python", "testing", "unittest"]'))]
        mock_provider.generate.return_value = mock_response
        mock_registry.get_provider.return_value = mock_provider

        # Mock collection.get to return empty
        provider.collection.get = Mock(return_value={"ids": []})

        provider.add_entry("test-1", "Python testing with unittest framework")

        # Verify keywords were added to metadata
        call_args = provider.collection.add.call_args
        metadata = call_args[1]["metadatas"][0]
        assert "keywords" in metadata
        assert json.loads(metadata["keywords"]) == ["python", "testing", "unittest"]

    @patch("providers.registry.ProviderType")
    @patch("providers.registry.ModelProviderRegistry")
    def test_add_entry_keyword_extraction_failure(self, mock_registry, mock_provider_type, provider):
        """Test entry addition continues even if keyword extraction fails."""
        # Mock keyword extraction failure
        mock_provider = Mock()
        mock_provider.generate.side_effect = Exception("AI service error")
        mock_registry.get_provider.return_value = mock_provider

        # Mock collection.get to return empty
        provider.collection.get = Mock(return_value={"ids": []})

        # Should not raise exception, just log warning
        provider.add_entry("test-1", "Some content")

        # Verify entry was still added with empty keywords
        call_args = provider.collection.add.call_args
        metadata = call_args[1]["metadatas"][0]
        assert metadata["keywords"] == "[]"

    def test_keyword_search_success(self, provider):
        """Test successful keyword search."""
        # Mock collection.get with entries containing keywords
        provider.collection.get = Mock(
            return_value={
                "ids": ["id1", "id2", "id3"],
                "documents": ["Python code", "JavaScript function", "Python class"],
                "metadatas": [
                    {
                        "entry_type": "code",
                        "timestamp": datetime.now().isoformat(),
                        "keywords": '["python", "code", "function"]',
                    },
                    {
                        "entry_type": "code",
                        "timestamp": datetime.now().isoformat(),
                        "keywords": '["javascript", "function", "async"]',
                    },
                    {
                        "entry_type": "code",
                        "timestamp": datetime.now().isoformat(),
                        "keywords": '["python", "class", "oop"]',
                    },
                ],
            }
        )

        results = provider.keyword_search(["python"], limit=10)

        assert len(results) == 2  # Should find 2 entries with "python" keyword
        assert all(r.entry.id in ["id1", "id3"] for r in results)
        assert all(r.score > 0 for r in results)

    def test_keyword_search_multiple_keywords(self, provider):
        """Test keyword search with multiple keywords."""
        provider.collection.get = Mock(
            return_value={
                "ids": ["id1", "id2"],
                "documents": ["Content 1", "Content 2"],
                "metadatas": [
                    {
                        "entry_type": "code",
                        "timestamp": datetime.now().isoformat(),
                        "keywords": '["python", "api", "rest"]',
                    },
                    {
                        "entry_type": "code",
                        "timestamp": datetime.now().isoformat(),
                        "keywords": '["python", "database"]',
                    },
                ],
            }
        )

        results = provider.keyword_search(["python", "api"])

        # First entry should have higher score (matches 2/2 keywords)
        assert len(results) == 2
        assert results[0].entry.id == "id1"
        assert results[0].score > results[1].score

    def test_keyword_search_with_filters(self, provider):
        """Test keyword search with entry type and metadata filters."""
        provider.collection.get = Mock(
            return_value={
                "ids": ["id1", "id2"],
                "documents": ["Doc content", "Code content"],
                "metadatas": [
                    {
                        "entry_type": "documentation",
                        "timestamp": datetime.now().isoformat(),
                        "keywords": '["python", "docs"]',
                    },
                    {"entry_type": "code", "timestamp": datetime.now().isoformat(), "keywords": '["python", "code"]'},
                ],
            }
        )

        # Search only in documentation
        provider.keyword_search(["python"], entry_types=[EntryType.DOCUMENTATION], metadata_filter={"project": "test"})

        # Verify filters were applied
        call_args = provider.collection.get.call_args[1]
        where_clause = call_args["where"]
        assert where_clause["entry_type"]["$in"] == ["documentation"]
        assert where_clause["project"] == "test"

    def test_keyword_search_invalid_json(self, provider):
        """Test keyword search handles invalid JSON in keywords field."""
        provider.collection.get = Mock(
            return_value={
                "ids": ["id1"],
                "documents": ["Content"],
                "metadatas": [
                    {"entry_type": "code", "timestamp": datetime.now().isoformat(), "keywords": "invalid json"},
                ],
            }
        )

        # Should handle gracefully and treat as no keywords
        results = provider.keyword_search(["python"])
        # Entry won't match any keywords because JSON is invalid, so results should be empty
        assert len(results) == 0

    def test_keyword_search_error_handling(self, provider):
        """Test error handling in keyword search."""
        provider.collection.get = Mock(side_effect=Exception("Database error"))

        with pytest.raises(RuntimeError, match="Failed to perform keyword search"):
            provider.keyword_search(["python"])
