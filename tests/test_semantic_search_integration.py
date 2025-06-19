"""Integration tests for semantic search functionality in Zen MCP Server.

This module tests the end-to-end integration of semantic search capabilities,
including ChromaDB vector store, hybrid search, fallback mechanisms, and migration.
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np

from tools.context import ContextTool, KnowledgeEntry
from utils.chroma_provider import ChromaProvider
from utils.migrate_storage import check_migration_needed, ensure_storage_directory, migrate_context_storage
from utils.vector_store import EntryType, QueryResult, VectorEntry


class TestSemanticSearchIntegration(unittest.TestCase):
    """Integration tests for semantic search functionality."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.kb_dir = os.path.join(self.test_dir, "kb")
        self.chroma_dir = os.path.join(self.test_dir, "chroma")

        # Set environment variables
        os.environ["ZEN_CONTEXT_KB_DIR"] = self.kb_dir
        os.environ["ENABLE_VECTOR_SEARCH"] = "true"

        # Create directories
        os.makedirs(self.kb_dir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        # Clean up environment variables
        if "ZEN_CONTEXT_KB_DIR" in os.environ:
            del os.environ["ZEN_CONTEXT_KB_DIR"]
        if "ENABLE_VECTOR_SEARCH" in os.environ:
            del os.environ["ENABLE_VECTOR_SEARCH"]

    @patch("providers.registry.ModelProviderRegistry")
    @patch("utils.chroma_provider.SentenceTransformer")
    @patch("utils.chroma_provider.chromadb")
    def test_end_to_end_context_operations(self, mock_chromadb, mock_transformer, mock_registry):
        """Test complete flow: add, search, update, delete operations."""
        # Set up mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1024)
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_transformer.return_value = mock_model

        # Mock keyword extraction
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='["semantic", "search", "test"]'))]
        mock_provider.generate.return_value = mock_response
        mock_registry.get_provider.return_value = mock_provider

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_collection.side_effect = ValueError("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Initialize context tool
        tool = ContextTool()
        self.assertIsNotNone(tool.vector_store)

        # Test 1: Add entry
        mock_collection.get.return_value = {"ids": []}  # No existing entry

        entry = KnowledgeEntry(
            content="This is a comprehensive test of semantic search capabilities in our system.",
            project_id="test_project",
            metadata={"tags": ["semantic", "search", "test"], "category": "testing", "importance": 0.8},
        )

        # Save entry
        saved_path = tool.save_entry(entry)
        self.assertTrue(os.path.exists(saved_path))

        # Verify vector store was called
        mock_collection.add.assert_called_once()
        add_call = mock_collection.add.call_args
        self.assertIn("ids", add_call[1])
        self.assertIn("embeddings", add_call[1])
        self.assertIn("documents", add_call[1])
        self.assertIn("metadatas", add_call[1])

        # Test 2: Search operations
        # Mock search results
        mock_query_result = {
            "ids": [[f"context_test_project_{entry.entry_id}"]],
            "distances": [[0.1]],
            "documents": [[entry.content]],
            "metadatas": [
                [
                    {
                        "entry_type": "context",
                        "timestamp": datetime.now().isoformat(),
                        "project_id": "test_project",
                        "keywords": '["semantic", "search", "test"]',
                    }
                ]
            ],
        }
        mock_collection.query.return_value = mock_query_result

        # Mock keyword search results
        mock_get_result = {
            "ids": [f"context_test_project_{entry.entry_id}"],
            "documents": [entry.content],
            "metadatas": [
                {
                    "entry_type": "context",
                    "timestamp": datetime.now().isoformat(),
                    "project_id": "test_project",
                    "keywords": '["semantic", "search", "test"]',
                }
            ],
        }
        mock_collection.get.return_value = mock_get_result

        # Perform search
        results = tool.search_entries("test_project", "semantic search")

        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, entry.content)

        # Test 3: Update entry (via load which updates access tracking)
        mock_collection.get.return_value = {"ids": [f"context_test_project_{entry.entry_id}"]}

        # Mock update_entry to prevent error
        mock_collection.update = Mock()

        loaded_entry = tool.load_entry("test_project", entry.entry_id)

        self.assertIsNotNone(loaded_entry)
        # Access count is 2 because it was already accessed during search
        self.assertEqual(loaded_entry.access_count, 2)
        self.assertIsNotNone(loaded_entry.last_accessed)

        # Test 4: List recent entries
        recent_entries = tool.list_recent_entries("test_project", limit=10)
        self.assertEqual(len(recent_entries), 1)
        self.assertEqual(recent_entries[0].entry_id, entry.entry_id)

        # Test 5: Export knowledge
        export_result = tool.export_knowledge("test_project")
        self.assertIn("export_file", export_result)
        self.assertEqual(export_result["entry_count"], 1)
        self.assertTrue(os.path.exists(export_result["export_file"]))

    @patch("utils.chroma_provider.SentenceTransformer")
    @patch("utils.chroma_provider.chromadb")
    @patch("providers.registry.ModelProviderRegistry")
    def test_hybrid_search_accuracy(self, mock_registry, mock_chromadb, mock_transformer):
        """Test hybrid search accuracy with semantic and keyword matching."""
        # Set up mocks
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_transformer.return_value = mock_model

        # Mock provider for keyword extraction
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='["python", "api", "rest", "framework"]'))]
        mock_provider.generate.return_value = mock_response
        mock_registry.get_provider.return_value = mock_provider

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_collection.side_effect = ValueError("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Initialize tool
        tool = ContextTool()

        # Create test entries with varying relevance
        entries = [
            KnowledgeEntry(
                content="Python REST API framework using FastAPI for building scalable web services",
                project_id="test_project",
                entry_id="entry1",
                metadata={"tags": ["python", "api", "rest", "fastapi"]},
            ),
            KnowledgeEntry(
                content="JavaScript frontend application with React and Redux state management",
                project_id="test_project",
                entry_id="entry2",
                metadata={"tags": ["javascript", "react", "frontend"]},
            ),
            KnowledgeEntry(
                content="Python Django REST framework tutorial for beginners",
                project_id="test_project",
                entry_id="entry3",
                metadata={"tags": ["python", "django", "rest", "tutorial"]},
            ),
            KnowledgeEntry(
                content="Database design patterns for microservices architecture",
                project_id="test_project",
                entry_id="entry4",
                metadata={"tags": ["database", "microservices", "architecture"]},
            ),
        ]

        # Mock vector store to not actually save
        mock_collection.get.return_value = {"ids": []}
        mock_collection.add.return_value = None

        # Save all entries
        for entry in entries:
            tool.save_entry(entry)

        # Set up search results
        # Semantic search finds entry1 and entry3 (both about Python APIs)
        mock_collection.query.return_value = {
            "ids": [["context_test_project_entry1", "context_test_project_entry3"]],
            "distances": [[0.1, 0.2]],
            "documents": [[entries[0].content, entries[2].content]],
            "metadatas": [
                [
                    {"entry_type": "context", "timestamp": datetime.now().isoformat()},
                    {"entry_type": "context", "timestamp": datetime.now().isoformat()},
                ]
            ],
        }

        # Mock the ChromaProvider instance methods
        mock_chroma_instance = Mock(spec=ChromaProvider)
        mock_chroma_instance.query.return_value = [
            QueryResult(
                entry=VectorEntry(
                    id="context_test_project_entry1",
                    content=entries[0].content,
                    vector=None,
                    metadata={},
                    entry_type=EntryType.CONTEXT,
                    timestamp=datetime.now(),
                ),
                score=0.9,
                distance=0.1,
            ),
            QueryResult(
                entry=VectorEntry(
                    id="context_test_project_entry3",
                    content=entries[2].content,
                    vector=None,
                    metadata={},
                    entry_type=EntryType.CONTEXT,
                    timestamp=datetime.now(),
                ),
                score=0.8,
                distance=0.2,
            ),
        ]

        mock_chroma_instance.keyword_search.return_value = [
            QueryResult(
                entry=VectorEntry(
                    id="context_test_project_entry1",
                    content=entries[0].content,
                    vector=None,
                    metadata={},
                    entry_type=EntryType.CONTEXT,
                    timestamp=datetime.now(),
                ),
                score=1.0,
                distance=0.0,
            ),
            QueryResult(
                entry=VectorEntry(
                    id="context_test_project_entry3",
                    content=entries[2].content,
                    vector=None,
                    metadata={},
                    entry_type=EntryType.CONTEXT,
                    timestamp=datetime.now(),
                ),
                score=0.67,
                distance=0.33,
            ),
            QueryResult(
                entry=VectorEntry(
                    id="context_test_project_entry4",
                    content=entries[3].content,
                    vector=None,
                    metadata={},
                    entry_type=EntryType.CONTEXT,
                    timestamp=datetime.now(),
                ),
                score=0.33,
                distance=0.67,
            ),
        ]

        tool.vector_store = mock_chroma_instance

        # Mock load_entry to return the appropriate entries
        def mock_load_entry(project_id, entry_id):
            for entry in entries:
                if entry.entry_id == entry_id:
                    return entry
            return None

        tool.load_entry = Mock(side_effect=mock_load_entry)

        # Perform hybrid search
        search_results = tool.search_entries("test_project", "python api framework", limit=3)

        # Verify results
        # Due to how the mock is set up, we should get the entries that were found
        # The exact number depends on how many unique entries are returned from both searches
        self.assertGreaterEqual(len(search_results), 2)  # At least 2 results
        self.assertLessEqual(len(search_results), 3)  # At most 3 results

        # Check that we got the expected entries (order may vary based on RRF scoring)
        result_ids = [r.entry_id for r in search_results]

        # Entry1 and entry3 should both be present (they appear in both search results)
        self.assertIn("entry1", result_ids)
        self.assertIn("entry3", result_ids)

        # If we have a third result, it should be entry4 (only in keyword results)
        if len(search_results) == 3:
            self.assertIn("entry4", result_ids)

    @patch("utils.chroma_provider.SentenceTransformer")
    @patch("utils.chroma_provider.chromadb")
    def test_fallback_mechanisms(self, mock_chromadb, mock_transformer):
        """Test fallback from vector search to keyword search on failures."""
        # Test Case 1: Vector store initialization failure
        os.environ["ENABLE_VECTOR_SEARCH"] = "true"
        mock_chromadb.PersistentClient.side_effect = Exception("ChromaDB connection failed")

        tool = ContextTool()
        self.assertIsNone(tool.vector_store)

        # Create and save entry (should work with file system only)
        entry = KnowledgeEntry(
            content="Test content for fallback mechanism testing",
            project_id="fallback_test",
            metadata={"tags": ["fallback", "test"]},
        )

        saved_path = tool.save_entry(entry)
        self.assertTrue(os.path.exists(saved_path))

        # Search should use legacy keyword search
        results = tool.search_entries("fallback_test", "fallback")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, entry.content)

        # Test Case 2: Vector search fails during operation
        # Reset mocks for new test
        mock_chromadb.PersistentClient.side_effect = None
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1024)
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_transformer.return_value = mock_model

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Create new tool instance with working vector store
        tool2 = ContextTool()
        self.assertIsNotNone(tool2.vector_store)

        # Make vector search operations fail
        tool2.vector_store.query = Mock(side_effect=Exception("Vector query failed"))
        tool2.vector_store.keyword_search = Mock(side_effect=Exception("Keyword search failed"))

        # Create and save entry
        entry2 = KnowledgeEntry(
            content="Another test for search fallback scenario",
            project_id="fallback_test2",
            metadata={"tags": ["search", "fallback"]},
        )

        # Mock add_entry to not fail during save
        tool2.vector_store.add_entry = Mock()
        saved_path2 = tool2.save_entry(entry2)
        self.assertTrue(os.path.exists(saved_path2))

        # Search should fall back to legacy keyword search
        results2 = tool2.search_entries("fallback_test2", "search fallback")
        self.assertEqual(len(results2), 1)
        self.assertEqual(results2[0].content, entry2.content)

        # Test Case 3: Keyword extraction failure during save
        mock_provider = Mock()
        mock_provider.generate.side_effect = Exception("AI service unavailable")

        with patch("providers.registry.ModelProviderRegistry") as mock_registry:
            mock_registry.get_provider.return_value = mock_provider

            # Reset vector store mock
            tool3 = ContextTool()
            tool3.vector_store = Mock(spec=ChromaProvider)
            tool3.vector_store.add_entry = Mock()

            entry3 = KnowledgeEntry(content="Test entry with keyword extraction failure", project_id="fallback_test3")

            # Should not raise exception
            saved_path3 = tool3.save_entry(entry3)
            self.assertTrue(os.path.exists(saved_path3))

            # Verify add_entry was still called (with empty keywords)
            tool3.vector_store.add_entry.assert_called_once()

    def test_migration_process(self):
        """Test storage migration from temporary to persistent location."""
        # Set up old and new locations
        old_kb_dir = os.path.join(self.test_dir, "old_kb")
        new_kb_dir = os.path.join(self.test_dir, "new_kb")

        # Create old knowledge base structure with data
        os.makedirs(os.path.join(old_kb_dir, "projects", "test_project", "entries"), exist_ok=True)
        os.makedirs(os.path.join(old_kb_dir, "projects", "test_project", ".index"), exist_ok=True)

        # Create test entries in old location
        entry1_data = {
            "entry_id": "test-entry-1",
            "timestamp": datetime.now().isoformat(),
            "content": "Test content for migration",
            "project_id": "test_project",
            "metadata": {"tags": ["migration", "test"]},
            "access_count": 5,
            "last_accessed": None,
        }

        entry2_data = {
            "entry_id": "test-entry-2",
            "timestamp": datetime.now().isoformat(),
            "content": "Another test entry for migration",
            "project_id": "test_project",
            "metadata": {"category": "testing"},
            "access_count": 2,
            "last_accessed": datetime.now().isoformat(),
        }

        # Save entries
        entry1_path = os.path.join(old_kb_dir, "projects", "test_project", "entries", "test-entry-1.json")
        entry2_path = os.path.join(old_kb_dir, "projects", "test_project", "entries", "test-entry-2.json")

        with open(entry1_path, "w") as f:
            json.dump(entry1_data, f)

        with open(entry2_path, "w") as f:
            json.dump(entry2_data, f)

        # Create index file
        index_data = {
            "entries": {
                "test-entry-1": {
                    "timestamp": entry1_data["timestamp"],
                    "summary": "Test content for migration",
                    "tags": ["migration", "test"],
                    "category": "general",
                },
                "test-entry-2": {
                    "timestamp": entry2_data["timestamp"],
                    "summary": "Another test entry for migration",
                    "tags": [],
                    "category": "testing",
                },
            },
            "tags": {"migration": ["test-entry-1"], "test": ["test-entry-1"]},
            "categories": {"general": ["test-entry-1"], "testing": ["test-entry-2"]},
        }

        index_path = os.path.join(old_kb_dir, "projects", "test_project", ".index", "entries.json")
        with open(index_path, "w") as f:
            json.dump(index_data, f)

        # Test 1: Check if migration is needed
        self.assertTrue(check_migration_needed(old_kb_dir, new_kb_dir))

        # Test 2: Perform migration
        success, error = migrate_context_storage(old_kb_dir, new_kb_dir)
        self.assertTrue(success)
        self.assertIsNone(error)

        # Test 3: Verify migration
        # Old location should not exist
        self.assertFalse(os.path.exists(old_kb_dir))

        # New location should exist with all data
        self.assertTrue(os.path.exists(new_kb_dir))
        self.assertTrue(os.path.exists(os.path.join(new_kb_dir, "projects", "test_project", "entries")))

        # Check entries were migrated
        migrated_entry1_path = os.path.join(new_kb_dir, "projects", "test_project", "entries", "test-entry-1.json")
        migrated_entry2_path = os.path.join(new_kb_dir, "projects", "test_project", "entries", "test-entry-2.json")

        self.assertTrue(os.path.exists(migrated_entry1_path))
        self.assertTrue(os.path.exists(migrated_entry2_path))

        # Verify content integrity
        with open(migrated_entry1_path) as f:
            migrated_entry1 = json.load(f)
        self.assertEqual(migrated_entry1["content"], entry1_data["content"])
        self.assertEqual(migrated_entry1["access_count"], 5)

        # Check migration marker
        marker_path = os.path.join(new_kb_dir, ".migrated_from_tmp")
        self.assertTrue(os.path.exists(marker_path))

        with open(marker_path) as f:
            migration_info = json.load(f)
        self.assertEqual(migration_info["migrated_from"], old_kb_dir)
        self.assertEqual(migration_info["migrated_to"], new_kb_dir)
        self.assertEqual(migration_info["project_count"], 1)
        self.assertEqual(migration_info["entry_count"], 2)

        # Test 4: Migration not needed after completion
        self.assertFalse(check_migration_needed(old_kb_dir, new_kb_dir))

        # Test 5: Force migration with existing destination
        # Create another entry in old location
        os.makedirs(os.path.join(old_kb_dir, "projects", "test_project", "entries"), exist_ok=True)
        entry3_path = os.path.join(old_kb_dir, "projects", "test_project", "entries", "test-entry-3.json")
        with open(entry3_path, "w") as f:
            json.dump({"entry_id": "test-entry-3", "content": "Force migration test"}, f)

        # Try without force (should fail)
        success, error = migrate_context_storage(old_kb_dir, new_kb_dir, force=False)
        self.assertFalse(success)
        self.assertIn("already contains data", error)

        # Try with force (should succeed)
        success, error = migrate_context_storage(old_kb_dir, new_kb_dir, force=True)
        self.assertTrue(success)
        self.assertIsNone(error)

        # Test 6: Ensure storage directory
        test_storage_path = os.path.join(self.test_dir, "test_storage", "deep", "path")
        self.assertTrue(ensure_storage_directory(test_storage_path))
        self.assertTrue(os.path.exists(test_storage_path))

        # Test write permissions
        test_file = os.path.join(test_storage_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        self.assertTrue(os.path.exists(test_file))

    @patch("utils.chroma_provider.SentenceTransformer")
    @patch("utils.chroma_provider.chromadb")
    @patch("providers.registry.ModelProviderRegistry")
    def test_vector_store_with_context_tool_integration(self, mock_registry, mock_chromadb, mock_transformer):
        """Test full integration between context tool and vector store operations."""
        # Set up comprehensive mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1024)
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_transformer.return_value = mock_model

        # Mock keyword extraction
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='["integration", "test", "vector", "search"]'))]
        mock_provider.generate.return_value = mock_response
        mock_registry.get_provider.return_value = mock_provider

        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_collection.side_effect = ValueError("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Initialize tool
        tool = ContextTool()

        # Test adding multiple entries with different metadata
        entries = []
        for i in range(5):
            entry = KnowledgeEntry(
                content=f"Integration test entry {i} with unique content for testing vector search accuracy",
                project_id="integration_test",
                entry_id=f"int-test-{i}",
                metadata={
                    "tags": [f"tag{i}", "integration"],
                    "category": "testing" if i % 2 == 0 else "development",
                    "importance": i / 10.0,
                },
            )
            entries.append(entry)

        # Mock vector store operations
        mock_collection.get.return_value = {"ids": []}  # No existing entries

        # Save all entries
        for entry in entries:
            saved_path = tool.save_entry(entry)
            self.assertTrue(os.path.exists(saved_path))

        # Verify vector store was called for each entry
        self.assertEqual(mock_collection.add.call_count, 5)

        # Test listing recent entries
        recent = tool.list_recent_entries("integration_test", limit=3)
        self.assertEqual(len(recent), 3)

        # Test export functionality
        export_result = tool.export_knowledge("integration_test")
        self.assertEqual(export_result["entry_count"], 5)

        # Verify export file exists and contains correct data
        with open(export_result["export_file"]) as f:
            export_data = json.load(f)

        self.assertEqual(export_data["project_id"], "integration_test")
        self.assertEqual(export_data["entry_count"], 5)
        self.assertEqual(len(export_data["entries"]), 5)

    def test_vector_search_disabled(self):
        """Test that system works correctly when vector search is disabled."""
        os.environ["ENABLE_VECTOR_SEARCH"] = "false"

        tool = ContextTool()
        self.assertIsNone(tool.vector_store)

        # All operations should still work with file-based storage
        entry = KnowledgeEntry(
            content="Test without vector search", project_id="no_vector_test", metadata={"tags": ["no-vector"]}
        )

        # Save should work
        saved_path = tool.save_entry(entry)
        self.assertTrue(os.path.exists(saved_path))

        # Search should use legacy keyword search
        results = tool.search_entries("no_vector_test", "vector")
        self.assertEqual(len(results), 1)

        # Load should work
        loaded = tool.load_entry("no_vector_test", entry.entry_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.content, entry.content)

        # List should work
        recent = tool.list_recent_entries("no_vector_test")
        self.assertEqual(len(recent), 1)

        # Export should work
        export_result = tool.export_knowledge("no_vector_test")
        self.assertEqual(export_result["entry_count"], 1)


if __name__ == "__main__":
    unittest.main()
