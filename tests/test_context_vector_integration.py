"""Unit tests for context tool vector store integration."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools.context import ContextTool, KnowledgeEntry
from utils.vector_store import EntryType


class TestContextVectorIntegration(unittest.TestCase):
    """Test vector store integration in ContextTool."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        os.environ["ZEN_CONTEXT_KB_DIR"] = self.temp_dir
        os.environ["ENABLE_VECTOR_SEARCH"] = "true"

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_vector_store_initialization_disabled(self):
        """Test that vector store is not initialized when disabled."""
        os.environ["ENABLE_VECTOR_SEARCH"] = "false"
        tool = ContextTool()
        self.assertIsNone(tool.vector_store)

    @patch("utils.chroma_provider.ChromaProvider")
    def test_vector_store_initialization_enabled(self, mock_chroma):
        """Test that vector store is initialized when enabled."""
        os.environ["ENABLE_VECTOR_SEARCH"] = "true"
        mock_instance = MagicMock()
        mock_chroma.return_value = mock_instance

        tool = ContextTool()

        self.assertIsNotNone(tool.vector_store)
        self.assertEqual(tool.vector_store, mock_instance)
        mock_chroma.assert_called_once()

    @patch("utils.chroma_provider.ChromaProvider")
    def test_save_entry_with_vector_indexing(self, mock_chroma):
        """Test that save_entry indexes in vector store when available."""
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        tool = ContextTool()
        entry = KnowledgeEntry(
            content="Test content for vector search",
            project_id="test_project",
            metadata={"tags": ["test", "vector"], "category": "testing"},
        )

        # Save entry
        tool.save_entry(entry)

        # Verify vector store was called
        mock_vector_store.add_entry.assert_called_once()
        call_args = mock_vector_store.add_entry.call_args
        self.assertEqual(call_args.kwargs["content"], "Test content for vector search")
        self.assertEqual(call_args.kwargs["entry_type"], EntryType.CONTEXT)
        self.assertIn("project_id", call_args.kwargs["metadata"])
        self.assertIn("access_count", call_args.kwargs["metadata"])

    @patch("utils.chroma_provider.ChromaProvider")
    def test_save_entry_handles_vector_store_failure(self, mock_chroma):
        """Test that save_entry continues if vector store fails."""
        mock_vector_store = MagicMock()
        mock_vector_store.add_entry.side_effect = Exception("Vector store error")
        mock_chroma.return_value = mock_vector_store

        tool = ContextTool()
        entry = KnowledgeEntry(content="Test content", project_id="test_project")

        # Should not raise exception
        entry_file = tool.save_entry(entry)
        self.assertIsNotNone(entry_file)

    @patch("utils.chroma_provider.ChromaProvider")
    def test_search_uses_vector_store(self, mock_chroma):
        """Test that search uses hybrid search with vector store when available."""
        mock_vector_store = MagicMock()
        mock_result = MagicMock()
        mock_result.entry.id = "context_test_project_entry123"

        # Mock both query and keyword_search for hybrid search
        mock_vector_store.query.return_value = [mock_result]
        mock_vector_store.keyword_search.return_value = [mock_result]
        mock_chroma.return_value = mock_vector_store

        tool = ContextTool()

        # Create and save an entry first
        entry = KnowledgeEntry(content="Test content", project_id="test_project", entry_id="entry123")
        tool.save_entry(entry)

        # Perform search
        tool.search_entries("test_project", "test query")

        # Verify both semantic and keyword search were used (hybrid search)
        mock_vector_store.query.assert_called_once()
        mock_vector_store.keyword_search.assert_called_once()

        # Check semantic search args
        query_args = mock_vector_store.query.call_args
        self.assertEqual(query_args.kwargs["query_text"], "test query")
        self.assertEqual(query_args.kwargs["entry_types"], [EntryType.CONTEXT])
        self.assertEqual(query_args.kwargs["metadata_filter"]["project_id"], "test_project")

        # Check keyword search args
        keyword_args = mock_vector_store.keyword_search.call_args
        self.assertEqual(keyword_args.kwargs["keywords"], ["test", "query"])
        self.assertEqual(keyword_args.kwargs["entry_types"], [EntryType.CONTEXT])
        self.assertEqual(keyword_args.kwargs["metadata_filter"]["project_id"], "test_project")

    @patch("utils.chroma_provider.ChromaProvider")
    def test_search_falls_back_to_keyword(self, mock_chroma):
        """Test that search falls back to legacy keyword search on vector store failure."""
        mock_vector_store = MagicMock()
        # Make hybrid search fail by raising exception on query
        mock_vector_store.query.side_effect = Exception("Vector query error")
        mock_vector_store.keyword_search.side_effect = Exception("Keyword search error")
        mock_chroma.return_value = mock_vector_store

        tool = ContextTool()

        # Create and save an entry
        entry = KnowledgeEntry(content="Test content with keywords", project_id="test_project")
        tool.save_entry(entry)

        # Perform search - should fall back to legacy keyword search
        results = tool.search_entries("test_project", "keywords")

        # Should find the entry via legacy keyword search
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Test content with keywords")

    @patch("utils.chroma_provider.ChromaProvider")
    def test_load_entry_updates_vector_metadata(self, mock_chroma):
        """Test that load_entry updates metadata in vector store."""
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store

        tool = ContextTool()

        # Create and save an entry
        entry = KnowledgeEntry(content="Test content", project_id="test_project")
        tool.save_entry(entry)

        # Reset mock to track update calls
        mock_vector_store.reset_mock()

        # Load entry (which updates access tracking)
        tool.load_entry("test_project", entry.entry_id)

        # Verify metadata was updated in vector store
        mock_vector_store.update_entry.assert_called_once()
        call_args = mock_vector_store.update_entry.call_args
        self.assertIn("access_count", call_args.kwargs["metadata"])
        self.assertIn("last_accessed", call_args.kwargs["metadata"])

    @patch("utils.chroma_provider.ChromaProvider")
    def test_hybrid_search_rrf_scoring(self, mock_chroma):
        """Test that hybrid search properly combines results using RRF scoring."""
        mock_vector_store = MagicMock()

        # Create different results for semantic and keyword search
        # Using simpler project_id without underscores to avoid parsing issues
        semantic_result1 = MagicMock()
        semantic_result1.entry.id = "context_testproject_entry1"
        semantic_result2 = MagicMock()
        semantic_result2.entry.id = "context_testproject_entry2"

        keyword_result1 = MagicMock()
        keyword_result1.entry.id = "context_testproject_entry2"  # Same as semantic #2
        keyword_result2 = MagicMock()
        keyword_result2.entry.id = "context_testproject_entry3"  # Different entry

        # Set up mock returns
        mock_vector_store.query.return_value = [semantic_result1, semantic_result2]
        mock_vector_store.keyword_search.return_value = [keyword_result1, keyword_result2]
        mock_chroma.return_value = mock_vector_store

        tool = ContextTool()

        # Create entries
        entry1 = KnowledgeEntry(content="First entry", project_id="testproject", entry_id="entry1")
        entry2 = KnowledgeEntry(content="Second entry", project_id="testproject", entry_id="entry2")
        entry3 = KnowledgeEntry(content="Third entry", project_id="testproject", entry_id="entry3")

        # Mock load_entry to return the appropriate entries
        def mock_load_entry(project_id, entry_id):
            if entry_id == "entry1":
                return entry1
            elif entry_id == "entry2":
                return entry2
            elif entry_id == "entry3":
                return entry3
            return None

        tool.load_entry = MagicMock(side_effect=mock_load_entry)

        # Perform hybrid search
        results = tool.search_entries("testproject", "test query", limit=3)

        # Verify RRF ranking: entry2 should be first (appears in both results)
        # Then entry1 and entry3 (appear in only one result each)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].entry_id, "entry2")  # Highest RRF score
        self.assertIn(results[1].entry_id, ["entry1", "entry3"])
        self.assertIn(results[2].entry_id, ["entry1", "entry3"])


if __name__ == "__main__":
    unittest.main()
