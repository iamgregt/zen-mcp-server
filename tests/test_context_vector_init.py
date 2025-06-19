"""
Unit tests for ContextTool vector store initialization
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.context import ContextTool


class TestContextVectorStoreInit(unittest.TestCase):
    """Test vector store initialization in ContextTool"""

    def setUp(self):
        """Set up test environment"""
        # Save original env vars
        self.original_enable = os.environ.get("ENABLE_VECTOR_SEARCH")
        self.original_kb_dir = os.environ.get("ZEN_CONTEXT_KB_DIR")

        # Set test KB directory
        self.test_kb_dir = "/tmp/test-context-vector"
        os.environ["ZEN_CONTEXT_KB_DIR"] = self.test_kb_dir

    def tearDown(self):
        """Restore original environment"""
        # Restore original env vars
        if self.original_enable is not None:
            os.environ["ENABLE_VECTOR_SEARCH"] = self.original_enable
        elif "ENABLE_VECTOR_SEARCH" in os.environ:
            del os.environ["ENABLE_VECTOR_SEARCH"]

        if self.original_kb_dir is not None:
            os.environ["ZEN_CONTEXT_KB_DIR"] = self.original_kb_dir
        elif "ZEN_CONTEXT_KB_DIR" in os.environ:
            del os.environ["ZEN_CONTEXT_KB_DIR"]

    def test_vector_store_disabled_by_default(self):
        """Test that vector store is disabled by default"""
        # Ensure ENABLE_VECTOR_SEARCH is not set
        if "ENABLE_VECTOR_SEARCH" in os.environ:
            del os.environ["ENABLE_VECTOR_SEARCH"]

        tool = ContextTool()
        self.assertIsNone(tool.vector_store)

    def test_vector_store_disabled_when_false(self):
        """Test that vector store is disabled when env var is false"""
        os.environ["ENABLE_VECTOR_SEARCH"] = "false"

        tool = ContextTool()
        self.assertIsNone(tool.vector_store)

    @patch("utils.chroma_provider.ChromaProvider")
    def test_vector_store_enabled(self, mock_chroma_provider):
        """Test that vector store is initialized when enabled"""
        # Mock ChromaProvider
        mock_instance = MagicMock()
        mock_chroma_provider.return_value = mock_instance

        # Enable vector search
        os.environ["ENABLE_VECTOR_SEARCH"] = "true"

        tool = ContextTool()

        # Verify ChromaProvider was called with correct arguments
        expected_persist_dir = str(Path(self.test_kb_dir) / ".chroma")
        mock_chroma_provider.assert_called_once_with(
            persist_directory=expected_persist_dir,
            collection_name="zen_context",
        )

        # Verify vector store is set
        self.assertEqual(tool.vector_store, mock_instance)

    @patch("tools.context.logger")
    def test_import_error_handling(self, mock_logger):
        """Test graceful handling of import errors"""
        # Enable vector search
        os.environ["ENABLE_VECTOR_SEARCH"] = "true"

        # Mock import error by making the import raise ImportError
        original_import = __builtins__.__import__

        def mock_import(name, *args, **kwargs):
            if name == "utils.chroma_provider":
                raise ImportError("ChromaDB not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            tool = ContextTool()

            # Verify vector store is None
            self.assertIsNone(tool.vector_store)

            # Verify appropriate warnings were logged
            mock_logger.warning.assert_any_call(
                "Vector search enabled but ChromaDB dependencies not available: ChromaDB not installed"
            )
            mock_logger.warning.assert_any_call("Falling back to keyword-only search")

    @patch("tools.context.logger")
    @patch("utils.chroma_provider.ChromaProvider")
    def test_initialization_error_handling(self, mock_chroma_provider, mock_logger):
        """Test graceful handling of initialization errors"""
        # Enable vector search
        os.environ["ENABLE_VECTOR_SEARCH"] = "true"

        # Mock initialization error
        mock_chroma_provider.side_effect = RuntimeError("Failed to connect to ChromaDB")

        tool = ContextTool()

        # Verify vector store is None
        self.assertIsNone(tool.vector_store)

        # Verify appropriate errors were logged
        mock_logger.error.assert_called_once()
        mock_logger.warning.assert_any_call("Falling back to keyword-only search")

    def test_case_insensitive_env_var(self):
        """Test that ENABLE_VECTOR_SEARCH is case-insensitive"""
        test_values = ["True", "TRUE", "true", "TrUe"]

        for value in test_values:
            os.environ["ENABLE_VECTOR_SEARCH"] = value

            # Need to create new instance to pick up env change
            with patch("utils.chroma_provider.ChromaProvider") as mock_chroma:
                mock_chroma.return_value = MagicMock()
                tool = ContextTool()
                self.assertIsNotNone(tool.vector_store, f"Failed for value: {value}")


if __name__ == "__main__":
    unittest.main()
