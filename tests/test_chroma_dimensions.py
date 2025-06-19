"""Test ChromaDB dimension flexibility for Gemini migration."""

import tempfile

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings


class TestChromaDimensions:
    """Test ChromaDB can handle different embedding dimensions."""

    def test_chroma_1024_dimensions(self):
        """Test ChromaDB with 1024 dimensions (current setup)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ChromaDB client
            client = chromadb.PersistentClient(
                path=temp_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create collection with 1024 dimensions
            collection = client.create_collection(
                name="test_1024",
                metadata={"embedding_dimension": 1024}
            )

            # Add test embeddings
            test_embedding = np.random.rand(1024).tolist()
            collection.add(
                embeddings=[test_embedding],
                documents=["Test document 1024"],
                ids=["test_1024_1"]
            )

            # Query to verify it works
            results = collection.query(
                query_embeddings=[test_embedding],
                n_results=1
            )

            assert len(results['documents'][0]) == 1
            assert results['documents'][0][0] == "Test document 1024"
            print("✓ ChromaDB successfully handles 1024 dimensions")

    def test_chroma_3072_dimensions(self):
        """Test ChromaDB with 3072 dimensions (Gemini embeddings)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ChromaDB client
            client = chromadb.PersistentClient(
                path=temp_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create collection with 3072 dimensions
            collection = client.create_collection(
                name="test_3072",
                metadata={"embedding_dimension": 3072}
            )

            # Add test embeddings
            test_embedding = np.random.rand(3072).tolist()
            collection.add(
                embeddings=[test_embedding],
                documents=["Test document 3072"],
                ids=["test_3072_1"]
            )

            # Query to verify it works
            results = collection.query(
                query_embeddings=[test_embedding],
                n_results=1
            )

            assert len(results['documents'][0]) == 1
            assert results['documents'][0][0] == "Test document 3072"
            print("✓ ChromaDB successfully handles 3072 dimensions")

    def test_mixed_dimensions_different_collections(self):
        """Test multiple collections with different dimensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ChromaDB client
            client = chromadb.PersistentClient(
                path=temp_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create two collections with different dimensions
            collection_1024 = client.create_collection(
                name="collection_1024",
                metadata={"embedding_dimension": 1024}
            )

            collection_3072 = client.create_collection(
                name="collection_3072",
                metadata={"embedding_dimension": 3072}
            )

            # Add embeddings to each
            embedding_1024 = np.random.rand(1024).tolist()
            embedding_3072 = np.random.rand(3072).tolist()

            collection_1024.add(
                embeddings=[embedding_1024],
                documents=["Doc in 1024"],
                ids=["doc_1024"]
            )

            collection_3072.add(
                embeddings=[embedding_3072],
                documents=["Doc in 3072"],
                ids=["doc_3072"]
            )

            # Query each collection
            results_1024 = collection_1024.query(
                query_embeddings=[embedding_1024],
                n_results=1
            )

            results_3072 = collection_3072.query(
                query_embeddings=[embedding_3072],
                n_results=1
            )

            assert results_1024['documents'][0][0] == "Doc in 1024"
            assert results_3072['documents'][0][0] == "Doc in 3072"
            print("✓ ChromaDB successfully handles multiple collections with different dimensions")

    def test_dimension_mismatch_error(self):
        """Test that ChromaDB properly rejects dimension mismatches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create ChromaDB client
            client = chromadb.PersistentClient(
                path=temp_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create collection and add initial embedding
            collection = client.create_collection(name="test_mismatch")

            # Add first embedding (this sets the dimension)
            embedding_1024 = np.random.rand(1024).tolist()
            collection.add(
                embeddings=[embedding_1024],
                documents=["First doc"],
                ids=["doc1"]
            )

            # Try to add embedding with different dimension
            embedding_3072 = np.random.rand(3072).tolist()
            with pytest.raises(Exception) as exc_info:
                collection.add(
                    embeddings=[embedding_3072],
                    documents=["Second doc"],
                    ids=["doc2"]
                )

            # Verify we got a dimension mismatch error
            assert "dimension" in str(exc_info.value).lower() or "shape" in str(exc_info.value).lower()
            print("✓ ChromaDB properly rejects dimension mismatches within a collection")


if __name__ == "__main__":
    test = TestChromaDimensions()

    print("Running ChromaDB dimension flexibility tests...")
    print("-" * 50)

    try:
        test.test_chroma_1024_dimensions()
        test.test_chroma_3072_dimensions()
        test.test_mixed_dimensions_different_collections()
        test.test_dimension_mismatch_error()

        print("-" * 50)
        print("✅ All tests passed! ChromaDB can handle different embedding dimensions.")
        print("\nKey findings:")
        print("- ChromaDB supports both 1024 and 3072 dimensional embeddings")
        print("- Different collections can have different dimensions")
        print("- Dimension mismatches within a collection are properly detected")
        print("- Migration will require creating a new collection for 3072 dimensions")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
