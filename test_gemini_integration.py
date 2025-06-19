#!/usr/bin/env python3
"""Test Gemini embedding integration with Context tool."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.context import ContextRequest, ContextTool, EntryType


async def test_gemini_integration():
    """Test the complete Gemini embedding integration."""
    print("🚀 Testing Gemini Embedding Integration")
    print("=" * 50)

    # Create temporary directory for knowledge base
    with tempfile.TemporaryDirectory() as temp_dir:
        kb_dir = Path(temp_dir) / "test_kb"
        kb_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Context tool
        print("\n1️⃣ Initializing Context Tool...")
        context_tool = ContextTool()
        context_tool.kb_dir = kb_dir

        # Force initialize vector store
        context_tool._init_vector_store()

        if context_tool.vector_store:
            print("✅ Vector store initialized successfully")
            print(f"   Collection: {context_tool.vector_store.collection_name}")
            print(f"   Using Gemini: {context_tool.vector_store.use_gemini}")
            print(f"   Embedding dimension: {context_tool.vector_store.embedding_dimension}")
        else:
            print("❌ Failed to initialize vector store")
            return

        # Add some test entries
        print("\n2️⃣ Adding test entries...")
        test_entries = [
            {
                "title": "Python Programming",
                "content": "Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                "tags": ["python", "programming", "language"]
            },
            {
                "title": "Machine Learning Basics",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning.",
                "tags": ["ml", "ai", "data-science"]
            },
            {
                "title": "Web Development",
                "content": "Modern web development involves HTML, CSS, and JavaScript for frontend, along with various backend technologies like Node.js, Python Django, or Ruby on Rails. React and Vue are popular frontend frameworks.",
                "tags": ["web", "frontend", "backend"]
            }
        ]

        for i, entry in enumerate(test_entries):
            request = ContextRequest(
                action="ADD",
                title=entry["title"],
                content=entry["content"],
                tags=entry["tags"],
                project_id="test_project",
                entry_type=EntryType.KNOWLEDGE
            )

            try:
                await context_tool.prepare_prompt(request)
                response = context_tool.format_response("Entry added", request)
                print(f"   ✅ Added: {entry['title']}")
            except Exception as e:
                print(f"   ❌ Failed to add {entry['title']}: {e}")

        # Test semantic search with Gemini embeddings
        print("\n3️⃣ Testing semantic search...")
        search_queries = [
            "programming languages for beginners",
            "artificial intelligence and machine learning",
            "building websites with JavaScript"
        ]

        for query in search_queries:
            print(f"\n   🔍 Searching for: '{query}'")
            request = ContextRequest(
                action="SEARCH",
                query=query,
                limit=3,
                project_id="test_project"
            )

            try:
                prompt = await context_tool.prepare_prompt(request)
                # Extract search results from prompt
                if "Search Results:" in prompt:
                    results_section = prompt.split("Search Results:")[1].split("\n\n")[0]
                    print(f"   📄 Results:\n{results_section}")
                else:
                    print("   📄 No results found")
            except Exception as e:
                print(f"   ❌ Search failed: {e}")

        # Check vector store stats
        print("\n4️⃣ Vector Store Statistics:")
        if context_tool.vector_store:
            stats = context_tool.vector_store.get_stats()
            print(f"   Total entries: {stats.total_entries}")
            print(f"   Embedding model: {stats.metadata.get('model_name', 'unknown')}")
            print(f"   Embedding dimension: {stats.metadata.get('embedding_dimension', 'unknown')}")
            print(f"   Using Gemini: {stats.metadata.get('using_gemini', False)}")

        # Test collection reset
        print("\n5️⃣ Testing collection reset...")
        if context_tool.vector_store:
            context_tool.vector_store.reset_collection()
            stats_after = context_tool.vector_store.get_stats()
            print(f"   ✅ Collection reset. Entries after reset: {stats_after.total_entries}")

        print("\n" + "=" * 50)
        print("✅ Gemini embedding integration test completed!")


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_gemini_integration())
