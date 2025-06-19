#!/usr/bin/env python3
"""Test script to verify monitoring and metrics are working properly"""

import asyncio
import logging
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.context import ContextRequest, ContextTool

# Configure logging to see all our metrics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_metrics():
    """Test that metrics are being logged properly"""

    # Enable vector search
    os.environ["ENABLE_VECTOR_SEARCH"] = "true"

    print("🔧 Initializing Context Tool...")
    context_tool = ContextTool()

    # Check if vector store initialized
    if context_tool.vector_store:
        print("✅ Vector store initialized successfully")
    else:
        print("❌ Vector store not initialized")

    # Test 1: Add some entries
    print("\n📝 Testing ADD operations...")
    for i in range(3):
        request = ContextRequest(
            operation="add",
            content=f"This is test knowledge entry {i+1}. It contains information about monitoring and metrics in the Zen MCP Server. We want to track performance and memory usage.",
            metadata={
                "tags": ["test", "monitoring", "metrics"],
                "category": "testing",
                "importance": 0.8
            },
            project_id="test_metrics"
        )

        # Use the tool's format_response method which handles the actual operation
        prompt = await context_tool.prepare_prompt(request)
        response = context_tool.format_response("Entry processed", request)
        print(f"Added entry {i+1}")

    # Test 2: Perform searches to generate search metrics
    print("\n🔍 Testing SEARCH operations...")
    search_queries = [
        "monitoring metrics",
        "performance tracking",
        "memory usage",
        "vector store statistics"
    ]

    for query in search_queries:
        request = ContextRequest(
            operation="search",
            content=query,
            limit=5,
            project_id="test_metrics"
        )

        await context_tool.prepare_prompt(request)
        context_tool.format_response("Search completed", request)
        print(f"Searched for: '{query}'")

    # Test 3: List entries
    print("\n📋 Testing LIST operation...")
    request = ContextRequest(
        operation="list",
        limit=10,
        project_id="test_metrics"
    )

    await context_tool.prepare_prompt(request)
    context_tool.format_response("List completed", request)

    # Test 4: Get metrics summary
    print("\n📊 Getting metrics summary...")
    metrics_summary = context_tool.get_metrics_summary()

    print("\nMetrics Summary:")
    print(f"- Total searches: {metrics_summary['total_searches']}")
    print(f"- Vector searches: {metrics_summary['vector_searches']}")
    print(f"- Keyword searches: {metrics_summary['keyword_searches']}")
    print(f"- Avg search latency: {metrics_summary['avg_search_latency']:.3f}s")
    print(f"- Avg add latency: {metrics_summary['avg_add_latency']:.3f}s")
    print(f"- Vector operations: {metrics_summary['vector_operations']}")
    print(f"- Vector failures: {metrics_summary['vector_failures']}")
    print(f"- Vector enabled: {metrics_summary['vector_enabled']}")

    if 'vector_store' in metrics_summary:
        print("\nVector Store Stats:")
        print(f"- Total entries: {metrics_summary['vector_store']['total_entries']}")
        print(f"- Entries by type: {metrics_summary['vector_store']['entries_by_type']}")
        if metrics_summary['vector_store']['storage_size_mb']:
            print(f"- Storage size: {metrics_summary['vector_store']['storage_size_mb']:.1f}MB")
        if metrics_summary['vector_store']['embedding_dimension']:
            print(f"- Embedding dimension: {metrics_summary['vector_store']['embedding_dimension']}")

    # Force a final metrics log
    print("\n📝 Forcing final metrics log...")
    context_tool._log_metrics(force=True)

    print("\n✅ Test completed! Check the logs above for metric outputs.")
    print("\nLook for log lines containing:")
    print("- 'Context Tool Metrics'")
    print("- 'ChromaProvider Memory Usage'")
    print("- 'CONTEXT_ADD'")
    print("- 'CONTEXT_SEARCH'")
    print("- 'Hybrid search returned'")
    print("- 'Successfully indexed entry'")

if __name__ == "__main__":
    asyncio.run(test_metrics())
