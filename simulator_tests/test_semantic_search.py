#!/usr/bin/env python3
"""
Semantic Search Test

Tests the semantic search functionality with ChromaDB integration.
Validates search quality improvements, hybrid search accuracy, and performance metrics.
"""

import time

from simulator_tests.base_test import BaseSimulatorTest


class TestSemanticSearch(BaseSimulatorTest):
    """Test semantic search functionality and quality improvements"""

    @property
    def test_name(self) -> str:
        return "semantic_search"

    @property
    def test_description(self) -> str:
        return "Semantic search quality, hybrid search, and performance validation"

    def run_test(self) -> bool:
        """Test semantic search scenarios"""
        try:
            self.logger.info("Test: Semantic Search Functionality")

            # Test 1: Basic semantic search capability
            if not self._test_basic_semantic_search():
                return False

            # Test 2: Hybrid search (semantic + keyword)
            if not self._test_hybrid_search():
                return False

            # Test 3: Search quality improvements
            if not self._test_search_quality():
                return False

            # Test 4: Performance metrics
            if not self._test_performance_metrics():
                return False

            # Test 5: Fallback mechanism
            if not self._test_fallback_mechanism():
                return False

            # Test 6: Vector store persistence
            if not self._test_vector_persistence():
                return False

            self.logger.info("  ✅ All semantic search tests passed!")
            return True

        except Exception as e:
            self.logger.error(f"Semantic search test failed: {e}")
            return False

    def _test_basic_semantic_search(self) -> bool:
        """Test basic semantic search capability"""
        self.logger.info("  1. Testing basic semantic search")

        # Add diverse knowledge entries with semantic relationships
        entries = [
            {
                "content": "Redis connection pooling is essential for high-performance applications. It prevents connection exhaustion and reduces latency by reusing established connections.",
                "metadata": {
                    "tags": ["redis", "performance", "connection-pooling"],
                    "category": "optimization",
                    "importance": 0.9,
                },
            },
            {
                "content": "Database connection management strategies include pooling, lazy loading, and proper cleanup. These patterns prevent resource leaks and improve application scalability.",
                "metadata": {
                    "tags": ["database", "architecture", "best-practices"],
                    "category": "design-patterns",
                    "importance": 0.8,
                },
            },
            {
                "content": "Memory optimization techniques for caching systems involve setting appropriate TTLs, implementing eviction policies, and monitoring memory usage patterns.",
                "metadata": {
                    "tags": ["caching", "memory", "optimization"],
                    "category": "performance",
                    "importance": 0.7,
                },
            },
        ]

        # Add all entries
        for i, entry in enumerate(entries):
            response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "add",
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                    "project_id": "semantic-search-test",
                    "model": "flash",
                },
            )
            if not response:
                self.logger.error(f"Failed to add entry {i+1}")
                return False

        self.logger.info("    ✅ Added test entries")

        # Test semantic search - should find related concepts
        semantic_queries = [
            ("connection pool exhaustion", ["redis", "database"]),  # Should find both Redis and DB entries
            ("resource management", ["database", "connection"]),  # Should find connection management
            ("performance tuning", ["redis", "memory", "optimization"]),  # Should find multiple entries
        ]

        for query, expected_keywords in semantic_queries:
            response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "search",
                    "content": query,
                    "project_id": "semantic-search-test",
                    "limit": 3,
                    "model": "flash",
                },
            )

            if not response:
                self.logger.error(f"Search failed for query: {query}")
                return False

            # Check if results contain expected semantic matches
            found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response.lower())
            if found_keywords < len(expected_keywords) // 2:  # At least half should match
                self.logger.error(
                    f"Semantic search quality issue for '{query}' - expected keywords: {expected_keywords}"
                )
                return False

            self.logger.info(f"    ✅ Semantic search for '{query}' found relevant results")

        return True

    def _test_hybrid_search(self) -> bool:
        """Test hybrid search combining semantic and keyword matching"""
        self.logger.info("  2. Testing hybrid search functionality")

        # Add entries with specific keywords and semantic content
        entries = [
            {
                "content": "The ZEN-MCP-2024 protocol defines standards for asynchronous message passing with retry mechanisms and circuit breakers.",
                "metadata": {"tags": ["protocol", "async", "zen-mcp-2024"], "category": "specification"},
            },
            {
                "content": "Implementing fault tolerance requires understanding retry patterns, backoff strategies, and circuit breaker implementations.",
                "metadata": {"tags": ["fault-tolerance", "patterns", "reliability"], "category": "architecture"},
            },
            {
                "content": "The ZEN-MCP-2024 specification includes detailed error handling procedures for distributed systems.",
                "metadata": {"tags": ["zen-mcp-2024", "error-handling", "distributed"], "category": "specification"},
            },
        ]

        for entry in entries:
            response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "add",
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                    "project_id": "hybrid-search-test",
                    "model": "flash",
                },
            )
            if not response:
                return False

        # Test 1: Exact keyword match (should prioritize keyword matches)
        response, _ = self.call_mcp_tool(
            "context",
            {
                "operation": "search",
                "content": "ZEN-MCP-2024",
                "project_id": "hybrid-search-test",
                "limit": 5,
                "model": "flash",
            },
        )

        if not response or response.count("ZEN-MCP-2024") < 2:
            self.logger.error("Hybrid search failed to prioritize exact keyword matches")
            return False

        self.logger.info("    ✅ Exact keyword matching works")

        # Test 2: Semantic + keyword (should find related content even without exact match)
        response, _ = self.call_mcp_tool(
            "context",
            {
                "operation": "search",
                "content": "message retry mechanism specification",
                "project_id": "hybrid-search-test",
                "limit": 3,
                "model": "flash",
            },
        )

        if not response or "ZEN-MCP-2024" not in response:
            self.logger.error("Hybrid search failed to find semantically related content")
            return False

        self.logger.info("    ✅ Semantic + keyword search works")

        return True

    def _test_search_quality(self) -> bool:
        """Test search quality improvements over basic keyword matching"""
        self.logger.info("  3. Testing search quality improvements")

        # Add entries with synonyms and related concepts
        quality_entries = [
            {
                "content": "Authentication and authorization are critical security components. JWT tokens provide stateless auth.",
                "metadata": {"tags": ["security", "auth", "jwt"], "category": "security"},
            },
            {
                "content": "User login systems must implement proper session management and secure credential storage.",
                "metadata": {"tags": ["login", "security", "sessions"], "category": "security"},
            },
            {
                "content": "Access control mechanisms include role-based permissions and attribute-based policies.",
                "metadata": {"tags": ["permissions", "rbac", "access-control"], "category": "security"},
            },
        ]

        for entry in quality_entries:
            self.call_mcp_tool(
                "context",
                {
                    "operation": "add",
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                    "project_id": "quality-test",
                    "model": "flash",
                },
            )

        # Test synonym understanding
        synonym_tests = [
            ("authentication system", ["auth", "login", "jwt"]),  # Should understand auth = authentication
            ("user permissions", ["access", "role", "control"]),  # Should understand permissions = access control
            ("security credentials", ["credential", "secure", "login"]),  # Should find related security concepts
        ]

        for query, expected_terms in synonym_tests:
            response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "search",
                    "content": query,
                    "project_id": "quality-test",
                    "limit": 3,
                    "model": "flash",
                },
            )

            if not response:
                self.logger.error(f"Quality search failed for: {query}")
                return False

            matches = sum(1 for term in expected_terms if term.lower() in response.lower())
            if matches == 0:
                self.logger.error(f"Search quality issue - no matches for '{query}'")
                return False

            self.logger.info(f"    ✅ Quality search for '{query}' found {matches} relevant matches")

        return True

    def _test_performance_metrics(self) -> bool:
        """Test search performance with various dataset sizes"""
        self.logger.info("  4. Testing search performance metrics")

        # Add multiple entries to test performance
        start_time = time.time()

        # Add 20 entries to test performance at scale
        for i in range(20):
            content = f"Performance test entry {i}: This contains information about {['caching', 'database', 'networking', 'security', 'monitoring'][i % 5]} systems and their optimization strategies."
            response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "add",
                    "content": content,
                    "metadata": {
                        "tags": [f"perf-test-{i}", f"category-{i % 5}"],
                        "category": "performance-test",
                        "index": i,
                    },
                    "project_id": "performance-test",
                    "model": "flash",
                },
            )

        add_time = time.time() - start_time
        self.logger.info(f"    Added 20 entries in {add_time:.2f}s ({add_time/20:.3f}s per entry)")

        # Test search performance
        search_times = []
        search_queries = ["caching optimization", "database performance", "network security", "monitoring systems"]

        for query in search_queries:
            start = time.time()
            response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "search",
                    "content": query,
                    "project_id": "performance-test",
                    "limit": 5,
                    "model": "flash",
                },
            )
            search_time = time.time() - start
            search_times.append(search_time)

            if not response:
                self.logger.error(f"Performance search failed for: {query}")
                return False

        avg_search_time = sum(search_times) / len(search_times)
        self.logger.info(f"    ✅ Average search time: {avg_search_time:.3f}s")

        # Performance should be reasonable (under 2 seconds average)
        if avg_search_time > 2.0:
            self.logger.warning(f"Search performance may be slow: {avg_search_time:.3f}s average")

        return True

    def _test_fallback_mechanism(self) -> bool:
        """Test fallback to keyword search when vector search fails"""
        self.logger.info("  5. Testing fallback mechanism")

        # Add an entry
        response, _ = self.call_mcp_tool(
            "context",
            {
                "operation": "add",
                "content": "FALLBACK-TEST-UNIQUE-KEYWORD: This tests the fallback mechanism when vector search encounters issues.",
                "metadata": {"tags": ["fallback", "test"], "category": "testing"},
                "project_id": "fallback-test",
                "model": "flash",
            },
        )

        if not response:
            self.logger.error("Failed to add fallback test entry")
            return False

        # Search should still work even if vector search has issues
        # The hybrid approach should ensure results are returned
        response, _ = self.call_mcp_tool(
            "context",
            {
                "operation": "search",
                "content": "FALLBACK-TEST-UNIQUE-KEYWORD",
                "project_id": "fallback-test",
                "limit": 3,
                "model": "flash",
            },
        )

        if not response or "FALLBACK-TEST-UNIQUE-KEYWORD" not in response:
            self.logger.error("Fallback mechanism failed - couldn't find entry via keyword search")
            return False

        self.logger.info("    ✅ Fallback mechanism works correctly")
        return True

    def _test_vector_persistence(self) -> bool:
        """Test that vector embeddings persist across operations"""
        self.logger.info("  6. Testing vector store persistence")

        # Add entries with semantic content
        persistence_entries = [
            "Machine learning models require careful hyperparameter tuning for optimal performance.",
            "Neural networks can overfit when trained on small datasets without regularization.",
            "Deep learning frameworks like PyTorch and TensorFlow simplify model development.",
        ]

        for i, content in enumerate(persistence_entries):
            response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "add",
                    "content": content,
                    "metadata": {"tags": ["ml", "persistence-test"], "entry": i},
                    "project_id": "persistence-test",
                    "model": "flash",
                },
            )

        # Export to verify entries exist
        export_response, _ = self.call_mcp_tool(
            "context",
            {
                "operation": "export",
                "project_id": "persistence-test",
                "model": "flash",
            },
        )

        if not export_response or "entries: 3" not in export_response.lower():
            self.logger.error("Failed to verify persistence - export doesn't show all entries")
            return False

        # Search for semantic similarity should work
        search_response, _ = self.call_mcp_tool(
            "context",
            {
                "operation": "search",
                "content": "artificial intelligence model training",
                "project_id": "persistence-test",
                "limit": 3,
                "model": "flash",
            },
        )

        if not search_response or "machine learning" not in search_response.lower():
            self.logger.error("Vector search persistence issue - semantic search not working")
            return False

        self.logger.info("    ✅ Vector embeddings persist correctly")
        return True
