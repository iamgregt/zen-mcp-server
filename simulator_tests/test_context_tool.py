"""
Simulator test for Context Tool functionality
"""

from simulator_tests.base_test import BaseSimulatorTest


class TestContextTool(BaseSimulatorTest):
    """Test the Context tool functionality"""

    @property
    def test_name(self) -> str:
        return "context_tool"

    @property
    def test_description(self) -> str:
        return "Context tool operations - add, search, list, export"

    def run_test(self) -> bool:
        """Test all context tool operations"""
        try:
            self.logger.info("Test: Context tool operations")
            
            # Test 1: Add knowledge entry
            self.logger.info("  1.1: Add knowledge entry")
            add_response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "add",
                    "content": "When debugging Redis timeout issues, always check connection pool settings and ensure proper cleanup in finally blocks. Common causes include: exhausted connection pools, missing cleanup, and incorrect timeout values.",
                    "metadata": {
                        "tags": ["redis", "debugging", "timeouts", "connection-pool"],
                        "category": "debugging",
                        "tool_source": "debug",
                        "importance": 0.8,
                    },
                    "project_id": "zen-mcp-test",
                    "model": "flash",
                },
            )

            if not add_response:
                self.logger.error("Failed to add knowledge entry")
                return False
            
            self.logger.info(f"  ✅ Knowledge entry added")

            # Test 2: Search for the entry
            self.logger.info("  1.2: Search for redis timeout")
            search_response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "search",
                    "content": "redis timeout",
                    "project_id": "zen-mcp-test",
                    "limit": 5,
                    "model": "flash",
                },
            )

            if not search_response:
                self.logger.error("Failed to search for knowledge")
                return False
            
            # Verify search found our entry
            if "redis" not in search_response.lower():
                self.logger.error("Search did not find the Redis entry we added")
                return False
            
            self.logger.info(f"  ✅ Search found Redis entry")

            # Test 3: List recent entries
            self.logger.info("  1.3: List recent entries")
            list_response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "list",
                    "project_id": "zen-mcp-test",
                    "limit": 10,
                    "model": "flash",
                },
            )

            if not list_response:
                self.logger.error("Failed to list entries")
                return False
            
            self.logger.info(f"  ✅ List operation successful")

            # Test 4: Add another entry
            self.logger.info("  1.4: Add second knowledge entry")
            add_response2, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "add",
                    "content": "Performance optimization tip: Use connection pooling for all database operations. Set appropriate pool sizes based on concurrent request patterns.",
                    "metadata": {
                        "tags": ["performance", "database", "connection-pool"],
                        "category": "optimization",
                        "importance": 0.7,
                    },
                    "project_id": "zen-mcp-test",
                    "model": "flash",
                },
            )

            if not add_response2:
                self.logger.error("Failed to add second knowledge entry")
                return False
            
            self.logger.info(f"  ✅ Second knowledge entry added")

            # Test 5: Export knowledge
            self.logger.info("  1.5: Export knowledge")
            export_response, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "export",
                    "project_id": "zen-mcp-test",
                    "model": "flash",
                },
            )

            if not export_response:
                self.logger.error("Failed to export knowledge")
                return False
            
            # Verify export contains 2 entries
            if "entries: 2" not in export_response.lower():
                self.logger.error("Export did not contain expected 2 entries")
                return False
            
            self.logger.info(f"  ✅ Export successful with 2 entries")

            # Test 6: Search for different term
            self.logger.info("  1.6: Search for performance")
            search_response2, _ = self.call_mcp_tool(
                "context",
                {
                    "operation": "search",
                    "content": "performance optimization",
                    "project_id": "zen-mcp-test",
                    "limit": 5,
                    "model": "flash",
                },
            )

            if not search_response2:
                self.logger.error("Failed to search for performance")
                return False
            
            if "performance" not in search_response2.lower():
                self.logger.error("Search did not find performance entry")
                return False
            
            self.logger.info(f"  ✅ Search found performance entry")

            # Test 7: Test with continuation_id (cross-tool memory)
            self.logger.info("  1.7: Test with continuation_id")
            
            # First, create a chat to establish context
            chat_response, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "I'm working on optimizing our Redis configuration. What should I focus on?",
                    "model": "flash",
                },
            )
            
            if continuation_id:
                self.logger.info(f"  Got continuation_id: {continuation_id}")
                
                # Add knowledge with continuation context
                context_with_continuation, _ = self.call_mcp_tool(
                    "context",
                    {
                        "operation": "add",
                        "content": "Based on our discussion: Redis optimization should focus on connection pooling, memory limits, and eviction policies.",
                        "metadata": {
                            "tags": ["redis", "optimization", "conversation"],
                            "category": "insights",
                            "from_conversation": True,
                        },
                        "project_id": "zen-mcp-test",
                        "continuation_id": continuation_id,
                        "model": "flash",
                    },
                )
                
                if not context_with_continuation:
                    self.logger.error("Failed to add knowledge with continuation_id")
                    return False
                
                self.logger.info(f"  ✅ Added knowledge with continuation context")
            
            self.logger.info("  ✅ All context tool operations passed!")
            return True

        except Exception as e:
            self.logger.error(f"Context tool test failed: {e}")
            return False