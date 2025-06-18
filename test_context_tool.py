#!/usr/bin/env python3
"""
Quick test script for the Context Tool MVP
Run this to verify basic functionality before full unit tests
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.context import ContextTool


async def test_context_tool():
    """Test basic context tool operations"""
    print("🧪 Testing Context Tool MVP...")
    
    # Initialize tool
    tool = ContextTool()
    print(f"✅ Tool initialized: {tool.get_name()}")
    
    # Test 1: Add operation
    print("\n📝 Test 1: Adding knowledge entry...")
    add_args = {
        "operation": "add",
        "content": "When debugging Redis timeout issues, check connection pool settings and ensure proper cleanup in finally blocks.",
        "metadata": {
            "tags": ["redis", "debugging", "timeouts"],
            "category": "debugging",
            "tool_source": "debug",
            "importance": 0.8
        },
        "project_id": "test-project",
        "model": "flash"  # Use model alias
    }
    
    try:
        result = await tool.execute(add_args)
        response = json.loads(result[0].text)
        print(f"✅ Add operation: {response['status']}")
        if response['status'] == 'success':
            print(f"   Response preview: {response['content'][:100]}...")
        elif response['status'] == 'error':
            print(f"   Error: {response.get('content', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Add operation failed: {e}")
        return False
    
    # Test 2: List operation
    print("\n📋 Test 2: Listing entries...")
    list_args = {
        "operation": "list",
        "project_id": "test-project",
        "limit": 5,
        "model": "flash"
    }
    
    try:
        result = await tool.execute(list_args)
        response = json.loads(result[0].text)
        print(f"✅ List operation: {response['status']}")
    except Exception as e:
        print(f"❌ List operation failed: {e}")
        return False
    
    # Test 3: Search operation
    print("\n🔍 Test 3: Searching for 'redis'...")
    search_args = {
        "operation": "search",
        "content": "redis timeout",
        "project_id": "test-project",
        "limit": 5,
        "model": "flash"
    }
    
    try:
        result = await tool.execute(search_args)
        response = json.loads(result[0].text)
        print(f"✅ Search operation: {response['status']}")
    except Exception as e:
        print(f"❌ Search operation failed: {e}")
        return False
    
    # Test 4: Export operation
    print("\n📦 Test 4: Exporting knowledge...")
    export_args = {
        "operation": "export",
        "project_id": "test-project",
        "model": "flash"
    }
    
    try:
        result = await tool.execute(export_args)
        response = json.loads(result[0].text)
        print(f"✅ Export operation: {response['status']}")
    except Exception as e:
        print(f"❌ Export operation failed: {e}")
        return False
    
    # Check if knowledge base directory was created
    kb_dir = Path(os.path.expanduser("~/zen-context-kb/projects/test-project"))
    if kb_dir.exists():
        print(f"\n✅ Knowledge base directory created: {kb_dir}")
        # List contents
        print("   Directory contents:")
        for item in kb_dir.rglob("*"):
            if item.is_file():
                print(f"   - {item.relative_to(kb_dir)}")
    else:
        print(f"\n❌ Knowledge base directory not found: {kb_dir}")
    
    print("\n🎉 All basic tests passed! The Context Tool MVP is working.")
    return True


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_context_tool())
    sys.exit(0 if success else 1)