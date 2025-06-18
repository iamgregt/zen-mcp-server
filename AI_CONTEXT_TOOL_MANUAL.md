# AI Context Management Tool - User Manual

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Usage Guide](#usage-guide)
5. [Operations Reference](#operations-reference)
6. [Administration & Maintenance](#administration--maintenance)
7. [Technical Details](#technical-details)
8. [Future Expansion Roadmap](#future-expansion-roadmap)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## Overview

The AI Context Management Tool is a specialized MCP (Model Context Protocol) tool designed to solve the critical problem of token waste when working with multiple AI assistants. By maintaining a persistent knowledge base across sessions, it reduces token usage by up to 70% while preserving important context and insights.

### Key Benefits
- **70% Token Reduction**: Dramatically reduces repeated context loading
- **Cross-Session Persistence**: Knowledge survives between AI conversations
- **Project Organization**: Separate knowledge bases per project
- **Simple File-Based Storage**: Easy to backup, migrate, and understand
- **Export/Import Capabilities**: Share knowledge between team members
- **Search Functionality**: Quickly find relevant past insights

### Current Version: MVP 1.0
- File-based storage in `/tmp/zen-context-kb/`
- JSON format for easy parsing and migration
- Basic keyword search functionality
- Project-based organization

## Architecture

### System Design
```
┌─────────────────────────────────────────────────────────┐
│                    AI Assistant (Claude)                 │
├─────────────────────────────────────────────────────────┤
│                    MCP Server Layer                      │
│  ┌─────────────────┐    ┌─────────────────┐           │
│  │  Redis Memory   │    │  Context Tool    │           │
│  │ (Session Cache) │    │ (Persistent KB)  │           │
│  └─────────────────┘    └─────────────────┘           │
├─────────────────────────────────────────────────────────┤
│                   Docker Container                       │
│              /tmp/zen-context-kb/                       │
└─────────────────────────────────────────────────────────┘
```

### Storage Structure
```
/tmp/zen-context-kb/
├── projects/
│   ├── [project-name]/
│   │   ├── entries/          # Individual knowledge entries
│   │   │   ├── [uuid].json   # Each entry as separate file
│   │   │   └── ...
│   │   ├── .index/           # Search indices
│   │   │   └── entries.json  # Master index file
│   │   └── exports/          # Exported knowledge dumps
│   │       └── knowledge-export-[timestamp].json
│   └── ...
```

### Knowledge Entry Schema
```json
{
  "entry_id": "uuid-v4",
  "timestamp": "ISO-8601 datetime",
  "content": "The actual knowledge content",
  "project_id": "project-name",
  "metadata": {
    "category": "architecture|implementation|bug-fix|decision|etc",
    "tags": ["array", "of", "relevant", "tags"],
    "importance": 0.0-1.0,
    "tool_source": "optional-source-tool",
    "related_files": ["optional", "file", "references"]
  },
  "access_count": 0,
  "last_accessed": "ISO-8601 datetime or null"
}
```

## Installation & Setup

### Prerequisites
- Zen MCP Server running in Docker
- Redis container (zen-mcp-redis) running
- Valid API keys for AI providers

### Initial Setup
1. The tool is automatically registered when the server starts
2. Storage directory is created automatically at `/tmp/zen-context-kb/`
3. No additional configuration required for MVP

### Environment Variables (Optional)
```bash
# Override default storage location (not recommended for Docker)
export ZEN_CONTEXT_KB_DIR="/custom/path"
```

## Usage Guide

### Basic Workflow

1. **Adding Knowledge**
   ```
   Use: mcp__zen__context
   Operation: add
   Content: "Important insight or decision to remember"
   Metadata: {
     "category": "architecture",
     "tags": ["api", "design", "rest"],
     "importance": 0.8
   }
   ```

2. **Searching Knowledge**
   ```
   Use: mcp__zen__context
   Operation: search
   Content: "api design patterns"
   ```

3. **Listing Recent Entries**
   ```
   Use: mcp__zen__context
   Operation: list
   Limit: 10
   ```

4. **Exporting Knowledge**
   ```
   Use: mcp__zen__context
   Operation: export
   ```

### Best Practices

#### When to Add Knowledge
- After making important architectural decisions
- When discovering non-obvious solutions to problems
- After debugging complex issues
- When learning project-specific patterns or conventions
- Before ending a long AI session with valuable context

#### Effective Tagging
- Use consistent tag naming (lowercase, hyphenated)
- Include both general and specific tags
- Common tag categories:
  - Technology: `redis`, `docker`, `react`
  - Type: `bug-fix`, `optimization`, `refactor`
  - Component: `auth`, `api`, `database`
  - Status: `todo`, `in-progress`, `completed`

#### Writing Good Content
- Be concise but complete
- Include "why" not just "what"
- Reference specific files or functions when relevant
- Include error messages or stack traces for bugs
- Note the context that led to the discovery

## Operations Reference

### ADD Operation
**Purpose**: Store new knowledge in the persistent knowledge base

**Parameters**:
- `operation`: "add" (required)
- `content`: The knowledge to store (required)
- `metadata`: Additional categorization (optional)
  - `category`: Type of knowledge
  - `tags`: Array of searchable tags
  - `importance`: 0.0-1.0 relevance score
  - `tool_source`: Which tool generated this
  - `related_files`: Associated code files
- `project_id`: Target project (optional, defaults to current)

**Example Response**:
```
✅ Knowledge saved successfully!
- Entry ID: `e79f0f89-fcd2-43ec-be43-d08bf9e4348e`
- Project: `app`
- Saved to: `/tmp/zen-context-kb/projects/app/entries/[id].json`
```

### SEARCH Operation
**Purpose**: Find relevant knowledge using keyword matching

**Parameters**:
- `operation`: "search" (required)
- `content`: Search query with keywords (required)
- `limit`: Maximum results (optional, default: 10)
- `project_id`: Target project (optional)

**Search Algorithm**:
- Simple keyword matching (MVP)
- Searches in content and metadata
- Returns most accessed entries first
- Case-insensitive matching

**Example Response**:
```
🔍 Found 3 matching entries:

### 1. Entry e79f0f89...
- Created: 2025-06-18 14:13
- Accessed: 5 times
- Tags: memory, redis, optimization
```

### LIST Operation
**Purpose**: Show recent knowledge entries

**Parameters**:
- `operation`: "list" (required)
- `limit`: Number of entries (optional, default: 10)
- `project_id`: Target project (optional)

**Sorting**: Most recently created first

### EXPORT Operation
**Purpose**: Create a portable backup of all knowledge

**Parameters**:
- `operation`: "export" (required)
- `project_id`: Target project (optional)

**Output Format**: JSON file containing all entries
**Location**: `/tmp/zen-context-kb/projects/[project]/exports/`

## Administration & Maintenance

### Backup Procedures

#### Manual Backup
```bash
# From host system
docker exec zen-mcp-server tar -czf - /tmp/zen-context-kb | gzip > kb_backup_$(date +%Y%m%d).tar.gz

# Verify backup
tar -tzf kb_backup_20250618.tar.gz
```

#### Automated Backup Script
```bash
#!/bin/bash
# backup_context_kb.sh
BACKUP_DIR="/Users/greg/backups/zen-context"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"
docker exec zen-mcp-server tar -czf - /tmp/zen-context-kb > "$BACKUP_DIR/kb_backup_$DATE.tar.gz"

# Keep only last 7 days
find "$BACKUP_DIR" -name "kb_backup_*.tar.gz" -mtime +7 -delete
```

### Restore Procedures
```bash
# Stop the server
docker stop zen-mcp-server

# Restore from backup
docker exec -i zen-mcp-server tar -xzf - -C / < kb_backup_20250618.tar.gz

# Restart server
./run-server.sh
```

### Monitoring & Health Checks

#### Check Storage Usage
```bash
docker exec zen-mcp-server du -sh /tmp/zen-context-kb/
docker exec zen-mcp-server find /tmp/zen-context-kb -type f | wc -l
```

#### View Recent Activity
```bash
# Check activity logs
docker exec zen-mcp-server grep "context" /tmp/mcp_activity.log | tail -20

# Find most accessed entries
docker exec zen-mcp-server find /tmp/zen-context-kb -name "*.json" -exec grep -H "access_count" {} \; | sort -nr -k2
```

### Cleanup Procedures

#### Remove Old Exports
```bash
# Remove exports older than 30 days
docker exec zen-mcp-server find /tmp/zen-context-kb -path "*/exports/*" -name "*.json" -mtime +30 -delete
```

#### Archive Inactive Projects
```bash
# Find projects not accessed in 90 days
docker exec zen-mcp-server find /tmp/zen-context-kb/projects -maxdepth 1 -type d -atime +90
```

### Migration Procedures

#### To Production Storage
```bash
# Future: Migrate from /tmp to persistent volume
docker exec zen-mcp-server cp -r /tmp/zen-context-kb /persistent/storage/
```

#### Between Servers
```bash
# Export on source
docker exec source-server tar -czf - /tmp/zen-context-kb > kb_export.tar.gz

# Import on destination
docker exec -i dest-server tar -xzf - -C / < kb_export.tar.gz
```

## Technical Details

### File Locations
- **Tool Implementation**: `/tools/context.py`
- **System Prompt**: `/systemprompts/context_prompt.py`
- **Test Suite**: `/test_context_tool.py`
- **Simulator Test**: `/simulator_tests/test_context_tool.py`

### Key Classes

#### KnowledgeEntry
```python
class KnowledgeEntry:
    entry_id: str
    timestamp: datetime
    content: str
    project_id: str
    metadata: dict
    access_count: int
    last_accessed: Optional[datetime]
```

#### ContextTool
- Inherits from `BaseTool`
- Manages file I/O operations
- Handles search indexing
- Provides operation routing

### Performance Characteristics
- **Add Operation**: O(1) - Direct file write
- **Search Operation**: O(n) - Linear scan (MVP)
- **List Operation**: O(n log n) - Sort by timestamp
- **Export Operation**: O(n) - Read all entries

### Security Considerations
- Files stored with mcpuser permissions
- No encryption in MVP (planned for v2)
- Project isolation through directory structure
- No authentication (relies on MCP security)

## Future Expansion Roadmap

### Phase 2: Enhanced Search (Q3 2024)
- **Vector Embeddings**: Semantic search using embeddings
- **Full-Text Search**: Better keyword matching
- **Relevance Scoring**: ML-based result ranking
- **Search History**: Track what users search for

### Phase 3: Advanced Features (Q4 2024)

#### Model Configuration
```python
# Proposed API
context_request = {
    "operation": "add",
    "content": "...",
    "model_config": {
        "model": "gpt-4",  # or "claude-3", "llama-3", etc.
        "temperature": 0.2,
        "max_tokens": 4000,
        "mode": "concise"  # or "detailed", "analytical"
    }
}
```

#### Local LLM Integration (Ollama)
```python
# Proposed configuration
{
    "llm_providers": {
        "ollama": {
            "endpoint": "http://localhost:11434",
            "models": ["llama3", "mistral", "codellama"],
            "default": "llama3"
        }
    }
}
```

#### Dynamic Model Switching
- Add `preferred_model` to metadata
- Auto-select model based on task type
- Cost optimization through model routing
- Fallback chains for availability

### Phase 4: Enterprise Features (2025)

#### Multi-User Support
- User-specific knowledge bases
- Shared team knowledge
- Access control and permissions
- Audit logging

#### Advanced Storage
- PostgreSQL backend option
- S3-compatible object storage
- Encryption at rest
- Compression for large entries

#### Integration Features
- Slack/Discord notifications
- GitHub issue integration
- JIRA ticket linking
- Custom webhook support

### Phase 5: Intelligence Layer (Future)

#### Knowledge Graph
- Automatic relationship detection
- Visual knowledge mapping
- Dependency tracking
- Impact analysis

#### Proactive Suggestions
- Context-aware reminders
- Similar problem detection
- Best practice recommendations
- Anti-pattern warnings

## Troubleshooting

### Common Issues

#### "No such file or directory" Error
```bash
# Check if directory exists
docker exec zen-mcp-server ls -la /tmp/zen-context-kb/

# Manually create if missing
docker exec zen-mcp-server mkdir -p /tmp/zen-context-kb/projects
```

#### Search Returns No Results
1. Verify entries exist: `list` operation
2. Check search terms are in content
3. Try broader search terms
4. Check project_id matches

#### Export Fails
```bash
# Check disk space
docker exec zen-mcp-server df -h /tmp

# Check permissions
docker exec zen-mcp-server ls -la /tmp/zen-context-kb/projects/
```

### Debug Commands

#### View Raw Entry
```bash
docker exec zen-mcp-server cat /tmp/zen-context-kb/projects/[project]/entries/[uuid].json | jq .
```

#### Check Index
```bash
docker exec zen-mcp-server cat /tmp/zen-context-kb/projects/[project]/.index/entries.json | jq .
```

#### Monitor Real-Time Activity
```bash
docker exec zen-mcp-server tail -f /tmp/mcp_activity.log | grep context
```

### Recovery Procedures

#### Rebuild Index
```python
# Emergency index rebuild script
import json
import os
from pathlib import Path

def rebuild_index(project_path):
    entries_dir = Path(project_path) / "entries"
    index = {"entries": {}, "tags": {}, "categories": {}}
    
    for entry_file in entries_dir.glob("*.json"):
        with open(entry_file) as f:
            entry = json.load(f)
            # Rebuild index entry
            index["entries"][entry["entry_id"]] = {
                "timestamp": entry["timestamp"],
                "summary": entry["content"][:100] + "...",
                "tags": entry["metadata"].get("tags", []),
                "category": entry["metadata"].get("category", "general")
            }
    
    # Save rebuilt index
    index_file = Path(project_path) / ".index" / "entries.json"
    index_file.parent.mkdir(exist_ok=True)
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)
```

## API Reference

### MCP Tool Interface
```python
# Tool name
"mcp__zen__context"

# Request format
{
    "operation": "add|search|list|export",
    "content": "string (for add/search)",
    "metadata": {dict},  # Optional
    "project_id": "string",  # Optional
    "limit": int,  # Optional
    "model": "model-name"  # Required
}

# Response format
{
    "status": "success|error|continuation_available",
    "content": "formatted response",
    "content_type": "markdown",
    "metadata": {
        "tool_name": "context",
        "thread_id": "uuid",
        "remaining_turns": int,
        "model_used": "string"
    },
    "continuation_offer": {...}  # Optional
}
```

### Error Codes
- `FileNotFoundError`: Storage directory not accessible
- `PermissionError`: Cannot write to storage
- `JSONDecodeError`: Corrupted entry file
- `ValueError`: Invalid operation or parameters

---

## Quick Reference Card

### Essential Commands
```bash
# Add knowledge
mcp__zen__context: add "Important insight" {"tags": ["topic"]}

# Search knowledge  
mcp__zen__context: search "keyword"

# List recent
mcp__zen__context: list

# Export all
mcp__zen__context: export

# Backup
docker exec zen-mcp-server tar -czf - /tmp/zen-context-kb > backup.tar.gz

# Check health
docker exec zen-mcp-server du -sh /tmp/zen-context-kb/
```

### File Paths
- Storage: `/tmp/zen-context-kb/`
- Entries: `/tmp/zen-context-kb/projects/[name]/entries/`
- Exports: `/tmp/zen-context-kb/projects/[name]/exports/`
- Index: `/tmp/zen-context-kb/projects/[name]/.index/`

### Support & Contribution
- GitHub Issues: Report bugs or request features
- Pull Requests: Contribute improvements
- Documentation: Help improve this manual

---

*Last Updated: June 18, 2025*
*Version: 1.0.0 (MVP)*