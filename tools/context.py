"""
Context tool - AI Context Management System for persistent knowledge
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Optional
from uuid import uuid4

from pydantic import Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from config import TEMPERATURE_ANALYTICAL
from systemprompts import CONTEXT_PROMPT

from .base import BaseTool, ToolRequest

# Default knowledge base directory
DEFAULT_KB_DIR = "/tmp/zen-context-kb"

# Field descriptions to avoid duplication
CONTEXT_FIELD_DESCRIPTIONS = {
    "operation": "Operation to perform: 'add' (add knowledge), 'search' (find context), 'list' (recent entries), 'export' (export knowledge)",
    "content": "For 'add': The knowledge/insight to store. For 'search': The search query",
    "metadata": "Optional metadata for the knowledge entry (tags, category, related files, etc.)",
    "project_id": "Project identifier (defaults to current directory name)",
    "limit": "Maximum number of results to return (for search and list operations)",
}


class ContextRequest(ToolRequest):
    """Request model for context management tool"""

    operation: Literal["add", "search", "list", "export"] = Field(
        ..., description=CONTEXT_FIELD_DESCRIPTIONS["operation"]
    )
    content: Optional[str] = Field(None, description=CONTEXT_FIELD_DESCRIPTIONS["content"])
    metadata: Optional[dict[str, Any]] = Field(None, description=CONTEXT_FIELD_DESCRIPTIONS["metadata"])
    project_id: Optional[str] = Field(None, description=CONTEXT_FIELD_DESCRIPTIONS["project_id"])
    limit: Optional[int] = Field(10, description=CONTEXT_FIELD_DESCRIPTIONS["limit"])


class KnowledgeEntry:
    """Represents a single knowledge entry in the context system"""

    def __init__(
        self,
        content: str,
        project_id: str,
        metadata: Optional[dict] = None,
        entry_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.entry_id = entry_id or str(uuid4())
        self.timestamp = timestamp or datetime.now()
        self.content = content
        self.project_id = project_id
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = None

    def to_dict(self) -> dict:
        """Convert entry to dictionary for storage"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "project_id": self.project_id,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeEntry":
        """Create entry from dictionary"""
        entry = cls(
            content=data["content"],
            project_id=data["project_id"],
            metadata=data.get("metadata", {}),
            entry_id=data["entry_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
        entry.access_count = data.get("access_count", 0)
        if data.get("last_accessed"):
            entry.last_accessed = datetime.fromisoformat(data["last_accessed"])
        return entry


class ContextTool(BaseTool):
    """AI Context Management System - Persistent knowledge across sessions"""

    def __init__(self):
        super().__init__()
        self.kb_dir = Path(os.getenv("ZEN_CONTEXT_KB_DIR", DEFAULT_KB_DIR))

    def get_name(self) -> str:
        return "context"

    def get_description(self) -> str:
        return (
            "AI CONTEXT MANAGEMENT - Manage persistent project knowledge base. "
            "Store insights from AI sessions, search past context, export knowledge, "
            "and dramatically reduce token usage by maintaining context across sessions. "
            "Operations: 'add' (store knowledge), 'search' (find relevant context), "
            "'list' (show recent entries), 'export' (export knowledge for sharing)."
        )

    def get_input_schema(self) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "search", "list", "export"],
                    "description": CONTEXT_FIELD_DESCRIPTIONS["operation"],
                },
                "content": {
                    "type": "string",
                    "description": CONTEXT_FIELD_DESCRIPTIONS["content"],
                },
                "metadata": {
                    "type": "object",
                    "description": CONTEXT_FIELD_DESCRIPTIONS["metadata"],
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "category": {"type": "string"},
                        "files": {"type": "array", "items": {"type": "string"}},
                        "tool_source": {"type": "string"},
                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "project_id": {
                    "type": "string",
                    "description": CONTEXT_FIELD_DESCRIPTIONS["project_id"],
                },
                "limit": {
                    "type": "integer",
                    "description": CONTEXT_FIELD_DESCRIPTIONS["limit"],
                    "minimum": 1,
                    "maximum": 100,
                },
                "model": self.get_model_field_schema(),
                "temperature": {
                    "type": "number",
                    "description": "Temperature for response (0-1, default 0.2)",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["operation"] + (["model"] if self.is_effective_auto_mode() else []),
        }
        return schema

    def get_system_prompt(self) -> str:
        return CONTEXT_PROMPT

    def get_default_temperature(self) -> float:
        return TEMPERATURE_ANALYTICAL

    def get_model_category(self) -> "ToolModelCategory":
        """Context tool needs balanced capabilities"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.BALANCED

    def get_request_model(self):
        return ContextRequest

    def get_project_id(self, request: ContextRequest) -> str:
        """Get project ID from request or current directory"""
        if request.project_id:
            return request.project_id
        # Default to current directory name
        return Path.cwd().name

    def get_project_dir(self, project_id: str) -> Path:
        """Get project directory, creating if needed"""
        project_dir = self.kb_dir / "projects" / project_id
        # Create directories if they don't exist
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "entries").mkdir(parents=True, exist_ok=True)
        (project_dir / ".index").mkdir(parents=True, exist_ok=True)
        return project_dir

    def save_entry(self, entry: KnowledgeEntry) -> str:
        """Save a knowledge entry to disk"""
        project_dir = self.get_project_dir(entry.project_id)
        entries_dir = project_dir / "entries"
        
        # Ensure directories exist (already created in get_project_dir)
        
        entry_file = entries_dir / f"{entry.entry_id}.json"
        
        with open(entry_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
        
        # Update index
        self._update_index(entry)
        
        return str(entry_file)

    def _update_index(self, entry: KnowledgeEntry):
        """Update the project index with new entry"""
        project_dir = self.get_project_dir(entry.project_id)
        index_file = project_dir / ".index" / "entries.json"
        
        # Load existing index
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
        else:
            index = {"entries": {}, "tags": {}, "categories": {}}
        
        # Add entry to index
        index["entries"][entry.entry_id] = {
            "timestamp": entry.timestamp.isoformat(),
            "summary": entry.content[:100] + "..." if len(entry.content) > 100 else entry.content,
            "tags": entry.metadata.get("tags", []),
            "category": entry.metadata.get("category", "general"),
        }
        
        # Update tag index
        for tag in entry.metadata.get("tags", []):
            if tag not in index["tags"]:
                index["tags"][tag] = []
            index["tags"][tag].append(entry.entry_id)
        
        # Update category index
        category = entry.metadata.get("category", "general")
        if category not in index["categories"]:
            index["categories"][category] = []
        index["categories"][category].append(entry.entry_id)
        
        # Save updated index
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

    def load_entry(self, project_id: str, entry_id: str) -> Optional[KnowledgeEntry]:
        """Load a knowledge entry from disk"""
        project_dir = self.get_project_dir(project_id)
        entry_file = project_dir / "entries" / f"{entry_id}.json"
        
        if not entry_file.exists():
            return None
        
        with open(entry_file) as f:
            data = json.load(f)
        
        entry = KnowledgeEntry.from_dict(data)
        
        # Update access tracking
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        self.save_entry(entry)  # Save updated access info
        
        return entry

    def search_entries(self, project_id: str, query: str, limit: int = 10) -> List[KnowledgeEntry]:
        """Search for entries using simple keyword matching"""
        project_dir = self.get_project_dir(project_id)
        index_file = project_dir / ".index" / "entries.json"
        
        if not index_file.exists():
            return []
        
        with open(index_file) as f:
            index = json.load(f)
        
        # Simple keyword search
        query_lower = query.lower()
        keywords = query_lower.split()
        
        matching_entries = []
        for entry_id, entry_info in index["entries"].items():
            # Check if any keyword matches summary
            summary_lower = entry_info["summary"].lower()
            if any(keyword in summary_lower for keyword in keywords):
                entry = self.load_entry(project_id, entry_id)
                if entry:
                    # Also check full content
                    content_lower = entry.content.lower()
                    if any(keyword in content_lower for keyword in keywords):
                        matching_entries.append(entry)
        
        # Sort by relevance (access count and recency)
        matching_entries.sort(
            key=lambda e: (e.access_count, e.timestamp),
            reverse=True
        )
        
        return matching_entries[:limit]

    def list_recent_entries(self, project_id: str, limit: int = 10) -> List[KnowledgeEntry]:
        """List most recent knowledge entries"""
        project_dir = self.get_project_dir(project_id)
        entries_dir = project_dir / "entries"
        
        if not entries_dir.exists():
            return []
        
        entries = []
        for entry_file in entries_dir.glob("*.json"):
            with open(entry_file) as f:
                data = json.load(f)
            entries.append(KnowledgeEntry.from_dict(data))
        
        # Sort by timestamp (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        
        return entries[:limit]

    def export_knowledge(self, project_id: str) -> dict:
        """Export all knowledge for a project"""
        entries = self.list_recent_entries(project_id, limit=1000)  # Get all entries
        
        export_data = {
            "project_id": project_id,
            "export_date": datetime.now().isoformat(),
            "entry_count": len(entries),
            "entries": [entry.to_dict() for entry in entries],
        }
        
        # Save export file
        project_dir = self.get_project_dir(project_id)
        export_dir = project_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        export_file = export_dir / f"knowledge-export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(export_file, "w") as f:
            json.dump(export_data, f, indent=2)
        
        return {
            "export_file": str(export_file),
            "entry_count": len(entries),
            "total_size": sum(len(e.content) for e in entries),
        }

    async def prepare_prompt(self, request: ContextRequest) -> str:
        """Prepare the prompt based on the operation"""
        operation = request.operation
        project_id = self.get_project_id(request)
        
        if operation == "add":
            if not request.content:
                raise ValueError("Content is required for 'add' operation")
            
            prompt = f"""Add the following knowledge to the project '{project_id}' knowledge base:

CONTENT TO STORE:
{request.content}

METADATA:
{json.dumps(request.metadata or {}, indent=2)}

Please:
1. Acknowledge the knowledge has been stored
2. Summarize what was captured
3. Suggest relevant tags if none were provided
4. Indicate how this knowledge might be useful in future sessions
"""

        elif operation == "search":
            if not request.content:
                raise ValueError("Search query is required for 'search' operation")
            
            prompt = f"""Search the project '{project_id}' knowledge base for: {request.content}

Please:
1. List all relevant entries found
2. For each entry, show:
   - Brief summary
   - When it was created
   - How many times it's been accessed
   - Why it's relevant to the search
3. If no entries found, suggest related queries
"""

        elif operation == "list":
            prompt = f"""List the {request.limit} most recent knowledge entries for project '{project_id}'.

For each entry show:
1. Entry ID and timestamp
2. Brief content summary
3. Tags and category
4. Access statistics
"""

        elif operation == "export":
            prompt = f"""Export all knowledge for project '{project_id}'.

Please:
1. Confirm the export was successful
2. Show export statistics (number of entries, total size)
3. Provide the export file location
4. Suggest how to use the exported knowledge
"""

        else:
            raise ValueError(f"Unknown operation: {operation}")

        return prompt

    def format_response(self, response: str, request: ContextRequest, model_info: Optional[dict] = None) -> str:
        """Format the response and perform the actual operation"""
        operation = request.operation
        project_id = self.get_project_id(request)
        
        result_header = f"## Context Management: {operation.title()}\n\n"
        
        if operation == "add":
            # Create and save the knowledge entry
            entry = KnowledgeEntry(
                content=request.content,
                project_id=project_id,
                metadata=request.metadata,
            )
            entry_file = self.save_entry(entry)
            
            result_header += f"✅ Knowledge saved successfully!\n"
            result_header += f"- Entry ID: `{entry.entry_id}`\n"
            result_header += f"- Project: `{project_id}`\n"
            result_header += f"- Saved to: `{entry_file}`\n\n"
            
        elif operation == "search":
            # Perform the search
            entries = self.search_entries(project_id, request.content, request.limit)
            
            result_header += f"🔍 Found {len(entries)} matching entries:\n\n"
            for i, entry in enumerate(entries, 1):
                result_header += f"### {i}. Entry {entry.entry_id[:8]}...\n"
                result_header += f"- Created: {entry.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                result_header += f"- Accessed: {entry.access_count} times\n"
                if entry.metadata.get("tags"):
                    result_header += f"- Tags: {', '.join(entry.metadata['tags'])}\n"
                result_header += f"\n```\n{entry.content[:200]}{'...' if len(entry.content) > 200 else ''}\n```\n\n"
            
        elif operation == "list":
            # List recent entries
            entries = self.list_recent_entries(project_id, request.limit)
            
            result_header += f"📋 Recent {len(entries)} entries for project `{project_id}`:\n\n"
            for i, entry in enumerate(entries, 1):
                result_header += f"### {i}. {entry.timestamp.strftime('%Y-%m-%d %H:%M')} - {entry.entry_id[:8]}...\n"
                result_header += f"- Category: {entry.metadata.get('category', 'general')}\n"
                result_header += f"- Access count: {entry.access_count}\n"
                if entry.metadata.get("tags"):
                    result_header += f"- Tags: {', '.join(entry.metadata['tags'])}\n"
                result_header += f"\n{entry.content[:100]}{'...' if len(entry.content) > 100 else ''}\n\n"
            
        elif operation == "export":
            # Export knowledge
            export_info = self.export_knowledge(project_id)
            
            result_header += f"📦 Knowledge exported successfully!\n"
            result_header += f"- Project: `{project_id}`\n"
            result_header += f"- Entries: {export_info['entry_count']}\n"
            result_header += f"- Total size: {export_info['total_size']:,} characters\n"
            result_header += f"- Export file: `{export_info['export_file']}`\n\n"
        
        return result_header + response