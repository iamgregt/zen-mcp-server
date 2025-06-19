"""
Context tool - AI Context Management System for persistent knowledge
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory


from config import TEMPERATURE_ANALYTICAL
from mcp.types import TextContent
from systemprompts import CONTEXT_PROMPT

from .base import BaseTool

# Default knowledge base directory
DEFAULT_KB_DIR = os.environ.get("STORAGE_DIR", "./data/kb")

logger = logging.getLogger(__name__)

# Field descriptions to avoid duplication
CONTEXT_FIELD_DESCRIPTIONS = {
    "operation": "Operation to perform: 'add' (add knowledge), 'search' (find context), 'list' (recent entries), 'export' (export knowledge)",
    "content": "For 'add': The knowledge/insight to store. For 'search': The search query",
    "metadata": "Optional metadata for the knowledge entry (tags, category, related files, etc.)",
    "project_id": "Project identifier (defaults to current directory name)",
    "limit": "Maximum number of results to return (for search and list operations)",
}


class ContextRequest(BaseModel):
    """Request model for context management tool"""

    operation: Literal["add", "search", "list", "export"] = Field(
        ..., description=CONTEXT_FIELD_DESCRIPTIONS["operation"]
    )
    content: Optional[str] = Field(None, description=CONTEXT_FIELD_DESCRIPTIONS["content"])
    metadata: Optional[dict[str, Any]] = Field(None, description=CONTEXT_FIELD_DESCRIPTIONS["metadata"])
    project_id: Optional[str] = Field(None, description=CONTEXT_FIELD_DESCRIPTIONS["project_id"])
    limit: Optional[int] = Field(10, description=CONTEXT_FIELD_DESCRIPTIONS["limit"])
    temperature: Optional[float] = Field(None, description="Temperature for response (0-1, default 0.2)")
    continuation_id: Optional[str] = Field(None, description="Thread continuation ID for multi-turn conversations")
    images: Optional[list[str]] = Field(None, description="Optional images for visual context")


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

        # Initialize vector store if enabled
        self.vector_store = None
        self._init_vector_store()

        # Metrics tracking
        self._metrics = {
            "total_searches": 0,
            "vector_searches": 0,
            "keyword_searches": 0,
            "search_latencies": [],
            "add_latencies": [],
            "vector_operations": 0,
            "vector_failures": 0,
            "last_stats_log": time.time()
        }
        self._stats_log_interval = 300  # Log stats every 5 minutes

    def _init_vector_store(self):
        """Initialize the vector store if enabled via environment variable"""
        # Hard code vector search to always be enabled
        enable_vector_search = True  # Always enable vector search

        if not enable_vector_search:
            logger.info("Vector search is disabled (ENABLE_VECTOR_SEARCH not set to true)")
            return

        try:
            # Import ChromaProvider dynamically to avoid dependency issues if not enabled
            from utils.chroma_provider import ChromaProvider

            # Initialize ChromaProvider with context-specific settings
            chroma_persist_dir = self.kb_dir / ".chroma"
            self.vector_store = ChromaProvider(
                persist_directory=str(chroma_persist_dir),
                collection_name="zen_context",
            )
            logger.info(f"Vector store initialized successfully at {chroma_persist_dir}")

        except ImportError as e:
            logger.warning(f"Vector search enabled but ChromaDB dependencies not available: {e}")
            logger.warning("Falling back to keyword-only search")
            self.vector_store = None

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            logger.warning("Falling back to keyword-only search")
            self.vector_store = None

    def _log_metrics(self, force: bool = False):
        """Log metrics periodically or when forced"""
        current_time = time.time()
        if not force and current_time - self._metrics["last_stats_log"] < self._stats_log_interval:
            return

        # Calculate average latencies
        avg_search_latency = (
            sum(self._metrics["search_latencies"]) / len(self._metrics["search_latencies"])
            if self._metrics["search_latencies"] else 0
        )
        avg_add_latency = (
            sum(self._metrics["add_latencies"]) / len(self._metrics["add_latencies"])
            if self._metrics["add_latencies"] else 0
        )

        # Log vector store statistics if available
        vector_stats = None
        if self.vector_store:
            try:
                vector_stats = self.vector_store.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get vector store stats: {e}")

        # Build metrics log message
        metrics_msg = "Context Tool Metrics - "
        metrics_msg += f"Searches: {self._metrics['total_searches']} total "
        metrics_msg += f"({self._metrics['vector_searches']} vector, {self._metrics['keyword_searches']} keyword), "
        metrics_msg += f"Avg Search Latency: {avg_search_latency:.3f}s, "
        metrics_msg += f"Avg Add Latency: {avg_add_latency:.3f}s, "
        metrics_msg += f"Vector Ops: {self._metrics['vector_operations']}, "
        metrics_msg += f"Vector Failures: {self._metrics['vector_failures']}"

        if vector_stats:
            metrics_msg += f", Vector Store: {vector_stats.total_entries} entries"
            if vector_stats.storage_size_bytes:
                metrics_msg += f", {vector_stats.storage_size_bytes / (1024*1024):.1f}MB storage"
            if vector_stats.metadata and "embedding_dimension" in vector_stats.metadata:
                metrics_msg += f", {vector_stats.metadata['embedding_dimension']}D embeddings"

        logger.info(metrics_msg)

        # Clear latency lists to prevent unbounded growth
        self._metrics["search_latencies"] = self._metrics["search_latencies"][-100:]  # Keep last 100
        self._metrics["add_latencies"] = self._metrics["add_latencies"][-100:]  # Keep last 100
        self._metrics["last_stats_log"] = current_time

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
                "temperature": {
                    "type": "number",
                    "description": "Temperature for response (0-1, default 0.2)",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["operation"],
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

    def get_model_field_schema(self) -> dict[str, Any]:
        """Override to remove model field from schema - context tool uses fixed model"""
        return {}

    def is_effective_auto_mode(self) -> bool:
        """Override to always return False - context tool doesn't use auto mode"""
        return False

    def _should_require_model_selection(self, model_name: str) -> bool:
        """Override to never require model selection - context tool uses fixed model"""
        return False

    async def execute(self, arguments: dict[str, Any]) -> list:
        """Override execute to use a fixed model (flash) for context operations"""
        # Always use 'flash' model for context operations - it's fast and efficient
        arguments['model'] = 'flash'

        # Call parent's execute method with the fixed model
        return await super().execute(arguments)

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
        """Save a knowledge entry to disk and index in vector store"""
        start_time = time.time()

        project_dir = self.get_project_dir(entry.project_id)
        entries_dir = project_dir / "entries"

        # Ensure directories exist (already created in get_project_dir)

        entry_file = entries_dir / f"{entry.entry_id}.json"

        with open(entry_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        # Update index
        self._update_index(entry)

        # Add to vector store if enabled
        if self.vector_store:
            try:
                vector_start = time.time()
                # Prepare metadata for vector store
                vector_metadata = entry.metadata.copy()
                vector_metadata["project_id"] = entry.project_id
                vector_metadata["access_count"] = entry.access_count

                # Add entry to vector store
                from utils.vector_store import EntryType

                self.vector_store.add_entry(
                    id=f"context_{entry.project_id}_{entry.entry_id}",
                    content=entry.content,
                    metadata=vector_metadata,
                    entry_type=EntryType.CONTEXT,
                )
                vector_time = time.time() - vector_start
                logger.info(f"Successfully indexed entry '{entry.entry_id}' in vector store (took {vector_time:.3f}s)")
                self._metrics["vector_operations"] += 1

            except Exception as e:
                # Log error but don't fail the save operation
                logger.error(f"Failed to index entry '{entry.entry_id}' in vector store: {e}")
                logger.warning("Entry was saved to disk but not indexed for semantic search")
                self._metrics["vector_failures"] += 1

        # Track add latency
        add_time = time.time() - start_time
        self._metrics["add_latencies"].append(add_time)

        # Log metrics periodically
        self._log_metrics()

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

        # Save to disk only (without re-indexing in vector store)
        with open(entry_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        # Update vector store metadata if available
        if self.vector_store:
            try:
                vector_id = f"context_{project_id}_{entry_id}"
                vector_metadata = entry.metadata.copy()
                vector_metadata["project_id"] = project_id
                vector_metadata["access_count"] = entry.access_count
                vector_metadata["last_accessed"] = entry.last_accessed.isoformat()

                # Update only metadata, not content
                self.vector_store.update_entry(id=vector_id, metadata=vector_metadata)
            except Exception as e:
                # Log but don't fail
                logger.debug(f"Could not update vector store metadata for entry '{entry_id}': {e}")

        return entry

    def search_entries(self, project_id: str, query: str, limit: int = 10) -> list[KnowledgeEntry]:
        """Search for entries using hybrid search (semantic + keyword) if vector store available"""
        start_time = time.time()
        self._metrics["total_searches"] += 1

        # Try hybrid search first if vector store is enabled
        if self.vector_store:
            try:
                # Perform hybrid search
                hybrid_start = time.time()
                entry_ids = self._hybrid_search(project_id, query, limit)
                hybrid_time = time.time() - hybrid_start

                # Load full entries
                matching_entries = []
                for entry_id in entry_ids:
                    entry = self.load_entry(project_id, entry_id)
                    if entry:
                        matching_entries.append(entry)

                if matching_entries:
                    search_time = time.time() - start_time
                    self._metrics["search_latencies"].append(search_time)
                    self._metrics["vector_searches"] += 1
                    logger.info(f"Hybrid search returned {len(matching_entries)} results in {search_time:.3f}s (vector search: {hybrid_time:.3f}s)")

                    # Log metrics periodically
                    self._log_metrics()
                    return matching_entries

            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to keyword search: {e}")
                self._metrics["vector_failures"] += 1

        # Fall back to legacy keyword search
        results = self._keyword_search_legacy(project_id, query, limit)
        search_time = time.time() - start_time
        self._metrics["search_latencies"].append(search_time)
        self._metrics["keyword_searches"] += 1

        logger.info(f"Keyword search returned {len(results)} results in {search_time:.3f}s")

        # Log metrics periodically
        self._log_metrics()
        return results

    def _hybrid_search(self, project_id: str, query: str, limit: int) -> list[str]:
        """Perform hybrid search using Reciprocal Rank Fusion (RRF) of semantic and keyword results"""
        from utils.vector_store import EntryType

        k = 60  # RRF constant (standard value for good fusion)

        # 1. Perform semantic search (get more results for better fusion)
        semantic_start = time.time()
        semantic_results = self.vector_store.query(
            query_text=query,
            limit=limit * 2,  # Get 2x results for better ranking
            entry_types=[EntryType.CONTEXT],
            metadata_filter={"project_id": project_id},
        )
        semantic_time = time.time() - semantic_start
        self._metrics["vector_operations"] += 1

        # 2. Perform keyword search
        keywords = query.lower().split()
        keyword_start = time.time()
        keyword_results = self.vector_store.keyword_search(
            keywords=keywords,
            limit=limit * 2,  # Get 2x results for better ranking
            entry_types=[EntryType.CONTEXT],
            metadata_filter={"project_id": project_id},
        )
        keyword_time = time.time() - keyword_start
        self._metrics["vector_operations"] += 1

        logger.debug(f"Hybrid search: semantic={len(semantic_results)} results in {semantic_time:.3f}s, keyword={len(keyword_results)} results in {keyword_time:.3f}s")

        # 3. Combine results using RRF scoring
        rrf_scores = {}

        # Add semantic search results with RRF scores
        for rank, result in enumerate(semantic_results):
            # Extract entry_id from the vector store id
            # Format: context_{project_id}_{entry_id}
            parts = result.entry.id.split("_", 2)
            if len(parts) >= 3:
                entry_id = parts[2]
                # RRF score formula: 1 / (k + rank + 1)
                rrf_scores[entry_id] = rrf_scores.get(entry_id, 0) + (1.0 / (k + rank + 1))

        # Add keyword search results with RRF scores
        for rank, result in enumerate(keyword_results):
            # Extract entry_id from the vector store id
            parts = result.entry.id.split("_", 2)
            if len(parts) >= 3:
                entry_id = parts[2]
                # RRF score formula: 1 / (k + rank + 1)
                rrf_scores[entry_id] = rrf_scores.get(entry_id, 0) + (1.0 / (k + rank + 1))

        # Sort by combined RRF score (highest first)
        sorted_entries = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top entry IDs
        return [entry_id for entry_id, _ in sorted_entries[:limit]]

    def _keyword_search_legacy(self, project_id: str, query: str, limit: int = 10) -> list[KnowledgeEntry]:
        """Legacy keyword search for when vector store is not available"""
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
        matching_entries.sort(key=lambda e: (e.access_count, e.timestamp), reverse=True)

        return matching_entries[:limit]

    def list_recent_entries(self, project_id: str, limit: int = 10) -> list[KnowledgeEntry]:
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

            result_header += "✅ Knowledge saved successfully!\n"
            result_header += f"- Entry ID: `{entry.entry_id}`\n"
            result_header += f"- Project: `{project_id}`\n"
            result_header += f"- Saved to: `{entry_file}`\n\n"

            # Log operation summary
            logger.info(f"CONTEXT_ADD: project={project_id}, entry_id={entry.entry_id}, content_size={len(request.content)}, metadata={request.metadata}")

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

            # Log operation summary
            logger.info(f"CONTEXT_SEARCH: project={project_id}, query='{request.content[:100]}', results={len(entries)}, limit={request.limit}")

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

            result_header += "📦 Knowledge exported successfully!\n"
            result_header += f"- Project: `{project_id}`\n"
            result_header += f"- Entries: {export_info['entry_count']}\n"
            result_header += f"- Total size: {export_info['total_size']:,} characters\n"
            result_header += f"- Export file: `{export_info['export_file']}`\n\n"

        return result_header + response

    def get_metrics_summary(self) -> dict:
        """Get a summary of current metrics for monitoring"""
        # Force log current metrics
        self._log_metrics(force=True)

        # Get vector store stats if available
        vector_stats = None
        if self.vector_store:
            try:
                vector_stats = self.vector_store.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get vector store stats: {e}")

        # Calculate averages
        avg_search_latency = (
            sum(self._metrics["search_latencies"]) / len(self._metrics["search_latencies"])
            if self._metrics["search_latencies"] else 0
        )
        avg_add_latency = (
            sum(self._metrics["add_latencies"]) / len(self._metrics["add_latencies"])
            if self._metrics["add_latencies"] else 0
        )

        summary = {
            "total_searches": self._metrics["total_searches"],
            "vector_searches": self._metrics["vector_searches"],
            "keyword_searches": self._metrics["keyword_searches"],
            "avg_search_latency": avg_search_latency,
            "avg_add_latency": avg_add_latency,
            "vector_operations": self._metrics["vector_operations"],
            "vector_failures": self._metrics["vector_failures"],
            "vector_enabled": self.vector_store is not None,
        }

        if vector_stats:
            summary["vector_store"] = {
                "total_entries": vector_stats.total_entries,
                "entries_by_type": vector_stats.entries_by_type,
                "storage_size_mb": vector_stats.storage_size_bytes / (1024*1024) if vector_stats.storage_size_bytes else None,
                "embedding_dimension": vector_stats.metadata.get("embedding_dimension") if vector_stats.metadata else None,
            }

        return summary
