"""Vector Store Provider Interface for Semantic Search in Zen MCP Server.

This module provides an abstract base class for vector store implementations,
enabling semantic search capabilities across the codebase and documentation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np


class EntryType(Enum):
    """Types of entries that can be stored in the vector store."""

    CODE = "code"
    DOCUMENTATION = "documentation"
    CONTEXT = "context"
    KNOWLEDGE = "knowledge"
    OTHER = "other"


@dataclass
class VectorEntry:
    """Represents an entry in the vector store."""

    id: str
    content: str
    vector: Optional[np.ndarray]
    metadata: dict[str, Any]
    entry_type: EntryType
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "vector": self.vector.tolist() if self.vector is not None else None,
            "metadata": self.metadata,
            "entry_type": self.entry_type.value,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VectorEntry":
        """Create entry from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            vector=np.array(data["vector"]) if data.get("vector") else None,
            metadata=data.get("metadata", {}),
            entry_type=EntryType(data["entry_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class QueryResult:
    """Represents a search result from the vector store."""

    entry: VectorEntry
    score: float
    distance: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {"entry": self.entry.to_dict(), "score": self.score, "distance": self.distance}


@dataclass
class VectorStoreStats:
    """Statistics about the vector store."""

    total_entries: int
    entries_by_type: dict[str, int]
    storage_size_bytes: Optional[int]
    index_size_bytes: Optional[int]
    last_updated: Optional[datetime]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_entries": self.total_entries,
            "entries_by_type": self.entries_by_type,
            "storage_size_bytes": self.storage_size_bytes,
            "index_size_bytes": self.index_size_bytes,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "metadata": self.metadata,
        }


class VectorStoreProvider(ABC):
    """Abstract base class for vector store implementations.

    This interface defines the contract that all vector store providers must implement
    to enable semantic search functionality in the Zen MCP Server.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the vector store provider with configuration options.

        Args:
            **kwargs: Provider-specific configuration options
        """
        pass

    @abstractmethod
    def add_entry(
        self,
        id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        entry_type: EntryType = EntryType.OTHER,
        vector: Optional[np.ndarray] = None,
    ) -> VectorEntry:
        """Add a single entry to the vector store.

        Args:
            id: Unique identifier for the entry
            content: Text content to be indexed
            metadata: Optional metadata associated with the entry
            entry_type: Type of entry being added
            vector: Pre-computed vector (if None, provider should compute it)

        Returns:
            VectorEntry: The added entry with computed vector

        Raises:
            ValueError: If id already exists or content is invalid
            RuntimeError: If vector store operation fails
        """
        pass

    @abstractmethod
    def add_entries_batch(
        self, entries: list[tuple[str, str, Optional[dict[str, Any]], EntryType]]
    ) -> list[VectorEntry]:
        """Add multiple entries to the vector store in batch.

        Args:
            entries: List of tuples (id, content, metadata, entry_type)

        Returns:
            list[VectorEntry]: List of added entries with computed vectors

        Raises:
            ValueError: If any id already exists or content is invalid
            RuntimeError: If vector store operation fails
        """
        pass

    @abstractmethod
    def query(
        self,
        query_text: str,
        limit: int = 10,
        entry_types: Optional[list[EntryType]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> list[QueryResult]:
        """Query the vector store for similar entries.

        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            entry_types: Filter results by entry types (None = all types)
            metadata_filter: Filter results by metadata fields
            threshold: Minimum similarity threshold (provider-specific scale)

        Returns:
            list[QueryResult]: Ordered list of results (highest score first)

        Raises:
            ValueError: If query parameters are invalid
            RuntimeError: If vector store operation fails
        """
        pass

    @abstractmethod
    def delete_entry(self, id: str) -> bool:
        """Delete an entry from the vector store.

        Args:
            id: Unique identifier of the entry to delete

        Returns:
            bool: True if entry was deleted, False if not found

        Raises:
            RuntimeError: If vector store operation fails
        """
        pass

    @abstractmethod
    def get_stats(self) -> VectorStoreStats:
        """Get statistics about the vector store.

        Returns:
            VectorStoreStats: Current statistics of the vector store

        Raises:
            RuntimeError: If unable to retrieve stats
        """
        pass

    @abstractmethod
    def clear(self, entry_types: Optional[list[EntryType]] = None) -> int:
        """Clear entries from the vector store.

        Args:
            entry_types: If specified, only clear entries of these types.
                        If None, clear all entries.

        Returns:
            int: Number of entries cleared

        Raises:
            RuntimeError: If vector store operation fails
        """
        pass

    @abstractmethod
    def update_entry(
        self,
        id: str,
        content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        vector: Optional[np.ndarray] = None,
    ) -> Optional[VectorEntry]:
        """Update an existing entry in the vector store.

        Args:
            id: Unique identifier of the entry to update
            content: New content (if None, keep existing)
            metadata: New metadata (if None, keep existing)
            vector: New vector (if None and content changed, recompute)

        Returns:
            Optional[VectorEntry]: Updated entry or None if not found

        Raises:
            RuntimeError: If vector store operation fails
        """
        pass

    @abstractmethod
    def get_entry(self, id: str) -> Optional[VectorEntry]:
        """Retrieve a single entry by ID.

        Args:
            id: Unique identifier of the entry

        Returns:
            Optional[VectorEntry]: The entry if found, None otherwise

        Raises:
            RuntimeError: If vector store operation fails
        """
        pass

    @abstractmethod
    def list_entries(
        self, entry_types: Optional[list[EntryType]] = None, offset: int = 0, limit: int = 100
    ) -> list[VectorEntry]:
        """List entries in the vector store.

        Args:
            entry_types: Filter by entry types (None = all types)
            offset: Number of entries to skip
            limit: Maximum number of entries to return

        Returns:
            list[VectorEntry]: List of entries

        Raises:
            ValueError: If offset or limit are invalid
            RuntimeError: If vector store operation fails
        """
        pass

    @abstractmethod
    def export_data(self, output_path: str) -> None:
        """Export vector store data to a file.

        Args:
            output_path: Path where to save the exported data

        Raises:
            IOError: If unable to write to output path
            RuntimeError: If export operation fails
        """
        pass

    @abstractmethod
    def import_data(self, input_path: str, merge: bool = False) -> int:
        """Import vector store data from a file.

        Args:
            input_path: Path to the data file to import
            merge: If True, merge with existing data. If False, replace.

        Returns:
            int: Number of entries imported

        Raises:
            IOError: If unable to read from input path
            ValueError: If data format is invalid
            RuntimeError: If import operation fails
        """
        pass
