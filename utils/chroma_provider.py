"""ChromaDB implementation of the VectorStoreProvider interface.

This module provides a ChromaDB-based vector store implementation for semantic search
in the Zen MCP Server, using Gemini embeddings for superior semantic understanding.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
import numpy as np
import psutil
from chromadb.config import Settings

from .vector_store import EntryType, QueryResult, VectorEntry, VectorStoreProvider, VectorStoreStats

logger = logging.getLogger(__name__)


class ChromaProvider(VectorStoreProvider):
    """ChromaDB implementation of the VectorStoreProvider interface.

    This provider uses Gemini embeddings (gemini-embedding-exp-03-07) for high-quality
    3072-dimensional embeddings and ChromaDB for vector storage and similarity search.
    """

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = "zen_context",
        **kwargs,
    ):
        """Initialize the ChromaDB vector store provider with Gemini embeddings.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            **kwargs: Additional configuration options
        """
        # Use environment variable or default to local data directory
        if persist_directory is None:
            persist_directory = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Initialize Gemini embeddings
        from providers.registry import ModelProviderRegistry, ProviderType

        self.embedding_provider = ModelProviderRegistry.get_provider(ProviderType.GOOGLE)
        if not self.embedding_provider:
            raise ValueError("Google provider not available for Gemini embeddings")

        logger.info("Using Gemini embeddings (gemini-embedding-exp-03-07, 3072 dimensions)")
        self.embedding_dimension = 3072
        self.model = None  # No local model needed

        # Update collection name to indicate Gemini embeddings
        self.collection_name = f"{collection_name}_gemini_3072"

        # Initialize or get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            logger.info(f"Collection {self.collection_name} not found, creating new collection")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_dimension": self.embedding_dimension,
                    "embedding_model": "gemini-embedding-exp-03-07"
                }
            )
            logger.info(f"Created new collection: {self.collection_name}")

        # Initialize keyword extraction cache
        self._keyword_cache = {}

        # Memory usage tracking
        self._process = psutil.Process()
        self._initial_memory = self._process.memory_info().rss / (1024 * 1024)  # MB
        self._last_memory_log = time.time()
        self._memory_log_interval = 300  # Log memory every 5 minutes

    def _log_memory_usage(self, operation: str = "", force: bool = False):
        """Log memory usage periodically or when forced"""
        current_time = time.time()
        if not force and current_time - self._last_memory_log < self._memory_log_interval:
            return

        try:
            memory_info = self._process.memory_info()
            current_memory = memory_info.rss / (1024 * 1024)  # MB
            memory_increase = current_memory - self._initial_memory

            # Get collection stats
            collection_count = self.collection.count()
            cache_size = len(self._keyword_cache)

            # Log memory usage
            memory_msg = "ChromaProvider Memory Usage"
            if operation:
                memory_msg += f" ({operation})"
            memory_msg += f": Current={current_memory:.1f}MB, "
            memory_msg += f"Increase={memory_increase:.1f}MB, "
            memory_msg += f"Collection={collection_count} entries, "
            memory_msg += f"KeywordCache={cache_size} items"

            # Add model info - Gemini is API-based, no local memory usage
            memory_msg += ", Model=Gemini (API-based)"

            logger.info(memory_msg)
            self._last_memory_log = current_time

        except Exception as e:
            logger.debug(f"Failed to log memory usage: {e}")

    def _format_instruction_query(self, text: str) -> str:
        """Format query text with instruction prefix for multilingual-e5-large-instruct.

        Args:
            text: The query text

        Returns:
            str: Formatted query with instruction prefix
        """
        return f"Instruct: Retrieve semantically similar technical content\nQuery: {text}"

    def _compute_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        """Compute embedding for the given text.

        Args:
            text: Text to embed
            is_query: Whether this is a query (applies instruction formatting)

        Returns:
            np.ndarray: Embedding vector
        """
        # Use appropriate task type based on query vs document
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"

        try:
            # Get embedding from Gemini
            embedding = self.embedding_provider.get_embedding(
                text=text,
                model="gemini-embedding-exp-03-07",
                task_type=task_type
            )
            return embedding
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise

    def _extract_keywords_ai(self, text: str) -> list[str]:
        """Extract keywords using Gemini Flash for intelligent understanding.

        Args:
            text: Text to extract keywords from

        Returns:
            list[str]: List of extracted keywords
        """
        # Check cache first
        cache_key = hash(text[:200])  # Use first 200 chars as cache key
        if cache_key in self._keyword_cache:
            return self._keyword_cache[cache_key]

        # Use Gemini Flash for extraction
        from providers.registry import ModelProviderRegistry, ProviderType

        provider = ModelProviderRegistry.get_provider(ProviderType.GOOGLE)
        if not provider:
            raise ValueError("Flash provider not available for keyword extraction")

        # Create optimized extraction prompt
        prompt = f"""Extract technical keywords from this text for search indexing.

Text: {text[:1000]}

Requirements:
- Extract 10-20 most important technical keywords
- Focus on: programming languages, frameworks, tools, libraries
- Include: technical concepts, patterns, methodologies
- Include: specific error types, function names, API endpoints if mentioned
- Exclude: common words, articles, prepositions

Return ONLY a JSON array, no other text:
["keyword1", "keyword2", "keyword3"]"""

        # Call Gemini Flash
        response = provider.generate_content(
            prompt=prompt,
            model_name="gemini-2.5-flash",  # Use latest stable flash model
            temperature=0.1,  # Low temperature for consistency
            max_output_tokens=200,
        )

        # Parse response
        content = response.content.strip()

        # Extract JSON array from response
        if "[" in content and "]" in content:
            start = content.find("[")
            end = content.rfind("]") + 1
            keywords = json.loads(content[start:end])
        else:
            # If response isn't valid JSON, raise error
            raise ValueError(f"Invalid keyword extraction response: {content}")

        # Cache and return result
        self._keyword_cache[cache_key] = keywords[:20]  # Limit to 20 keywords
        return keywords[:20]

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
            vector: Pre-computed vector (if None, provider will compute it)

        Returns:
            VectorEntry: The added entry with computed vector

        Raises:
            ValueError: If id already exists or content is invalid
            RuntimeError: If vector store operation fails
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Check if ID already exists
        existing = self.collection.get(ids=[id])
        if existing and existing["ids"]:
            raise ValueError(f"Entry with id '{id}' already exists")

        # Compute vector if not provided
        if vector is None:
            vector = self._compute_embedding(content)

        # Prepare metadata
        entry_metadata = (metadata or {}).copy()
        entry_metadata["entry_type"] = entry_type.value
        entry_metadata["timestamp"] = datetime.now().isoformat()

        # Extract keywords using AI
        try:
            keywords = self._extract_keywords_ai(content)
            # Store keywords as JSON string for ChromaDB compatibility
            entry_metadata["keywords"] = json.dumps(keywords)
            logger.debug(f"Extracted {len(keywords)} keywords for entry '{id}'")
        except Exception as e:
            logger.warning(f"Keyword extraction failed for entry '{id}': {e}")
            # Don't fail the entire operation if keyword extraction fails
            entry_metadata["keywords"] = "[]"

        # Add to ChromaDB
        try:
            self.collection.add(ids=[id], embeddings=[vector.tolist()], documents=[content], metadatas=[entry_metadata])
        except Exception as e:
            logger.error(f"Failed to add entry to ChromaDB: {e}")
            raise RuntimeError(f"Failed to add entry: {e}")

        # Create and return VectorEntry
        entry = VectorEntry(
            id=id,
            content=content,
            vector=vector,
            metadata=metadata or {},
            entry_type=entry_type,
            timestamp=datetime.now(),
        )

        logger.info(f"Added entry '{id}' of type {entry_type.value}")

        # Log memory usage periodically
        self._log_memory_usage("add_entry")

        return entry

    def add_entries_batch(
        self, entries: list[tuple[str, str, Optional[dict[str, Any]], EntryType]], batch_size: int = 100
    ) -> list[VectorEntry]:
        """Add multiple entries to the vector store in batch with optimized processing.

        Args:
            entries: List of tuples (id, content, metadata, entry_type)
            batch_size: Size of sub-batches for processing (default: 100)

        Returns:
            list[VectorEntry]: List of added entries with computed vectors

        Raises:
            ValueError: If any id already exists or content is invalid
            RuntimeError: If vector store operation fails
        """
        if not entries:
            return []

        # Import tqdm here for optional progress bar support
        try:
            from tqdm import tqdm

            use_progress = True
        except ImportError:
            use_progress = False
            logger.debug("tqdm not available, progress bars disabled")

        # Validate entries
        ids = []
        contents = []
        metadatas = []
        entry_types = []

        for entry in entries:
            if len(entry) != 4:
                raise ValueError("Each entry must be a tuple of (id, content, metadata, entry_type)")

            id, content, metadata, entry_type = entry

            if not content or not content.strip():
                raise ValueError(f"Content cannot be empty for id '{id}'")

            ids.append(id)
            contents.append(content)
            metadatas.append(metadata or {})
            entry_types.append(entry_type)

        # Check for existing IDs in batches to avoid large queries
        logger.info(f"Checking for existing IDs among {len(ids)} entries...")
        existing_ids = []
        for i in range(0, len(ids), 1000):  # Check 1000 IDs at a time
            batch_ids = ids[i : i + 1000]
            existing = self.collection.get(ids=batch_ids)
            if existing and existing["ids"]:
                existing_ids.extend(existing["ids"])

        if existing_ids:
            raise ValueError(
                f"Entries with ids {existing_ids[:10]}{'...' if len(existing_ids) > 10 else ''} already exist"
            )

        # Process in optimized sub-batches
        all_vector_entries = []
        total_entries = len(entries)

        # Create progress bar if available
        if use_progress and total_entries > 10:
            pbar = tqdm(total=total_entries, desc="Processing entries", unit="entries")
        else:
            pbar = None

        for batch_start in range(0, total_entries, batch_size):
            batch_end = min(batch_start + batch_size, total_entries)
            batch_contents = contents[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            batch_metadatas = metadatas[batch_start:batch_end]
            batch_entry_types = entry_types[batch_start:batch_end]

            # Compute embeddings for this batch
            logger.debug(
                f"Computing embeddings for batch {batch_start//batch_size + 1} ({len(batch_contents)} entries)..."
            )

            # Compute Gemini embeddings one by one due to rate limiting
            batch_embeddings = []
            for content in batch_contents:
                try:
                    embedding = self._compute_embedding(content, is_query=False)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to compute Gemini embedding: {e}")
                    raise
            batch_embeddings = np.array(batch_embeddings)

            # Prepare metadata and extract keywords in parallel if possible
            chromadb_metadatas = []

            # Process keywords in smaller sub-batches to avoid API rate limits
            keyword_batch_size = 10
            for i in range(0, len(batch_contents), keyword_batch_size):
                sub_batch_end = min(i + keyword_batch_size, len(batch_contents))

                # Extract keywords for sub-batch
                for j in range(i, sub_batch_end):
                    entry_metadata = batch_metadatas[j].copy()
                    entry_metadata["entry_type"] = batch_entry_types[j].value
                    entry_metadata["timestamp"] = datetime.now().isoformat()

                    # Extract keywords with error handling
                    try:
                        keywords = self._extract_keywords_ai(batch_contents[j])
                        entry_metadata["keywords"] = json.dumps(keywords)
                    except Exception as e:
                        logger.warning(f"Keyword extraction failed for entry {batch_start + j}: {e}")
                        entry_metadata["keywords"] = "[]"

                    chromadb_metadatas.append(entry_metadata)

            # Add to ChromaDB with optimized batch size
            try:
                # ChromaDB performs best with batches of 100-500 items
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_contents,
                    metadatas=chromadb_metadatas,
                )
            except Exception as e:
                logger.error(f"Failed to add batch entries to ChromaDB: {e}")
                if pbar:
                    pbar.close()
                raise RuntimeError(f"Failed to add batch entries: {e}")

            # Create VectorEntry objects for this batch
            for i in range(len(batch_ids)):
                entry = VectorEntry(
                    id=batch_ids[i],
                    content=batch_contents[i],
                    vector=batch_embeddings[i],
                    metadata=batch_metadatas[i],
                    entry_type=batch_entry_types[i],
                    timestamp=datetime.now(),
                )
                all_vector_entries.append(entry)

            # Update progress
            if pbar:
                pbar.update(len(batch_ids))

        if pbar:
            pbar.close()

        logger.info(
            f"Successfully added {len(all_vector_entries)} entries in {(total_entries + batch_size - 1) // batch_size} batches"
        )

        # Log memory usage after batch operation
        self._log_memory_usage("add_entries_batch", force=True)

        return all_vector_entries

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
            threshold: Minimum similarity threshold (0-1, higher is more similar)

        Returns:
            list[QueryResult]: Ordered list of results (highest score first)

        Raises:
            ValueError: If query parameters are invalid
            RuntimeError: If vector store operation fails
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        if limit <= 0:
            raise ValueError("Limit must be positive")

        # Compute query embedding
        query_embedding = self._compute_embedding(query_text, is_query=True)

        # Build where clause for filtering
        where = {}
        where_conditions = []

        if entry_types:
            where_conditions.append({"entry_type": {"$in": [et.value for et in entry_types]}})

        if metadata_filter:
            # Add each metadata filter as a separate condition
            for key, value in metadata_filter.items():
                where_conditions.append({key: value})

        # ChromaDB expects either a single condition or $and/$or operator
        if len(where_conditions) == 0:
            where = None
        elif len(where_conditions) == 1:
            where = where_conditions[0]
        else:
            where = {"$and": where_conditions}

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                where=where if where else None,
            )
        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}")
            raise RuntimeError(f"Failed to query: {e}")

        # Process results
        query_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # Calculate similarity score (1 - distance for cosine)
                distance = results["distances"][0][i]
                score = 1.0 - distance

                # Apply threshold filter
                if threshold is not None and score < threshold:
                    continue

                # Create VectorEntry
                metadata = results["metadatas"][0][i]
                entry_type_str = metadata.pop("entry_type", EntryType.OTHER.value)
                timestamp_str = metadata.pop("timestamp", datetime.now().isoformat())

                entry = VectorEntry(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    vector=None,  # Don't return embeddings in query results
                    metadata=metadata,
                    entry_type=EntryType(entry_type_str),
                    timestamp=datetime.fromisoformat(timestamp_str),
                )

                query_result = QueryResult(entry=entry, score=score, distance=distance)
                query_results.append(query_result)

        logger.info(f"Query returned {len(query_results)} results")

        # Log memory usage periodically
        self._log_memory_usage("query")

        return query_results

    def keyword_search(
        self,
        keywords: list[str],
        limit: int = 10,
        entry_types: Optional[list[EntryType]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[QueryResult]:
        """Search for entries containing specific keywords.

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results
            entry_types: Filter by entry types
            metadata_filter: Additional metadata filters

        Returns:
            list[QueryResult]: Matching entries with relevance scores
        """
        # Build where clause for basic filters
        where = {}

        # Add entry type filter
        if entry_types:
            where["entry_type"] = {"$in": [et.value for et in entry_types]}

        # Add metadata filters
        if metadata_filter:
            where.update(metadata_filter)

        # Get all entries matching basic filters (we'll filter by keywords manually)
        try:
            results = self.collection.get(
                where=where if where else None,
                limit=limit * 10,  # Get more to filter by keywords
                include=["metadatas", "documents"],
            )
        except Exception as e:
            logger.error(f"Failed to perform keyword search: {e}")
            raise RuntimeError(f"Failed to perform keyword search: {e}")

        # Convert to QueryResult format and filter by keywords
        query_results = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i].copy()

                # Parse keywords from JSON string
                entry_keywords_str = metadata.get("keywords", "[]")
                try:
                    entry_keywords = json.loads(entry_keywords_str)
                except json.JSONDecodeError:
                    entry_keywords = []

                # Check if any search keywords match
                if not keywords or any(any(kw.lower() in ek.lower() for ek in entry_keywords) for kw in keywords):
                    entry_type_str = metadata.pop("entry_type", EntryType.OTHER.value)
                    timestamp_str = metadata.pop("timestamp", datetime.now().isoformat())

                    # Calculate relevance score based on keyword matches
                    matched_keywords = sum(
                        1 for kw in keywords if any(kw.lower() in ek.lower() for ek in entry_keywords)
                    )
                    score = matched_keywords / len(keywords) if keywords else 1.0

                    entry = VectorEntry(
                        id=results["ids"][i],
                        content=results["documents"][i],
                        vector=None,
                        metadata=metadata,
                        entry_type=EntryType(entry_type_str),
                        timestamp=datetime.fromisoformat(timestamp_str),
                    )

                    query_result = QueryResult(
                        entry=entry, score=score, distance=1.0 - score  # Convert to distance metric
                    )
                    query_results.append(query_result)

        # Sort by score (highest first) and limit
        query_results.sort(key=lambda r: r.score, reverse=True)
        return query_results[:limit]

    def delete_entry(self, id: str) -> bool:
        """Delete an entry from the vector store.

        Args:
            id: Unique identifier of the entry to delete

        Returns:
            bool: True if entry was deleted, False if not found

        Raises:
            RuntimeError: If vector store operation fails
        """
        try:
            # Check if entry exists
            existing = self.collection.get(ids=[id])
            if not existing or not existing["ids"]:
                return False

            # Delete from ChromaDB
            self.collection.delete(ids=[id])
            logger.info(f"Deleted entry '{id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete entry from ChromaDB: {e}")
            raise RuntimeError(f"Failed to delete entry: {e}")

    def get_stats(self) -> VectorStoreStats:
        """Get statistics about the vector store.

        Returns:
            VectorStoreStats: Current statistics of the vector store

        Raises:
            RuntimeError: If unable to retrieve stats
        """
        try:
            # Get total count
            total_entries = self.collection.count()

            # Get all entries to calculate stats by type
            all_entries = self.collection.get()
            entries_by_type = {}

            if all_entries and all_entries["metadatas"]:
                for metadata in all_entries["metadatas"]:
                    entry_type = metadata.get("entry_type", EntryType.OTHER.value)
                    entries_by_type[entry_type] = entries_by_type.get(entry_type, 0) + 1

            # Estimate storage size (ChromaDB doesn't provide direct size info)
            storage_size = None
            if os.path.exists(self.persist_directory):
                storage_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(self.persist_directory)
                    for filename in filenames
                )

            stats = VectorStoreStats(
                total_entries=total_entries,
                entries_by_type=entries_by_type,
                storage_size_bytes=storage_size,
                index_size_bytes=None,  # ChromaDB doesn't separate index size
                last_updated=datetime.now(),
                metadata={
                    "collection_name": self.collection_name,
                    "model_name": "gemini-embedding-exp-03-07",
                    "embedding_dimension": self.embedding_dimension,
                    "using_gemini": True,
                },
            )

            return stats
        except Exception as e:
            logger.error(f"Failed to get stats from ChromaDB: {e}")
            raise RuntimeError(f"Failed to get stats: {e}")

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
        try:
            if entry_types is None:
                # Clear all entries
                count = self.collection.count()
                self.client.delete_collection(name=self.collection_name)
                # Recreate empty collection
                self.collection = self.client.create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Cleared all {count} entries")
                return count
            else:
                # Clear specific entry types
                where = {"entry_type": {"$in": [et.value for et in entry_types]}}

                # Get IDs to delete
                results = self.collection.get(where=where)
                if not results or not results["ids"]:
                    return 0

                ids_to_delete = results["ids"]
                count = len(ids_to_delete)

                # Delete entries
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Cleared {count} entries of types {[et.value for et in entry_types]}")
                return count
        except Exception as e:
            logger.error(f"Failed to clear entries from ChromaDB: {e}")
            raise RuntimeError(f"Failed to clear entries: {e}")

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
        try:
            # Get existing entry
            existing = self.collection.get(ids=[id])
            if not existing or not existing["ids"]:
                return None

            # Extract current values
            current_content = existing["documents"][0]
            current_metadata = existing["metadatas"][0]

            # Determine what to update
            new_content = content if content is not None else current_content
            new_metadata = current_metadata.copy()
            if metadata is not None:
                new_metadata.update(metadata)

            # Update timestamp
            new_metadata["timestamp"] = datetime.now().isoformat()

            # Compute new vector if content changed and vector not provided
            if content is not None and vector is None:
                vector = self._compute_embedding(new_content)

            # Update in ChromaDB
            update_args = {"ids": [id], "documents": [new_content], "metadatas": [new_metadata]}

            if vector is not None:
                update_args["embeddings"] = [vector.tolist()]

            self.collection.update(**update_args)

            # Create updated VectorEntry
            entry_type_str = new_metadata.get("entry_type", EntryType.OTHER.value)
            timestamp_str = new_metadata.get("timestamp", datetime.now().isoformat())

            # Remove internal metadata fields
            clean_metadata = new_metadata.copy()
            clean_metadata.pop("entry_type", None)
            clean_metadata.pop("timestamp", None)

            updated_entry = VectorEntry(
                id=id,
                content=new_content,
                vector=vector,
                metadata=clean_metadata,
                entry_type=EntryType(entry_type_str),
                timestamp=datetime.fromisoformat(timestamp_str),
            )

            logger.info(f"Updated entry '{id}'")
            return updated_entry
        except Exception as e:
            logger.error(f"Failed to update entry in ChromaDB: {e}")
            raise RuntimeError(f"Failed to update entry: {e}")

    def get_entry(self, id: str) -> Optional[VectorEntry]:
        """Retrieve a single entry by ID.

        Args:
            id: Unique identifier of the entry

        Returns:
            Optional[VectorEntry]: The entry if found, None otherwise

        Raises:
            RuntimeError: If vector store operation fails
        """
        try:
            results = self.collection.get(ids=[id])
            if not results or not results["ids"]:
                return None

            # Extract entry data
            content = results["documents"][0]
            metadata = results["metadatas"][0]

            entry_type_str = metadata.get("entry_type", EntryType.OTHER.value)
            timestamp_str = metadata.get("timestamp", datetime.now().isoformat())

            # Remove internal metadata fields
            clean_metadata = metadata.copy()
            clean_metadata.pop("entry_type", None)
            clean_metadata.pop("timestamp", None)

            entry = VectorEntry(
                id=id,
                content=content,
                vector=None,  # Don't return embeddings for single get
                metadata=clean_metadata,
                entry_type=EntryType(entry_type_str),
                timestamp=datetime.fromisoformat(timestamp_str),
            )

            return entry
        except Exception as e:
            logger.error(f"Failed to get entry from ChromaDB: {e}")
            raise RuntimeError(f"Failed to get entry: {e}")

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
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        if limit <= 0:
            raise ValueError("Limit must be positive")

        try:
            # Build where clause
            where = None
            if entry_types:
                where = {"entry_type": {"$in": [et.value for et in entry_types]}}

            # Get all matching entries (ChromaDB doesn't support offset/limit directly)
            results = self.collection.get(where=where)

            if not results or not results["ids"]:
                return []

            # Create VectorEntry objects
            entries = []
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i]
                entry_type_str = metadata.get("entry_type", EntryType.OTHER.value)
                timestamp_str = metadata.get("timestamp", datetime.now().isoformat())

                # Remove internal metadata fields
                clean_metadata = metadata.copy()
                clean_metadata.pop("entry_type", None)
                clean_metadata.pop("timestamp", None)

                entry = VectorEntry(
                    id=results["ids"][i],
                    content=results["documents"][i],
                    vector=None,  # Don't return embeddings in list
                    metadata=clean_metadata,
                    entry_type=EntryType(entry_type_str),
                    timestamp=datetime.fromisoformat(timestamp_str),
                )
                entries.append(entry)

            # Sort by timestamp (newest first) and apply offset/limit
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            return entries[offset : offset + limit]
        except Exception as e:
            logger.error(f"Failed to list entries from ChromaDB: {e}")
            raise RuntimeError(f"Failed to list entries: {e}")

    def export_data(self, output_path: str) -> None:
        """Export vector store data to a file.

        Args:
            output_path: Path where to save the exported data

        Raises:
            IOError: If unable to write to output path
            RuntimeError: If export operation fails
        """
        try:
            # Get all entries
            results = self.collection.get()

            if not results or not results["ids"]:
                # Create empty export
                export_data = {
                    "version": "1.0",
                    "model": "gemini-embedding-exp-03-07",
                    "collection": self.collection_name,
                    "exported_at": datetime.now().isoformat(),
                    "entries": [],
                }
            else:
                # Create export data
                entries = []
                for i in range(len(results["ids"])):
                    entry_data = {
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "embedding": results["embeddings"][i] if results.get("embeddings") else None,
                    }
                    entries.append(entry_data)

                export_data = {
                    "version": "1.0",
                    "model": "gemini-embedding-exp-03-07",
                    "collection": self.collection_name,
                    "exported_at": datetime.now().isoformat(),
                    "entries": entries,
                }

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(export_data['entries'])} entries to {output_path}")
        except OSError as e:
            logger.error(f"Failed to write export file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to export data from ChromaDB: {e}")
            raise RuntimeError(f"Failed to export data: {e}")

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
        try:
            # Read import file
            with open(input_path, encoding="utf-8") as f:
                import_data = json.load(f)

            # Validate format
            if not isinstance(import_data, dict) or "entries" not in import_data:
                raise ValueError("Invalid import file format")

            entries = import_data["entries"]
            if not isinstance(entries, list):
                raise ValueError("Invalid entries format in import file")

            # Clear existing data if not merging
            if not merge:
                self.clear()

            # Import entries
            if not entries:
                return 0

            # Prepare batch data
            ids = []
            contents = []
            metadatas = []
            embeddings = []

            for entry in entries:
                if not isinstance(entry, dict) or "id" not in entry or "content" not in entry:
                    raise ValueError("Invalid entry format in import file")

                ids.append(entry["id"])
                contents.append(entry["content"])
                metadatas.append(entry.get("metadata", {}))

                # Use existing embedding if available, otherwise we'll compute
                if entry.get("embedding"):
                    embeddings.append(entry["embedding"])
                else:
                    embeddings = None  # Will compute all

            # Compute embeddings if needed
            if embeddings is None:
                logger.info(f"Computing embeddings for {len(contents)} imported entries...")
                # Process in optimized batches
                computed_embeddings = []
                batch_size = 100

                # Use tqdm if available
                try:
                    from tqdm import tqdm

                    pbar = tqdm(total=len(contents), desc="Computing embeddings", unit="entries")
                except ImportError:
                    pbar = None

                for i in range(0, len(contents), batch_size):
                    batch = contents[i : i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=32,  # Optimal for transformer models
                    )
                    computed_embeddings.extend(batch_embeddings.tolist())

                    if pbar:
                        pbar.update(len(batch))

                if pbar:
                    pbar.close()

                embeddings = computed_embeddings

            # Add to ChromaDB in optimized batches
            logger.info(f"Adding {len(ids)} entries to ChromaDB...")
            batch_size = 500  # ChromaDB optimal batch size

            try:
                from tqdm import tqdm

                pbar = tqdm(total=len(ids), desc="Importing to ChromaDB", unit="entries")
            except ImportError:
                pbar = None

            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))
                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=contents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                )

                if pbar:
                    pbar.update(batch_end - i)

            if pbar:
                pbar.close()

            logger.info(f"Imported {len(ids)} entries from {input_path}")
            return len(ids)
        except OSError as e:
            logger.error(f"Failed to read import file: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in import file: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.error(f"Failed to import data to ChromaDB: {e}")
            raise RuntimeError(f"Failed to import data: {e}")

    def reset_collection(self) -> None:
        """Delete and recreate the current collection.

        WARNING: This will delete all data in the collection!
        """
        logger.warning(f"Resetting collection '{self.collection_name}' - all data will be deleted!")

        # Delete the collection
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection (may not exist): {e}")

        # Recreate the collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_dimension": self.embedding_dimension,
                "embedding_model": "gemini-embedding-exp-03-07"
            }
        )
        logger.info(f"Created new collection: {self.collection_name}")

        # Clear keyword cache
        self._keyword_cache.clear()
        logger.info("Collection reset complete")
