# Gemini Embedding Migration Guide

## Overview
This guide provides step-by-step instructions for migrating the MCP Context Tool from using local SentenceTransformer embeddings to Google's Gemini-embedding-001 API.

### Current State
- **Model**: intfloat/multilingual-e5-large-instruct (SentenceTransformer)
- **Dimensions**: 1024
- **Max Tokens**: 514
- **Storage**: Local model (~1GB)
- **Location**: `/utils/chroma_provider.py`

### Target State
- **Model**: gemini-embedding-exp-03-07 (Google API)
- **Dimensions**: 3072
- **Max Tokens**: 2048
- **Storage**: API-based (no local storage)
- **Benefits**: 3x semantic resolution, 4x text capacity, latest experimental model

## Prerequisites
- Google API key with Gemini access (already configured in environment)
- Python environment with existing dependencies
- Access to modify `/providers/google.py` and `/utils/chroma_provider.py`

## Task Types for Optimization

Gemini embeddings support different task types for optimization:
- **SEMANTIC_SIMILARITY**: Optimized for assessing text similarity
- **CLASSIFICATION**: Optimized for classifying texts
- **CLUSTERING**: Optimized for clustering based on similarities
- **RETRIEVAL_DOCUMENT**: For indexing documents (use for adding entries)
- **RETRIEVAL_QUERY**: For search queries (use when searching)
- **QUESTION_ANSWERING**: For Q&A systems
- **FACT_VERIFICATION**: For fact checking
- **CODE_RETRIEVAL_QUERY**: For retrieving code blocks

For the Context Tool, we primarily use RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY.

## Implementation Steps

### Step 1: Add Embedding Support to Google Provider

First, add embedding capability to the Google provider at `/providers/google.py`:

```python
# Add to imports
import numpy as np
import time
from datetime import datetime, timedelta
from collections import deque
from google import genai
from google.genai import types

# Add to GoogleProvider class
def __init__(self, ...):
    # ... existing init code ...
    
    # Rate limiting for embeddings
    self.embedding_rate_limiter = {
        'requests': deque(maxlen=10),  # Track last 10 requests
        'daily_count': 0,
        'daily_reset': datetime.now().date()
    }

# Add this method to GoogleProvider class
def get_embedding(self, text: str, model: str = "gemini-embedding-exp-03-07", task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Get embeddings using Gemini embedding model with rate limiting.
    
    Rate limits: 10 RPM, 1000 RPD
    
    Args:
        text: Text to embed
        model: Model name (default: gemini-embedding-exp-03-07)
        task_type: Task type for embedding optimization
        
    Returns:
        np.ndarray: Embedding vector
    """
    # Check and reset daily counter if needed
    current_date = datetime.now().date()
    if current_date > self.embedding_rate_limiter['daily_reset']:
        self.embedding_rate_limiter['daily_count'] = 0
        self.embedding_rate_limiter['daily_reset'] = current_date
    
    # Check daily limit (1000 requests per day)
    if self.embedding_rate_limiter['daily_count'] >= 1000:
        raise RuntimeError("Daily embedding limit (1000) reached. Try again tomorrow.")
    
    # Implement rate limiting (10 requests per minute)
    now = datetime.now()
    requests = self.embedding_rate_limiter['requests']
    
    # Remove requests older than 1 minute
    while requests and (now - requests[0]) > timedelta(minutes=1):
        requests.popleft()
    
    # If we've made 10 requests in the last minute, wait
    if len(requests) >= 10:
        wait_time = 60 - (now - requests[0]).total_seconds()
        if wait_time > 0:
            logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time + 0.1)  # Add small buffer
    
    try:
        # Use the new client API
        client = genai.Client(api_key=self.api_key)
        
        # Get embedding
        result = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        
        # Extract embedding from result
        embedding = np.array(result.embeddings[0].values)
        
        # Record successful request
        self.embedding_rate_limiter['requests'].append(now)
        self.embedding_rate_limiter['daily_count'] += 1
        
        return embedding
        
    except Exception as e:
        if "429" in str(e):
            logger.warning("Rate limit hit despite protection. Waiting 60 seconds...")
            time.sleep(60)
            # Recursive retry
            return self.get_embedding(text, model, task_type)
        
        logger.error(f"Failed to get Gemini embedding: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")
```

### Step 2: Modify ChromaProvider

Update `/utils/chroma_provider.py` to use Gemini embeddings:

#### 2.1 Update Imports and Initialization

```python
# Replace/modify in __init__ method:
def __init__(
    self,
    persist_directory: str = None,
    collection_name: str = "zen_context",
    use_gemini: bool = True,  # New parameter
    **kwargs,
):
    """Initialize ChromaDB with optional Gemini embeddings."""
    # ... existing setup code ...
    
    self.use_gemini = use_gemini
    
    if use_gemini:
        # Use Gemini embeddings
        from providers.registry import ModelProviderRegistry, ProviderType
        self.embedding_provider = ModelProviderRegistry.get_provider(ProviderType.GOOGLE)
        if not self.embedding_provider:
            raise ValueError("Google provider not available for Gemini embeddings")
        logger.info("Using gemini-embedding-exp-03-07 (3072 dimensions)")
        self.embedding_dimension = 3072
    else:
        # Fallback to SentenceTransformer
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
```

#### 2.2 Update _compute_embedding Method

```python
def _compute_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
    """Compute embedding using Gemini or SentenceTransformer."""
    if self.use_gemini:
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
            logger.error(f"Gemini embedding failed, falling back to local model: {e}")
            # Could fall back to SentenceTransformer here if needed
            raise
    else:
        # Original SentenceTransformer logic
        if is_query:
            text = self._format_instruction_query(text)
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding
```

### Step 3: Handle Collection Migration

Add a migration utility to handle existing embeddings:

```python
# Add to chroma_provider.py or create separate migration script
def migrate_to_gemini_embeddings(self):
    """Re-embed all entries using Gemini embeddings."""
    if not self.use_gemini:
        raise ValueError("Migration requires Gemini embeddings to be enabled")
    
    # Create new collection with different name
    migration_collection_name = f"{self.collection_name}_gemini_3072"
    
    try:
        # Create new collection for migrated embeddings
        migrated_collection = self.client.create_collection(
            name=migration_collection_name,
            metadata={"hnsw:space": "cosine", "embedding_dimension": "3072"}
        )
        
        # Get all entries from current collection
        all_entries = self.collection.get()
        
        if not all_entries['ids']:
            logger.info("No entries to migrate")
            return
        
        logger.info(f"Migrating {len(all_entries['ids'])} entries to Gemini embeddings...")
        logger.info("Rate limits: 10 RPM, 1000 RPD. Migration will be throttled accordingly.")
        
        # Calculate migration time estimate
        total_entries = len(all_entries['ids'])
        minutes_needed = total_entries / 10  # 10 per minute
        logger.info(f"Estimated migration time: {minutes_needed:.1f} minutes ({minutes_needed/60:.1f} hours)")
        
        # Re-embed with rate limiting (max 10 per minute)
        batch_size = 5  # Process 5 at a time, twice per minute
        batch_count = 0
        start_time = time.time()
        
        for i in range(0, len(all_entries['ids']), batch_size):
            batch_start = time.time()
            
            batch_ids = all_entries['ids'][i:i+batch_size]
            batch_docs = all_entries['documents'][i:i+batch_size]
            batch_metadatas = all_entries['metadatas'][i:i+batch_size]
            
            # Compute new embeddings
            new_embeddings = []
            for j, doc in enumerate(batch_docs):
                try:
                    embedding = self._compute_embedding(doc)
                    new_embeddings.append(embedding.tolist())
                    
                    # Progress indicator
                    current_entry = i + j + 1
                    if current_entry % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = current_entry / elapsed * 60
                        eta = (total_entries - current_entry) / rate if rate > 0 else 0
                        logger.info(f"Progress: {current_entry}/{total_entries} entries "
                                  f"({current_entry/total_entries*100:.1f}%) - "
                                  f"Rate: {rate:.1f}/min - ETA: {eta:.1f} min")
                
                except Exception as e:
                    logger.error(f"Failed to embed document {batch_ids[j]}: {e}")
                    # You might want to handle this differently
                    raise
            
            # Add to new collection
            migrated_collection.add(
                ids=batch_ids,
                embeddings=new_embeddings,
                documents=batch_docs,
                metadatas=batch_metadatas
            )
            
            batch_count += 1
            
            # Rate limiting: ensure we don't exceed 10 requests per minute
            # We process 5 items per batch, so 2 batches = 10 items = 1 minute
            if batch_count % 2 == 0:
                batch_time = time.time() - batch_start
                sleep_time = max(60 - batch_time, 0)
                if sleep_time > 0 and i + batch_size < len(all_entries['ids']):
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
        
        # Swap collections
        old_collection_name = f"{self.collection_name}_old_1024"
        self.client.delete_collection(old_collection_name)  # Delete if exists
        self.collection.modify(name=old_collection_name)  # Rename current
        migrated_collection.modify(name=self.collection_name)  # Rename new
        self.collection = migrated_collection
        
        logger.info("Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        # Clean up if migration failed
        try:
            self.client.delete_collection(migration_collection_name)
        except:
            pass
        raise
```

### Step 4: Update Environment Configuration

Add to `.env` or environment:
```bash
# Enable Gemini embeddings
USE_GEMINI_EMBEDDINGS=true

# Ensure Google API key is set
GOOGLE_API_KEY=your_key_here
```

### Step 5: Test Implementation

Create a test script to verify the migration:

```python
# test_gemini_embeddings.py
import numpy as np
from utils.chroma_provider import ChromaProvider

def test_gemini_embeddings():
    # Initialize with Gemini
    provider = ChromaProvider(
        persist_directory="./data/test_gemini",
        collection_name="test_gemini",
        use_gemini=True
    )
    
    # Test embedding generation
    test_text = "This is a test of Gemini embeddings"
    embedding = provider._compute_embedding(test_text)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Expected: (3072,)")
    assert embedding.shape == (3072,), f"Wrong shape: {embedding.shape}"
    
    # Test query vs document embeddings
    query_embedding = provider._compute_embedding("test query", is_query=True)
    assert query_embedding.shape == (3072,)
    
    # Test adding and searching
    provider.add_entry(
        id="test_001",
        content=test_text,
        metadata={"test": True}
    )
    
    results = provider.query("test Gemini", limit=1)
    assert len(results) > 0
    assert results[0].entry.id == "test_001"
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_gemini_embeddings()
```

## Migration Execution Plan

### Phase 1: Preparation (30 min)
1. Backup current ChromaDB data: `cp -r ./data/kb ./data/kb_backup_$(date +%Y%m%d)`
2. Review and understand current embedding usage
3. Ensure Google API credentials are properly configured

### Phase 2: Implementation (2-3 hours)
1. Implement Step 1: Add embedding support to Google provider
2. Implement Step 2: Modify ChromaProvider
3. Implement Step 3: Add migration utility
4. Run tests to verify implementation

### Phase 3: Migration (1 hour)
1. Run migration script on test data first
2. Monitor API usage and rate limits
3. Execute full migration on production data
4. Verify search functionality still works

### Phase 4: Cleanup (30 min)
1. Remove SentenceTransformer dependency if no longer needed
2. Update documentation
3. Clean up old collections after verification period

## Rollback Plan

If issues occur, rollback is straightforward:
1. Set `use_gemini=False` in ChromaProvider initialization
2. Restore from backup: `mv ./data/kb_backup_[date] ./data/kb`
3. Restart services

## Rate Limits & Performance Considerations

### Gemini Embedding Rate Limits
- **10 RPM (Requests Per Minute)**: Maximum 10 embedding requests per minute
- **1,000 RPD (Requests Per Day)**: Maximum 1,000 embedding requests per day

### Implementation Strategy
1. **Built-in Rate Limiter**: The `get_embedding` method tracks requests using a deque
2. **Automatic Waiting**: If 10 requests made in last minute, waits before next request
3. **Daily Counter**: Tracks total requests per day, resets at midnight
4. **Migration Throttling**: Processes 5 entries per batch, 2 batches per minute (10 total)

### Time Estimates
- **100 entries**: ~10 minutes
- **1,000 entries**: ~100 minutes (1.7 hours) 
- **Daily limit**: Can only migrate 1,000 entries per day

### Other Considerations
- **Latency**: Each API call takes ~100-300ms (vs <10ms for local model)
- **Cost**: Check current Gemini pricing for embedding requests
- **Real-time Impact**: 10 RPM limit means max 10 searches/additions per minute
- **Fallback Strategy**: Consider keeping SentenceTransformer as fallback for high-volume periods

## Verification Checklist

- [ ] Google provider can generate embeddings
- [ ] ChromaProvider correctly uses 3072-dimensional embeddings
- [ ] Search functionality works with new embeddings
- [ ] Migration script successfully processes all entries
- [ ] Performance is acceptable for real-time usage
- [ ] Error handling works for API failures

## Common Issues and Solutions

### Issue: "Google provider not available"
**Solution**: Ensure GOOGLE_API_KEY is set and valid

### Issue: Dimension mismatch errors
**Solution**: Ensure collection is recreated with new dimensions, not just updated

### Issue: 429 Rate Limit Error
**Solution**: The code already handles this with:
- Built-in rate limiter prevents most 429 errors
- If 429 still occurs, automatic 60-second wait and retry
- For migration, ensure batch_size ≤ 5 and proper sleep timing

### Issue: Daily limit reached (1000 requests)
**Solution**: 
- Migration must be split across multiple days for >1000 entries
- Consider hybrid approach: Gemini for new content, local for bulk operations
- Track daily usage in logs

### Issue: Search quality degraded
**Solution**: Verify task_type is set correctly for queries vs documents

## Next Steps After Migration

1. Monitor search quality and gather user feedback
2. Optimize query formulation for Gemini embeddings
3. Consider implementing embedding caching for frequently accessed content
4. Document new embedding characteristics for future reference

---

This guide provides everything needed to implement the Gemini embedding migration. The modular approach allows for testing at each step and easy rollback if needed.