# Vector Search Setup for Context Tool

## Overview
The Context Tool now supports optional vector search capabilities using ChromaDB and multilingual-e5-large-instruct embeddings. This feature enhances the search functionality with semantic search capabilities while maintaining backward compatibility with the existing keyword-based search.

## Configuration

### Environment Variables

- **ENABLE_VECTOR_SEARCH**: Set to `true` to enable vector search functionality (default: `false`)
- **ZEN_CONTEXT_KB_DIR**: Knowledge base directory path (default: `/data/kb`)

### Enabling Vector Search

To enable vector search, set the environment variable before starting the server:

```bash
export ENABLE_VECTOR_SEARCH=true
./run-server.sh
```

Or add it to your Docker configuration:

```yaml
environment:
  - ENABLE_VECTOR_SEARCH=true
```

## How It Works

1. **Initialization**: When `ENABLE_VECTOR_SEARCH=true`, the ContextTool initializes a ChromaProvider instance
2. **Storage**: Vector embeddings are stored in `{KB_DIR}/.chroma` directory
3. **Fallback**: If initialization fails (missing dependencies, connection errors), the tool gracefully falls back to keyword-only search
4. **Backward Compatibility**: When disabled, the tool works exactly as before with keyword-based search

## Features

- **Semantic Search**: Find related content even when exact keywords don't match
- **Multilingual Support**: Uses multilingual-e5-large-instruct model for cross-language search
- **AI-Powered Keywords**: Automatically extracts technical keywords using Gemini Flash
- **Persistent Storage**: Vector embeddings are stored persistently alongside the knowledge base

## Requirements

When vector search is enabled, the following dependencies are required:
- ChromaDB
- Sentence Transformers
- ~2.2GB disk space for the multilingual-e5-large-instruct model

These are already included in the project's requirements.txt.

## Testing

Run the vector store initialization tests:

```bash
# Integration test
python test_vector_store_init.py

# Unit tests
python -m pytest tests/test_context_vector_init.py -v
```

## Next Steps

With vector store initialization complete, the next tasks are:
- Task 2.2.2: Update save_entry method to index in vector store
- Task 2.2.3: Implement hybrid search combining semantic and keyword search