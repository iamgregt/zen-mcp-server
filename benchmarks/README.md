# Semantic Search Performance Benchmarks

This directory contains performance benchmarking tools for the Zen MCP Server's semantic search implementation.

## Overview

The `semantic_search_performance.py` script provides comprehensive benchmarking of:

- **Search Performance**: Measures query latency for semantic, keyword, and hybrid search methods
- **Indexing Speed**: Tests different batch sizes to find optimal indexing throughput
- **Legacy Comparison**: Compares semantic search with legacy keyword search to quantify improvements
- **Resource Usage**: Monitors memory consumption and CPU usage during operations

## Running Benchmarks

### Quick Test
To verify the benchmark is working correctly with minimal data:
```bash
python benchmarks/test_benchmark.py
```

### Full Benchmark
To run a complete performance benchmark:
```bash
python benchmarks/semantic_search_performance.py
```

### Custom Configuration
```bash
# Specify custom data size (default: 1000 entries)
python benchmarks/semantic_search_performance.py --data-size 5000

# Specify output directory (default: benchmarks/results)
python benchmarks/semantic_search_performance.py --output-dir /path/to/results

# Specify ChromaDB persistence directory
python benchmarks/semantic_search_performance.py --persist-dir /tmp/custom_chroma
```

## Benchmark Metrics

### Search Performance
- **Semantic Search**: Pure vector similarity search using embeddings
- **Keyword Search**: Traditional keyword matching with extracted terms
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) combining semantic and keyword results

### Indexing Performance
Tests batch sizes: 1, 10, 50, 100, 500 items per batch
Measures:
- Items indexed per second
- Memory usage per item
- Optimal batch size for throughput

### Comparison Metrics
- **Speedup Factor**: How much faster/slower semantic search is vs legacy
- **Improvement Percentage**: Relative performance gain
- **Query Distribution**: Which queries benefit most from semantic search

## Output Files

The benchmark generates two files:

1. **JSON Report** (`semantic_search_benchmark_YYYYMMDD_HHMMSS.json`):
   - Complete benchmark data
   - All individual test results
   - System information
   - Detailed metadata

2. **Summary Report** (`semantic_search_benchmark_YYYYMMDD_HHMMSS_summary.txt`):
   - Human-readable summary
   - Key performance metrics
   - Recommendations based on results

## Performance Recommendations

The benchmark automatically generates recommendations based on results:

- Optimal batch sizes for indexing
- When to use hybrid vs pure semantic search
- Caching strategies for frequently searched queries
- Memory optimization suggestions
- Query routing strategies

## Interpreting Results

### Good Performance Indicators:
- Semantic search < 100ms average latency
- Hybrid search outperforms both pure methods
- Indexing throughput > 100 items/second
- Memory usage < 1MB per item

### Areas for Optimization:
- High latency (> 200ms) suggests need for caching
- Low indexing throughput indicates batch size tuning needed
- High memory usage may require model optimization
- Slower semantic search may need index optimization

## Technical Details

### Test Data Generation
- Creates realistic technical documentation entries
- Includes programming languages, frameworks, and error types
- Generates diverse metadata for filtering tests
- Creates representative search queries

### Measurement Methodology
- Uses high-resolution timing for accuracy
- Monitors system resources with psutil
- Averages results across multiple runs
- Accounts for warm-up effects

### Hybrid Search Implementation
- Uses Reciprocal Rank Fusion (RRF) with k=60
- Combines semantic and keyword rankings
- Provides balanced results leveraging both methods

## Troubleshooting

### Import Errors
Ensure you're running from the project root or the script's directory.

### Memory Issues
Reduce `--data-size` parameter for systems with limited memory.

### ChromaDB Errors
Check that the persistence directory is writable and has sufficient space.

### Missing Dependencies
Install required packages:
```bash
pip install sentence-transformers chromadb psutil numpy
```