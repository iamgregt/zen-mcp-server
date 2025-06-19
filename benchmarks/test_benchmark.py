#!/usr/bin/env python3
"""
Quick test of the semantic search benchmark with minimal data
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.semantic_search_performance import SemanticSearchBenchmark


def test_benchmark():
    """Run a quick benchmark test with minimal data"""
    print("Running quick benchmark test...")

    # Create benchmark with small test size
    benchmark = SemanticSearchBenchmark(
        persist_dir="/tmp/test_benchmark_chroma", test_data_size=50  # Small size for quick test
    )

    try:
        # Setup
        benchmark.setup()
        print("✓ Setup complete")

        # Test search performance
        search_results = benchmark.benchmark_search_performance()
        print(f"✓ Search benchmarks complete: {len(search_results)} results")

        # Test indexing speed
        indexing_results = benchmark.benchmark_indexing_speed()
        print(f"✓ Indexing benchmarks complete: {len(indexing_results)} results")

        # Test comparison
        comparison_results = benchmark.benchmark_legacy_comparison()
        print(f"✓ Comparison benchmarks complete: {len(comparison_results)} results")

        # Generate report
        report = benchmark.generate_report(search_results, indexing_results, comparison_results)
        print("✓ Report generated")

        # Save report
        benchmark.save_report(report, "/tmp/test_benchmark_report.json")
        print("✓ Report saved to /tmp/test_benchmark_report.json")

        print("\nTest successful! The benchmark script is working correctly.")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        benchmark._cleanup()


if __name__ == "__main__":
    test_benchmark()
