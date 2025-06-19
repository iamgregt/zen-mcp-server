#!/usr/bin/env python3
"""
Semantic Search Performance Benchmarks for Zen MCP Server

This script benchmarks the performance of the semantic search implementation
including:
- Search query performance (semantic, keyword, hybrid)
- Indexing speed for different batch sizes
- Comparison with legacy keyword search
- Memory usage and resource consumption
- Scalability tests with different corpus sizes
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tools.context import ContextTool, KnowledgeEntry
from utils.chroma_provider import ChromaProvider
from utils.vector_store import EntryType

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Stores results from a single benchmark run"""

    name: str
    operation: str
    duration_ms: float
    items_processed: int
    items_per_second: float
    memory_used_mb: float
    cpu_percent: float
    metadata: dict[str, Any]


@dataclass
class BenchmarkReport:
    """Complete benchmark report with all results"""

    timestamp: datetime
    system_info: dict[str, Any]
    search_results: list[BenchmarkResult]
    indexing_results: list[BenchmarkResult]
    comparison_results: list[BenchmarkResult]
    summary: dict[str, Any]


class SemanticSearchBenchmark:
    """Benchmarking suite for semantic search performance"""

    def __init__(self, persist_dir: str = "/tmp/benchmark_chroma", test_data_size: int = 1000):
        self.persist_dir = persist_dir
        self.test_data_size = test_data_size
        self.vector_store = None
        self.context_tool = None
        self.test_entries = []
        self.test_queries = []

        # Clean up any existing benchmark data
        self._cleanup()

    def _cleanup(self):
        """Clean up benchmark directories"""
        import shutil

        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        if os.path.exists("/tmp/benchmark_kb"):
            shutil.rmtree("/tmp/benchmark_kb")

    def setup(self):
        """Initialize components for benchmarking"""
        logger.info("Setting up benchmark environment...")

        # Initialize vector store
        self.vector_store = ChromaProvider(persist_directory=self.persist_dir, collection_name="benchmark_collection")

        # Initialize context tool with temporary KB directory
        os.environ["ZEN_KB_DIR"] = "/tmp/benchmark_kb"
        self.context_tool = ContextTool()

        # Generate test data
        self._generate_test_data()

        logger.info(f"Setup complete. Generated {len(self.test_entries)} test entries.")

    def _generate_test_data(self):
        """Generate synthetic test data for benchmarking"""
        logger.info(f"Generating {self.test_data_size} test entries...")

        # Common technical terms for realistic data
        languages = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C++"]
        frameworks = ["React", "Vue", "Django", "FastAPI", "Express", "Spring", "Rails"]
        concepts = [
            "API",
            "database",
            "authentication",
            "caching",
            "optimization",
            "security",
            "testing",
            "deployment",
            "monitoring",
            "scaling",
        ]
        errors = [
            "NullPointerException",
            "TypeError",
            "IndexError",
            "KeyError",
            "SyntaxError",
            "RuntimeError",
            "ValueError",
        ]

        # Generate diverse test entries
        for i in range(self.test_data_size):
            # Create realistic technical content
            lang = languages[i % len(languages)]
            framework = frameworks[i % len(frameworks)]
            concept = concepts[i % len(concepts)]
            error = errors[i % len(errors)]

            content = f"""
            Technical Issue #{i}: {lang} {framework} {concept} Implementation

            Problem Description:
            When implementing {concept} in {framework} using {lang}, we encountered a {error}.
            The issue occurs when processing large datasets with concurrent requests.

            Solution:
            We resolved this by implementing proper error handling and adding caching mechanisms.
            The solution involves using async/await patterns and connection pooling.

            Code snippet:
            async def handle_{concept.lower()}(request):
                try:
                    result = await process_data(request)
                    return cache.set(key, result)
                except {error}:
                    logger.error("Failed to process {concept}")
                    return None

            Performance improved by 85% after implementing this solution.
            Memory usage reduced from 2GB to 500MB.
            """

            metadata = {
                "language": lang,
                "framework": framework,
                "concept": concept,
                "error_type": error,
                "category": "technical",
                "tags": [lang.lower(), framework.lower(), concept.lower()],
                "importance": 0.5 + (i % 5) * 0.1,
            }

            self.test_entries.append(
                {"id": f"test_entry_{i}", "content": content, "metadata": metadata, "entry_type": EntryType.KNOWLEDGE}
            )

        # Generate test queries
        self.test_queries = [
            "Python FastAPI authentication implementation",
            "JavaScript React TypeError debugging",
            "database optimization techniques",
            "async await patterns",
            "connection pooling best practices",
            "NullPointerException Java Spring",
            "caching mechanisms performance",
            "concurrent request handling",
            "memory usage optimization",
            "error handling strategies",
        ]

    def _measure_performance(self, func, *args, **kwargs) -> tuple[Any, BenchmarkResult]:
        """Measure performance of a function call"""
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent(interval=0.1)

        # Time the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Get final resource usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent(interval=0.1)

        # Calculate metrics
        duration_ms = (end_time - start_time) * 1000
        memory_used = final_memory - initial_memory
        cpu_used = (final_cpu + initial_cpu) / 2

        return result, BenchmarkResult(
            name=func.__name__,
            operation=kwargs.get("operation", "unknown"),
            duration_ms=duration_ms,
            items_processed=kwargs.get("items_processed", 1),
            items_per_second=kwargs.get("items_processed", 1) / (duration_ms / 1000),
            memory_used_mb=memory_used,
            cpu_percent=cpu_used,
            metadata=kwargs.get("metadata", {}),
        )

    def benchmark_search_performance(self) -> list[BenchmarkResult]:
        """Benchmark different search methods"""
        logger.info("Benchmarking search performance...")
        results = []

        # First, index some test data
        logger.info("Indexing test data for search benchmarks...")
        batch_size = 100
        for i in range(0, min(1000, len(self.test_entries)), batch_size):
            batch = self.test_entries[i : i + batch_size]
            entries_to_add = [(e["id"], e["content"], e["metadata"], e["entry_type"]) for e in batch]
            self.vector_store.add_entries_batch(entries_to_add)

        # Benchmark semantic search
        logger.info("Testing semantic search...")
        for query in self.test_queries[:5]:  # Test first 5 queries
            _, result = self._measure_performance(
                self.vector_store.query,
                query_text=query,
                limit=10,
                operation="semantic_search",
                metadata={"query": query},
            )
            results.append(result)

        # Benchmark keyword search
        logger.info("Testing keyword search...")
        for query in self.test_queries[:5]:
            keywords = query.split()[:3]  # Use first 3 words as keywords
            _, result = self._measure_performance(
                self.vector_store.keyword_search,
                keywords=keywords,
                limit=10,
                operation="keyword_search",
                metadata={"keywords": keywords},
            )
            results.append(result)

        # Benchmark hybrid search through context tool
        logger.info("Testing hybrid search...")
        # First add some entries to context tool
        project_id = "benchmark_project"
        for i in range(100):
            entry = KnowledgeEntry(
                content=self.test_entries[i]["content"],
                project_id=project_id,
                metadata=self.test_entries[i]["metadata"],
            )
            self.context_tool.save_entry(entry)

        for query in self.test_queries[:5]:
            _, result = self._measure_performance(
                self.context_tool._hybrid_search,
                project_id=project_id,
                query=query,
                limit=10,
                operation="hybrid_search",
                metadata={"query": query},
            )
            results.append(result)

        return results

    def benchmark_indexing_speed(self) -> list[BenchmarkResult]:
        """Benchmark indexing performance with different batch sizes"""
        logger.info("Benchmarking indexing speed...")
        results = []

        # Test different batch sizes - including optimized sizes
        batch_sizes = [1, 10, 32, 50, 100, 200, 500]

        for batch_size in batch_sizes:
            # Clear collection for clean test
            self.vector_store.clear()

            # Test batch indexing
            total_indexed = 0
            batch_results = []

            for i in range(0, min(1000, len(self.test_entries)), batch_size):
                batch = self.test_entries[i : i + batch_size]
                entries_to_add = [(e["id"], e["content"], e["metadata"], e["entry_type"]) for e in batch]

                _, result = self._measure_performance(
                    self.vector_store.add_entries_batch,
                    entries=entries_to_add,
                    operation="batch_indexing",
                    items_processed=len(entries_to_add),
                    metadata={"batch_size": batch_size},
                )

                batch_results.append(result)
                total_indexed += len(entries_to_add)

                # Stop after indexing 500 items for each batch size
                if total_indexed >= 500:
                    break

            # Calculate average performance for this batch size
            avg_duration = np.mean([r.duration_ms for r in batch_results])
            avg_items_per_sec = np.mean([r.items_per_second for r in batch_results])
            avg_memory = np.mean([r.memory_used_mb for r in batch_results])

            results.append(
                BenchmarkResult(
                    name="batch_indexing_average",
                    operation="batch_indexing",
                    duration_ms=avg_duration,
                    items_processed=batch_size,
                    items_per_second=avg_items_per_sec,
                    memory_used_mb=avg_memory,
                    cpu_percent=np.mean([r.cpu_percent for r in batch_results]),
                    metadata={
                        "batch_size": batch_size,
                        "total_indexed": total_indexed,
                        "num_batches": len(batch_results),
                    },
                )
            )

        return results

    def benchmark_legacy_comparison(self) -> list[BenchmarkResult]:
        """Compare performance between semantic and legacy keyword search"""
        logger.info("Comparing semantic vs legacy search...")
        results = []

        # Setup test data in context tool
        project_id = "comparison_project"

        # Index entries
        for i in range(500):
            entry = KnowledgeEntry(
                content=self.test_entries[i]["content"],
                project_id=project_id,
                metadata=self.test_entries[i]["metadata"],
            )
            self.context_tool.save_entry(entry)

        # Test each query with both methods
        for query in self.test_queries:
            # Legacy keyword search
            _, legacy_result = self._measure_performance(
                self.context_tool._keyword_search_legacy,
                project_id=project_id,
                query=query,
                limit=10,
                operation="legacy_search",
                metadata={"query": query, "method": "legacy"},
            )

            # Hybrid search (semantic + keyword)
            _, hybrid_result = self._measure_performance(
                self.context_tool._hybrid_search,
                project_id=project_id,
                query=query,
                limit=10,
                operation="hybrid_search_comparison",
                metadata={"query": query, "method": "hybrid"},
            )

            # Calculate improvement
            speedup = legacy_result.duration_ms / hybrid_result.duration_ms if hybrid_result.duration_ms > 0 else 0

            comparison_result = BenchmarkResult(
                name="search_comparison",
                operation="comparison",
                duration_ms=hybrid_result.duration_ms - legacy_result.duration_ms,
                items_processed=1,
                items_per_second=speedup,  # Using this field for speedup ratio
                memory_used_mb=hybrid_result.memory_used_mb - legacy_result.memory_used_mb,
                cpu_percent=hybrid_result.cpu_percent - legacy_result.cpu_percent,
                metadata={
                    "query": query,
                    "legacy_ms": legacy_result.duration_ms,
                    "hybrid_ms": hybrid_result.duration_ms,
                    "speedup": speedup,
                    "improvement_percent": (
                        ((legacy_result.duration_ms - hybrid_result.duration_ms) / legacy_result.duration_ms * 100)
                        if legacy_result.duration_ms > 0
                        else 0
                    ),
                },
            )

            results.append(comparison_result)

        return results

    def generate_report(
        self,
        search_results: list[BenchmarkResult],
        indexing_results: list[BenchmarkResult],
        comparison_results: list[BenchmarkResult],
    ) -> BenchmarkReport:
        """Generate comprehensive benchmark report"""
        logger.info("Generating benchmark report...")

        # System information
        system_info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "model_name": "intfloat/multilingual-e5-large-instruct",
            "vector_store": "ChromaDB",
            "test_data_size": self.test_data_size,
        }

        # Calculate summaries
        search_summary = {
            "semantic_avg_ms": np.mean([r.duration_ms for r in search_results if r.operation == "semantic_search"]),
            "keyword_avg_ms": np.mean([r.duration_ms for r in search_results if r.operation == "keyword_search"]),
            "hybrid_avg_ms": np.mean([r.duration_ms for r in search_results if r.operation == "hybrid_search"]),
        }

        indexing_summary = {
            "optimal_batch_size": max(indexing_results, key=lambda r: r.items_per_second).metadata["batch_size"],
            "max_items_per_second": max(r.items_per_second for r in indexing_results),
            "avg_memory_per_item_mb": np.mean([r.memory_used_mb / r.items_processed for r in indexing_results]),
        }

        comparison_summary = {
            "avg_speedup": np.mean([r.metadata["speedup"] for r in comparison_results]),
            "avg_improvement_percent": np.mean([r.metadata["improvement_percent"] for r in comparison_results]),
            "queries_faster_with_semantic": sum(1 for r in comparison_results if r.metadata["speedup"] > 1),
            "queries_slower_with_semantic": sum(1 for r in comparison_results if r.metadata["speedup"] < 1),
        }

        summary = {
            "search": search_summary,
            "indexing": indexing_summary,
            "comparison": comparison_summary,
            "recommendations": self._generate_recommendations(search_summary, indexing_summary, comparison_summary),
        }

        return BenchmarkReport(
            timestamp=datetime.now(),
            system_info=system_info,
            search_results=search_results,
            indexing_results=indexing_results,
            comparison_results=comparison_results,
            summary=summary,
        )

    def _generate_recommendations(
        self, search_summary: dict, indexing_summary: dict, comparison_summary: dict
    ) -> list[str]:
        """Generate performance recommendations based on results"""
        recommendations = []

        # Search performance recommendations
        if search_summary["hybrid_avg_ms"] < search_summary["semantic_avg_ms"]:
            recommendations.append("Hybrid search outperforms pure semantic search - continue using RRF fusion")

        if search_summary["semantic_avg_ms"] > 100:
            recommendations.append("Consider caching frequently searched queries to improve response time")

        # Indexing recommendations
        if indexing_summary["optimal_batch_size"] >= 100:
            recommendations.append(
                f"Use batch size of {indexing_summary['optimal_batch_size']} for optimal indexing performance"
            )

        if indexing_summary["avg_memory_per_item_mb"] > 1:
            recommendations.append("Consider optimizing embedding model or using dimensionality reduction")

        # Comparison recommendations
        if comparison_summary["avg_speedup"] < 1:
            recommendations.append(
                "Semantic search is slower than legacy - consider optimizing vector store configuration"
            )
        elif comparison_summary["avg_speedup"] > 2:
            recommendations.append(
                "Semantic search shows significant performance gains - consider migrating all search operations"
            )

        if comparison_summary["queries_slower_with_semantic"] > comparison_summary["queries_faster_with_semantic"]:
            recommendations.append("Many queries perform better with legacy search - consider query-specific routing")

        return recommendations

    def save_report(self, report: BenchmarkReport, output_path: str):
        """Save benchmark report to file"""
        logger.info(f"Saving report to {output_path}...")

        # Convert report to dict
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "system_info": report.system_info,
            "search_results": [
                {
                    "name": r.name,
                    "operation": r.operation,
                    "duration_ms": r.duration_ms,
                    "items_per_second": r.items_per_second,
                    "memory_used_mb": r.memory_used_mb,
                    "cpu_percent": r.cpu_percent,
                    "metadata": r.metadata,
                }
                for r in report.search_results
            ],
            "indexing_results": [
                {
                    "name": r.name,
                    "operation": r.operation,
                    "duration_ms": r.duration_ms,
                    "items_processed": r.items_processed,
                    "items_per_second": r.items_per_second,
                    "memory_used_mb": r.memory_used_mb,
                    "cpu_percent": r.cpu_percent,
                    "metadata": r.metadata,
                }
                for r in report.indexing_results
            ],
            "comparison_results": [
                {
                    "name": r.name,
                    "operation": r.operation,
                    "duration_ms": r.duration_ms,
                    "speedup": r.items_per_second,  # We used this field for speedup
                    "memory_diff_mb": r.memory_used_mb,
                    "cpu_diff_percent": r.cpu_percent,
                    "metadata": r.metadata,
                }
                for r in report.comparison_results
            ],
            "summary": report.summary,
        }

        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        # Also create a human-readable summary
        summary_path = output_path.replace(".json", "_summary.txt")
        with open(summary_path, "w") as f:
            f.write("SEMANTIC SEARCH PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Data Size: {report.system_info['test_data_size']} entries\n")
            f.write(f"Model: {report.system_info['model_name']}\n\n")

            f.write("SEARCH PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Semantic Search Avg: {report.summary['search']['semantic_avg_ms']:.2f}ms\n")
            f.write(f"Keyword Search Avg: {report.summary['search']['keyword_avg_ms']:.2f}ms\n")
            f.write(f"Hybrid Search Avg: {report.summary['search']['hybrid_avg_ms']:.2f}ms\n\n")

            f.write("INDEXING PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Optimal Batch Size: {report.summary['indexing']['optimal_batch_size']}\n")
            f.write(f"Max Throughput: {report.summary['indexing']['max_items_per_second']:.2f} items/second\n")
            f.write(f"Avg Memory per Item: {report.summary['indexing']['avg_memory_per_item_mb']:.2f}MB\n\n")

            f.write("COMPARISON WITH LEGACY SEARCH\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Speedup: {report.summary['comparison']['avg_speedup']:.2f}x\n")
            f.write(f"Average Improvement: {report.summary['comparison']['avg_improvement_percent']:.1f}%\n")
            f.write(f"Queries Faster: {report.summary['comparison']['queries_faster_with_semantic']}\n")
            f.write(f"Queries Slower: {report.summary['comparison']['queries_slower_with_semantic']}\n\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(report.summary["recommendations"], 1):
                f.write(f"{i}. {rec}\n")

        logger.info(f"Report saved to {output_path} and {summary_path}")

    def run(self, output_dir: str = "benchmarks/results"):
        """Run complete benchmark suite"""
        try:
            # Setup
            self.setup()

            # Run benchmarks
            search_results = self.benchmark_search_performance()
            indexing_results = self.benchmark_indexing_speed()
            comparison_results = self.benchmark_legacy_comparison()

            # Generate report
            report = self.generate_report(search_results, indexing_results, comparison_results)

            # Save report
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"semantic_search_benchmark_{timestamp}.json")
            self.save_report(report, output_path)

            # Print summary to console
            print("\n" + "=" * 50)
            print("BENCHMARK COMPLETE")
            print("=" * 50)
            print("\nSearch Performance:")
            print(f"  - Semantic: {report.summary['search']['semantic_avg_ms']:.2f}ms avg")
            print(f"  - Keyword: {report.summary['search']['keyword_avg_ms']:.2f}ms avg")
            print(f"  - Hybrid: {report.summary['search']['hybrid_avg_ms']:.2f}ms avg")
            print("\nIndexing Performance:")
            print(f"  - Optimal batch size: {report.summary['indexing']['optimal_batch_size']}")
            print(f"  - Max throughput: {report.summary['indexing']['max_items_per_second']:.2f} items/sec")
            print("\nVs Legacy Search:")
            print(f"  - Average speedup: {report.summary['comparison']['avg_speedup']:.2f}x")
            print(f"  - Average improvement: {report.summary['comparison']['avg_improvement_percent']:.1f}%")
            print(f"\nFull report saved to: {output_path}")

        finally:
            # Cleanup
            self._cleanup()


def main():
    """Main entry point for benchmark script"""
    parser = argparse.ArgumentParser(description="Benchmark semantic search performance")
    parser.add_argument(
        "--data-size", type=int, default=1000, help="Number of test entries to generate (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to save benchmark results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="/tmp/benchmark_chroma",
        help="Directory for ChromaDB persistence during benchmark",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = SemanticSearchBenchmark(persist_dir=args.persist_dir, test_data_size=args.data_size)
    benchmark.run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
