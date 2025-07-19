"""Sample BenchmarkConfigs for all benchmarks using 1 subset each.

This file contains example configurations for all implemented benchmarks,
each using a single subset to demonstrate the configuration format.
"""

from benchmark.executor_config import BenchmarkConfig

# Sample BenchmarkConfigs for all benchmarks using 1 subset each

# 1. InfiniteBench - using passkey task
infinite_bench_config = BenchmarkConfig(
    benchmark_name="infinite_bench",
    subsets=["passkey"]
)

# 2. Ruler - using 4096 context length
ruler_config = BenchmarkConfig(
    benchmark_name="ruler",
    subsets=["4096"]
)

# 3. Loogle - using shortdep_qa task
loogle_config = BenchmarkConfig(
    benchmark_name="loogle",
    subsets=["shortdep_qa"]
)

# 4. ZeroScrolls - using gov_report task
zero_scrolls_config = BenchmarkConfig(
    benchmark_name="zero_scrolls",
    subsets=["gov_report"]
)

# 5. LongBenchv2 - using 0shot task
longbenchv2_config = BenchmarkConfig(
    benchmark_name="longbenchv2",
    subsets=["0shot"]
)

# 6. AIME2024 - using single task
aime2024_config = BenchmarkConfig(
    benchmark_name="aime2024",
    subsets=["aime2024"]
)

# 7. AIME2025 - using single task
aime2025_config = BenchmarkConfig(
    benchmark_name="aime2025",
    subsets=["aime2025"]
)

# 8. LongBench (existing) - using narrativeqa task
longbench_config = BenchmarkConfig(
    benchmark_name="longbench",
    subsets=["narrativeqa"]
)

# 9. Mock Benchmark (existing) - using single task
mock_benchmark_config = BenchmarkConfig(
    benchmark_name="mock_benchmark",
    subsets=["mock_task"]
)

# List of all sample configurations
all_sample_configs = [
    infinite_bench_config,
    ruler_config,
    loogle_config,
    zero_scrolls_config,
    longbenchv2_config,
    aime2024_config,
    aime2025_config,
    longbench_config,
    mock_benchmark_config
]

# Example usage with executor
def get_sample_configs():
    """Return all sample benchmark configurations.
    
    Returns:
        List of BenchmarkConfig instances, one for each benchmark with a single subset.
    """
    return all_sample_configs

# Example of how to use these configurations
if __name__ == "__main__":
    print("Sample BenchmarkConfigs for all benchmarks:")
    print("=" * 50)
    
    for config in all_sample_configs:
        print(f"Benchmark: {config.benchmark_name}")
        print(f"Subsets: {config.subsets}")
        print("-" * 30)
    
    print(f"\nTotal configurations: {len(all_sample_configs)}")
    
    # Example of validation
    from benchmark.benchmark_registry import create_benchmark_instance
    
    print("\nValidating configurations...")
    for config in all_sample_configs:
        try:
            benchmark = create_benchmark_instance(config.benchmark_name)
            config.validate_with_benchmark_instance(benchmark)
            print(f"✅ {config.benchmark_name}: Valid")
        except Exception as e:
            print(f"❌ {config.benchmark_name}: {str(e)}") 