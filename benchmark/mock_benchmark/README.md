# Mock Benchmark

A simple benchmark for testing and demonstration purposes, designed to validate the benchmark framework with minimal data.

## Overview

The Mock Benchmark contains 5 carefully crafted samples with:
- **Short contexts**: All contexts are under 250 words (127-136 words each)
- **Simple questions**: Basic reading comprehension questions
- **Context sharing**: Samples 1&2 share one context, samples 4&5 share another context
- **Clear answers**: Straightforward, factual answers for easy evaluation

## Data Structure

The benchmark includes 3 unique contexts covering different domains:

1. **Science Context** (132 words): About photosynthesis
   - Sample 1: "What are the two main stages of photosynthesis?"
   - Sample 2: "What gas is produced as a byproduct of photosynthesis?"

2. **History Context** (136 words): About the Renaissance
   - Sample 3: "In which century did the Renaissance begin?"

3. **Geography Context** (127 words): About the Amazon rainforest
   - Sample 4: "Why is the Amazon rainforest called the 'lungs of the Earth'?"
   - Sample 5: "Which river flows through the Amazon rainforest?"

## Usage

```python
from benchmark.mock_benchmark import MockBenchmark

# Create benchmark instance
mock_benchmark = MockBenchmark()

# Run with a model adapter
results = mock_benchmark.run_benchmark(adapter, result_dir="/path/to/results")

# View results
print(f"Accuracy: {results['accuracy']}")
print(f"Correct predictions: {results['correct_predictions']}/{results['total_samples']}")
```

## Evaluation

The benchmark uses simple exact match and substring matching for evaluation:
- Exact match with ground truth answers
- Case-insensitive substring matching
- Multiple acceptable answers per question

## Purpose

This mock benchmark is ideal for:
- Testing the benchmark framework
- Validating model adapters
- Demonstrating benchmark usage
- Quick evaluation during development 