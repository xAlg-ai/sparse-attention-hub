# Test Suite for Sparse Attention Hub

This directory contains the comprehensive test suite for the sparse attention hub project.

## Directory Structure

```
tests/
├── conftest.py                    # Pytest configuration and shared fixtures
├── fixtures/                      # Shared test data and utilities
│   ├── __init__.py
│   └── sample_data.py
├── unit/                          # Fast, isolated unit tests
│   ├── test_sparse_attention/     # Tests for sparse attention module
│   ├── test_metrics/              # Tests for metrics module
│   ├── test_model_hub/            # Tests for model hub module
│   ├── test_pipeline/             # Tests for pipeline module
│   ├── test_plotting/             # Tests for plotting module
│   └── test_benchmark/            # Tests for benchmark module
├── integration/                   # Integration tests
│   └── test_end_to_end.py
└── performance/                   # Performance and stress tests
    └── test_benchmarks.py
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test types
```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests only
pytest -m integration

# Performance tests only
pytest -m performance

# Slow tests (performance + stress)
pytest -m slow
```

### Run tests for specific modules
```bash
# Test sparse attention module
pytest tests/unit/test_sparse_attention/

# Test specific file
pytest tests/unit/test_sparse_attention/test_base.py

# Test specific function
pytest tests/unit/test_sparse_attention/test_base.py::TestBaseSparseAttention::test_initialization
```

### Run with coverage
```bash
pytest --cov=sparse_attention_hub --cov-report=html
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions, classes, and methods in isolation
- **Speed**: Fast (< 1 second per test)
- **Dependencies**: Minimal external dependencies
- **Scope**: Single module or class

### Integration Tests (`tests/integration/`)
- **Purpose**: Test interactions between multiple components
- **Speed**: Medium (1-10 seconds per test)
- **Dependencies**: May require external services or complex setup
- **Scope**: Multiple modules working together

### Performance Tests (`tests/performance/`)
- **Purpose**: Test speed, memory usage, and scalability
- **Speed**: Slow (10+ seconds per test)
- **Dependencies**: May require significant computational resources
- **Scope**: End-to-end performance validation

## Writing Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Using Fixtures
Fixtures are defined in `conftest.py` and can be used across all tests:

```python
def test_my_function(sample_attention_scores, batch_size):
    # Use the fixtures
    assert sample_attention_scores.shape[0] == batch_size
```

### Test Markers
Use appropriate markers for test categorization:

```python
@pytest.mark.unit
def test_fast_function():
    pass

@pytest.mark.integration
def test_slow_integration():
    pass

@pytest.mark.slow
def test_performance():
    pass
```

## Best Practices

1. **Mirror Package Structure**: Unit tests should mirror your main package structure
2. **Use Descriptive Names**: Test names should clearly describe what they're testing
3. **One Assertion Per Test**: Each test should verify one specific behavior
4. **Use Fixtures**: Share common test data and setup code
5. **Test Edge Cases**: Include tests for boundary conditions and error cases
6. **Keep Tests Fast**: Unit tests should run quickly for rapid feedback
7. **Document Complex Tests**: Add docstrings to explain complex test logic

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

- Unit tests run on every commit
- Integration tests run on pull requests
- Performance tests run on scheduled intervals or releases 