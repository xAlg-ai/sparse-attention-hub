# Benchmark Implementation Plan

## Overview
This document outlines the detailed implementation plan for the Benchmark functionality in the sparse-attention-hub project. The design follows an object-oriented approach with a base abstract class and concrete implementations for specific benchmarks.

## Class Hierarchy

```
Benchmark (Abstract Base Class)
│
├── LongBench
├── ZeroScrolls  
├── Ruler
├── InfiniteBench
├── LongBenchV2
├── Loogle
├── AIME2024
└── AIME2025
```

## Base Class: `Benchmark`

### Location
- File: `sparse-attention-hub/benchmark/base.py`

### Class Attributes (to be overridden in subclasses)
```python
all_datasets: List[str]              # All available datasets for this benchmark
benchmark_name: str                  # Name identifier for the benchmark
huggingface_dataset_id: str         # HuggingFace dataset identifier
```

### Instance Attributes
```python
subsets_to_run: List[str]           # Datasets to run (subset of all_datasets)
```

### Constructor
```python
def __init__(self, subsets_to_run: Optional[List[str]] = None)
```
- **Parameters:**
  - `subsets_to_run`: Optional list of dataset names to run. If None, uses all_datasets
- **Behavior:**
  - Validates that all subsets_to_run exist in all_datasets
  - Sets default to all_datasets if not provided

### Abstract Methods (must be implemented by subclasses)

#### `post_run_evaluate`
```python
@abstractmethod
def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]
```
- **Purpose:** Compute evaluation metrics on benchmark results
- **Parameters:**
  - `results_df`: DataFrame containing input data and model outputs
- **Returns:** Dictionary containing computed metrics
- **Implementation:** Specific to each benchmark's evaluation methodology

### Concrete Methods

#### `_load_datasets`
```python
def _load_datasets(self) -> pd.DataFrame
```
- **Purpose:** Load and combine all specified datasets into a single DataFrame
- **Returns:** Combined pandas DataFrame with all samples
- **Implementation:**
  - Downloads/loads datasets from HuggingFace using `huggingface_dataset_id`
  - Filters to only include datasets in `subsets_to_run`
  - Combines into single DataFrame with consistent schema
  - Adds metadata columns (e.g., dataset_name, sample_id)

#### `_process_all_requests`
```python
def _process_all_requests(self, 
                         adapter: ModelHubAdapterInterface, 
                         generation_kwargs: Dict[str, Any],
                         dataset_df: pd.DataFrame) -> pd.DataFrame
```
- **Purpose:** Process all samples through the model adapter using context grouping for efficiency
- **Parameters:**
  - `adapter`: Model adapter implementing ModelHubAdapterInterface
  - `generation_kwargs`: Parameters for text generation (passed to adapter)
  - `dataset_df`: DataFrame containing the benchmark dataset
- **Returns:** DataFrame with added 'predicted_answer' column
- **Implementation:**
  - Group dataset by context (following HashAttention approach)
  - For each unique context:
    - Collect all questions for that context
    - Extract answer_prefix and max_new_tokens from first row in group
    - **Request Population Strategy** (simplified approach):
      ```python
      # HashAttention pipeline handles: context, questions, answer_prefix, max_new_tokens
      # Current Request class only supports: context, questions
      
      # SIMPLIFIED APPROACH: Use basic Request class without modifications
      request = Request(context=context, questions=questions_list)
      
      # NOTE: answer_prefix is currently ignored
      # Future enhancement needed: Determine how to handle answer_prefix
      # Options could include:
      # - Adapter interface enhancement to support answer_prefix
      # - Post-processing to add answer_prefix to generated responses  
      # - Pre-processing within adapter implementation
      ```
    - Call adapter.process_request(request)
    - Assign responses back to corresponding DataFrame rows
  - Handles error cases gracefully
  - Includes progress tracking with tqdm
  - Validates that all questions get responses

#### `run_benchmark`
```python
def run_benchmark(self, 
                 adapter: ModelHubAdapterInterface, 
                 generation_kwargs: Dict[str, Any],
                 result_dir: str) -> Dict[str, Any]
```
- **Purpose:** Main orchestration method for running complete benchmark
- **Parameters:**
  - `adapter`: Model adapter implementing ModelHubAdapterInterface
  - `generation_kwargs`: Generation parameters (passed through to adapter)
  - `result_dir`: Directory to save results
- **Returns:** Dictionary containing evaluation results and metadata
- **Implementation:**
  1. Call `_load_datasets()` to get input data
  2. Validate dataset size (<10K rows, warn if larger)
  3. Call `_process_all_requests()` to get model outputs (adds 'predicted_answer' column)
     - Uses simplified Request(context, questions) interface
     - Ignores answer_prefix for now (future enhancement)
  4. Call `post_run_evaluate()` to compute metrics
  5. Save results DataFrame to `result_dir/raw_results.csv`
  6. Save metrics to `result_dir/metrics.json`
  7. Return metrics dictionary

### Utility Methods

#### `get_available_datasets`
```python
def get_available_datasets(self) -> List[str]
```
- **Purpose:** Return list of all available datasets for this benchmark
- **Returns:** Copy of all_datasets

#### `validate_subsets`
```python
def _validate_subsets(self, subsets: List[str]) -> None
```
- **Purpose:** Validate that requested subsets exist in all_datasets
- **Raises:** ValueError if invalid subset specified

## Concrete Class: `LongBench`

### Location
- File: `sparse-attention-hub/benchmark/longbench/longbench.py`

### Class Attributes (overrides)
```python
all_datasets: List[str] = [
    # Standard LongBench datasets
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
    "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", 
    "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", 
    "lsht", "passage_count", "passage_retrieval_en", 
    "passage_retrieval_zh", "lcc", "repobench-p",
    # LongBench-E (extended) datasets
    "qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e", 
    "gov_report_e", "multi_news_e", "trec_e", "triviaqa_e", 
    "samsum_e", "passage_count_e", "passage_retrieval_en_e", 
    "lcc_e", "repobench-p_e"
]
benchmark_name: str = "longbench"
huggingface_dataset_id: str = "Xnhyacinth/LongBench"  # aditya(use /data/apdesai/code/HashAttention/evaluation/evaluate.py to know the dataset_id of most datasets except AIME2024, AIME2025)
```

### Dataset Structure
Each LongBench dataset sample contains the following columns:
- `context`: Pre-formatted context with task-specific template
- `question`: The input question/prompt for the task
- `answer_prefix`: Expected prefix for the model response
- `answers`: List of ground truth answers
- `all_classes`: Available classes for classification tasks (if applicable)
- `task`: Dataset/task name identifier
- `max_new_tokens`: Recommended maximum tokens for generation
- `length`: Input length in tokens (for extended datasets)

### Implementation of Abstract Methods

#### `post_run_evaluate`
```python
def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]
```
- **Implementation:**
  - Import `calculate_metrics` and `calculate_metrics_e` from `calculate_metrics.py`
  - Group results by dataset/task name
  - For standard datasets: use `calculate_metrics(subset_df)`
  - For extended datasets (_e suffix): use `calculate_metrics_e(subset_df)`
  - Aggregate metrics across all datasets
  - Return dictionary with:
    - Individual dataset scores
    - Overall average score
    - Length-based metrics for extended datasets (0-4k, 4-8k, 8k+ tokens)
    - Task type breakdown (QA, summarization, classification, etc.)

## Implementation Strategy

### Phase 1: Base Class Implementation
1. Create abstract base class in `benchmark/base.py`
2. Implement all concrete methods
3. Define clear interfaces for abstract methods
4. Add comprehensive type hints and docstrings

### Phase 2: LongBench Implementation
1. Analyze existing `create_huggingface_dataset.py` to understand dataset structure
2. Populate `all_datasets` list
3. Implement `post_run_evaluate` using existing `calculate_metrics.py`
4. Test with sample data

### Phase 3: Integration Points
1. Ensure compatibility with existing adapter interfaces
2. Define standard format for generation_kwargs
3. Establish result storage conventions
4. Add error handling and logging

### Phase 4: Testing Strategy
1. Unit tests for each method
2. Integration tests with mock adapters
3. End-to-end tests with actual models (subset)
4. Performance benchmarking

## Key Design Decisions

### Dataset Loading
- Use pandas DataFrame as standard data format
- Lazy loading only when needed
- Cache datasets to avoid repeated downloads

### Error Handling
- Graceful degradation for failed samples
- Comprehensive logging for debugging
- Clear error messages for configuration issues

### Extensibility
- Abstract base class allows easy addition of new benchmarks
- Standardized interfaces for adapters
- Flexible result storage format

### Performance Considerations
- Batch processing where possible
- Progress tracking for long-running benchmarks
- Memory-efficient data handling for large datasets

## Dependencies

### Required Packages
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
from pathlib import Path
```

### Internal Dependencies
```python
from .longbench.calculate_metrics import calculate_metrics  # For LongBench
from sparse_attention_hub.adapters.base import Request, RequestResponse  # For adapter interface
# Additional imports for other benchmark implementations
```

## Dataset ID Mapping
Based on HashAttention evaluation, the correct dataset IDs are:
```python
DATASET_DICT = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler", 
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "Xnhyacinth/LongBench-v2",
    "aime2025": "xAlg-AI/att-hub-aime2025",  # Now available as processed dataset
    "aime2024": "xAlg-AI/att-hub-aime2024",  # Now available as processed dataset
}
```

## Future Extensions

### Additional Benchmarks
Each new benchmark will follow the same pattern:
1. Create subdirectory under `benchmark/`
2. Implement concrete class extending `Benchmark`
3. Override class attributes and `post_run_evaluate`
4. Add any benchmark-specific utilities

### Adapter Interface
Define standard adapter interface that all model adapters must implement:
```python
class BaseAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
```

### Result Analysis Tools
- Cross-benchmark comparison utilities
- Visualization tools for results
- Statistical significance testing

## LongBench Evaluation Metrics

The LongBench implementation uses different metrics for different task types:

### Metric Types
- **QA F1 Score**: For most English QA tasks (narrativeqa, qasper, hotpotqa, etc.)
- **QA F1 Score (Chinese)**: For Chinese QA tasks (multifieldqa_zh) using jieba tokenization
- **ROUGE Score**: For summarization tasks (gov_report, qmsum, multi_news, samsum)
- **ROUGE Score (Chinese)**: For Chinese summarization tasks (dureader, vcsum)
- **Classification Score**: For classification tasks (trec, lsht)
- **Retrieval Score**: For passage retrieval tasks (passage_retrieval_en/zh)
- **Count Score**: For counting tasks (passage_count)
- **Code Similarity Score**: For code completion tasks (lcc, repobench-p)

### Task Categories
1. **Question Answering**: narrativeqa, qasper, multifieldqa_en/zh, hotpotqa, 2wikimqa, musique, triviaqa
2. **Summarization**: gov_report, qmsum, multi_news, vcsum, samsum
3. **Classification**: trec, lsht
4. **Retrieval**: passage_retrieval_en, passage_retrieval_zh
5. **Counting**: passage_count
6. **Code Completion**: lcc, repobench-p
7. **Reading Comprehension**: dureader

## Resolved Design Questions

### 1. Dataset Size and Processing Strategy
**Decision**: Support datasets < 10K rows. For larger datasets, raise a warning: "Repository not expected to handle large datasets. If needed, request this feature."
- Load entire dataset into memory at once
- Process all samples and dump results at the end
- No streaming or incremental processing needed

### 2. Prompt Format Handling - Request Class Integration
**Challenge**: Adapter's Request class expects `context` and `questions`, but benchmark datasets have various formats.

**Solution Options Analyzed** (from HashAttention evaluation):

**Option A: Context Grouping Approach** (Recommended)
```python
def _process_all_requests(self, adapter, generation_kwargs):
    # Group by context like HashAttention does
    df_context = self.dataset_df.groupby("context")
    for context, df_group in df_context:
        questions = df_group["question"].to_list()
        request = Request(context=context, questions=questions)
        response = adapter.process_request(request) 
        # Assign responses back to DataFrame
```
aditya(generation_kwargs need to go to process_requestion. However, currently adapter does not take this argument. we will need to make sure adapter takes this and passes it to generate correctly. Also note that process_request  in adapterHF needs to fall back to model.generate api since it implements all the fancy generation. Make note of this for fixing after this plan is implemented.)


**Option B: Individual Request Approach**
```python
def _process_all_requests(self, adapter, generation_kwargs):
    for idx, row in self.dataset_df.iterrows():
        request = Request(context=row["context"], questions=[row["question"]])
        response = adapter.process_request(request)
        self.dataset_df.loc[idx, "predicted_answer"] = response.responses[0]
```

**Option C: Batch Processing with Context Reuse**
```python
def _process_all_requests(self, adapter, generation_kwargs):
    # Process contexts in batches, reusing KV cache when possible
    for context in unique_contexts:
        context_questions = get_questions_for_context(context)
        request = Request(context=context, questions=context_questions)
        # Process entire context batch at once
```

**Recommended**: Option A (Context Grouping) - matches HashAttention approach and maximizes efficiency

### 3. Caching Strategy
**Decision**: No caching needed
- Datasets are small enough to load entirely into memory
- Process once and save results
- Simple and sufficient for current requirements

### 4. Adapter Interface
**Decision**: Adapters passed as instances
- Follow existing pattern: `adapter.process_request(request: Request) -> RequestResponse`
- Request contains `context: str` and `questions: Union[str, List[str]]`
- Response contains `responses: Union[str, List[str]]`

### 5. Generation Parameters
**Decision**: Pass-through only
- Benchmark class doesn't handle generation parameters
- They're only inputs to `run_benchmark()` function
- Passed directly to adapter without modification

### 6. Custom Evaluation Metrics
**Decision**: Not supported
- Use only built-in benchmark evaluation methods
- Keeps implementation focused and simple

### 7. Special Preprocessing Requirements
**Decision**: No special preprocessing needed
- **AIME2024**: Now available directly from `xAlg-AI/att-hub-aime2024` 
- **AIME2025**: Now available directly from `xAlg-AI/att-hub-aime2025`
- **All other datasets**: Use standard HuggingFace dataset loading

**Implementation Strategy**:
```python
def _load_datasets(self):
    # Standard loading for all benchmarks - no special preprocessing needed
    return load_dataset(self.huggingface_dataset_id, split="test").to_pandas()
```

## Updated Implementation Strategy

### Phase 1: Base Class Implementation ✅ COMPLETED
1. ✅ Create abstract base class in `benchmark/base.py`
2. ✅ Implement dataset size validation (warn if >10K rows)
3. ✅ Implement context grouping approach in `_process_all_requests()`
4. ✅ Add comprehensive type hints and Google-style docstrings
5. ✅ Standard dataset loading for all benchmarks (no special preprocessing needed)
6. ✅ Add TODO notes for future generation_kwargs and answer_prefix support

### Phase 2: LongBench Implementation ✅ COMPLETED
1. ✅ Create `benchmark/longbench/longbench.py`
2. ✅ Use dataset ID: "Xnhyacinth/LongBench"
3. ✅ Implement all 35 datasets (22 standard + 13 extended)
4. ✅ Implement `post_run_evaluate()` using existing `calculate_metrics.py`
5. ✅ Handle both standard and extended (_e) datasets
6. ✅ Export LongBench class in package __init__.py files

### Phase 3: Integration & Testing ✅ COMPLETED
1. ✅ Test with existing adapters in `sparse_attention_hub/adapters/`
2. ✅ Validate Request/RequestResponse interface compatibility
3. ✅ Test context grouping efficiency vs individual requests
4. ✅ Add error handling and logging
5. ✅ Create comprehensive unit tests in `tests/unit/benchmark/`
6. ✅ Create integration tests in `tests/integration/`

### Phase 4: Extension to Other Benchmarks
1. Follow same pattern for other benchmarks (Ruler, ZeroScrolls, etc.)
2. Use dataset IDs from DATASET_DICT mapping
3. Implement benchmark-specific evaluation methods

## Key Implementation Details

### Context Grouping Implementation
```python
def _process_all_requests(self, adapter, generation_kwargs, dataset_df):
    from tqdm import tqdm
    
    # Initialize predicted_answer column
    dataset_df["predicted_answer"] = None
    
    # Group by context for efficiency
    df_context = dataset_df.groupby("context")
    
    for context, df_group in tqdm(df_context, total=dataset_df["context"].nunique()):
        questions = df_group["question"].to_list()
        
        # Create request using current adapter interface (simplified)
        request = Request(context=context, questions=questions)
        
        # Process through adapter
        # NOTE: generation_kwargs currently not passed to adapter.process_request()
        # NOTE: answer_prefix currently ignored - future enhancement needed
        # This will be fixed in post-implementation adapter updates
        response = adapter.process_request(request)
        
        # Assign responses back to DataFrame
        dataset_df.loc[df_group.index, "predicted_answer"] = response.responses
        
        # Memory cleanup for large contexts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return dataset_df
```

### Dataset Size Validation
```python
def _validate_dataset_size(self, df):
    if len(df) > 10000:
        import warnings
        warnings.warn(
            f"Dataset has {len(df)} rows (>10K). Repository not expected to handle "
            "large datasets. If needed, request this feature.",
            UserWarning
        )
```

## Known Issues and Post-Implementation Fixes #aditya(Do not implement in this plan. Keep for strictly future work)

### ⚠️ Adapter Interface Limitations

#### 1. Missing generation_kwargs Support
**Issue**: Current adapter interface doesn't support generation_kwargs
- `adapter.process_request(request)` doesn't accept generation parameters
- `generation_kwargs` need to be passed to underlying `model.generate()` API
- HuggingFace adapter should use `model.generate()` for full generation feature support

#### 2. Missing answer_prefix Support
**Issue**: Current Request class only supports context and questions, but benchmarks use answer_prefix
- HashAttention pipeline uses: `context`, `questions`, `answer_prefix` (plus generation parameters)
- Current Request class only has: `context`, `questions`
- Missing: `answer_prefix` support
- Note: `max_new_tokens` and similar parameters belong in `generation_kwargs`, not the Request class

**Required Fixes** (after base implementation):

1. **Update Adapter Interface** to support generation_kwargs:
```python
# Current interface
def process_request(self, request: Request) -> RequestResponse:

# Needed interface  
def process_request(self, request: Request, generation_kwargs: Dict[str, Any] = None) -> RequestResponse:
```

2. **Add answer_prefix Support** (multiple possible approaches):
   - **Option A**: Add answer_prefix field to Request class
   - **Option B**: Handle answer_prefix within adapter implementation  
   - **Option C**: Post-process responses to add answer_prefix
   - **Option D**: Let evaluation handle answer_prefix formatting

3. **Update HuggingFace Adapter**:
   - Accept `generation_kwargs` parameter
   - Pass `generation_kwargs` to `model.generate()` API
   - Support all generation features (temperature, top_p, max_tokens, etc.)
   - Implement chosen answer_prefix handling approach

4. **Update Benchmark Implementation**:
```python
# After adapter interface is fixed
request = Request(context=context, questions=questions)
response = adapter.process_request(request, generation_kwargs)
```

**Current Approach**: For initial implementation, ignore answer_prefix:
```python
# Simple implementation - answer_prefix handling deferred
request = Request(context=context, questions=questions)
response = adapter.process_request(request)
```

**Impact**: For now, benchmark will work with simplified Request interface and default generation settings. Full parameter control and answer_prefix support will be available after adapter interface updates.

## Next Steps - Ready for Implementation

1. ✅ Implementation plan finalized
2. ✅ All design questions resolved  
3. ✅ Adapter interface compatibility confirmed
4. ✅ Dataset IDs and preprocessing requirements identified
5. ✅ AIME datasets now available as processed datasets
6. ✅ Simplified approach using basic Request(context, questions) interface
7. ⚠️ generation_kwargs and answer_prefix limitations documented for post-implementation fix
8. **Next**: Begin coding the base class implementation

---

# Implementation TODO List

## Phase 1: Base Class Implementation

### 1.1 Create File Structure
- [ ] Create `benchmark/base.py`
- [ ] Ensure proper Python package structure with `__init__.py`

### 1.2 Implement Abstract Base Class `Benchmark`
- [ ] Add imports:
  ```python
  from abc import ABC, abstractmethod
  from typing import List, Dict, Any, Optional
  import pandas as pd
  import json
  import os
  import warnings
  from pathlib import Path
  from datasets import load_dataset
  from tqdm import tqdm
  from sparse_attention_hub.adapters.base import Request, RequestResponse, ModelHubAdapterInterface
  ```

- [ ] Define class attributes (to be overridden by subclasses):
  ```python
  all_datasets: List[str]
  benchmark_name: str  
  huggingface_dataset_id: str
  ```

- [ ] Implement `__init__(self, subsets_to_run: Optional[List[str]] = None)`
  - [ ] Validate subsets_to_run against all_datasets
  - [ ] Set default to all_datasets if None provided
  - [ ] Add proper error handling

- [ ] Implement `_load_datasets(self) -> pd.DataFrame`
  - [ ] Load dataset from HuggingFace using huggingface_dataset_id
  - [ ] Filter to subsets_to_run
  - [ ] Return combined DataFrame

- [ ] Implement `_validate_dataset_size(self, df: pd.DataFrame) -> None`
  - [ ] Check if dataset has >10K rows
  - [ ] Issue warning if too large
  - [ ] Include helpful error message

- [ ] Implement `_process_all_requests(self, adapter, generation_kwargs, dataset_df) -> pd.DataFrame`
  - [ ] Initialize predicted_answer column
  - [ ] Group by context using df.groupby("context")
  - [ ] Use tqdm for progress tracking
  - [ ] Create Request(context=context, questions=questions) for each group
  - [ ] Call adapter.process_request(request)
  - [ ] Assign responses back to DataFrame
  - [ ] Add memory cleanup with torch.cuda.empty_cache()
  - [ ] Return modified DataFrame

- [ ] Implement `run_benchmark(self, adapter, generation_kwargs, result_dir) -> Dict[str, Any]`
  - [ ] Call _load_datasets()
  - [ ] Call _validate_dataset_size()
  - [ ] Call _process_all_requests()
  - [ ] Call post_run_evaluate()
  - [ ] Save results to CSV and JSON
  - [ ] Return metrics dictionary

- [ ] Implement utility methods:
  - [ ] `get_available_datasets(self) -> List[str]`
  - [ ] `_validate_subsets(self, subsets: List[str]) -> None`

- [ ] Define abstract method:
  - [ ] `@abstractmethod post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]`

### 1.3 Add Documentation
- [ ] Add comprehensive Google-style docstrings to all methods
- [ ] Add class-level docstring explaining usage
- [ ] Add type hints to all method signatures
- [ ] Add inline comments for complex logic

### 1.4 Error Handling
- [ ] Add try-catch blocks for dataset loading
- [ ] Add validation for required DataFrame columns
- [ ] Add meaningful error messages
- [ ] Handle edge cases (empty datasets, missing columns, etc.)

## Phase 2: LongBench Implementation

### 2.1 Create LongBench Class
- [ ] Create `benchmark/longbench/longbench.py`
- [ ] Import base Benchmark class
- [ ] Import calculate_metrics functions

### 2.2 Implement LongBench Class
- [ ] Define class attributes:
  ```python
  all_datasets: List[str] = [
      "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
      "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", 
      "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", 
      "lsht", "passage_count", "passage_retrieval_en", 
      "passage_retrieval_zh", "lcc", "repobench-p",
      "qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e", 
      "gov_report_e", "multi_news_e", "trec_e", "triviaqa_e", 
      "samsum_e", "passage_count_e", "passage_retrieval_en_e", 
      "lcc_e", "repobench-p_e"
  ]
  benchmark_name: str = "longbench"
  huggingface_dataset_id: str = "Xnhyacinth/LongBench"
  ```

- [ ] Implement `post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]`
  - [ ] Import calculate_metrics and calculate_metrics_e
  - [ ] Group results by task/dataset name
  - [ ] Use calculate_metrics for standard datasets
  - [ ] Use calculate_metrics_e for extended (_e) datasets
  - [ ] Aggregate metrics across all datasets
  - [ ] Return comprehensive metrics dictionary

### 2.3 Add LongBench Documentation
- [ ] Add class docstring explaining LongBench specifics
- [ ] Document the 35 datasets supported
- [ ] Add examples of usage
- [ ] Document evaluation metrics used

## Phase 3: Integration & Testing

### 3.1 Create Test Framework
- [ ] Create `tests/test_benchmark_base.py`
- [ ] Create `tests/test_longbench.py`
- [ ] Set up test fixtures and mock data

### 3.2 Unit Tests
- [ ] Test Benchmark.__init__ with various subsets
- [ ] Test _load_datasets with mock HuggingFace datasets
- [ ] Test _validate_dataset_size with large/small datasets
- [ ] Test _process_all_requests with mock adapter
- [ ] Test run_benchmark end-to-end with mock data

### 3.3 Integration Tests
- [ ] Test with real HuggingFace adapter
- [ ] Test with small subset of real LongBench data
- [ ] Verify Request/RequestResponse interface compatibility
- [ ] Test file saving functionality

### 3.4 Error Handling Tests
- [ ] Test with invalid dataset IDs
- [ ] Test with malformed DataFrames
- [ ] Test with adapter failures
- [ ] Test with missing directories

### 3.5 Performance Testing
- [ ] Compare context grouping vs individual requests
- [ ] Test memory usage with large contexts
- [ ] Verify progress tracking works correctly

## Phase 4: Documentation & Examples #aditya(SKIP)

### 4.1 Usage Documentation
- [ ] Create usage examples in README
- [ ] Document how to extend for new benchmarks
- [ ] Add configuration examples

### 4.2 Create Example Scripts
- [ ] Create example script for running LongBench
- [ ] Create example for custom subsets
- [ ] Add example output analysis

## Phase 5: Integration with Repository

### 5.1 Update Repository Structure  
- [ ] Update main benchmark/__init__.py to export classes
- [ ] Add benchmark imports to main package
- [ ] Update repository README

### 5.2 CI/CD Integration
- [ ] Add benchmark tests to CI pipeline
- [ ] Add linting checks for new code
- [ ] Verify conda environment compatibility

## Completion Criteria

### Ready for Use
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Example usage works
- [ ] LongBench evaluation produces reasonable metrics
- [ ] Memory usage is acceptable
- [ ] Error handling is robust

### Ready for Extension
- [ ] Base class is easily extensible
- [ ] Pattern is clear for adding new benchmarks
- [ ] Interface is well-documented
- [ ] Future enhancements are clearly documented 