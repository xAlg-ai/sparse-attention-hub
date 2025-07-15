# AIME2025 Benchmark

This benchmark evaluates language models on problems from the American Invitational Mathematics Examination (AIME) 2025.

## Dataset Description

The AIME2025 dataset contains 30 mathematical competition problems from the 2025 AIME competition. Each problem requires a numerical answer between 0 and 999.

**Source**: `yentinglin/aime_2025` on Hugging Face

**Key Features**:
- 30 high-quality mathematical problems
- Problems cover various mathematical topics (algebra, geometry, number theory, combinatorics, etc.)
- Answers are integers in the range [0, 999]
- Problems require multi-step reasoning and mathematical insight

## Evaluation Format

### Input Format
Each problem is presented with:
- The problem statement
- Clear instructions to wrap the final answer in `\boxed{...}` format
- Encouragement to show step-by-step work

### Expected Output Format
Models should:
1. Show their mathematical reasoning step by step
2. Provide the final numerical answer wrapped in `\boxed{answer}` format
3. Ensure the answer is an integer between 0 and 999

Example:
```
Let me solve this step by step.

[Mathematical reasoning and calculations]

Therefore, the answer is \boxed{42}.
```

## Evaluation Metrics

### Primary Metric
- **Exact Match**: Percentage of problems where the extracted numerical answer exactly matches the ground truth

### Additional Metrics
- **Extraction Rate**: Percentage of responses from which a numerical answer could be extracted
- **Boxed Format Rate**: Percentage of responses that used the required `\boxed{...}` format
- **Error Analysis**: Breakdown of error types (no answer extracted, wrong answer, out of range, format issues)

## Answer Extraction Logic

The evaluation uses a robust answer extraction system with multiple fallback strategies:

1. **Primary**: Extract from `\boxed{...}` format
2. **Fallback 1**: Look for "answer is X" or "answer: X" patterns
3. **Fallback 2**: Extract numbers from the end of the response
4. **Fallback 3**: Find any valid AIME-range number (0-999) in the text

## Usage

### Running the Benchmark
```python
from evaluation.evaluate import evaluate

# Run AIME2025 evaluation
evaluate(
    dataset="aime2025",
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=4096,  # Allow comprehensive step-by-step solutions
)
```

### Dataset Processing
```python
from evaluation.AIME2025.create_huggingface_dataset import create_aime2025_dataset

# Process the raw dataset
dataset = create_aime2025_dataset()
```

### Metrics Calculation
```python
from evaluation.AIME2025.calculate_metrics import calculate_metrics
import pandas as pd

# Calculate metrics on results
df = pd.read_csv("results.csv")
metrics = calculate_metrics(df)
print(metrics)
```

## Expected Performance

AIME problems are designed to be challenging for high school students and typically require:
- Strong mathematical reasoning abilities
- Multi-step problem solving
- Knowledge across various mathematical domains
- Careful numerical computation

Even strong language models may find these problems challenging, with accuracy varying significantly based on:
- Model size and training
- Mathematical reasoning capabilities
- Ability to follow the boxed answer format
- Numerical computation accuracy

## Notes

- The benchmark emphasizes both mathematical reasoning and format compliance
- The `\boxed{...}` format is standard in mathematical competitions and helps ensure clear answer identification
- Problems are from a recent competition (2025), reducing the likelihood of training data contamination
- The relatively small dataset size (30 problems) means results should be interpreted carefully, especially for statistical significance