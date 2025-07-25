#!/usr/bin/env python3
"""
Example usage of the AIME2025 benchmark.

This script demonstrates how to use the AIME2025 benchmark
for evaluating mathematical reasoning capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AIME2025.create_huggingface_dataset import create_aime2025_dataset
from AIME2025.calculate_metrics import calculate_metrics
import pandas as pd


def show_sample_problems():
    """Display sample problems from the AIME2025 dataset."""
    print("AIME2025 Sample Problems")
    print("=" * 50)
    
    dataset = create_aime2025_dataset()
    
    # Show first 3 problems
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        print(f"\nProblem {i+1}:")
        print("-" * 30)
        
        # Extract just the problem statement from the context
        context = example['context']
        problem_start = context.find("Problem: ") + len("Problem: ")
        problem_end = context.find("\n\nInstructions:")
        problem = context[problem_start:problem_end]
        
        print(f"Problem: {problem}")
        print(f"Answer: {example['answer'][0]}")


def simulate_model_responses():
    """Simulate model responses and calculate metrics."""
    print("\n" + "=" * 50)
    print("Simulated Model Evaluation")
    print("=" * 50)
    
    # Create some simulated model responses
    simulated_responses = [
        {
            'predicted_answer': "Let me work through this step by step.\n\nAfter calculating the bases, I find that the sum is \\boxed{70}.",
            'answer': ['70']  # Correct
        },
        {
            'predicted_answer': "This is a complex problem. After working through it, the answer is \\boxed{42}.",
            'answer': ['588']  # Wrong
        },
        {
            'predicted_answer': "The calculation gives us 588 as the final result.",
            'answer': ['588']  # Correct, but no boxed format
        },
        {
            'predicted_answer': "Answer: 16",
            'answer': ['16']  # Correct, no boxed format
        },
        {
            'predicted_answer': "This problem is too difficult to solve.",
            'answer': ['100']  # No answer extracted
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(simulated_responses)
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    print("Simulated Results:")
    print(f"Total problems: {len(simulated_responses)}")
    print(f"Exact match accuracy: {metrics['exact_match']:.1%}")
    print(f"Answer extraction rate: {metrics['extraction_rate']:.1%}")
    print(f"Boxed format usage: {metrics['boxed_format_rate']:.1%}")
    
    print("\nDetailed breakdown:")
    for i, response in enumerate(simulated_responses):
        from AIME2025.calculate_metrics import extract_numerical_answer
        extracted = extract_numerical_answer(response['predicted_answer'])
        correct = extracted in response['answer']
        status = "✓ Correct" if correct else "✗ Wrong"
        
        print(f"Problem {i+1}: {status}")
        print(f"  Expected: {response['answer'][0]}")
        print(f"  Extracted: '{extracted}'")
        print(f"  Response: {response['predicted_answer'][:60]}...")
        print()


def show_usage_instructions():
    """Show how to use the benchmark in practice."""
    print("=" * 50)
    print("Usage Instructions")
    print("=" * 50)
    
    print("\n1. Command Line Usage:")
    print("   python evaluation/evaluate.py --dataset aime2025 --base_model meta-llama/Llama-3.1-8B-Instruct")
    
    print("\n2. Programmatic Usage:")
    print("""
   from evaluation.evaluate import evaluate
   
   # Run evaluation
   evaluate(
       dataset="aime2025",
       base_model="meta-llama/Llama-3.1-8B-Instruct",
       max_new_tokens=4096,
       num_samples=10  # Optional: evaluate on subset
   )
   """)
    
    print("\n3. Custom Evaluation:")
    print("""
   from evaluation.AIME2025 import create_aime2025_dataset, calculate_metrics
   import pandas as pd
   
   # Load dataset
   dataset = create_aime2025_dataset()
   
   # Run your model on the problems
   # ... (your model inference code)
   
   # Calculate metrics
   results_df = pd.DataFrame({
       'predicted_answer': model_responses,
       'answer': [example['answer'] for example in dataset]
   })
   metrics = calculate_metrics(results_df)
   print(metrics)
   """)
    
    print("\n4. Expected Output Format:")
    print("""
   Models should respond with step-by-step reasoning and wrap
   the final answer in \\boxed{...} format:
   
   Example:
   "Let me solve this step by step.
   
   First, I need to find the bases where 17_b divides 97_b.
   This means (1×b + 7) divides (9×b + 7).
   
   [mathematical reasoning...]
   
   Therefore, the sum of all valid bases is \\boxed{70}."
   """)


def main():
    """Run the example demonstration."""
    show_sample_problems()
    simulate_model_responses()
    show_usage_instructions()
    
    print("\n" + "=" * 50)
    print("AIME2025 Benchmark Ready!")
    print("=" * 50)


if __name__ == "__main__":
    main()