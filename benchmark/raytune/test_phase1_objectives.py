#!/usr/bin/env python3
"""
Test script to verify Phase 1 works with different objective functions.
"""

import subprocess
import sys
import os
import argparse

def test_objective_function(objective_name, show_full_output=False):
    """Test Phase 1 with a specific objective function."""
    print(f"\n{'='*60}")
    print(f"Testing Phase 1 with objective: {objective_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "benchmark/raytune/run_two_phase_benchmark.py",
        "--phase", "1",
        "--debug",  # Use debug mode for faster testing
        "--objective", objective_name,
        "--num-samples", "5",  # Fewer samples for testing
        "--search-timeout", "300",
        "--force-search"  # Force re-search to test the objective
    ]
    
    try:
        # Run subprocess with real-time output
        print("\n--- Starting Phase 1 run ---")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        output_lines = []
        found_objective = False
        found_score_logging = False
        
        # Stream output line by line
        for line in process.stdout:
            # Store for later analysis
            output_lines.append(line.rstrip())
            
            # Check for key indicators
            if f"Objective Function: {objective_name}" in line:
                found_objective = True
            if "Error:" in line and "Density:" in line and "Score:" in line:
                found_score_logging = True
            
            # Print based on preference
            if show_full_output:
                print(line.rstrip())
            else:
                # Only print important lines for default mode
                if any(keyword in line for keyword in [
                    "Objective Function:", "Objective:", "Error:", "Density:", "Score:",
                    "Targeting", "Formula", "Best score:", "✓", "✗", "Phase 1 complete",
                    "ERROR", "Exception", "Traceback", "Failed", "Warning"
                ]):
                    print(f"  > {line.rstrip()}")
        
        # Wait for process to complete
        return_code = process.wait()
        print("--- Phase 1 run completed ---\n")
        
        if return_code == 0:
            print("✓ Phase 1 completed successfully")
            
            if found_objective:
                print(f"✓ Objective function '{objective_name}' was properly logged")
            else:
                print(f"✗ Objective function '{objective_name}' was not found in output")
                
            if found_score_logging:
                print("✓ Density, error, and score logging is working")
            else:
                print("✗ Score logging not detected")
                
            return True
        else:
            print(f"✗ Phase 1 failed with exit code {return_code}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False

def main():
    """Test different objective functions."""
    parser = argparse.ArgumentParser(description="Test Phase 1 with different objective functions")
    parser.add_argument("--full-output", action="store_true",
                       help="Show full output from each test run instead of just key lines")
    parser.add_argument("--objectives", nargs="+", 
                       default=["default", "sparsity_5", "sparsity_10", "sparsity_15"],
                       help="List of objectives to test")
    args = parser.parse_args()
    
    print("Testing Phase 1 with different objective functions")
    if args.full_output:
        print("(Full output mode enabled)")
    
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)
    
    # Test different objectives
    objectives_to_test = args.objectives
    
    results = {}
    for obj in objectives_to_test:
        results[obj] = test_objective_function(obj, show_full_output=args.full_output)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for obj, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{obj}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
