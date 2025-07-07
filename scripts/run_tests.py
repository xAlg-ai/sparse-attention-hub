#!/usr/bin/env python3
"""
Script to run tests for the Sparse Attention Hub.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Main function to run tests."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    # Check if tests directory exists
    if not tests_dir.exists():
        print(f"Error: Tests directory not found at {tests_dir}")
        return 1
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest", 
        str(tests_dir),
        "-v",
        "--tb=short"
    ]
    
    print(f"Running tests in: {tests_dir}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
