#!/usr/bin/env python3
"""
Script to run tests for the Sparse Attention Hub.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Run tests for Sparse Attention Hub")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration"],
        default="all",
        help="Type of tests to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Get the project root directory
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    # Check if tests directory exists
    if not tests_dir.exists():
        print(f"Error: Tests directory not found at {tests_dir}")
        return 1

    # Determine test path based on type
    if args.type == "unit":
        test_path = tests_dir / "unit"
    elif args.type == "integration":
        test_path = tests_dir / "integration"
    else:  # all
        test_path = tests_dir

    # Check if the specific test path exists
    if not test_path.exists():
        print(f"Error: Test path not found at {test_path}")
        return 1

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", str(test_path)]

    # Add verbose flag if requested
    if args.verbose:
        cmd.append("-v")

    # Always add short traceback for cleaner output
    cmd.append("--tb=short")

    print(f"Running {args.type} tests in: {test_path}")
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
