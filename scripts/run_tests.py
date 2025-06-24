#!/usr/bin/env python3
"""
Script to run all tests for the Sparse Attention Hub.
"""

import argparse
import sys

from sparse_attention_hub.testing import Tester


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Run Sparse Attention Hub tests")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "specific"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--test-path", help="Specific test path (required when type=specific)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--discover", action="store_true", help="Discover and list available tests"
    )

    args = parser.parse_args()

    tester = Tester()

    if args.discover:
        print("Discovering tests...")
        discovered = tester.discover_tests()

        print(f"\nUnit tests ({len(discovered['unit'])}): ")
        for test in discovered["unit"]:
            print(f" - {test}")

        print(f"\nIntegration tests ({len(discovered['integration'])}): ")
        for test in discovered["integration"]:
            print(f" - {test}")

        return 0

    success = True

    if args.type == "all":
        results = tester.execute_all_tests(verbose=args.verbose)
        success = results["all_passed"]
    elif args.type == "unit":
        success = tester.execute_unit_tests(verbose=args.verbose)
    elif args.type == "integration":
        success = tester.execute_integration_tests(verbose=args.verbose)
    elif args.type == "specific":
        if not args.test_path:
            print("Error: --test-path required when type=specific")
            return 1
        success = tester.execute_specific_test(args.test_path, verbose=args.verbose)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
