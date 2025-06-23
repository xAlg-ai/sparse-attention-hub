"""Test execution and management."""

import subprocess
import sys
from typing import Dict, List, Optional


class Tester:
    """Manages and executes tests for the sparse attention hub."""

    def __init__(self, test_directory: str = "tests"):
        self.test_directory = test_directory
        self.unit_test_dir = f"{test_directory}/unit"
        self.integration_test_dir = f"{test_directory}/integration"

    def execute_all_tests(self, verbose: bool = True) -> Dict[str, bool]:
        """Execute all tests (unit and integration).

        Args:
            verbose: Whether to print detailed output

        Returns:
            Dictionary with test results
        """
        results = {}

        print("Running all tests...")

        # Run unit tests
        unit_result = self.execute_unit_tests(verbose=verbose)
        results["unit_tests"] = unit_result

        # Run integration tests
        integration_result = self.execute_integration_tests(verbose=verbose)
        results["integration_tests"] = integration_result

        # Overall result
        results["all_passed"] = unit_result and integration_result

        if verbose:
            self._print_summary(results)

        return results

    def execute_unit_tests(self, verbose: bool = True) -> bool:
        """Execute unit tests.

        Args:
            verbose: Whether to print detailed output

        Returns:
            True if all unit tests passed, False otherwise
        """
        if verbose:
            print("Running unit tests...")

        try:
            # Run pytest on unit test directory
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    self.unit_test_dir,
                    "-v" if verbose else "-q",
                    "--tb=short",
                ],
                capture_output=not verbose,
                text=True,
            )

            success = result.returncode == 0

            if verbose:
                if success:
                    print("✓ Unit tests passed")
                else:
                    print("✗ Unit tests failed")
                    if result.stdout:
                        print("STDOUT:", result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)

            return success

        except Exception as e:
            if verbose:
                print(f"✗ Error running unit tests: {e}")
            return False

    def execute_integration_tests(self, verbose: bool = True) -> bool:
        """Execute integration tests.

        Args:
            verbose: Whether to print detailed output

        Returns:
            True if all integration tests passed, False otherwise
        """
        if verbose:
            print("Running integration tests...")

        try:
            # Run pytest on integration test directory
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    self.integration_test_dir,
                    "-v" if verbose else "-q",
                    "--tb=short",
                ],
                capture_output=not verbose,
                text=True,
            )

            success = result.returncode == 0

            if verbose:
                if success:
                    print("✓ Integration tests passed")
                else:
                    print("✗ Integration tests failed")
                    if result.stdout:
                        print("STDOUT:", result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)

            return success

        except Exception as e:
            if verbose:
                print(f"✗ Error running integration tests: {e}")
            return False

    def execute_specific_test(self, test_path: str, verbose: bool = True) -> bool:
        """Execute a specific test file or test function.

        Args:
            test_path: Path to test file or test function
            verbose: Whether to print detailed output

        Returns:
            True if test passed, False otherwise
        """
        if verbose:
            print(f"Running specific test: {test_path}")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    test_path,
                    "-v" if verbose else "-q",
                    "--tb=short",
                ],
                capture_output=not verbose,
                text=True,
            )

            success = result.returncode == 0

            if verbose:
                if success:
                    print(f"✓ Test {test_path} passed")
                else:
                    print(f"✗ Test {test_path} failed")
                    if result.stdout:
                        print("STDOUT:", result.stdout)
                    if result.stderr:
                        print("STDERR:", result.stderr)

            return success

        except Exception as e:
            if verbose:
                print(f"✗ Error running test {test_path}: {e}")
            return False

    def _print_summary(self, results: Dict[str, bool]) -> None:
        """Print test summary.

        Args:
            results: Test results dictionary
        """
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)

        unit_status = "PASSED" if results["unit_tests"] else "FAILED"
        integration_status = "PASSED" if results["integration_tests"] else "FAILED"
        overall_status = "PASSED" if results["all_passed"] else "FAILED"

        print(f"Unit Tests:        {unit_status}")
        print(f"Integration Tests: {integration_status}")
        print(f"Overall:           {overall_status}")
        print("=" * 50)

    def discover_tests(self) -> Dict[str, List[str]]:
        """Discover available tests.

        Returns:
            Dictionary mapping test types to test file lists
        """
        import os

        discovered = {"unit": [], "integration": []}

        # Discover unit tests
        if os.path.exists(self.unit_test_dir):
            for root, dirs, files in os.walk(self.unit_test_dir):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        discovered["unit"].append(os.path.join(root, file))

        # Discover integration tests
        if os.path.exists(self.integration_test_dir):
            for root, dirs, files in os.walk(self.integration_test_dir):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        discovered["integration"].append(os.path.join(root, file))

        return discovered
