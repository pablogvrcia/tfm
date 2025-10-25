#!/usr/bin/env python3
"""
Comprehensive test runner for Open-Vocabulary Segmentation Pipeline.
Automatically detects CPU/GPU and runs all tests with detailed reporting.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --fast             # Run only fast tests
    python run_tests.py --module models    # Run tests for specific module
    python run_tests.py --verbose          # Verbose output
"""

import sys
import os
import unittest
import argparse
import time
import torch
from io import StringIO
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""

    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_times = {}

    def startTest(self, test):
        super().startTest(test)
        self.test_times[test] = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        elapsed = time.time() - self.test_times[test]
        if self.showAll:
            self.stream.write(f"{self.GREEN}✓ PASS{self.RESET} ({elapsed:.3f}s)\n")

    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.write(f"{self.RED}✗ ERROR{self.RESET}\n")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.write(f"{self.RED}✗ FAIL{self.RESET}\n")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.write(f"{self.YELLOW}⊘ SKIP{self.RESET} - {reason}\n")


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output."""
    resultclass = ColoredTextTestResult


def print_header(text, char='='):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f" {text}")
    print(f"{char * 70}\n")


def detect_environment():
    """Detect the execution environment (CPU/GPU)."""
    print_header("Environment Detection")

    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")

    # Check PyTorch
    try:
        print(f"PyTorch version: {torch.__version__}")
    except:
        print("PyTorch: Not installed")
        return None

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        device = "cuda"
    else:
        print(f"CUDA available: No")
        print(f"Running on: CPU")
        device = "cpu"

    # Check for models
    print("\nModel availability:")
    models_status = {}

    try:
        import open_clip
        print("  ✓ CLIP (open_clip)")
        models_status['clip'] = True
    except ImportError:
        print("  ✗ CLIP not available")
        models_status['clip'] = False

    try:
        from sam2 import SAM2
        print("  ✓ SAM 2")
        models_status['sam2'] = True
    except:
        print("  ✗ SAM 2 not available (will use mock)")
        models_status['sam2'] = False

    try:
        from diffusers import StableDiffusionInpaintPipeline
        print("  ✓ Stable Diffusion")
        models_status['stable_diffusion'] = True
    except ImportError:
        print("  ✗ Stable Diffusion not available (will use mock)")
        models_status['stable_diffusion'] = False

    return {
        'device': device,
        'cuda_available': cuda_available,
        'models': models_status
    }


def discover_tests(test_dir='tests', pattern='test_*.py', module=None):
    """Discover test cases."""
    if module:
        pattern = f'test_{module}.py'

    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)

    return suite


def count_tests(suite):
    """Count total number of tests in suite."""
    count = 0
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            count += count_tests(test)
        else:
            count += 1
    return count


def run_tests(suite, verbosity=2):
    """Run test suite with custom runner."""
    runner = ColoredTextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result


def print_summary(result, elapsed_time, env_info):
    """Print test summary."""
    print_header("Test Summary", '=')

    total = result.testsRun
    passed = total - len(result.failures) - len(result.errors) - len(result.skipped)
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)

    print(f"Environment: {env_info['device'].upper()}")
    print(f"Total tests run: {total}")
    print(f"  ✓ Passed: {passed}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")
    if errors > 0:
        print(f"  ✗ Errors: {errors}")
    if skipped > 0:
        print(f"  ⊘ Skipped: {skipped}")

    print(f"\nTotal time: {elapsed_time:.2f}s")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description='Run tests for Open-Vocabulary Segmentation Pipeline'
    )
    parser.add_argument(
        '--module',
        choices=['models', 'pipeline', 'config', 'utils', 'main'],
        help='Run tests for specific module only'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run only fast tests (skip slow model tests)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    parser.add_argument(
        '--failfast',
        action='store_true',
        help='Stop on first failure'
    )

    args = parser.parse_args()

    # Detect environment
    env_info = detect_environment()
    if env_info is None:
        print("ERROR: PyTorch not installed. Cannot run tests.")
        return 1

    # Set verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1

    # Discover tests
    print_header("Discovering Tests")

    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = discover_tests(test_dir, module=args.module)

    total_tests = count_tests(suite)
    print(f"Found {total_tests} tests")

    if args.module:
        print(f"Running tests for module: {args.module}")

    # Run tests
    print_header("Running Tests")

    start_time = time.time()

    # Set environment variable for fast mode
    if args.fast:
        os.environ['FAST_TESTS'] = '1'
        print("Running in FAST mode (skipping slow tests)\n")

    result = run_tests(suite, verbosity=verbosity)

    elapsed_time = time.time() - start_time

    # Print summary
    exit_code = print_summary(result, elapsed_time, env_info)

    # Print failures and errors if any
    if result.failures or result.errors:
        print_header("Failed Tests Details", '-')

        for test, traceback in result.failures:
            print(f"\nFAILURE: {test}")
            print(traceback)

        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(traceback)

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
