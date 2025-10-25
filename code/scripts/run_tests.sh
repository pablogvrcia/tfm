#!/bin/bash
# Bash wrapper for test runner - works on both CPU and GPU

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE} Open-Vocabulary Segmentation Pipeline - Test Suite${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check if we're in a Docker container
if [ -f /.dockerenv ]; then
    echo -e "${YELLOW}Running inside Docker container${NC}"
fi

# Check Python version
python3 --version

# Run Python test runner
echo ""
python3 run_tests.py "$@"

exit_code=$?

# Print final message
echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✓ Test suite completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed. See details above.${NC}"
fi

exit $exit_code
