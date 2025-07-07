#!/usr/bin/env bash
# Linting script for sparse-attention-hub
#
# Usage:
#    # Run all linting checks
#    bash scripts/lint.sh
#
#    # Run specific linter
#    bash scripts/lint.sh --flake8
#    bash scripts/lint.sh --mypy
#    bash scripts/lint.sh --pylint
#    bash scripts/lint.sh --bandit

# Cause the script to exit if a single command fails
set -eo pipefail

# Navigate to project root
builtin cd "$(dirname "${BASH_SOURCE:-$0}")/.."
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if tool is installed
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed. Please run: pip install -r requirements-dev.txt${NC}"
        return 1
    fi
    return 0
}

# Run flake8
run_flake8() {
    echo -e "${BLUE}Running flake8...${NC}"
    if check_tool "flake8"; then
        flake8 \
            --max-line-length=88 \
            --extend-ignore=E203,W503,E501 \
            --exclude=build,dist,.git,__pycache__,.pytest_cache,.venv \
            sparse_attention_hub tests scripts
        echo -e "${GREEN}✅ flake8 passed${NC}"
    else
        return 1
    fi
}

# Run mypy
run_mypy() {
    echo -e "${BLUE}Running mypy...${NC}"
    if check_tool "mypy"; then
        mypy \
            --ignore-missing-imports \
            --no-strict-optional \
            --warn-redundant-casts \
            --warn-unused-ignores \
            sparse_attention_hub
        echo -e "${GREEN}✅ mypy passed${NC}"
    else
        return 1
    fi
}

# Run pylint
run_pylint() {
    echo -e "${BLUE}Running pylint...${NC}"
    if check_tool "pylint"; then
        pylint --rcfile=.pylintrc sparse_attention_hub
        echo -e "${GREEN}✅ pylint passed${NC}"
    else
        return 1
    fi
}

# Run bandit (security linter)
run_bandit() {
    echo -e "${BLUE}Running bandit (security linter)...${NC}"
    if check_tool "bandit"; then
        bandit -r sparse_attention_hub/ -f json -o bandit-report.json || true
        bandit -r sparse_attention_hub/
        echo -e "${GREEN}✅ bandit completed${NC}"
    else
        return 1
    fi
}

# Run black check
run_black_check() {
    echo -e "${BLUE}Running black (format check)...${NC}"
    if check_tool "black"; then
        black --check --line-length=88 sparse_attention_hub tests scripts
        echo -e "${GREEN}✅ black format check passed${NC}"
    else
        return 1
    fi
}

# Run isort check
run_isort_check() {
    echo -e "${BLUE}Running isort (import order check)...${NC}"
    if check_tool "isort"; then
        isort \
            --check-only \
            --profile=black \
            --line-length=88 \
            --multi-line=3 \
            -p sparse_attention_hub \
            sparse_attention_hub tests scripts
        echo -e "${GREEN}✅ isort check passed${NC}"
    else
        return 1
    fi
}

# Run all linting checks
run_all() {
    echo -e "${YELLOW}Running all linting checks for sparse-attention-hub...${NC}"
    echo "=================================================="
    
    local failed=0
    
    # Format checks
    run_black_check || failed=1
    echo ""
    
    run_isort_check || failed=1
    echo ""
    
    # Linting checks
    run_flake8 || failed=1
    echo ""
    
    run_mypy || failed=1
    echo ""
    
    run_pylint || failed=1
    echo ""
    
    run_bandit || failed=1
    echo ""
    
    if [[ $failed -eq 0 ]]; then
        echo -e "${GREEN}🎉 All linting checks passed!${NC}"
        return 0
    else
        echo -e "${RED}❌ Some linting checks failed${NC}"
        return 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    --flake8)
        run_flake8
        ;;
    --mypy)
        run_mypy
        ;;
    --pylint)
        run_pylint
        ;;
    --bandit)
        run_bandit
        ;;
    --black)
        run_black_check
        ;;
    --isort)
        run_isort_check
        ;;
    --help|-h)
        echo "Usage: $0 [--flake8|--mypy|--pylint|--bandit|--black|--isort|--help]"
        echo ""
        echo "Options:"
        echo "  --flake8    Run flake8 linter"
        echo "  --mypy      Run mypy type checker"
        echo "  --pylint    Run pylint linter"
        echo "  --bandit    Run bandit security linter"
        echo "  --black     Run black format checker"
        echo "  --isort     Run isort import order checker"
        echo "  --help      Show this help message"
        echo ""
        echo "If no option is provided, all checks will be run."
        ;;
    "")
        run_all
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
