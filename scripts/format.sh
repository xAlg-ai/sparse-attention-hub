#!/usr/bin/env bash
# Formatting script for sparse-attention-hub
# Adapted from SkyPilot's format.sh
#
# Usage:
#    # Format files that differ from origin/main
#    bash scripts/format.sh
#
#    # Format all files
#    bash scripts/format.sh --all
#
#    # Format specific files
#    bash scripts/format.sh --files file1.py file2.py

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")/.."
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

# Check if development dependencies are installed
check_tool_installed() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed. Please run: pip install -r requirements-dev.txt"
        exit 1
    fi
}

# Check required tools
check_tool_installed "black"
check_tool_installed "isort"
check_tool_installed "flake8"
check_tool_installed "mypy"
check_tool_installed "pylint"

# Version checking function
tool_version_check() {
    local tool_name=$1
    local installed_version=$2
    local required_version=$3
    
    if [[ "$installed_version" != "$required_version" ]]; then
        echo "Warning: $tool_name version mismatch. Required: $required_version, Installed: $installed_version"
        echo "Consider running: pip install -r requirements-dev.txt"
    fi
}

# Get versions
BLACK_VERSION=$(black --version | head -n 1 | awk '{print $2}')
ISORT_VERSION=$(isort --version | head -n 1 | awk '{print $2}')
FLAKE8_VERSION=$(flake8 --version | head -n 1 | awk '{print $1}')
MYPY_VERSION=$(mypy --version | awk '{print $2}')
PYLINT_VERSION=$(pylint --version | head -n 1 | awk '{print $2}')

# Check versions against requirements-dev.txt
if [[ -f "requirements-dev.txt" ]]; then
    tool_version_check "black" "$BLACK_VERSION" "$(grep "black==" requirements-dev.txt | cut -d'=' -f3)"
    tool_version_check "isort" "$ISORT_VERSION" "$(grep "isort==" requirements-dev.txt | cut -d'=' -f3)"
    tool_version_check "flake8" "$FLAKE8_VERSION" "$(grep "flake8==" requirements-dev.txt | cut -d'=' -f3)"
    tool_version_check "mypy" "$MYPY_VERSION" "$(grep "mypy==" requirements-dev.txt | cut -d'=' -f3)"
    tool_version_check "pylint" "$PYLINT_VERSION" "$(grep "pylint==" requirements-dev.txt | cut -d'=' -f3)"
fi

# Formatting flags
BLACK_FLAGS=(
    '--line-length=88'
)

ISORT_FLAGS=(
    '--profile=black'
    '--line-length=88'
    '--multi-line=3'
    '-p' 'sparse_attention_hub'
)

FLAKE8_FLAGS=(
    '--max-line-length=88'
    '--extend-ignore=E203,W503,E501'
    '--exclude=build,dist,.git,__pycache__,.pytest_cache,.venv'
)

PYLINT_FLAGS=(
    '--rcfile=.pylintrc'
)

MYPY_FLAGS=(
    '--ignore-missing-imports'
    '--no-strict-optional'
    '--warn-redundant-casts'
    '--warn-unused-ignores'
)

# Directories to format
PYTHON_DIRS=(
    'sparse_attention_hub'
    'tests'
    'examples'
    'scripts'
)

# Format specified files
format_files() {
    echo "Formatting specified files..."
    black "${BLACK_FLAGS[@]}" "$@"
    isort "${ISORT_FLAGS[@]}" "$@"
}

# Format files that differ from main branch
format_changed() {
    echo "Formatting changed files..."
    
    # Get merge base with origin/main or main
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
        MERGEBASE="$(git merge-base origin/main HEAD)"
    elif git rev-parse --verify main >/dev/null 2>&1; then
        MERGEBASE="$(git merge-base main HEAD)"
    else
        echo "Warning: No main branch found, formatting all files"
        format_all
        return
    fi

    # Format Python files that have changed
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        echo "Found changed Python files, formatting..."
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | \
            tr '\n' '\0' | xargs -P 5 -0 \
            black "${BLACK_FLAGS[@]}"
        
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | \
            tr '\n' '\0' | xargs -P 5 -0 \
            isort "${ISORT_FLAGS[@]}"
    else
        echo "No Python files changed."
    fi
}

# Format all files
format_all() {
    echo "Formatting all Python files..."
    black "${BLACK_FLAGS[@]}" "${PYTHON_DIRS[@]}"
    isort "${ISORT_FLAGS[@]}" "${PYTHON_DIRS[@]}"
}

# Run linting
run_linting() {
    echo "Running linting checks..."
    
    echo "Running flake8..."
    flake8 "${FLAKE8_FLAGS[@]}" "${PYTHON_DIRS[@]}"
    
    echo "Running mypy..."
    mypy "${MYPY_FLAGS[@]}" sparse_attention_hub
    
    echo "Running pylint..."
    pylint "${PYLINT_FLAGS[@]}" sparse_attention_hub
}

# Main execution
echo "Sparse Attention Hub Code Formatting"
echo "====================================="

# Parse command line arguments
if [[ "$1" == '--files' ]]; then
    format_files "${@:2}"
elif [[ "$1" == '--all' ]]; then
    format_all
elif [[ "$1" == '--lint-only' ]]; then
    run_linting
    exit 0
else
    format_changed
fi

echo "Formatting complete!"

# Run linting unless --no-lint is specified
if [[ "$*" != *"--no-lint"* ]]; then
    echo ""
    run_linting
    echo "Linting complete!"
fi

# Check if there are any changes after formatting
if ! git diff --quiet &>/dev/null; then
    echo ""
    echo "⚠️  Formatting made changes to the following files:"
    git --no-pager diff --name-only
    echo ""
    echo "Please review and commit these changes."
    exit 1
else
    echo ""
    echo "✅ All files are properly formatted!"
fi