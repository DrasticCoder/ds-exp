#!/bin/bash

# 🧪 Local CI/CD Test Script
# Mimics the GitHub Actions workflow locally

echo "🚀 Running Local CI/CD Tests..."
echo "==============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print status
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✅]${NC} $1"
}

print_error() {
    echo -e "${RED}[❌]${NC} $1"
}

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    print_step "Using virtual environment"
fi

# Step 1: Install tools (like in CI)
print_step "Installing tools..."
python -m pip install --upgrade pip
pip install ruff==0.6.7 black==24.8.0

# Step 2: Ruff check (lint + isort) - Only production files
print_step "Running Ruff (lint + isort)..."
if ruff check dashboard/ app/ *.py --exclude="notebooks/"; then
    print_success "Ruff checks passed"
else
    print_error "Ruff checks failed"
    RUFF_FAILED=1
fi

# Step 3: Black format check - Only production files  
print_step "Running Black (format check)..."
if black --check dashboard/ app/ *.py --exclude="notebooks/"; then
    print_success "Black formatting check passed"
else
    print_error "Black formatting check failed"
    echo "💡 Run 'black .' to auto-fix formatting issues"
    BLACK_FAILED=1
fi

# Step 4: Install app dependencies (like in CI test job)
print_step "Installing app dependencies..."
pip install -r requirements.txt
pip install pytest==8.3.2 httpx==0.27.2

# Step 5: Run tests
print_step "Running tests..."
if pytest -q; then
    print_success "Tests passed"
else
    print_error "Tests failed"
    TEST_FAILED=1
fi

# Summary
echo ""
echo "==============================="
if [ -z "$RUFF_FAILED" ] && [ -z "$BLACK_FAILED" ] && [ -z "$TEST_FAILED" ]; then
    print_success "🎉 All CI/CD checks passed! Ready to push."
    exit 0
else
    print_error "❌ Some checks failed:"
    if [ -n "$RUFF_FAILED" ]; then
        echo "   - Ruff linting issues"
        echo "     Run: ruff check . --fix (to auto-fix)"
    fi
    if [ -n "$BLACK_FAILED" ]; then
        echo "   - Black formatting issues" 
        echo "     Run: black . (to auto-fix)"
    fi
    if [ -n "$TEST_FAILED" ]; then
        echo "   - Test failures"
        echo "     Run: pytest -v (for details)"
    fi
    exit 1
fi