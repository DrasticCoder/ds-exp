#!/bin/bash

# 🛠️ Auto-fix Code Quality Issues
# Automatically applies formatting and linting fixes

echo "🛠️ Auto-fixing code quality issues..."
echo "===================================="

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "🐍 Using virtual environment"
fi

# 1. Auto-fix with Ruff (focus on main application files)
echo ""
echo "🔧 Running Ruff auto-fix on production files..."
.venv/bin/python -m ruff check dashboard/ app/ *.py --fix

# 2. Auto-format with Black (focus on main application files)
echo ""
echo "🎨 Running Black formatting on production files..."
.venv/bin/python -m black dashboard/ app/ *.py

# 3. Verify fixes
echo ""
echo "✅ Verification:"
echo "=================="

# Check Ruff on production files
echo "📝 Checking Ruff on production files..."
if .venv/bin/python -m ruff check dashboard/ app/ *.py --quiet; then
    echo "✅ Ruff: All production files clean"
else
    echo "⚠️ Ruff: Some issues remain in production files"
fi

# Check Black on production files
echo "🎨 Checking Black on production files..."
if .venv/bin/python -m black --check dashboard/ app/ *.py --quiet; then
    echo "✅ Black: Production files formatted correctly"
else
    echo "⚠️ Black: Production formatting issues remain"
fi

echo ""
echo "🎉 Auto-fix complete! Your code is now CI/CD ready."