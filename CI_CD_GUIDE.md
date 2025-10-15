# 🚀 CI/CD Testing Guide for Disease Outbreak Dashboard

## 📋 Overview

Your repository: **https://github.com/DrasticCoder/ds-exp**

This guide helps you test your CI/CD pipeline locally before pushing to GitHub Actions.

## 🛠️ Available Scripts

### 1. **Auto-Fix Script** (`./fix_code.sh`)

Automatically fixes most formatting and linting issues:

```bash
./fix_code.sh
```

### 2. **Quick Test Script** (`./test_ci.sh`)

Runs the same checks as your GitHub Actions locally:

```bash
./test_ci.sh
```

## 🔧 Manual Commands

### Install Dependencies

```bash
# Install CI/CD tools
pip install ruff==0.6.7 black==24.8.0

# Install app dependencies
pip install -r requirements.txt
```

### Linting & Formatting

```bash
# Check formatting with Black
black --check .

# Auto-fix formatting
black .

# Check linting with Ruff
ruff check .

# Auto-fix linting issues
ruff check . --fix
```

### Testing

```bash
# Run tests
pytest -q

# Import validation
python -c "import dashboard.app; import app.main; print('✅ All imports successful')"
```

## 📊 GitHub Actions Workflow

Your CI pipeline in `.github/workflows/ci.yml` runs 3 jobs:

### Job 1: **Lint** 🔍

- ✅ Ruff (linting + import sorting)
- ✅ Black (code formatting)

### Job 2: **Test** 🧪

- ✅ Install dependencies
- ✅ DVC artifact pulling (optional)
- ✅ Run pytest

### Job 3: **Docker Build** 🐳

- ✅ Build Docker image
- ✅ Push to GitHub Container Registry (on main branch)

## 🚀 Workflow to Push Changes

### 1. **Local Testing**

```bash
# Step 1: Auto-fix issues
./fix_code.sh

# Step 2: Test CI pipeline locally
./test_ci.sh

# Step 3: Manual verification (optional)
ruff check .
black --check .
pytest -q
```

### 2. **Push to GitHub**

```bash
# Commit your changes
git add .
git commit -m "feat: your changes description"

# Push to trigger CI
git push origin main
```

### 3. **Monitor GitHub Actions**

Visit: https://github.com/DrasticCoder/ds-exp/actions

## 🐛 Common Issues & Fixes

### **Ruff Issues**

```bash
# Fix automatically
ruff check . --fix

# Manual fixes needed for complex issues
```

### **Black Formatting**

```bash
# Auto-format all files
black .

# Check specific file
black --check dashboard/app.py
```

### **Import Errors**

```bash
# Test imports manually
python -c "import dashboard.app"
python -c "import app.main"
```

### **Test Failures**

```bash
# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_api.py -v
```

## 🎯 Production Deployment

### Streamlit Cloud

1. **Auto-deploys** from your main branch
2. Uses `requirements.txt` for dependencies
3. Uses `.streamlit/config.toml` for configuration

### Docker (via GitHub Actions)

- ✅ Builds on every push to main
- ✅ Pushes to `ghcr.io/drasticcoder/disease-outbreak-api`
- ✅ Available at: https://github.com/DrasticCoder/ds-exp/pkgs/container/disease-outbreak-api

## 📈 CI/CD Success Indicators

### ✅ **All Green Checks**

- Linting passes
- Formatting is correct
- Tests pass
- Docker builds successfully

### ❌ **Common Failures**

- **Red X on Lint**: Run `./fix_code.sh`
- **Red X on Test**: Check `pytest -v` output
- **Red X on Docker**: Check Dockerfile and dependencies

## 🎉 Next Steps

1. **Test locally**: `./test_ci.sh`
2. **Fix issues**: `./fix_code.sh`
3. **Push changes**: `git push origin main`
4. **Monitor**: https://github.com/DrasticCoder/ds-exp/actions
5. **Deploy**: Automatic on successful CI

---

**Your Disease Outbreak Dashboard is ready for production! 🦠📊**
