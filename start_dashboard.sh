#!/bin/bash

# Production startup script for Disease Outbreak Risk Dashboard
# This script ensures clean startup without browser auto-opening issues

echo "🦠 Starting Disease Outbreak Risk Dashboard..."
echo "📍 Production Mode: Clean startup without browser auto-opening"

# Set environment variables for production
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false

# Start the dashboard with production settings
/Users/deepbansode/Desktop/code/learning/ds-exp/.venv/bin/python -m streamlit run \
    dashboard/app.py \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false

echo "🎉 Dashboard started successfully!"
echo "🌐 Access at: http://localhost:8501"