#!/bin/bash

echo "🧠 Starting Parkinson's Disease Detection Web App..."
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

# Start Streamlit app
echo "Starting web interface..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

echo "App started successfully!"
echo "Access the app at: http://localhost:8501"
