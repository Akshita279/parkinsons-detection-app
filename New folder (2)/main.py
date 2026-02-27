#!/usr/bin/env python3
"""
Main launcher for Parkinson's Disease Detection System
"""

import subprocess
import sys
from pathlib import Path

def run_data_collection():
    """Run data collection pipeline"""
    print("🚀 Running data collection...")
    result = subprocess.run([sys.executable, "scripts/collect_data.py"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Data collection completed")
        return True
    else:
        print(f"❌ Data collection failed: {result.stderr}")
        return False

def run_model_training():
    """Run model training pipeline"""
    print("🚀 Running model training...")
    result = subprocess.run([sys.executable, "scripts/train_models.py"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Model training completed")
        return True
    else:
        print(f"❌ Model training failed: {result.stderr}")
        return False

def launch_web_app():
    """Launch Streamlit web application"""
    print("🚀 Launching web application...")
    subprocess.run([
        "streamlit", "run", "src/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def main():
    print("🧠 Parkinson's Disease Detection System")
    print("=" * 50)
    
    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists() or not list(models_dir.glob("best_model_*.pkl")):
        print("📊 No trained models found. Setting up system...")
        
        # Run data collection
        if not run_data_collection():
            return
        
        # Run model training
        if not run_model_training():
            return
    else:
        print("✅ Pre-trained models found")
    
    # Launch web app
    print("\n🌐 Starting web interface...")
    print("Access the app at: http://localhost:8501")
    launch_web_app()

if __name__ == "__main__":
    main()
