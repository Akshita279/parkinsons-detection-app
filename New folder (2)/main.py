"""
Parkinson's Disease Detection System - Streamlit Entry Point
Run with: streamlit run main.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path so imports work correctly
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Change working directory to this file's location so relative paths work
os.chdir(Path(__file__).parent)

# Import and run the actual Streamlit app
from streamlit_app import main

main()
