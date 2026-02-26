"""
DS Project - HicXAI-Style Main Application
Clean, question-driven interface for DS intent classification
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Configure page - HicXAI style
st.set_page_config(
    page_title="Customer Service Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add project root to path (MUST be done before any local imports)
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set working directory to project root
os.chdir(project_root)

# Import the simple banking assistant interface
try:
    from src.streamlit_app.simple_banking_assistant import main as banking_main
except ImportError as e:
    st.error(f"Failed to import banking assistant: {e}")
    st.error("Please check that all files are in the correct locations")
    st.stop()

def main():
    """Run the simple banking assistant"""
    banking_main()

if __name__ == "__main__":
    main()