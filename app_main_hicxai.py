"""
DS Project - HicXAI-Style Main Application
Clean, question-driven interface for DS intent classification
"""

import streamlit as st
from pathlib import Path
import sys

# Configure page - HicXAI style
st.set_page_config(
    page_title="Banking Assistant - DS Reasoning",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the simple banking assistant interface
from src.streamlit_app.simple_banking_assistant import main as banking_main

def main():
    """Run the simple banking assistant"""
    banking_main()

if __name__ == "__main__":
    main()