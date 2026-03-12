"""
DS Project - HicXAI-Style Main Application (original setup)

Loads study queries from outputs/user_study/workflow_demo/ — the original
equal-split banking77 / clinc150 sets produced by the workflow notebook.
Controlled by STUDY_SET env var (small / medium / large / full).
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

# The hicxai query CSVs live at the root of /ds_project_queries/ in Dropbox
# (no subfolder), so override the Dropbox folder to the root directly.
os.environ.setdefault('STUDY_SET_DIR', 'outputs/user_study/workflow_demo')
os.environ.setdefault('STUDY_SET_DROPBOX_FOLDER', '/ds_project_queries')
os.environ.setdefault('STUDY_SET', 'small')

# Import the simple banking assistant interface
try:
    from src.streamlit_app.simple_banking_assistant import main as banking_main
except ImportError as e:
    st.error(f"Failed to import banking assistant: {e}")
    st.error("Please check that all files are in the correct locations")
    st.stop()

def main():
    """Run the simple banking assistant (original equal-split query sets)."""
    banking_main()

if __name__ == "__main__":
    main()