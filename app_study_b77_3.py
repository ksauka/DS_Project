"""
DS Project - Banking77 Study App — Slice 3 of 4

100% Banking77 queries, DS system.
Participants in this app see queries 9–12 (small), 17–24 (medium), or 25–36 (large).
Deploy as its own Streamlit Cloud app; set STUDY_SET in Streamlit secrets.

STUDY_SET (Streamlit secret): small | medium | large    default: small
"""

import streamlit as st
from pathlib import Path
import sys
import os

st.set_page_config(
    page_title="Customer Service Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

os.environ['STUDY_SET_DIR'] = 'outputs/user_study/study_b77only'
os.environ['QUERY_SLICE'] = '2'
os.environ.setdefault('STUDY_SET', 'small')

try:
    from src.streamlit_app.simple_banking_assistant import main as banking_main
except ImportError as e:
    st.error(f"Failed to import banking assistant: {e}")
    st.stop()

def main():
    banking_main()

if __name__ == "__main__":
    main()
