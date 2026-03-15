"""
DS Project - CLINC150 HicXAI Study App — Slice 4 of 4

100% CLINC150 queries, HicXAI system.
Participants in this app see queries 13–16 (small), 25–32 (medium), or 37–46 (large).
Deploy as its own Streamlit Cloud app; set STUDY_SET in Streamlit secrets.

Note: large set has 46 queries total (not 48) because the problematic_level1
group only has 10 available CLINC150 queries. Slice 4 will have 10 queries for large.

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

os.environ['STUDY_SET_DIR'] = 'outputs/user_study/study_clinc150only'
os.environ['STUDY_SET_DROPBOX_FOLDER'] = '/ds_project_queries/study_clinc150only'
os.environ['QUERY_SLICE'] = '3'
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
