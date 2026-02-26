"""Session management for Streamlit app."""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, Any

class StreamlitSessionManager:
    """Manages session state persistence and recovery."""
    
    def __init__(self, session_dir: Path = None):
        """
        Initialize session manager.
        
        Args:
            session_dir: Directory to store sessions (default: outputs/sessions)
        """
        self.session_dir = session_dir or Path("outputs/sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, session_id: str, data: Dict[str, Any]) -> Path:
        """
        Save session to JSON file.
        
        Args:
            session_id: Unique session identifier
            data: Session data to save
        
        Returns:
            Path to saved session file
        """
        session_file = self.session_dir / f"{session_id}.json"
        
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        return session_file
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session from JSON file.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            Session data or None if not found
        """
        session_file = self.session_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        return session_data.get('data')
    
    def list_sessions(self) -> list:
        """
        List all available sessions.
        
        Returns:
            List of session IDs
        """
        return [f.stem for f in self.session_dir.glob("*.json")]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session to delete
        
        Returns:
            True if deleted, False if not found
        """
        session_file = self.session_dir / f"{session_id}.json"
        
        if session_file.exists():
            session_file.unlink()
            return True
        
        return False
    
    def export_session(self, session_id: str) -> str:
        """
        Export session as JSON string.
        
        Args:
            session_id: Session to export
        
        Returns:
            JSON string
        """
        data = self.load_session(session_id)
        return json.dumps(data, indent=2, default=str) if data else ""

def initialize_session():
    """Initialize Streamlit session state."""
    
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = StreamlitSessionManager()
    
    if 'sessions_list' not in st.session_state:
        st.session_state.sessions_list = st.session_state.session_manager.list_sessions()

def save_current_session():
    """Save current session to disk."""
    
    session_manager: StreamlitSessionManager = st.session_state.session_manager
    session_id = st.session_state.current_session_id
    
    session_data = {
        'dataset': st.session_state.current_dataset,
        'model_loaded': st.session_state.model_loaded,
        'conversation_history': st.session_state.conversation_history,
        'belief_history': st.session_state.belief_history,
    }
    
    session_manager.save_session(session_id, session_data)

def load_session(session_id: str):
    """Load a session."""
    
    session_manager: StreamlitSessionManager = st.session_state.session_manager
    data = session_manager.load_session(session_id)
    
    if data:
        st.session_state.current_dataset = data.get('current_dataset', 'banking77')
        st.session_state.model_loaded = data.get('model_loaded', False)
        st.session_state.conversation_history = data.get('conversation_history', [])
        st.session_state.belief_history = data.get('belief_history', [])
        st.session_state.current_session_id = session_id
        return True
    
    return False
