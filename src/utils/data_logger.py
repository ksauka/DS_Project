"""Data Logger for DS_Project User Studies

Tracks user interactions, query results, and feedback metrics.
Saves to private GitHub repository for centralized analysis.
"""

import json
import os
from datetime import datetime, date
from typing import Optional, Dict, Any, List


class _SafeEncoder(json.JSONEncoder):
    """Handle numpy scalars, datetimes and other non-serializable types."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

from pathlib import Path
import streamlit as st

from .github_saver import save_to_github


class DataLogger:
    """Logs user study sessions and saves to private GitHub repository"""
    
    def __init__(self, participant_id: str, condition: str, session_id: str):
        """
        Initialize data logger.
        
        Args:
            participant_id: Unique participant identifier (pid or PROLIFIC_PID)
            condition: Experimental condition (e.g., 'control', 'experimental')
            session_id: Unique session identifier
        """
        self.participant_id = participant_id
        self.condition = condition
        self.session_id = session_id
        self.session_start = datetime.now().isoformat()
        
        # Data collections
        self.query_results: List[Dict] = []
        self.final_feedback: Optional[Dict] = None
        
        # Behavior metrics
        self.behavior_metrics = {
            "total_queries": 0,
            "total_clarifications": 0,
            "total_why_questions": 0,
            "total_interaction_time": 0.0,
            "queries_correct": 0,
            "queries_incorrect": 0
        }
    
    def log_query_result(self, result: Dict[str, Any]):
        """
        Log a completed query interaction.
        
        Args:
            result: Query result dictionary with prediction, confidence, feedback, etc.
        """
        self.query_results.append(result)
        self.behavior_metrics["total_queries"] += 1
        
        if result.get('is_correct', False):
            self.behavior_metrics["queries_correct"] += 1
        else:
            self.behavior_metrics["queries_incorrect"] += 1
        
        self.behavior_metrics["total_clarifications"] += result.get('num_clarification_turns', 0)
        self.behavior_metrics["total_interaction_time"] += result.get('interaction_time_seconds', 0)
    
    def log_why_question(self):
        """Log when user asks 'why' for explanation"""
        self.behavior_metrics["total_why_questions"] += 1
    
    def set_final_feedback(self, feedback: Dict[str, Any]):
        """
        Set final survey feedback.
        
        Args:
            feedback: Final feedback dictionary
        """
        self.final_feedback = feedback
    
    def build_final_data(self) -> Dict[str, Any]:
        """
        Build complete data structure for saving.
        
        Returns:
            Complete session data dictionary
        """
        session_end = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(self.session_start)
        end_dt = datetime.fromisoformat(session_end)
        duration_seconds = (end_dt - start_dt).total_seconds()
        
        # Use either directly-logged query results or the ones embedded in final_feedback
        qr = self.query_results or (self.final_feedback.get('query_results', []) if self.final_feedback else [])
        n = len(qr)

        # Calculate summary statistics
        if qr:
            queries_correct = sum(1 for r in qr if r.get('is_correct', False))
            queries_incorrect = n - queries_correct
            accuracy = queries_correct / n
            avg_clarifications = sum(r.get('num_clarification_turns', 0) for r in qr) / n
            avg_time = sum(r.get('interaction_time_seconds', 0) for r in qr) / n
            total_clarifications = sum(r.get('num_clarification_turns', 0) for r in qr)
            total_time = sum(r.get('interaction_time_seconds', 0) for r in qr)

            clarity_ratings = [r['feedback_clarity'] for r in qr if r.get('feedback_clarity') is not None]
            confidence_ratings = [r['feedback_confidence'] for r in qr if r.get('feedback_confidence') is not None]
            avg_clarity = sum(clarity_ratings) / len(clarity_ratings) if clarity_ratings else None
            avg_confidence = sum(confidence_ratings) / len(confidence_ratings) if confidence_ratings else None
        else:
            queries_correct = 0
            queries_incorrect = 0
            accuracy = 0
            avg_clarifications = 0
            avg_time = 0
            total_clarifications = 0
            total_time = 0.0
            avg_clarity = None
            avg_confidence = None
        
        # Derive dataset & system from STUDY_SET_DIR env var
        study_dir = os.getenv('STUDY_SET_DIR', 'study_b77only')
        if 'clinc150' in study_dir:
            dataset_name = 'clinc150'
            system_name = 'HicXAI'
        elif 'b77' in study_dir:
            dataset_name = 'banking77'
            system_name = 'DS_hierarchical_intent_classification'
        else:
            dataset_name = 'mixed'
            system_name = 'DS_hierarchical_intent_classification'

        return {
            "metadata": {
                "participant_id": self.participant_id,
                "condition": self.condition,
                "session_id": self.session_id,
                "session_start": self.session_start,
                "session_end": session_end,
                "duration_seconds": duration_seconds,
                "dataset": dataset_name,
                "system": system_name
            },
            "summary_statistics": {
                "total_queries": n,
                "queries_correct": queries_correct,
                "queries_incorrect": queries_incorrect,
                "accuracy": accuracy,
                "total_clarifications": total_clarifications,
                "avg_clarifications_per_query": avg_clarifications,
                "total_why_questions": self.behavior_metrics["total_why_questions"],
                "total_interaction_time_seconds": total_time,
                "avg_time_per_query_seconds": avg_time,
                "avg_feedback_clarity": avg_clarity,
                "avg_feedback_confidence": avg_confidence
            },
            "query_results": qr,
            "final_feedback": self.final_feedback,
            "behavior_metrics": self.behavior_metrics
        }
    
    def save_to_github(self, repo: str, github_token: str) -> bool:
        """
        Save session data to private GitHub repository.
        
        Args:
            repo: GitHub repository (e.g., 'ksauka/hicxai-data-private')
            github_token: GitHub personal access token
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build file path: sessions/{study_type}/{date}/{participant_id}_{condition}_{timestamp}.json
            study_dir = os.getenv('STUDY_SET_DIR', 'study_b77only')
            if 'clinc150' in study_dir:
                study_folder = 'clinc150'
            elif 'b77' in study_dir:
                study_folder = 'b77'
            else:
                study_folder = 'mixed'
            # QUERY_SLICE is 0-based (set per app); store as app_1..app_4
            slice_idx = int(os.getenv('QUERY_SLICE', '0'))
            app_folder = f"app_{slice_idx + 1}"
            date_str = datetime.now().strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sessions/{study_folder}/{app_folder}/{date_str}/{self.participant_id}_{self.condition}_{timestamp}.json"
            
            # Build data
            data = self.build_final_data()
            content = json.dumps(data, indent=2, cls=_SafeEncoder)
            
            # Save to GitHub
            commit_message = f"Session data: {self.participant_id} condition {self.condition} - {self.behavior_metrics['total_queries']} queries"
            
            success, error_message = save_to_github(
                repo=repo,
                path=filename,
                content=content,
                commit_message=commit_message,
                github_token=github_token
            )
            
            if success:
                print(f"✅ Session saved to GitHub: {filename}")
            else:
                if error_message:
                    st.session_state["github_save_error"] = error_message
                # Fallback to local
                print("⚠️ GitHub save failed, falling back to local save")
                self._save_local()
            
            return success
        except Exception as e:
            st.session_state["github_save_error"] = f"GitHub save error: {e}"
            print(f"❌ GitHub save error: {e}")
            return self._save_local()
    
    def _save_local(self) -> bool:
        """
        Fallback: Save to local file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory structure
            data_dir = Path("data/sessions")
            date_str = datetime.now().strftime('%Y-%m-%d')
            session_dir = data_dir / date_str
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Build filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = session_dir / f"{self.participant_id}_{self.condition}_{timestamp}.json"
            
            # Build and save data
            data = self.build_final_data()
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, cls=_SafeEncoder)
            
            print(f"✅ Session saved locally: {filename}")
            return True
        except Exception as e:
            print(f"❌ Local save failed: {e}")
            return False


def init_logger() -> Optional[DataLogger]:
    """
    Initialize data logger from session state and URL parameters.
    
    Returns:
        DataLogger instance or None if initialization fails
    """
    # Return existing logger if already initialized
    if "data_logger" in st.session_state:
        return st.session_state.data_logger
    
    try:
        # Extract participant info from session state
        participant_id = st.session_state.get("pid") or st.session_state.get("prolific_pid", "unknown")
        condition = st.session_state.get("cond", "default")
        session_id = st.session_state.get("session_id", "unknown")
        
        # Create logger
        logger = DataLogger(participant_id, condition, session_id)
        st.session_state.data_logger = logger
        
        print(f"✅ Data logger initialized: {participant_id} | {condition} | {session_id}")
        return logger
    except Exception as e:
        print(f"❌ Failed to initialize logger: {e}")
        return None


def _normalize_github_repo(repo: str) -> str:
    """
    Normalize GitHub repo URL to username/repo format.
    
    Args:
        repo: Repository in various formats:
            - 'username/repo'
            - 'https://github.com/username/repo'
            - 'https://github.com/username/repo.git'
    
    Returns:
        Normalized 'username/repo' format
    """
    if not repo:
        return repo
    
    # Already in username/repo format
    if '/' in repo and not repo.startswith('http'):
        return repo
    
    # Extract from URL
    if 'github.com' in repo:
        # Remove .git suffix if present
        if repo.endswith('.git'):
            repo = repo[:-4]
        # Extract username/repo from URL
        parts = repo.split('github.com/')[-1]
        return parts
    
    return repo


def _get_secret_value(*keys: str):
    """Look up a secret from Streamlit using common flat and sectioned layouts."""
    try:
        secrets = st.secrets
    except Exception:
        return None

    for key in keys:
        if not key:
            continue

        try:
            value = secrets.get(key)
            if value:
                return value
        except Exception:
            pass

        # Support sectioned secrets such as:
        # [github]
        # token = "..."
        # repo = "owner/name"
        if "." in key:
            section_name, nested_key = key.split(".", 1)
            section_names = (section_name, section_name.lower(), section_name.upper())
            nested_names = (nested_key, nested_key.lower(), nested_key.upper())
            for section in section_names:
                try:
                    section_obj = secrets.get(section)
                except Exception:
                    section_obj = None
                if not section_obj:
                    continue
                for nested in nested_names:
                    try:
                        value = section_obj.get(nested)
                    except Exception:
                        value = None
                    if value:
                        return value

    return None


def save_session_to_github():
    """
    Save current session to GitHub repository.
    
    Retrieves credentials from Streamlit secrets or environment variables.
    Falls back to local save if GitHub is not configured.
    
    Note: Session data goes to GitHub only. Dropbox is used for model files.
    """
    logger = st.session_state.get("data_logger")
    if not logger:
        print("⚠️ No data logger found in session")
        return False
    
    st.session_state["github_save_error"] = None

    # Match AnthroKit's simpler deployment model: the data repo is fixed.
    repo = "ksauka/hicxai-data-private"
    github_token = _get_secret_value("GITHUB_DATA_TOKEN", "GITHUB_TOKEN", "github.token")
    if not github_token:
        github_token = os.getenv("GITHUB_DATA_TOKEN") or os.getenv("GITHUB_TOKEN")
    
    print(f"🔑 Secrets check — repo=SET ({repo}), token={'SET' if github_token else 'MISSING'}")
    if github_token:
        print(f"📤 Attempting to save to GitHub repo: {repo}")
        result = logger.save_to_github(repo, github_token)
        print(f"📤 save_to_github result: {result}")
        if not result and not st.session_state.get("github_save_error"):
            st.session_state["github_save_error"] = (
                f"GitHub token was found, but the upload to `{repo}` failed. "
                "Check that the token is valid and that the repo is reachable."
            )
        return result
    else:
        print("⚠️ GitHub not configured: token={}".format('SET' if github_token else 'MISSING'))
        st.session_state["github_save_error"] = (
            "Missing GitHub token in Streamlit secrets. "
            "Expected `GITHUB_TOKEN` or `GITHUB_DATA_TOKEN`."
        )
        return False
