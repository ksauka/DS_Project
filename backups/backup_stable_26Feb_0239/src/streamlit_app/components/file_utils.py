"""File utilities for Streamlit app."""

import streamlit as st
import json
from pathlib import Path
from typing import Optional, Dict, Any

def save_uploaded_hierarchy(uploaded_file) -> Optional[Path]:
    """
    Save uploaded hierarchy JSON file to config/hierarchies/.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Path to saved file or None if error
    """
    
    try:
        # Validate JSON
        hierarchy = json.load(uploaded_file)
        
        # Save to config directory
        config_dir = Path("config/hierarchies")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        filename = uploaded_file.name
        filepath = config_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(hierarchy, f, indent=2)
        
        return filepath
    
    except json.JSONDecodeError:
        st.error("Invalid JSON file")
        return None
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def save_uploaded_config(uploaded_file, config_type: str) -> Optional[Path]:
    """
    Save uploaded configuration file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        config_type: Type of config ('hierarchy', 'intents', 'thresholds')
    
    Returns:
        Path to saved file or None if error
    """
    
    config_dirs = {
        'hierarchy': Path("config/hierarchies"),
        'intents': Path("config/hierarchies"),
        'thresholds': Path("config/thresholds")
    }
    
    try:
        config_dir = config_dirs.get(config_type, Path("config"))
        config_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = config_dir / uploaded_file.name
        
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return filepath
    
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def load_json_file(filepath: Path) -> Optional[Dict]:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Parsed JSON or None if error
    """
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def download_session_as_json(session_data: Dict) -> str:
    """
    Convert session data to downloadable JSON.
    
    Args:
        session_data: Session dictionary
    
    Returns:
        JSON string
    """
    
    return json.dumps(session_data, indent=2, default=str)

def list_config_files(config_type: str) -> list:
    """
    List available configuration files.
    
    Args:
        config_type: Type of config to list
    
    Returns:
        List of config file paths
    """
    
    config_dirs = {
        'hierarchy': Path("config/hierarchies"),
        'intents': Path("config/hierarchies"),
        'thresholds': Path("config/thresholds")
    }
    
    config_dir = config_dirs.get(config_type)
    
    if not config_dir or not config_dir.exists():
        return []
    
    return sorted(list(config_dir.glob("*.json")))

def get_model_paths(dataset: str) -> Optional[Dict[str, Path]]:
    """
    Get paths to trained models for a dataset.
    
    Args:
        dataset: Dataset name
    
    Returns:
        Dict with 'model' and 'metadata' paths
    """
    
    exp_dir = Path(f"experiments/{dataset}")
    
    if not exp_dir.exists():
        return None
    
    # Find latest trained model
    model_files = sorted(exp_dir.glob("*/model.pkl"), reverse=True)
    
    if not model_files:
        return None
    
    model_path = model_files[0]
    metadata_path = model_path.parent / "metadata.json"
    
    return {
        'model': model_path,
        'metadata': metadata_path if metadata_path.exists() else None
    }
