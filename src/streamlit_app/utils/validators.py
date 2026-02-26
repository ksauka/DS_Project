"""Data validation utilities."""

import streamlit as st
from typing import Dict, List, Optional, Tuple
import json

def validate_hierarchy(hierarchy: Dict) -> Tuple[bool, str]:
    """
    Validate hierarchy structure.
    
    Args:
        hierarchy: Hierarchy dictionary
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    if not isinstance(hierarchy, dict):
        return False, "Hierarchy must be a dictionary"
    
    if not hierarchy:
        return False, "Hierarchy is empty"
    
    # Check for valid structure (parent -> children)
    for parent, children in hierarchy.items():
        if not isinstance(children, list):
            return False, f"Children of '{parent}' must be a list"
        
        for child in children:
            if not isinstance(child, str):
                return False, f"Child '{child}' must be a string"
    
    return True, ""

def validate_thresholds(thresholds: Dict) -> Tuple[bool, str]:
    """
    Validate threshold configuration.
    
    Args:
        thresholds: Thresholds dictionary
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    if not isinstance(thresholds, dict):
        return False, "Thresholds must be a dictionary"
    
    for intent, threshold in thresholds.items():
        if not isinstance(intent, str):
            return False, f"Intent name must be string, got {type(intent)}"
        
        try:
            threshold_val = float(threshold)
            if not 0.0 <= threshold_val <= 1.0:
                return False, f"Threshold for '{intent}' must be between 0.0 and 1.0"
        except (TypeError, ValueError):
            return False, f"Threshold for '{intent}' must be numeric"
    
    return True, ""

def validate_json_file(file_obj) -> Tuple[bool, str, Optional[Dict]]:
    """
    Validate and parse JSON file.
    
    Args:
        file_obj: File object from st.file_uploader
    
    Returns:
        Tuple of (is_valid, message, parsed_data)
    """
    
    try:
        data = json.load(file_obj)
        return True, "Valid JSON", data
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", None
    except Exception as e:
        return False, f"Error reading file: {e}", None

def validate_dataset_name(dataset: str) -> Tuple[bool, str]:
    """
    Validate dataset name.
    
    Args:
        dataset: Dataset name
    
    Returns:
        Tuple of (is_valid, message)
    """
    
    valid_datasets = ['banking77', 'clinc150', 'snips', 'atis', 'topv2']
    
    if dataset.lower() not in valid_datasets:
        return False, f"Dataset must be one of {valid_datasets}"
    
    return True, ""

def validate_classifier_config(config: Dict) -> Tuple[bool, str]:
    """
    Validate classifier configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (is_valid, message)
    """
    
    required_fields = ['classifier_type', 'embedding_model']
    
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"
    
    valid_classifiers = ['logistic', 'svm']
    if config['classifier_type'].lower() not in valid_classifiers:
        return False, f"Classifier must be one of {valid_classifiers}"
    
    return True, ""

def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> Tuple[bool, str]:
    """
    Validate user query.
    
    Args:
        query: Query text
        min_length: Minimum length
        max_length: Maximum length
    
    Returns:
        Tuple of (is_valid, message)
    """
    
    query = query.strip()
    
    if not query:
        return False, "Query cannot be empty"
    
    if len(query) < min_length:
        return False, f"Query must be at least {min_length} characters"
    
    if len(query) > max_length:
        return False, f"Query must be at most {max_length} characters"
    
    return True, ""
