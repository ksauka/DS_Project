"""Hierarchy configuration loader."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_hierarchy_from_json(filepath: str) -> Dict[str, List[str]]:
    """Load hierarchy from JSON file.

    Args:
        filepath: Path to hierarchy JSON file

    Returns:
        Hierarchy dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Hierarchy file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            hierarchy = json.load(f)

        logger.info(f"Loaded hierarchy from {filepath}")
        return hierarchy

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filepath}: {e}")


def load_hierarchical_intents_from_json(filepath: str) -> Dict[str, str]:
    """Load hierarchical intent descriptions from JSON file.

    Args:
        filepath: Path to hierarchical intents JSON file

    Returns:
        Dictionary mapping intent names to descriptions

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Hierarchical intents file not found: {filepath}"
        )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            hierarchical_intents = json.load(f)

        logger.info(f"Loaded hierarchical intents from {filepath}")
        return hierarchical_intents

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filepath}: {e}")


def save_hierarchy_to_json(
    hierarchy: Dict[str, List[str]],
    filepath: str,
    indent: int = 4
):
    """Save hierarchy to JSON file.

    Args:
        hierarchy: Hierarchy dictionary
        filepath: Path to save JSON file
        indent: Indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=indent, ensure_ascii=False)

    logger.info(f"Saved hierarchy to {filepath}")


def validate_hierarchy(hierarchy: Dict[str, List[str]]) -> bool:
    """Validate hierarchy structure.

    Args:
        hierarchy: Hierarchy dictionary

    Returns:
        True if valid, False otherwise
    """
    # Check if all children exist as keys (leaves should have empty lists)
    all_nodes = set(hierarchy.keys())
    all_children = set()

    for children in hierarchy.values():
        if not isinstance(children, list):
            logger.error("Hierarchy values must be lists")
            return False
        all_children.update(children)

    # All children should either be keys themselves or be leaf nodes
    for child in all_children:
        if child not in all_nodes:
            logger.warning(f"Child '{child}' not found as key in hierarchy")

    logger.info("Hierarchy validation passed")
    return True


def get_leaf_nodes(hierarchy: Dict[str, List[str]]) -> List[str]:
    """Get all leaf nodes from hierarchy.

    Args:
        hierarchy: Hierarchy dictionary

    Returns:
        List of leaf node names
    """
    return [node for node, children in hierarchy.items() if not children]


def get_parent_nodes(hierarchy: Dict[str, List[str]]) -> List[str]:
    """Get all parent (non-leaf) nodes from hierarchy.

    Args:
        hierarchy: Hierarchy dictionary

    Returns:
        List of parent node names
    """
    return [node for node, children in hierarchy.items() if children]


def get_node_depth(hierarchy: Dict[str, List[str]], node: str) -> int:
    """Calculate depth of a node in hierarchy.

    Args:
        hierarchy: Hierarchy dictionary
        node: Node name

    Returns:
        Depth of node (0 for root-level nodes)
    """
    depth = 0
    current = node

    while current in hierarchy and hierarchy[current]:
        depth += 1
        current = hierarchy[current][0]

    return depth
