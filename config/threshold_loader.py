"""Threshold configuration loader."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def load_thresholds_from_json(filepath: str) -> Dict[str, float]:
    """Load confidence thresholds from JSON file.

    Args:
        filepath: Path to thresholds JSON file

    Returns:
        Dictionary mapping intent names to threshold values

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Thresholds file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle nested format (threshold + f1_score)
        if isinstance(next(iter(data.values())), dict):
            thresholds = {
                intent: values.get('threshold', 0.5)
                for intent, values in data.items()
            }
        else:
            thresholds = data

        logger.info(f"Loaded thresholds for {len(thresholds)} intents")
        return thresholds

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filepath}: {e}")


def save_thresholds_to_json(
    thresholds: Dict[str, float],
    filepath: str,
    include_metadata: bool = False,
    metadata: Optional[Dict[str, Dict]] = None,
    indent: int = 4
):
    """Save thresholds to JSON file.

    Args:
        thresholds: Dictionary of thresholds
        filepath: Path to save JSON file
        include_metadata: Whether to include metadata (f1_score, etc.)
        metadata: Optional metadata dictionary
        indent: Indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if include_metadata and metadata:
        data = metadata
    else:
        data = thresholds

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.info(f"Saved thresholds to {filepath}")


def validate_thresholds(
    thresholds: Dict[str, float],
    min_value: float = 0.0,
    max_value: float = 1.0
) -> bool:
    """Validate threshold values.

    Args:
        thresholds: Dictionary of thresholds
        min_value: Minimum valid threshold
        max_value: Maximum valid threshold

    Returns:
        True if all thresholds are valid
    """
    invalid_thresholds = []

    for intent, threshold in thresholds.items():
        if not isinstance(threshold, (int, float)):
            logger.error(f"Invalid threshold type for {intent}: {type(threshold)}")
            return False

        if not (min_value <= threshold <= max_value):
            invalid_thresholds.append((intent, threshold))

    if invalid_thresholds:
        logger.warning(f"Found {len(invalid_thresholds)} invalid thresholds")
        for intent, threshold in invalid_thresholds[:5]:
            logger.warning(f"  {intent}: {threshold}")
        return False

    logger.info("All thresholds are valid")
    return True


def merge_thresholds(
    default_thresholds: Dict[str, float],
    custom_thresholds: Dict[str, float]
) -> Dict[str, float]:
    """Merge custom thresholds with default thresholds.

    Args:
        default_thresholds: Default threshold values
        custom_thresholds: Custom threshold values (override defaults)

    Returns:
        Merged threshold dictionary
    """
    merged = default_thresholds.copy()
    merged.update(custom_thresholds)

    logger.info(
        f"Merged thresholds: {len(default_thresholds)} defaults + "
        f"{len(custom_thresholds)} custom = {len(merged)} total"
    )

    return merged


def get_default_thresholds(
    intent_list: list,
    default_value: float = 0.5
) -> Dict[str, float]:
    """Generate default thresholds for a list of intents.

    Args:
        intent_list: List of intent names
        default_value: Default threshold value

    Returns:
        Dictionary of default thresholds
    """
    return {intent: default_value for intent in intent_list}
