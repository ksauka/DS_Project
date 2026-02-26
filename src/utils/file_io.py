"""File I/O utilities following DRY principle."""

import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

logger = logging.getLogger(__name__)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from {filepath}")
    return data


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 4):
    """Save dictionary as JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
        indent: Indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.debug(f"Saved JSON to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    logger.debug(f"Loaded pickle from {filepath}")
    return data


def save_pickle(obj: Any, filepath: Union[str, Path]):
    """Save object as pickle file.

    Args:
        obj: Object to save
        filepath: Path to save pickle file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    logger.debug(f"Saved pickle to {filepath}")


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load CSV file as DataFrame.

    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, **kwargs)
    logger.debug(f"Loaded CSV from {filepath} ({len(df)} rows)")
    return df


def save_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs):
    """Save DataFrame as CSV file.

    Args:
        df: DataFrame to save
        filepath: Path to save CSV file
        **kwargs: Additional arguments for df.to_csv
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, **kwargs)
    logger.debug(f"Saved CSV to {filepath} ({len(df)} rows)")


def ensure_dir(dirpath: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        dirpath: Directory path

    Returns:
        Path object
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> list:
    """List files in directory matching pattern.

    Args:
        directory: Directory path
        pattern: Glob pattern
        recursive: Whether to search recursively

    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    return [f for f in files if f.is_file()]
