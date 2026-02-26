"""Dataset configuration definitions."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

    name: str
    huggingface_path: str
    subset: Optional[str] = None
    text_field: str = "text"
    label_field: str = "label"
    has_oos: bool = False
    oos_label: Optional[str] = None
    custom_loader: Optional[str] = None
    preprocessing_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.preprocessing_args is None:
            self.preprocessing_args = {}


# Predefined dataset configurations
DATASET_CONFIGS = {
    "banking77": DatasetConfig(
        name="banking77",
        huggingface_path="banking77",
        subset=None,
        text_field="text",
        label_field="label",
        has_oos=False,
        oos_label=None,
    ),
    "clinc150": DatasetConfig(
        name="clinc150",
        huggingface_path="clinc_oos",
        subset="plus",
        text_field="text",
        label_field="intent",
        has_oos=True,
        oos_label="oos",
    ),
    "snips": DatasetConfig(
        name="snips",
        huggingface_path="snips_built_in_intents",
        subset=None,
        text_field="text",
        label_field="label",
        has_oos=False,
        oos_label=None,
    ),
    "atis": DatasetConfig(
        name="atis",
        huggingface_path="tuetschek/atis",
        subset=None,
        text_field="text",
        label_field="intent",
        has_oos=False,
        oos_label=None,
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetConfig object

    Raises:
        ValueError: If dataset name is not recognized
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]
