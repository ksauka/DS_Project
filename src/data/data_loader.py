"""Dataset loading utilities following DRY and FAIR principles."""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datasets import load_dataset, Dataset

from .dataset_config import DatasetConfig, get_dataset_config

logger = logging.getLogger(__name__)


class DataLoader:
    """Data-agnostic loader for intent classification datasets."""

    def __init__(self, dataset_name: str, config: Optional[DatasetConfig] = None):
        """Initialize DataLoader.

        Args:
            dataset_name: Name of the dataset to load
            config: Optional custom dataset configuration
        """
        self.dataset_name = dataset_name
        self.config = config or get_dataset_config(dataset_name)
        self.dataset = None
        self.full_intent_names = []
        self.intent_names = []
        self.index_to_name = {}
        self.name_to_index = {}

    def load(self) -> Dict[str, Dataset]:
        """Load dataset from HuggingFace or custom source.

        Returns:
            Dictionary with train/test/validation splits
        """
        logger.info(f"Loading dataset: {self.config.name}")

        try:
            # Load from HuggingFace
            if self.config.subset:
                self.dataset = load_dataset(
                    self.config.huggingface_path,
                    self.config.subset
                )
            else:
                self.dataset = load_dataset(self.config.huggingface_path)

            # Extract intent mappings
            self._extract_intent_mappings()

            logger.info(
                f"Loaded {self.config.name}: "
                f"{len(self.intent_names)} intents"
            )
            return self.dataset

        except Exception as e:
            logger.error(f"Error loading dataset {self.config.name}: {e}")
            raise

    def _extract_intent_mappings(self):
        """Extract intent name mappings from dataset."""
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Get label feature
        train_data = self.dataset.get("train", self.dataset.get("training"))
        if not train_data:
            raise ValueError("No training split found in dataset")

        label_feature = train_data.features.get(self.config.label_field)

        if hasattr(label_feature, 'names'):
            # ClassLabel feature
            self.full_intent_names = list(label_feature.names)
        elif hasattr(label_feature, '_int2str'):
            # Some datasets use _int2str
            self.full_intent_names = list(label_feature._int2str.values())
        else:
            # Fallback: extract unique labels
            labels = train_data[self.config.label_field]
            self.full_intent_names = sorted(list(set(labels)))

        # Filter out OOS if specified
        if self.config.has_oos and self.config.oos_label:
            self.intent_names = [
                name for name in self.full_intent_names
                if name != self.config.oos_label
            ]
        else:
            self.intent_names = list(self.full_intent_names)

        # Create mappings based on full list to preserve raw label indices
        self.index_to_name = {
            i: name for i, name in enumerate(self.full_intent_names)
        }
        self.name_to_index = {
            name: i for i, name in self.index_to_name.items()
        }

    def get_split_data(
        self,
        split: str = "test",
        filter_oos: bool = True
    ) -> Tuple[List[str], List[str], List[int]]:
        """Get data from a specific split.

        Args:
            split: Split name (train/test/validation)
            filter_oos: Whether to filter out-of-scope samples

        Returns:
            Tuple of (texts, intent_names, intent_indices)
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load() first.")

        split_data = self.dataset.get(split)
        if not split_data:
            raise ValueError(f"Split '{split}' not found in dataset")

        texts = split_data[self.config.text_field]
        labels = split_data[self.config.label_field]

        # Convert labels to intent names if needed
        if isinstance(labels[0], int):
            intent_names = [self.index_to_name.get(lbl, "unknown") for lbl in labels]
        else:
            intent_names = labels

        # Filter OOS if requested
        if filter_oos and self.config.has_oos:
            filtered_data = [
                (text, intent, idx)
                for text, intent, idx in zip(texts, intent_names, labels)
                if intent != self.config.oos_label
            ]
            if filtered_data:
                texts, intent_names, labels = zip(*filtered_data)
                texts = list(texts)
                intent_names = list(intent_names)
                labels = list(labels)

        return texts, intent_names, labels

    def get_formatted_data(
        self,
        split: str = "test"
    ) -> List[Dict[str, str]]:
        """Get formatted data as list of dictionaries.

        Args:
            split: Split name (train/test/validation)

        Returns:
            List of dictionaries with 'Example' and 'Label' keys
        """
        texts, intent_names, _ = self.get_split_data(split)
        return [
            {"Example": text, "Label": intent}
            for text, intent in zip(texts, intent_names)
        ]

    def get_pandas_dataframe(
        self,
        split: str = "test"
    ) -> pd.DataFrame:
        """Get data as pandas DataFrame.

        Args:
            split: Split name (train/test/validation)

        Returns:
            DataFrame with text and label columns
        """
        texts, intent_names, labels = self.get_split_data(split)
        return pd.DataFrame({
            "text": texts,
            "intent_name": intent_names,
            "intent_id": labels
        })

    def get_intent_names(self) -> List[str]:
        """Get list of all intent names.

        Returns:
            List of intent names
        """
        return self.intent_names

    def get_intent_mappings(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        """Get intent mapping dictionaries.

        Returns:
            Tuple of (index_to_name, name_to_index) dictionaries
        """
        return self.index_to_name, self.name_to_index


def load_banking77(split: str = "test") -> Tuple[List[str], List[str]]:
    """Convenience function to load Banking77 dataset.

    Args:
        split: Split name (train/test)

    Returns:
        Tuple of (texts, intent_names)
    """
    loader = DataLoader("banking77")
    loader.load()
    texts, intents, _ = loader.get_split_data(split)
    return texts, intents


def load_clinc150(
    split: str = "test",
    filter_oos: bool = True
) -> Tuple[List[str], List[str]]:
    """Convenience function to load CLINC150 dataset.

    Args:
        split: Split name (train/test/validation)
        filter_oos: Whether to filter out-of-scope samples

    Returns:
        Tuple of (texts, intent_names)
    """
    loader = DataLoader("clinc150")
    loader.load()
    texts, intents, _ = loader.get_split_data(split, filter_oos=filter_oos)
    return texts, intents
