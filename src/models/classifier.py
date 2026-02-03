"""Intent classifier wrapper for various ML models."""

import logging
import pickle
from typing import Optional, Union
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Wrapper for intent classification models following OOP principles."""

    def __init__(
        self,
        classifier_type: str = "logistic",
        **kwargs
    ):
        """Initialize intent classifier.

        Args:
            classifier_type: Type of classifier ('logistic' or 'svm')
            **kwargs: Additional arguments for the classifier
        """
        self.classifier_type = classifier_type.lower()
        self.model = self._create_classifier(**kwargs)
        self.is_trained = False
        logger.info(f"Initialized {classifier_type} classifier")

    def _create_classifier(self, **kwargs):
        """Create classifier based on type.

        Args:
            **kwargs: Classifier-specific parameters

        Returns:
            Initialized classifier

        Raises:
            ValueError: If classifier type is unknown
        """
        if self.classifier_type == "logistic":
            default_params = {
                'multi_class': 'multinomial',
                'max_iter': 1000,
                'random_state': 42
            }
            default_params.update(kwargs)
            return LogisticRegression(**default_params)

        elif self.classifier_type == "svm":
            default_params = {
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            }
            default_params.update(kwargs)
            return SVC(**default_params)

        else:
            raise ValueError(
                f"Unknown classifier type: {self.classifier_type}. "
                f"Use 'logistic' or 'svm'"
            )

    def train(
        self,
        embeddings: np.ndarray,
        labels: Union[list, np.ndarray]
    ):
        """Train the classifier.

        Args:
            embeddings: Training embeddings (n_samples, embedding_dim)
            labels: Training labels (n_samples,)
        """
        logger.info(
            f"Training {self.classifier_type} classifier on "
            f"{len(labels)} samples"
        )
        self.model.fit(embeddings, labels)
        self.is_trained = True
        logger.info("Training completed")

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict intent labels.

        Args:
            embeddings: Input embeddings (n_samples, embedding_dim)

        Returns:
            Predicted labels

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict intent probabilities.

        Args:
            embeddings: Input embeddings (n_samples, embedding_dim)

        Returns:
            Probability distributions over intents

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(embeddings)

    def get_classes(self) -> np.ndarray:
        """Get classifier classes.

        Returns:
            Array of class labels

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.classes_

    def save(self, filepath: Union[str, Path]):
        """Save trained model to disk.

        Args:
            filepath: Path to save the model

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Union[str, Path]):
        """Load trained model from disk.

        Args:
            filepath: Path to load the model from

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)

        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

    @classmethod
    def from_pretrained(
        cls,
        filepath: Union[str, Path],
        classifier_type: str = "logistic"
    ) -> 'IntentClassifier':
        """Load a pretrained classifier.

        Args:
            filepath: Path to the saved model
            classifier_type: Type of classifier

        Returns:
            Loaded IntentClassifier instance
        """
        classifier = cls(classifier_type=classifier_type)
        classifier.load(filepath)
        return classifier
