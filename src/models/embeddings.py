"""Sentence embedding models for intent classification."""

import logging
from typing import Optional, Dict
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceEmbedder:
    """Wrapper for sentence embedding models following DRY principle."""

    def __init__(
        self,
        model_name: str = 'intfloat/e5-base',
        device: Optional[str] = None
    ):
        """Initialize sentence embedder.

        Args:
            model_name: Name of the sentence transformer model
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()
        logger.info(f"Loaded embedding model: {model_name}")

    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model.

        Returns:
            Loaded SentenceTransformer model
        """
        return SentenceTransformer(self.model_name, device=self.device)

    def get_embedding(self, text: str, prepend_query: bool = False) -> np.ndarray:
        """Generate embedding for a given text.

        Args:
            text: Input text string
            prepend_query: Whether to prepend 'query: ' to text (for e5 models)
                          Default is False for consistency across training/inference.
                          
                          NOTE: Old notebook had a bug - trained WITHOUT prefix but
                          used WITH prefix in DS inference, causing train/test mismatch.
                          This implementation is consistent (no prefix everywhere).

        Returns:
            Embedding vector as numpy array
        """
        if prepend_query and 'e5' in self.model_name.lower():
            formatted_text = f"query: {text.strip()}"
        else:
            formatted_text = text.strip()

        return self.model.encode(formatted_text, show_progress_bar=False)

    def get_embeddings_batch(
        self,
        texts: list,
        batch_size: int = 64,
        show_progress: bool = True,
        prepend_query: bool = False  # Default False to match old notebook
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            prepend_query: Whether to prepend 'query: ' to texts

        Returns:
            Array of embedding vectors
        """
        if prepend_query and 'e5' in self.model_name.lower():
            formatted_texts = [f"query: {text.strip()}" for text in texts]
        else:
            formatted_texts = [text.strip() for text in texts]

        return self.model.encode(
            formatted_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )

    def save(self, path: str):
        """Save embedder configuration.
        
        Note: SentenceTransformer models are cached, so we only save the model name.
        
        Args:
            path: Path to save configuration
        """
        import json
        from pathlib import Path
        
        config = {
            'model_name': self.model_name,
            'device': self.device
        }
        
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved embedder config to {path}")

    @classmethod
    def load(cls, path: str):
        """Load embedder from configuration.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Loaded SentenceEmbedder instance
        """
        import json
        from pathlib import Path
        
        path = Path(path)
        with open(path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded embedder config from {path}")
        return cls(
            model_name=config['model_name'],
            device=config.get('device')
        )


class IntentEmbeddings:
    """Manage embeddings for all intents in the hierarchy."""

    def __init__(
        self,
        intents: Dict[str, str],
        embedder: Optional[SentenceEmbedder] = None,
        model_name: str = 'intfloat/e5-base'
    ):
        """Initialize intent embeddings.

        Args:
            intents: Dictionary mapping intent names to descriptions
            embedder: Optional pre-initialized embedder
            model_name: Model name if creating new embedder
        """
        self.intents = intents
        self.embedder = embedder or SentenceEmbedder(model_name)
        self.intent_embeddings = self._compute_intent_embeddings()
        logger.info(f"Computed embeddings for {len(self.intent_embeddings)} intents")

    def _compute_intent_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for all intents.

        Returns:
            Dictionary mapping intent names to embeddings
        """
        embeddings = {}
        for intent, description in self.intents.items():
            # Combine intent name and description for richer representation
            combined_text = f"{intent}: {description}"
            embeddings[intent] = self.embedder.get_embedding(combined_text)
        return embeddings

    def get_embedding(self, intent: str) -> Optional[np.ndarray]:
        """Get embedding for a specific intent.

        Args:
            intent: Intent name

        Returns:
            Embedding vector or None if intent not found
        """
        return self.intent_embeddings.get(intent)

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all intent embeddings.

        Returns:
            Dictionary of all intent embeddings
        """
        return self.intent_embeddings

    def add_intent(self, intent: str, description: str):
        """Add a new intent and compute its embedding.

        Args:
            intent: Intent name
            description: Intent description
        """
        self.intents[intent] = description
        combined_text = f"{intent}: {description}"
        self.intent_embeddings[intent] = self.embedder.get_embedding(combined_text)
        logger.info(f"Added new intent: {intent}")

    def remove_intent(self, intent: str):
        """Remove an intent and its embedding.

        Args:
            intent: Intent name to remove
        """
        if intent in self.intents:
            del self.intents[intent]
        if intent in self.intent_embeddings:
            del self.intent_embeddings[intent]
            logger.info(f"Removed intent: {intent}")
