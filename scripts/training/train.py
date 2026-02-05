"""Training script for intent classification models."""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader
from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
from src.models.classifier import IntentClassifier
from src.utils.file_io import save_json, save_pickle, ensure_dir
from config.hierarchy_loader import (
    load_hierarchy_from_json,
    load_hierarchical_intents_from_json
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    dataset: str,
    classifier_type: str = 'logistic',
    embedding_model: str = 'intfloat/e5-base',
    output_dir: Path = None,
    experiment_name: str = None,
    hierarchy_file: Path = None,
    intents_file: Path = None,
    max_iter: int = 1000,
    batch_size: int = 64,
    train_samples: int = None,
    eval_samples: int = None
):
    """
    Train an intent classification model.
    
    Args:
        dataset: Dataset name ('banking77', 'clinc150', 'snips', 'atis')
        classifier_type: Type of classifier ('logistic' or 'svm')
        embedding_model: Name of the sentence transformer model
        output_dir: Directory to save model and results
        experiment_name: Name for this experiment (uses timestamp if None)
        hierarchy_file: Path to hierarchy JSON file (optional)
        intents_file: Path to intents JSON file (optional)
        max_iter: Maximum iterations for logistic regression
        batch_size: Batch size for embedding generation
        train_samples: Number of training samples to use (None = all)
        eval_samples: Number of evaluation samples to use (None = all)
    
    Returns:
        dict: Training results including model paths and metrics
    """
    logger.info(f"Starting training for {dataset}")
    logger.info(f"Classifier: {classifier_type}")
    logger.info(f"Embedding model: {embedding_model}")

    # Set default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "experiments" / dataset
    output_dir = ensure_dir(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use experiment name if provided, otherwise use timestamp
    if experiment_name:
        run_name = experiment_name
        logger.info(f"Experiment name: {experiment_name}")
    else:
        run_name = f"{dataset}_{classifier_type}_{timestamp}"
        logger.info(f"Using timestamp-based run name: {run_name}")
    
    run_dir = ensure_dir(output_dir / run_name)

    # Load dataset
    logger.info("Loading dataset...")
    data_loader = DataLoader(dataset)
    data_loader.load()

    # Get training data
    train_texts, train_labels_names, train_indices = data_loader.get_split_data('train')
    
    # Limit training samples if requested
    if train_samples:
        train_texts = train_texts[:train_samples]
        train_labels_names = train_labels_names[:train_samples]
        train_indices = train_indices[:train_samples]
        logger.info(f"Using {train_samples} training samples")
    
    logger.info(f"Training samples: {len(train_texts)}")

    # Load hierarchy and intents if provided
    if hierarchy_file and intents_file:
        logger.info("Loading hierarchy and intent descriptions...")
        hierarchy = load_hierarchy_from_json(hierarchy_file)
        hierarchical_intents = load_hierarchical_intents_from_json(intents_file)
        logger.info("Using hierarchical training")
        train_labels = train_labels_names
    else:
        logger.info("No hierarchy provided. Training vanilla baseline with intent names (STRING labels)...")
        train_labels = train_labels_names
        hierarchy = None
        hierarchical_intents = None

    # Initialize embedder
    logger.info("Initializing sentence embedder...")
    embedder = SentenceEmbedder(model_name=embedding_model)

    # Generate embeddings for training data
    # NOTE: Using default prepend_query=False for consistency.
    # Old notebook: trained without prefix but used WITH prefix in DS inference (bug).
    # This implementation: no prefix everywhere (consistent).
    logger.info("Generating embeddings for training data...")
    train_embeddings = embedder.get_embeddings_batch(
        train_texts,
        batch_size=batch_size,
        show_progress=True
    )

    # Generate intent embeddings only if using hierarchy
    if hierarchical_intents:
        logger.info("Generating intent embeddings...")
        intent_embeddings = IntentEmbeddings(
            hierarchical_intents,
            embedder=embedder
        )
    else:
        logger.info("Skipping intent embeddings (vanilla baseline)")
        intent_embeddings = None

    # Train classifier
    logger.info("Training classifier...")
    if classifier_type == 'logistic':
        classifier = IntentClassifier(
            classifier_type='logistic',
            max_iter=max_iter
        )
    else:
        classifier = IntentClassifier(classifier_type='svm')

    classifier.train(train_embeddings, train_labels)

    # Save model
    model_path = output_dir / f"{dataset}_{classifier_type}_model.pkl"
    embedder_path = output_dir / f"{dataset}_embedder.json"
    
    logger.info(f"Saving model to {model_path}")
    classifier.save(model_path)
    
    logger.info(f"Saving embedder to {embedder_path}")
    embedder.save(embedder_path)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_texts, test_labels, test_indices = data_loader.get_split_data('test')
    
    # Limit eval samples if requested
    if eval_samples:
        test_texts = test_texts[:eval_samples]
        test_labels = test_labels[:eval_samples]
        test_indices = test_indices[:eval_samples]
        logger.info(f"Using {eval_samples} evaluation samples")
    
    test_embeddings = embedder.get_embeddings_batch(
        test_texts,
        batch_size=batch_size,
        show_progress=False
    )
    predictions = classifier.predict(test_embeddings)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np
    
    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(test_labels, predictions, average='macro', zero_division=0)
    
    # Compute confidence statistics
    probabilities = classifier.predict_proba(test_embeddings)
    confidences = np.max(probabilities, axis=1)
    correct_mask = predictions == test_labels
    
    avg_confidence = confidences.mean()
    avg_confidence_correct = confidences[correct_mask].mean() if correct_mask.sum() > 0 else 0
    avg_confidence_incorrect = confidences[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
    
    # Print formatted results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Model: {classifier_type.upper()} Classifier + BERT")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Samples: {len(test_labels)}")
    logger.info("-"*60)
    logger.info(f"Accuracy:           {accuracy:.4f}")
    logger.info(f"F1 (macro):         {f1_macro:.4f}")
    logger.info(f"F1 (weighted):      {f1_weighted:.4f}")
    logger.info(f"Precision (macro):  {precision:.4f}")
    logger.info(f"Recall (macro):     {recall:.4f}")
    logger.info("-"*60)
    logger.info(f"Avg Confidence:     {avg_confidence:.4f}")
    logger.info(f"  Correct:          {avg_confidence_correct:.4f}")
    logger.info(f"  Incorrect:        {avg_confidence_incorrect:.4f}")
    logger.info("-"*60)
    logger.info(f"Correct Predictions:   {correct_mask.sum()}")
    logger.info(f"Incorrect Predictions: {(~correct_mask).sum()}")
    logger.info("="*60)

    logger.info(f"\nTraining completed! Model saved to {output_dir}")
    
    # Return results
    return {
        'model_path': model_path,
        'embedder_path': embedder_path,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision': float(precision),
        'recall': float(recall),
        'avg_confidence': float(avg_confidence),
        'classifier': classifier,
        'embedder': embedder,
        'data_loader': data_loader
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train intent classification model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['banking77', 'clinc150', 'snips', 'atis'],
        help='Dataset name'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        default='logistic',
        choices=['logistic', 'svm'],
        help='Classifier type'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='intfloat/e5-base',
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--hierarchy-file',
        type=str,
        default=None,
        help='Path to hierarchy JSON file (optional, for hierarchical models)'
    )
    parser.add_argument(
        '--intents-file',
        type=str,
        default=None,
        help='Path to hierarchical intents JSON file (optional, for hierarchical models)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Name for this experiment run (default: uses timestamp)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=1000,
        help='Maximum iterations for logistic regression'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for embedding generation'
    )
    parser.add_argument(
        '--train-samples',
        type=int,
        default=None,
        help='Number of training samples to use (None = all)'
    )
    parser.add_argument(
        '--eval-samples',
        type=int,
        default=None,
        help='Number of evaluation samples to use (None = all)'
    )
    return parser.parse_args()


def main():
    """Main training function - CLI wrapper for train_model()."""
    args = parse_args()
    
    # Call the importable train_model function
    train_model(
        dataset=args.dataset,
        classifier_type=args.classifier,
        embedding_model=args.embedding_model,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        experiment_name=args.experiment_name,
        hierarchy_file=Path(args.hierarchy_file) if args.hierarchy_file else None,
        intents_file=Path(args.intents_file) if args.intents_file else None,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples
    )


if __name__ == '__main__':
    main()
