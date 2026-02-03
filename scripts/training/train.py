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
        required=True,
        help='Path to hierarchy JSON file'
    )
    parser.add_argument(
        '--intents-file',
        type=str,
        required=True,
        help='Path to hierarchical intents JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for trained model'
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
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    logger.info(f"Starting training for {args.dataset}")
    logger.info(f"Classifier: {args.classifier}")
    logger.info(f"Embedding model: {args.embedding_model}")

    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_dir / f"{args.dataset}_{args.classifier}_{timestamp}")

    # Load dataset
    logger.info("Loading dataset...")
    data_loader = DataLoader(args.dataset)
    data_loader.load()

    # Get training data
    train_texts, train_labels, _ = data_loader.get_split_data('train')
    logger.info(f"Training samples: {len(train_texts)}")

    # Load hierarchy and intents
    logger.info("Loading hierarchy and intent descriptions...")
    hierarchy = load_hierarchy_from_json(args.hierarchy_file)
    hierarchical_intents = load_hierarchical_intents_from_json(args.intents_file)

    # Initialize embedder
    logger.info("Initializing sentence embedder...")
    embedder = SentenceEmbedder(model_name=args.embedding_model)

    # Generate embeddings for training data
    logger.info("Generating embeddings for training data...")
    train_embeddings = embedder.get_embeddings_batch(
        train_texts,
        batch_size=args.batch_size,
        show_progress=True
    )

    # Generate intent embeddings
    logger.info("Generating intent embeddings...")
    intent_embeddings = IntentEmbeddings(
        hierarchical_intents,
        embedder=embedder
    )

    # Train classifier
    logger.info("Training classifier...")
    if args.classifier == 'logistic':
        classifier = IntentClassifier(
            classifier_type='logistic',
            max_iter=args.max_iter
        )
    else:
        classifier = IntentClassifier(classifier_type='svm')

    classifier.train(train_embeddings, train_labels)

    # Save model
    model_path = run_dir / f"{args.dataset}_{args.classifier}_model.pkl"
    logger.info(f"Saving model to {model_path}")
    classifier.save(model_path)

    # Save configuration
    config = {
        'dataset': args.dataset,
        'classifier': args.classifier,
        'embedding_model': args.embedding_model,
        'training_samples': len(train_texts),
        'num_intents': len(data_loader.get_intent_names()),
        'hierarchy_file': args.hierarchy_file,
        'intents_file': args.intents_file,
        'timestamp': timestamp
    }
    config_path = run_dir / 'config.json'
    save_json(config, config_path)
    logger.info(f"Saved configuration to {config_path}")

    # Test on validation/test split
    logger.info("Evaluating on test set...")
    test_texts, test_labels, _ = data_loader.get_split_data('test')
    test_embeddings = embedder.get_embeddings_batch(
        test_texts,
        batch_size=args.batch_size,
        show_progress=True
    )

    predictions = classifier.predict(test_embeddings)

    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(test_labels, predictions)
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    report = classification_report(test_labels, predictions)
    logger.info(f"\nClassification Report:\n{report}")

    # Save predictions
    import pandas as pd
    results_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels,
        'predicted_label': predictions
    })
    results_path = run_dir / 'test_predictions.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved predictions to {results_path}")

    logger.info(f"\nTraining completed! Model saved to {run_dir}")


if __name__ == '__main__':
    main()
