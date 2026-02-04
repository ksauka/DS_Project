"""Evaluate vanilla baseline (Logistic Regression + BERT embeddings only, no hierarchy)."""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.classifier import IntentClassifier
from src.models.embeddings import SentenceEmbedder
from src.utils.file_io import save_csv, save_json, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate vanilla baseline (pure Logistic Regression + BERT)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='banking77',
        choices=['banking77', 'clinc150', 'snips', 'atis'],
        help='Dataset name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/vanilla_baseline',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of test samples to evaluate (None for all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to pretrained model (if None, will train new one)'
    )
    return parser.parse_args()


def train_vanilla_baseline(dataset_name='banking77', model_path=None, batch_size=64):
    """Train vanilla baseline model with STRING labels (matching old notebook).
    
    Returns:
        classifier: Trained IntentClassifier
        embedder: SentenceEmbedder instance
        intent_names: List of intent names
    """
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)
    train_data = dataset["train"]
    
    logger.info("Initializing Sentence-BERT embedder...")
    embedder = SentenceEmbedder()
    
    logger.info("Encoding training texts...")
    train_texts = [ex["text"] for ex in train_data]
    train_embeddings = embedder.get_embeddings_batch(train_texts, batch_size=batch_size, show_progress=True)
    
    # Get intent labels as STRINGS (matching old notebook exactly)
    intent_names = dataset["train"].features["label"].names
    train_labels = [intent_names[ex["label"]] for ex in train_data]
    
    logger.info(f"Training Logistic Regression on {len(train_labels)} samples with STRING labels...")
    logger.info(f"Number of intents: {len(set(train_labels))}")
    logger.info(f"Sample labels: {train_labels[:3]}")
    
    # Train classifier with STRING labels
    classifier = IntentClassifier(model_type='logistic')
    classifier.train(train_embeddings, train_labels)
    
    # Save model
    if model_path is None:
        model_path = f"experiments/{dataset_name}/banking77_logistic_model.pkl"
    
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_path)
    
    logger.info(f"Model saved to {model_path}")
    return classifier, embedder, intent_names


def load_vanilla_baseline(model_path, dataset_name='banking77'):
    """Load pretrained vanilla baseline model."""
    logger.info("Initializing Sentence-BERT embedder...")
    embedder = SentenceEmbedder()
    
    logger.info(f"Loading classifier from {model_path}")
    classifier = IntentClassifier.from_pretrained(model_path)
    
    # Load intent names from dataset
    dataset = load_dataset(dataset_name)
    intent_names = dataset["train"].features["label"].names
    
    return classifier, embedder, intent_names


def evaluate_vanilla_baseline(
    classifier,
    embedder,
    intent_names,
    dataset_name='banking77',
    num_samples=None,
    batch_size=64,
    output_dir=None
):
    """Evaluate vanilla baseline on test set.
    
    Args:
        classifier: Trained IntentClassifier
        embedder: SentenceEmbedder instance
        intent_names: List of intent names
        dataset_name: Dataset name
        num_samples: Number of samples to evaluate
        batch_size: Batch size for encoding
        output_dir: Output directory for results
        
    Returns:
        Dictionary with metrics
    """
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)
    test_data = dataset["test"]
    
    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    logger.info(f"Evaluating on {len(test_data)} test samples...")
    
    # Prepare test data
    test_texts = [ex["text"] for ex in test_data]
    test_labels = [intent_names[ex["label"]] for ex in test_data]
    
    # Encode test texts
    logger.info("Encoding test texts...")
    test_embeddings = embedder.get_embeddings_batch(test_texts, batch_size=batch_size, show_progress=True)
    
    # Predict
    logger.info("Making predictions...")
    predictions = classifier.predict(test_embeddings)
    probabilities = classifier.predict_proba(test_embeddings)
    
    logger.info(f"Predictions type: {type(predictions[0])}, sample: {predictions[0]}")
    logger.info(f"Test labels type: {type(test_labels[0])}, sample: {test_labels[0]}")
    
    # Get confidence scores
    confidences = np.max(probabilities, axis=1)
    
    # Store results
    results = []
    for i, (text, true_label, pred_label, conf) in enumerate(zip(
        test_texts, test_labels, predictions, confidences
    )):
        results.append({
            'query': text,
            'true_intent': true_label,
            'predicted_intent': pred_label,
            'confidence': float(conf),
            'correct': int(pred_label == true_label),
        })
    
    # Save predictions
    if output_dir:
        results_df = pd.DataFrame(results)
        predictions_path = output_dir / f'{dataset_name}_vanilla_predictions.csv'
        save_csv(results_df, predictions_path, index=False)
        logger.info(f"Saved predictions to {predictions_path}")
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    precision_macro = precision_score(test_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(test_labels, predictions, average='macro', zero_division=0)
    
    # Confidence statistics
    correct_mask = np.array(predictions) == np.array(test_labels)
    avg_confidence = float(np.mean(confidences))
    avg_correct_confidence = float(np.mean(confidences[correct_mask])) if correct_mask.sum() > 0 else 0.0
    avg_incorrect_confidence = float(np.mean(confidences[~correct_mask])) if (~correct_mask).sum() > 0 else 0.0
    
    metrics = {
        'model': 'Vanilla Logistic Regression + BERT',
        'dataset': dataset_name,
        'num_samples': len(test_data),
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'avg_confidence': avg_confidence,
        'avg_correct_confidence': avg_correct_confidence,
        'avg_incorrect_confidence': avg_incorrect_confidence,
        'num_correct': int(correct_mask.sum()),
        'num_incorrect': int((~correct_mask).sum()),
    }
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    logger.info(f"Starting vanilla baseline evaluation for {args.dataset}")
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Train or load model
    if args.model_path and Path(args.model_path).exists():
        logger.info(f"Loading pretrained model from {args.model_path}")
        classifier, embedder, intent_names = load_vanilla_baseline(args.model_path, args.dataset)
    else:
        logger.info("Training new model...")
        classifier, embedder, intent_names = train_vanilla_baseline(args.dataset, args.model_path, args.batch_size)
    
    # Evaluate
    metrics = evaluate_vanilla_baseline(
        classifier=classifier,
        embedder=embedder,
        intent_names=intent_names,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=output_dir
    )
    
    # Save metrics
    metrics_path = output_dir / f'{args.dataset}_vanilla_metrics.json'
    save_json(metrics, metrics_path)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("VANILLA BASELINE RESULTS")
    logger.info("="*60)
    logger.info(f"Model: {metrics['model']}")
    logger.info(f"Dataset: {metrics['dataset']}")
    logger.info(f"Samples: {metrics['num_samples']}")
    logger.info("-"*60)
    logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
    logger.info(f"F1 (macro):         {metrics['f1_macro']:.4f}")
    logger.info(f"F1 (weighted):      {metrics['f1_weighted']:.4f}")
    logger.info(f"Precision (macro):  {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (macro):     {metrics['recall_macro']:.4f}")
    logger.info("-"*60)
    logger.info(f"Avg Confidence:     {metrics['avg_confidence']:.4f}")
    logger.info(f"  Correct:          {metrics['avg_correct_confidence']:.4f}")
    logger.info(f"  Incorrect:        {metrics['avg_incorrect_confidence']:.4f}")
    logger.info("-"*60)
    logger.info(f"Correct Predictions:   {metrics['num_correct']}")
    logger.info(f"Incorrect Predictions: {metrics['num_incorrect']}")
    logger.info("="*60)
    logger.info(f"\nMetrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
