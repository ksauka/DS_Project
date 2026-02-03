"""Utility functions for evaluation metrics."""

import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

logger = logging.getLogger(__name__)


def compute_accuracy(
    y_true: List[str],
    y_pred: List[str]
) -> float:
    """Compute accuracy score.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def compute_f1_score(
    y_true: List[str],
    y_pred: List[str],
    average: str = 'macro'
) -> float:
    """Compute F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'micro', 'weighted')

    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average)


def compute_precision_recall(
    y_true: List[str],
    y_pred: List[str],
    average: str = 'macro'
) -> Tuple[float, float]:
    """Compute precision and recall.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method

    Returns:
        Tuple of (precision, recall)
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    return precision, recall


def compute_all_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    accuracy = compute_accuracy(y_true, y_pred)
    f1_macro = compute_f1_score(y_true, y_pred, average='macro')
    f1_weighted = compute_f1_score(y_true, y_pred, average='weighted')
    precision, recall = compute_precision_recall(y_true, y_pred, average='macro')

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }

    logger.info(f"Metrics: {metrics}")
    return metrics


def get_classification_report(
    y_true: List[str],
    y_pred: List[str],
    target_names: List[str] = None
) -> str:
    """Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of target class names

    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=target_names)


def analyze_predictions(
    predictions: List[Tuple[str, float]],
    true_labels: List[str]
) -> Dict[str, any]:
    """Analyze prediction results with confidence scores.

    Args:
        predictions: List of (predicted_intent, confidence) tuples
        true_labels: List of true intent labels

    Returns:
        Dictionary with analysis results
    """
    pred_labels = [pred[0] for pred in predictions]
    confidences = [pred[1] for pred in predictions]

    # Separate correct and incorrect predictions
    correct_confidences = []
    incorrect_confidences = []

    for pred_label, true_label, conf in zip(pred_labels, true_labels, confidences):
        if pred_label == true_label:
            correct_confidences.append(conf)
        else:
            incorrect_confidences.append(conf)

    analysis = {
        'accuracy': compute_accuracy(true_labels, pred_labels),
        'f1_macro': compute_f1_score(true_labels, pred_labels, average='macro'),
        'avg_confidence': np.mean(confidences),
        'avg_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0.0,
        'avg_incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0.0,
        'num_correct': len(correct_confidences),
        'num_incorrect': len(incorrect_confidences),
        'total': len(predictions)
    }

    return analysis


def count_interactions(conversation_histories: List[str]) -> Dict[str, float]:
    """Count chatbot interactions in conversation histories.

    Args:
        conversation_histories: List of conversation history strings

    Returns:
        Dictionary with interaction statistics
    """
    total_chatbot_turns = 0
    total_conversations = 0

    for history in conversation_histories:
        if history:
            chatbot_turns = history.count("Chatbot:")
            total_chatbot_turns += chatbot_turns
            total_conversations += 1

    avg_turns = total_chatbot_turns / total_conversations if total_conversations > 0 else 0

    return {
        'total_chatbot_turns': total_chatbot_turns,
        'total_conversations': total_conversations,
        'avg_turns_per_conversation': avg_turns
    }
