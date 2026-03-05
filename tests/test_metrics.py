"""Unit tests for src/utils/metrics.py"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import (
    compute_accuracy,
    compute_f1_score,
    compute_precision_recall,
    compute_all_metrics,
    get_classification_report,
    analyze_predictions,
    count_interactions,
)


# ---------- compute_accuracy ----------

def test_compute_accuracy_perfect():
    assert compute_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_compute_accuracy_zero():
    assert compute_accuracy(["a", "b"], ["b", "a"]) == 0.0


def test_compute_accuracy_partial():
    result = compute_accuracy(["a", "a", "b"], ["a", "b", "b"])
    assert abs(result - 2/3) < 1e-9


# ---------- compute_f1_score ----------

def test_compute_f1_score_perfect():
    y = ["a", "b", "c"]
    assert compute_f1_score(y, y) == pytest.approx(1.0)


def test_compute_f1_score_macro():
    y_true = ["a", "b"]
    y_pred = ["a", "a"]
    score = compute_f1_score(y_true, y_pred, average="macro")
    assert 0.0 <= score <= 1.0


def test_compute_f1_score_weighted():
    y_true = ["a", "a", "b"]
    y_pred = ["a", "b", "b"]
    score = compute_f1_score(y_true, y_pred, average="weighted")
    assert 0.0 <= score <= 1.0


# ---------- compute_precision_recall ----------

def test_compute_precision_recall_perfect():
    y = ["a", "b", "c"]
    precision, recall = compute_precision_recall(y, y)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_compute_precision_recall_returns_tuple():
    result = compute_precision_recall(["a", "b"], ["a", "a"])
    assert isinstance(result, tuple)
    assert len(result) == 2


# ---------- compute_all_metrics ----------

def test_compute_all_metrics_keys():
    y = ["a", "b", "c"]
    metrics = compute_all_metrics(y, y)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


def test_compute_all_metrics_perfect_scores():
    y = ["a", "b", "c"]
    metrics = compute_all_metrics(y, y)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


# ---------- get_classification_report ----------

def test_get_classification_report_returns_string():
    report = get_classification_report(["a", "b"], ["a", "b"])
    assert isinstance(report, str)
    assert len(report) > 0


def test_get_classification_report_with_target_names():
    report = get_classification_report(["a", "b"], ["a", "b"], target_names=["a", "b"])
    assert "a" in report
    assert "b" in report


# ---------- analyze_predictions ----------

def test_analyze_predictions_all_correct():
    predictions = [("a", 0.9), ("b", 0.8)]
    true_labels = ["a", "b"]
    result = analyze_predictions(predictions, true_labels)
    assert result["num_correct"] == 2
    assert result["num_incorrect"] == 0
    assert result["total"] == 2
    assert result["accuracy"] == pytest.approx(1.0)


def test_analyze_predictions_all_incorrect():
    predictions = [("b", 0.9), ("a", 0.8)]
    true_labels = ["a", "b"]
    result = analyze_predictions(predictions, true_labels)
    assert result["num_correct"] == 0
    assert result["num_incorrect"] == 2


def test_analyze_predictions_avg_confidence():
    predictions = [("a", 0.6), ("b", 0.4)]
    true_labels = ["a", "b"]
    result = analyze_predictions(predictions, true_labels)
    assert result["avg_confidence"] == pytest.approx(0.5)


def test_analyze_predictions_empty_incorrect():
    predictions = [("a", 0.9)]
    true_labels = ["a"]
    result = analyze_predictions(predictions, true_labels)
    assert result["avg_incorrect_confidence"] == 0.0


# ---------- count_interactions ----------

def test_count_interactions_basic():
    histories = [
        "User: hello\nChatbot: hi\nUser: bye",
        "User: test\nChatbot: yes\nChatbot: ok",
    ]
    result = count_interactions(histories)
    assert result["total_chatbot_turns"] == 3
    assert result["total_conversations"] == 2
    assert result["avg_turns_per_conversation"] == pytest.approx(1.5)


def test_count_interactions_empty_list():
    result = count_interactions([])
    assert result["total_chatbot_turns"] == 0
    assert result["total_conversations"] == 0
    assert result["avg_turns_per_conversation"] == 0


def test_count_interactions_empty_string_histories():
    result = count_interactions(["", ""])
    assert result["total_chatbot_turns"] == 0
    # Empty strings are falsy — loop skips them, 0 conversations counted
    assert result["total_conversations"] == 0
    assert result["avg_turns_per_conversation"] == 0.0


def test_count_interactions_no_chatbot_turns():
    result = count_interactions(["User: hello"])
    assert result["total_chatbot_turns"] == 0
    assert result["avg_turns_per_conversation"] == 0.0
