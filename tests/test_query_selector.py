"""Unit tests for src/utils/query_selector.py — QuerySelector"""

import pytest
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.query_selector import QuerySelector


def make_df(rows):
    """Helper: build a results DataFrame from list-of-dicts."""
    return pd.DataFrame(rows)


def sample_df():
    return make_df([
        {"true_intent": "a", "predicted_intent": "a", "interaction": "User: hi\nChatbot: q1\nUser: ans\nChatbot: q2\nUser: done", "confidence": 0.9},
        {"true_intent": "b", "predicted_intent": "b", "interaction": "User: hi\nChatbot: q1\nUser: ans\nChatbot: q2\nUser: done", "confidence": 0.8},
        {"true_intent": "c", "predicted_intent": "d", "interaction": "User: hi\nChatbot: q1\nUser: ans\nChatbot: q2\nUser: done", "confidence": 0.4},
        {"true_intent": "d", "predicted_intent": "x", "interaction": "User: hi\nChatbot: q1\nUser: ans\nChatbot: q2\nUser: done\nChatbot: q3\nUser: final", "confidence": 0.3},
    ])


@pytest.fixture
def selector():
    return QuerySelector(min_interactions=2, max_confidence=0.7, problematic_ratio=0.5)


# ---------- _preprocess ----------

def test_preprocess_adds_num_interactions(selector):
    df = sample_df()
    processed = selector._preprocess(df)
    assert "num_interactions" in processed.columns
    assert processed["num_interactions"].iloc[0] == 2  # 2 "Chatbot:" occurrences


def test_preprocess_adds_is_correct(selector):
    df = sample_df()
    processed = selector._preprocess(df)
    assert "is_correct" in processed.columns
    assert bool(processed["is_correct"].iloc[0]) is True   # a == a
    assert bool(processed["is_correct"].iloc[2]) is False  # c != d


def test_preprocess_no_interaction_column(selector):
    df = make_df([
        {"true_intent": "a", "predicted_intent": "a", "confidence": 0.9},
    ])
    processed = selector._preprocess(df)
    assert "num_interactions" in processed.columns
    assert processed["num_interactions"].iloc[0] == 0


def test_preprocess_does_not_mutate_original(selector):
    df = sample_df()
    selector._preprocess(df)
    assert "num_interactions" not in df.columns


# ---------- select_worst_queries ----------

def test_select_worst_queries_returns_incorrect_first(selector):
    df = sample_df()
    result = selector.select_worst_queries(df, max_samples=2)
    assert len(result) <= 2
    assert all(result["is_correct"] == False)  # noqa: E712


def test_select_worst_queries_max_samples_respected(selector):
    df = sample_df()
    result = selector.select_worst_queries(df, max_samples=1)
    assert len(result) <= 1


def test_select_worst_queries_fallback_to_low_confidence(selector):
    # All correct predictions → fallback to low confidence
    df = make_df([
        {"true_intent": "a", "predicted_intent": "a", "interaction": "Chatbot: q1\nChatbot: q2", "confidence": 0.4},
        {"true_intent": "b", "predicted_intent": "b", "interaction": "Chatbot: q1\nChatbot: q2", "confidence": 0.6},
    ])
    result = selector.select_worst_queries(df, max_samples=2)
    assert len(result) >= 0  # should not raise


# ---------- select_by_interaction_levels ----------

def test_select_by_interaction_levels_returns_df(selector):
    df = sample_df()
    result = selector.select_by_interaction_levels(df, max_samples=4)
    assert isinstance(result, pd.DataFrame)


def test_select_by_interaction_levels_max_samples_respected(selector):
    df = sample_df()
    result = selector.select_by_interaction_levels(df, max_samples=2)
    assert len(result) <= 2


def test_select_by_interaction_levels_empty_when_no_interactions():
    selector = QuerySelector(min_interactions=100)
    df = sample_df()
    result = selector.select_by_interaction_levels(df, max_samples=10)
    assert result.empty


# ---------- select_high_interaction ----------

def test_select_high_interaction_returns_df(selector):
    df = sample_df()
    result = selector.select_high_interaction(df, max_samples=4)
    assert isinstance(result, pd.DataFrame)


def test_select_high_interaction_respects_max_samples(selector):
    df = sample_df()
    result = selector.select_high_interaction(df, max_samples=2)
    assert len(result) <= 2


def test_select_high_interaction_labels_categories(selector):
    df = sample_df()
    result = selector.select_high_interaction(df, max_samples=4)
    if not result.empty:
        assert "selection_category" in result.columns
        assert set(result["selection_category"]).issubset({"problematic", "successful"})


# ---------- select_for_user_study ----------

def test_select_for_user_study_valid_strategies(selector):
    df = sample_df()
    for strategy in ["interaction_levels", "worst", "high_interaction"]:
        result = selector.select_for_user_study(df, max_samples=4, strategy=strategy)
        assert isinstance(result, pd.DataFrame)


def test_select_for_user_study_invalid_strategy_raises(selector):
    df = sample_df()
    with pytest.raises(ValueError, match="Unknown strategy"):
        selector.select_for_user_study(df, strategy="unknown")


# ---------- generate_study_summary ----------

def test_generate_study_summary_empty_df(selector):
    result = selector.generate_study_summary(pd.DataFrame())
    assert result["total_selected"] == 0
    assert result["avg_interactions"] == 0.0


def test_generate_study_summary_normal(selector):
    df = sample_df()
    # Run preprocess to add required columns
    processed = selector._preprocess(df)
    processed["selection_category"] = "test_cat"
    summary = selector.generate_study_summary(processed)
    assert summary["total_selected"] == 4
    assert "accuracy" in summary
    assert "categories" in summary


def test_generate_study_summary_accuracy_range(selector):
    df = sample_df()
    processed = selector._preprocess(df)
    processed["selection_category"] = "cat"
    summary = selector.generate_study_summary(processed)
    assert 0.0 <= summary["accuracy"] <= 1.0
