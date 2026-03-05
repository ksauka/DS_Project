"""Unit tests for src/utils/faithfulness.py — FaithfulnessValidator pure logic methods"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.faithfulness import FaithfulnessValidator


@pytest.fixture
def validator():
    return FaithfulnessValidator()


# ---------- test_prediction_belief_alignment ----------

def test_alignment_passes_when_prediction_has_highest_belief(validator):
    belief = {"intent_a": 0.8, "intent_b": 0.2}
    result = validator.test_prediction_belief_alignment("intent_a", belief)
    assert result["passed"] is True
    assert result["predicted_intent"] == "intent_a"


def test_alignment_fails_when_different_intent_has_highest_belief(validator):
    belief = {"intent_a": 0.3, "intent_b": 0.7}
    result = validator.test_prediction_belief_alignment("intent_a", belief)
    assert result["passed"] is False
    assert "reason" in result


def test_alignment_empty_belief_fails(validator):
    result = validator.test_prediction_belief_alignment("intent_a", {})
    assert result["passed"] is False
    assert result["highest_belief_intent"] is None


def test_alignment_includes_query_id(validator):
    belief = {"a": 0.9}
    result = validator.test_prediction_belief_alignment("a", belief, query_id="q1")
    assert result["query_id"] == "q1"


def test_alignment_belief_rank_is_one_for_correct(validator):
    belief = {"a": 0.9, "b": 0.1}
    result = validator.test_prediction_belief_alignment("a", belief)
    assert result["belief_rank"] == 1


# ---------- test_belief_monotonicity ----------

def test_monotonicity_passes_with_increasing_belief(validator):
    history = [
        ({"intent_a": 0.2, "intent_b": 0.8}, "T1"),
        ({"intent_a": 0.5, "intent_b": 0.5}, "T2"),
        ({"intent_a": 0.8, "intent_b": 0.2}, "T3"),
    ]
    result = validator.test_belief_monotonicity(history, "intent_a")
    assert result["passed"] is True
    assert result["net_change"] == pytest.approx(0.6)


def test_monotonicity_fails_with_decreasing_belief(validator):
    history = [
        ({"intent_a": 0.8}, "T1"),
        ({"intent_a": 0.5}, "T2"),
        ({"intent_a": 0.2}, "T3"),
    ]
    result = validator.test_belief_monotonicity(history, "intent_a")
    assert result["passed"] is False


def test_monotonicity_single_turn_always_passes(validator):
    history = [({"intent_a": 0.5}, "T1")]
    result = validator.test_belief_monotonicity(history, "intent_a")
    assert result["passed"] is True


def test_monotonicity_empty_history_fails(validator):
    result = validator.test_belief_monotonicity([], "intent_a")
    assert result["passed"] is False


def test_monotonicity_missing_true_intent_treated_as_zero(validator):
    history = [
        ({"intent_b": 0.9}, "T1"),
        ({"intent_b": 0.5}, "T2"),
    ]
    result = validator.test_belief_monotonicity(history, "intent_a")
    assert "belief_progression" in result
    assert result["belief_progression"] == [0.0, 0.0]


# ---------- compute_belief_delta ----------

def test_compute_belief_delta_basic(validator):
    before = {"a": 0.3, "b": 0.7}
    after = {"a": 0.6, "b": 0.4}
    deltas = validator.compute_belief_delta(before, after)
    assert "a" in deltas
    assert "b" in deltas
    assert deltas["a"] == pytest.approx(0.3)
    assert deltas["b"] == pytest.approx(-0.3)


def test_compute_belief_delta_top_k_limits_results(validator):
    before = {f"intent_{i}": 0.1 for i in range(10)}
    after = {f"intent_{i}": 0.2 for i in range(10)}
    deltas = validator.compute_belief_delta(before, after, top_k=3)
    assert len(deltas) == 3


def test_compute_belief_delta_new_intents_in_after(validator):
    before = {"a": 0.5}
    after = {"a": 0.3, "b": 0.7}
    deltas = validator.compute_belief_delta(before, after, top_k=5)
    assert "b" in deltas
    assert deltas["b"] == pytest.approx(0.7)


def test_compute_belief_delta_sorted_by_abs_change(validator):
    before = {"a": 0.9, "b": 0.1}
    after = {"a": 0.1, "b": 0.9}
    deltas = validator.compute_belief_delta(before, after, top_k=2)
    values = list(deltas.values())
    assert abs(values[0]) >= abs(values[1])


# ---------- test_uncertainty_reduction ----------

def test_uncertainty_reduction_passes_when_decreasing(validator):
    history = [
        ({"Uncertainty": 0.8}, "T1"),
        ({"Uncertainty": 0.4}, "T2"),
        ({"Uncertainty": 0.1}, "T3"),
    ]
    result = validator.test_uncertainty_reduction(history)
    assert result["passed"] is True
    assert result["net_reduction"] == pytest.approx(0.7)


def test_uncertainty_reduction_fails_when_increasing(validator):
    history = [
        ({"Uncertainty": 0.1}, "T1"),
        ({"Uncertainty": 0.9}, "T2"),
    ]
    result = validator.test_uncertainty_reduction(history)
    assert result["passed"] is False


def test_uncertainty_reduction_passes_when_final_low(validator):
    # Even if increased, passes if final < 0.2
    history = [
        ({"Uncertainty": 0.05}, "T1"),
        ({"Uncertainty": 0.15}, "T2"),
    ]
    result = validator.test_uncertainty_reduction(history)
    assert result["passed"] is True


def test_uncertainty_reduction_empty_history_fails(validator):
    result = validator.test_uncertainty_reduction([])
    assert result["passed"] is False


def test_uncertainty_reduction_single_turn_passes(validator):
    history = [({"Uncertainty": 0.5}, "T1")]
    result = validator.test_uncertainty_reduction(history)
    assert result["passed"] is True


def test_uncertainty_reduction_missing_key_treated_as_zero(validator):
    history = [
        ({"intent_a": 0.9}, "T1"),
        ({"intent_a": 0.95}, "T2"),
    ]
    result = validator.test_uncertainty_reduction(history)
    assert result["uncertainty_progression"] == [0.0, 0.0]
    assert result["passed"] is True
