"""Unit tests for src/utils/explainability.py — BeliefTracker and BeliefVisualizer.generate_belief_summary"""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.explainability import BeliefTracker, BeliefVisualizer


BELIEF_1 = {"intent_a": 0.7, "intent_b": 0.2, "Uncertainty": 0.1}
BELIEF_2 = {"intent_a": 0.5, "intent_b": 0.4, "Uncertainty": 0.1}
BELIEF_3 = {"intent_a": 0.8, "intent_b": 0.15, "Uncertainty": 0.05}


# ---------- BeliefTracker.record_belief / get_history ----------

def test_record_belief_appends_entry():
    tracker = BeliefTracker()
    tracker.record_belief(BELIEF_1, "Turn 1")
    history = tracker.get_history()
    assert len(history) == 1
    assert history[0][1] == "Turn 1"


def test_record_belief_copies_dict():
    tracker = BeliefTracker()
    belief = {"a": 0.9}
    tracker.record_belief(belief, "T1")
    belief["a"] = 0.0  # mutate original
    assert tracker.get_history()[0][0]["a"] == 0.9  # stored copy unchanged


def test_get_history_returns_copy():
    tracker = BeliefTracker()
    tracker.record_belief(BELIEF_1, "T1")
    h1 = tracker.get_history()
    h1.clear()
    assert len(tracker.get_history()) == 1  # internal list unaffected


# ---------- BeliefTracker.clear_history ----------

def test_clear_history_empties_list():
    tracker = BeliefTracker()
    tracker.record_belief(BELIEF_1, "T1")
    tracker.clear_history()
    assert tracker.get_history() == []


# ---------- BeliefTracker.get_latest_belief ----------

def test_get_latest_belief_none_when_empty():
    tracker = BeliefTracker()
    assert tracker.get_latest_belief() is None


def test_get_latest_belief_returns_last():
    tracker = BeliefTracker()
    tracker.record_belief(BELIEF_1, "T1")
    tracker.record_belief(BELIEF_2, "T2")
    assert tracker.get_latest_belief() == BELIEF_2


# ---------- BeliefTracker.get_final_belief ----------

def test_get_final_belief_none_when_empty():
    assert BeliefTracker().get_final_belief() is None


def test_get_final_belief_returns_copy_of_last():
    tracker = BeliefTracker()
    tracker.record_belief(BELIEF_1, "T1")
    tracker.record_belief(BELIEF_3, "T3")
    fb = tracker.get_final_belief()
    assert fb == BELIEF_3
    fb["intent_a"] = 0.0  # mutate returned copy
    assert tracker.get_final_belief()["intent_a"] == BELIEF_3["intent_a"]  # original unchanged


# ---------- BeliefTracker.get_belief_at_turn ----------

def test_get_belief_at_turn_valid_index():
    tracker = BeliefTracker()
    tracker.record_belief(BELIEF_1, "T1")
    tracker.record_belief(BELIEF_2, "T2")
    assert tracker.get_belief_at_turn(0) == BELIEF_1
    assert tracker.get_belief_at_turn(1) == BELIEF_2


def test_get_belief_at_turn_out_of_bounds_returns_none():
    tracker = BeliefTracker()
    tracker.record_belief(BELIEF_1, "T1")
    assert tracker.get_belief_at_turn(5) is None
    assert tracker.get_belief_at_turn(-1) is None


# ---------- BeliefTracker.save_to_json / load_from_json ----------

def test_save_and_load_json_roundtrip(tmp_path):
    tracker = BeliefTracker()
    tracker.record_belief(BELIEF_1, "Turn 1")
    tracker.record_belief(BELIEF_2, "Turn 2")
    p = str(tmp_path / "belief_log.json")
    tracker.save_to_json(p)

    tracker2 = BeliefTracker()
    tracker2.load_from_json(p)
    history = tracker2.get_history()
    assert len(history) == 2
    assert history[0][1] == "Turn 1"
    assert history[1][0] == BELIEF_2


def test_save_to_json_file_format(tmp_path):
    tracker = BeliefTracker()
    tracker.record_belief({"x": 0.5}, "T1")
    p = tmp_path / "log.json"
    tracker.save_to_json(str(p))
    data = json.loads(p.read_text())
    assert isinstance(data, list)
    assert data[0]["turn"] == "T1"
    assert data[0]["belief"] == {"x": 0.5}


# ---------- BeliefVisualizer.generate_belief_summary ----------

def test_generate_belief_summary_empty():
    result = BeliefVisualizer.generate_belief_summary([])
    assert "error" in result


def test_generate_belief_summary_single_turn():
    history = [(BELIEF_1, "T1")]
    result = BeliefVisualizer.generate_belief_summary(history)
    assert result["num_turns"] == 1
    assert result["initial_top_intent"] == "intent_a"
    assert result["final_top_intent"] == "intent_a"


def test_generate_belief_summary_multiple_turns():
    history = [(BELIEF_1, "T1"), (BELIEF_2, "T2"), (BELIEF_3, "T3")]
    result = BeliefVisualizer.generate_belief_summary(history)
    assert result["num_turns"] == 3
    assert len(result["top_intents_per_turn"]) == 3
    assert result["initial_uncertainty"] == pytest.approx(0.1)
    assert result["final_uncertainty"] == pytest.approx(0.05)


def test_generate_belief_summary_uncertainty_progression():
    history = [
        ({"a": 0.3, "Uncertainty": 0.7}, "T1"),
        ({"a": 0.8, "Uncertainty": 0.2}, "T2"),
    ]
    result = BeliefVisualizer.generate_belief_summary(history)
    assert result["initial_uncertainty"] == pytest.approx(0.7)
    assert result["final_uncertainty"] == pytest.approx(0.2)
