"""Unit tests for DSMassFunction pure logic methods (no ML model required).

conftest.py stubs torch/sentence_transformers before this module is collected.
"""

import pytest
import sys
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly to avoid __init__.py eagerly loading embeddings
import importlib
_ds_mod = importlib.import_module("src.models.ds_mass_function")
DSMassFunction = _ds_mod.DSMassFunction


# ---------- Fixtures ----------

HIERARCHY = {
    "root": ["cat_a", "cat_b"],
    "cat_a": ["leaf_1", "leaf_2"],
    "cat_b": ["leaf_3"],
    "leaf_1": [],
    "leaf_2": [],
    "leaf_3": [],
}

# Minimal intent embeddings (keys only; values not needed for pure logic tests)
INTENT_EMBEDDINGS = {k: np.zeros(10) for k in HIERARCHY}


def make_ds(custom_thresholds=None):
    """Create DSMassFunction with a mocked classifier and embedder."""
    classifier = MagicMock()
    classifier.get_classes.return_value = list(HIERARCHY.keys())
    # predict_proba returns uniform probabilities
    n = len(HIERARCHY)
    classifier.predict_proba.return_value = np.array([[1/n] * n])

    embedder = MagicMock()
    embedder.get_embedding.return_value = np.zeros(10)

    return DSMassFunction(
        intent_embeddings=INTENT_EMBEDDINGS,
        hierarchy=HIERARCHY,
        classifier=classifier,
        custom_thresholds=custom_thresholds,
        embedder=embedder,
        enable_belief_tracking=False,
    )


@pytest.fixture
def ds():
    return make_ds()


# ---------- is_leaf ----------

def test_is_leaf_leaf_node(ds):
    assert ds.is_leaf("leaf_1") is True
    assert ds.is_leaf("leaf_2") is True
    assert ds.is_leaf("leaf_3") is True


def test_is_leaf_non_leaf_node(ds):
    assert ds.is_leaf("root") is False
    assert ds.is_leaf("cat_a") is False


def test_is_leaf_unknown_node(ds):
    assert ds.is_leaf("nonexistent") is True  # empty list returned by .get()


# ---------- get_threshold ----------

def test_get_threshold_leaf(ds):
    assert ds.get_threshold("leaf_1") == 0.1


def test_get_threshold_parent_in_hierarchy(ds):
    assert ds.get_threshold("cat_a") == 0.2


def test_get_threshold_unknown(ds):
    # Unknown nodes are treated as leaves by is_leaf() → returns leaf threshold
    assert ds.get_threshold("unknown_node") == 0.1


# ---------- get_confidence_threshold ----------

def test_get_confidence_threshold_no_custom_returns_zero(ds):
    # custom_thresholds is empty → always 0.0 (baseline mode)
    assert ds.get_confidence_threshold("leaf_1") == 0.0
    assert ds.get_confidence_threshold("root") == 0.0


def test_get_confidence_threshold_custom_leaf():
    ds = make_ds(custom_thresholds={"leaf_1": 0.55, "cat_a": 0.65})
    assert ds.get_confidence_threshold("leaf_1") == 0.55
    assert ds.get_confidence_threshold("cat_a") == 0.65


def test_get_confidence_threshold_custom_fallback_to_defaults():
    # custom dict present but does NOT contain the queried intent → use defaults
    ds = make_ds(custom_thresholds={"other_intent": 0.9})
    # leaf_1 not in custom dict, falls to default leaf threshold 0.3
    assert ds.get_confidence_threshold("leaf_1") == 0.3


# ---------- get_node_depth ----------

def test_get_node_depth_leaf(ds):
    assert ds.get_node_depth("leaf_1") == 0
    assert ds.get_node_depth("leaf_3") == 0


def test_get_node_depth_intermediate(ds):
    assert ds.get_node_depth("cat_a") == 1
    assert ds.get_node_depth("cat_b") == 1


def test_get_node_depth_root(ds):
    assert ds.get_node_depth("root") == 2


def test_get_node_depth_unknown(ds):
    assert ds.get_node_depth("unknown") == 0


# ---------- get_all_descendants ----------

def test_get_all_descendants_leaf_is_itself(ds):
    assert ds.get_all_descendants("leaf_1") == {"leaf_1"}


def test_get_all_descendants_cat_a(ds):
    assert ds.get_all_descendants("cat_a") == {"cat_a", "leaf_1", "leaf_2"}


def test_get_all_descendants_root(ds):
    expected = set(HIERARCHY.keys())
    assert ds.get_all_descendants("root") == expected


# ---------- find_lowest_common_ancestor ----------

def test_find_lca_siblings_under_cat_a(ds):
    lca = ds.find_lowest_common_ancestor(["leaf_1", "leaf_2"])
    assert lca == "cat_a"


def test_find_lca_nodes_under_different_cats(ds):
    lca = ds.find_lowest_common_ancestor(["leaf_1", "leaf_3"])
    assert lca == "root"


def test_find_lca_single_node(ds):
    lca = ds.find_lowest_common_ancestor(["leaf_1"])
    assert lca == "leaf_1"


def test_find_lca_empty_list(ds):
    assert ds.find_lowest_common_ancestor([]) is None


# ---------- compute_belief ----------

def test_compute_belief_leaf_equals_mass(ds):
    mass = {"leaf_1": 0.6, "leaf_2": 0.2, "leaf_3": 0.1, "root": 0.1, "cat_a": 0.0, "cat_b": 0.0}
    belief = ds.compute_belief(mass)
    assert belief["leaf_1"] == pytest.approx(0.6)
    assert belief["leaf_2"] == pytest.approx(0.2)


def test_compute_belief_parent_aggregates_children(ds):
    mass = {k: 0.0 for k in HIERARCHY}
    mass["leaf_1"] = 0.4
    mass["leaf_2"] = 0.3
    mass["leaf_3"] = 0.2
    belief = ds.compute_belief(mass)
    assert belief["cat_a"] == pytest.approx(0.7)  # leaf_1 + leaf_2
    assert belief["cat_b"] == pytest.approx(0.2)  # leaf_3
    assert belief["root"] == pytest.approx(0.9)   # cat_a + cat_b


# ---------- ask_clarification ----------

def test_ask_clarification_small_children(ds):
    # cat_a has 2 children < 4 → all included
    parent_nodes = [("cat_a", 0.7)]
    belief = {k: 0.1 for k in HIERARCHY}
    queries = ds.ask_clarification(parent_nodes, belief)
    assert len(queries) == 1
    parent, children = queries[0]
    assert parent == "cat_a"
    assert set(children) == {"leaf_1", "leaf_2"}


def test_ask_clarification_large_children():
    """When a parent has >= 4 children, only top-3 by belief are returned."""
    hierarchy = {
        "root": ["c1", "c2", "c3", "c4", "c5"],
        "c1": [], "c2": [], "c3": [], "c4": [], "c5": [],
    }
    embeddings = {k: np.zeros(10) for k in hierarchy}
    classifier = MagicMock()
    classifier.get_classes.return_value = list(hierarchy.keys())
    classifier.predict_proba.return_value = np.array([[0.2] * 5])
    embedder = MagicMock()
    embedder.get_embedding.return_value = np.zeros(10)

    ds2 = DSMassFunction(
        intent_embeddings=embeddings,
        hierarchy=hierarchy,
        classifier=classifier,
        embedder=embedder,
        enable_belief_tracking=False,
    )
    belief = {"c1": 0.5, "c2": 0.3, "c3": 0.1, "c4": 0.05, "c5": 0.05, "root": 0.0}
    queries = ds2.ask_clarification([("root", 0.5)], belief)
    _, children = queries[0]
    assert len(children) == 3


# ---------- evaluate_hierarchy ----------

def test_evaluate_hierarchy_returns_confident_nodes(ds):
    # With no custom thresholds, threshold = 0.0 → every node with mass > 0 is confident
    mass = {"leaf_1": 0.6, "leaf_2": 0.2, "leaf_3": 0.2, "root": 0.0, "cat_a": 0.0, "cat_b": 0.0}
    leaf_nodes = ["leaf_1", "leaf_2", "leaf_3"]
    confident, belief = ds.evaluate_hierarchy(leaf_nodes, mass)
    confident_intents = [n for n, _ in confident]
    assert "leaf_1" in confident_intents
    assert "leaf_2" in confident_intents


def test_evaluate_hierarchy_respects_threshold():
    ds = make_ds(custom_thresholds={"leaf_1": 0.5, "leaf_2": 0.5, "leaf_3": 0.5})
    mass = {"leaf_1": 0.6, "leaf_2": 0.1, "leaf_3": 0.1, "root": 0.0, "cat_a": 0.1, "cat_b": 0.1}
    confident, _ = ds.evaluate_hierarchy(["leaf_1", "leaf_2", "leaf_3"], mass)
    confident_intents = [n for n, _ in confident]
    assert "leaf_1" in confident_intents
    assert "leaf_2" not in confident_intents


# ---------- get_prediction_from_mass ----------

def test_get_prediction_from_mass_returns_highest_leaf(ds):
    mass = {"leaf_1": 0.8, "leaf_2": 0.1, "leaf_3": 0.05, "root": 0.0, "cat_a": 0.0, "cat_b": 0.05}
    intent, confidence = ds.get_prediction_from_mass(mass)
    assert intent == "leaf_1"
    assert confidence == pytest.approx(0.8)


def test_get_prediction_from_mass_all_zero_returns_first(ds):
    mass = {k: 0.0 for k in HIERARCHY}
    intent, confidence = ds.get_prediction_from_mass(mass)
    # Some leaf returned
    assert intent in {"leaf_1", "leaf_2", "leaf_3"}
    assert confidence == pytest.approx(0.0)


# ---------- should_ask_clarification ----------

def test_should_ask_clarification_no_thresholds_returns_false(ds):
    # With threshold=0.0, any mass > 0 satisfies → no clarification needed
    mass = {"leaf_1": 0.6, "leaf_2": 0.2, "leaf_3": 0.2, "root": 0.0, "cat_a": 0.0, "cat_b": 0.0}
    assert ds.should_ask_clarification(mass) is False


def test_should_ask_clarification_with_high_threshold_returns_true():
    ds = make_ds(custom_thresholds={"leaf_1": 0.99, "leaf_2": 0.99, "leaf_3": 0.99})
    mass = {"leaf_1": 0.3, "leaf_2": 0.3, "leaf_3": 0.4, "root": 0.0, "cat_a": 0.0, "cat_b": 0.0}
    assert ds.should_ask_clarification(mass) is True


# ---------- clear_belief_history / get_belief_tracker ----------

def test_get_belief_tracker_when_disabled(ds):
    assert ds.get_belief_tracker() is None


def test_get_belief_tracker_when_enabled():
    classifier = MagicMock()
    classifier.get_classes.return_value = list(HIERARCHY.keys())
    classifier.predict_proba.return_value = np.array([[1/6] * 6])
    embedder = MagicMock()
    embedder.get_embedding.return_value = np.zeros(10)

    ds_tracked = DSMassFunction(
        intent_embeddings=INTENT_EMBEDDINGS,
        hierarchy=HIERARCHY,
        classifier=classifier,
        embedder=embedder,
        enable_belief_tracking=True,
    )
    assert ds_tracked.get_belief_tracker() is not None


def test_clear_belief_history_no_error_when_tracking_disabled(ds):
    ds.clear_belief_history()  # should not raise
