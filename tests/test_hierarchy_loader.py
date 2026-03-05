"""Unit tests for config/hierarchy_loader.py"""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.hierarchy_loader import (
    load_hierarchy_from_json,
    load_hierarchical_intents_from_json,
    save_hierarchy_to_json,
    validate_hierarchy,
    get_leaf_nodes,
    get_parent_nodes,
    get_node_depth,
)

SAMPLE_HIERARCHY = {
    "root": ["child_a", "child_b"],
    "child_a": ["leaf_1", "leaf_2"],
    "child_b": ["leaf_3"],
    "leaf_1": [],
    "leaf_2": [],
    "leaf_3": [],
}


# ---------- load_hierarchy_from_json ----------

def test_load_hierarchy_from_json_returns_dict(tmp_path):
    p = tmp_path / "hierarchy.json"
    p.write_text(json.dumps(SAMPLE_HIERARCHY))
    result = load_hierarchy_from_json(str(p))
    assert result == SAMPLE_HIERARCHY


def test_load_hierarchy_from_json_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_hierarchy_from_json("/nonexistent/path/hierarchy.json")


def test_load_hierarchy_from_json_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not valid json {")
    with pytest.raises(ValueError):
        load_hierarchy_from_json(str(p))


# ---------- load_hierarchical_intents_from_json ----------

def test_load_hierarchical_intents_from_json_returns_dict(tmp_path):
    intents = {"intent_a": "description a", "intent_b": "description b"}
    p = tmp_path / "intents.json"
    p.write_text(json.dumps(intents))
    result = load_hierarchical_intents_from_json(str(p))
    assert result == intents


def test_load_hierarchical_intents_from_json_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_hierarchical_intents_from_json("/nonexistent/intents.json")


def test_load_hierarchical_intents_from_json_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{broken")
    with pytest.raises(ValueError):
        load_hierarchical_intents_from_json(str(p))


# ---------- save_hierarchy_to_json ----------

def test_save_hierarchy_to_json_creates_file(tmp_path):
    p = tmp_path / "out.json"
    save_hierarchy_to_json(SAMPLE_HIERARCHY, str(p))
    assert p.exists()
    loaded = json.loads(p.read_text())
    assert loaded == SAMPLE_HIERARCHY


def test_save_hierarchy_to_json_creates_parent_dirs(tmp_path):
    p = tmp_path / "nested" / "deep" / "hierarchy.json"
    save_hierarchy_to_json(SAMPLE_HIERARCHY, str(p))
    assert p.exists()


# ---------- validate_hierarchy ----------

def test_validate_hierarchy_valid():
    assert validate_hierarchy(SAMPLE_HIERARCHY) is True


def test_validate_hierarchy_empty():
    assert validate_hierarchy({}) is True


def test_validate_hierarchy_non_list_values():
    bad = {"node": "not_a_list"}
    assert validate_hierarchy(bad) is False


def test_validate_hierarchy_child_not_in_keys():
    # Child missing as key — warns but still returns True
    h = {"root": ["leaf_1"]}  # leaf_1 not a key
    result = validate_hierarchy(h)
    assert result is True  # warning logged, but not False


# ---------- get_leaf_nodes ----------

def test_get_leaf_nodes_returns_only_leaves():
    leaves = get_leaf_nodes(SAMPLE_HIERARCHY)
    assert set(leaves) == {"leaf_1", "leaf_2", "leaf_3"}


def test_get_leaf_nodes_empty_hierarchy():
    assert get_leaf_nodes({}) == []


def test_get_leaf_nodes_all_leaves():
    h = {"a": [], "b": [], "c": []}
    assert set(get_leaf_nodes(h)) == {"a", "b", "c"}


# ---------- get_parent_nodes ----------

def test_get_parent_nodes_returns_only_parents():
    parents = get_parent_nodes(SAMPLE_HIERARCHY)
    assert set(parents) == {"root", "child_a", "child_b"}


def test_get_parent_nodes_empty_hierarchy():
    assert get_parent_nodes({}) == []


# ---------- get_node_depth ----------

def test_get_node_depth_leaf_is_zero():
    assert get_node_depth(SAMPLE_HIERARCHY, "leaf_1") == 0


def test_get_node_depth_direct_child_is_one():
    assert get_node_depth(SAMPLE_HIERARCHY, "child_a") == 1


def test_get_node_depth_root_is_two():
    # root -> child_a -> leaf_1: root has 2 levels below it
    assert get_node_depth(SAMPLE_HIERARCHY, "root") == 2


def test_get_node_depth_node_not_in_hierarchy():
    assert get_node_depth(SAMPLE_HIERARCHY, "nonexistent") == 0
