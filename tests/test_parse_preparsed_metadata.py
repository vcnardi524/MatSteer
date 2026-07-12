"""Tests for get_by_path and derive_col_names in parse_preparsed_metadata.py."""
import math

import pytest

from parse_preparsed_metadata import derive_col_names, get_by_path


# ── get_by_path ───────────────────────────────────────────────────────────────

def test_simple_path():
    obj = {"a": {"b": 42}}
    assert get_by_path(obj, "a.b") == 42


def test_single_key():
    obj = {"x": 7}
    assert get_by_path(obj, "x") == 7


def test_missing_key_returns_none():
    obj = {"a": {"b": 1}}
    assert get_by_path(obj, "a.c") is None


def test_missing_intermediate_returns_none():
    obj = {"a": 1}
    assert get_by_path(obj, "a.b.c") is None


def test_list_at_intermediate_uses_first_element():
    obj = {"a": [{"b": 99}]}
    assert get_by_path(obj, "a.b") == 99


def test_list_at_leaf_returns_first_element():
    obj = {"a": [10, 20, 30]}
    assert get_by_path(obj, "a") == 10


def test_empty_list_returns_none():
    obj = {"a": []}
    assert get_by_path(obj, "a") is None


def test_deeply_nested():
    obj = {"p": {"q": {"r": {"s": "deep"}}}}
    assert get_by_path(obj, "p.q.r.s") == "deep"


def test_none_value_at_key():
    obj = {"a": None}
    assert get_by_path(obj, "a") is None


def test_non_dict_intermediate():
    obj = {"a": 5}
    # "a" is an int, so descending further must return None
    assert get_by_path(obj, "a.b") is None


# ── derive_col_names ──────────────────────────────────────────────────────────

def test_single_path_last_segment():
    paths = ["material.symmetry.point_group"]
    mapping = derive_col_names(paths)
    assert mapping["material.symmetry.point_group"] == "point_group"


def test_collision_resolved_by_more_segments():
    paths = [
        "a.b.name",
        "a.c.name",  # same last segment → needs 2 segments
    ]
    mapping = derive_col_names(paths)
    values = list(mapping.values())
    # both must be unique
    assert len(set(values)) == 2
    # shorter path gets the 1-segment name first
    assert "name" in values


def test_no_collision_uses_minimal_suffix():
    paths = ["foo.bar.baz", "x.y.z"]
    mapping = derive_col_names(paths)
    assert mapping["foo.bar.baz"] == "baz"
    assert mapping["x.y.z"] == "z"


def test_all_unique_last_segments():
    paths = ["a.x", "b.y", "c.z"]
    mapping = derive_col_names(paths)
    assert mapping == {"a.x": "x", "b.y": "y", "c.z": "z"}


def test_returns_full_path_when_no_unique_suffix_possible():
    # Three paths that share the same tail at every depth
    paths = ["a.name", "b.name"]
    mapping = derive_col_names(paths)
    values = list(mapping.values())
    assert len(set(values)) == 2


def test_single_segment_path():
    paths = ["toplevel"]
    mapping = derive_col_names(paths)
    assert mapping["toplevel"] == "toplevel"
