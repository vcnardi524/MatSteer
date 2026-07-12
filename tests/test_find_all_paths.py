"""Tests for get_leaf_paths in find_all_paths.py.

find_all_paths.py has module-level side effects (reads a parquet, writes a file),
so we load it via importlib under patches to isolate the pure function.
"""
import importlib.util
import os
import sys
import unittest.mock as mock

import pandas as pd
import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "find_all_paths.py")


@pytest.fixture(scope="module")
def fap():
    empty_df = pd.DataFrame({"id": [], "results": []})
    with mock.patch("pandas.read_parquet", return_value=empty_df), \
         mock.patch("builtins.open", mock.mock_open()), \
         mock.patch("builtins.print"):
        spec = importlib.util.spec_from_file_location("find_all_paths", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


# ── get_leaf_paths ────────────────────────────────────────────────────────────

def test_flat_dict(fap):
    data = {"a": 1, "b": 2}
    assert set(fap.get_leaf_paths(data)) == {"a", "b"}


def test_nested_dict(fap):
    data = {"a": {"b": {"c": 1}}}
    assert fap.get_leaf_paths(data) == ["a.b.c"]


def test_mixed_dict_and_list(fap):
    data = {"x": [{"y": 1}, {"y": 2}]}
    # list items are traversed; both have the same path
    paths = fap.get_leaf_paths(data)
    assert paths.count("x.y") == 2


def test_empty_dict_is_skipped(fap):
    data = {"a": {}, "b": 1}
    # empty dict has no children → only "b" is a leaf
    assert fap.get_leaf_paths(data) == ["b"]


def test_empty_list_is_skipped(fap):
    data = {"a": [], "b": 2}
    assert fap.get_leaf_paths(data) == ["b"]


def test_scalar_at_root(fap):
    # a bare scalar (not dict/list) returns []
    assert fap.get_leaf_paths(42) == []
    assert fap.get_leaf_paths("hello") == []


def test_empty_input(fap):
    assert fap.get_leaf_paths({}) == []
    assert fap.get_leaf_paths([]) == []


def test_list_of_dicts_at_root(fap):
    data = [{"a": 1}, {"b": 2}]
    assert set(fap.get_leaf_paths(data)) == {"a", "b"}


def test_deeply_nested_with_list(fap):
    data = {"results": {"properties": {"electronic": [{"band_gap": 1.2}]}}}
    paths = fap.get_leaf_paths(data)
    assert "results.properties.electronic.band_gap" in paths
