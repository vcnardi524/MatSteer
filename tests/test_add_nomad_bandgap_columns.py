"""Tests for as_dict and first_value in add_nomad_bandgap_columns.py."""
import pytest

from add_nomad_bandgap_columns import as_dict, first_value


# ── as_dict ───────────────────────────────────────────────────────────────────

def test_as_dict_returns_dict_unchanged():
    d = {"key": "val"}
    assert as_dict(d) is d


def test_as_dict_list_returns_first_element():
    d = {"a": 1}
    assert as_dict([d]) is d


def test_as_dict_empty_list_returns_none():
    assert as_dict([]) is None


def test_as_dict_scalar_passes_through():
    assert as_dict(42) == 42
    assert as_dict(None) is None
    assert as_dict("str") == "str"


# ── first_value ───────────────────────────────────────────────────────────────

def test_first_value_single_element():
    val, n = first_value([{"value": 1.5e-19}])
    assert val == 1.5e-19
    assert n == 1


def test_first_value_returns_minimum_across_channels():
    items = [{"value": 3.0e-19}, {"value": 1.0e-19}, {"value": 2.0e-19}]
    val, n = first_value(items)
    assert val == 1.0e-19
    assert n == 3


def test_first_value_skips_missing_value_key():
    items = [{"value": None}, {"value": 2.0e-19}]
    val, n = first_value(items)
    assert val == 2.0e-19
    assert n == 2


def test_first_value_all_none_values():
    items = [{"value": None}, {"other": 1}]
    val, n = first_value(items)
    assert val is None
    assert n == 2


def test_first_value_empty_list():
    val, n = first_value([])
    assert val is None
    assert n == 0


def test_first_value_not_a_list():
    val, n = first_value(None)
    assert val is None
    assert n == 0

    val, n = first_value("not a list")
    assert val is None
    assert n == 0


def test_first_value_zero_is_valid():
    # band_gap == 0.0 (metal) must be preserved, not treated as missing
    val, n = first_value([{"value": 0.0}])
    assert val == 0.0
    assert n == 1


def test_first_value_non_dict_items_ignored():
    items = ["bad", {"value": 5.0e-19}]
    val, n = first_value(items)
    assert val == 5.0e-19
