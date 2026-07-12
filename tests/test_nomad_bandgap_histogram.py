"""Tests for the dig() helper in nomad_bandgap_histogram.py."""
import pytest

from nomad_bandgap_histogram import dig


def test_simple_nested_path():
    d = {"a": {"b": {"c": 42}}}
    assert dig(d, "a.b.c") == 42


def test_missing_key_returns_none():
    d = {"a": {"b": 1}}
    assert dig(d, "a.x") is None


def test_missing_top_level_key():
    assert dig({}, "a") is None


def test_single_key():
    d = {"x": 99}
    assert dig(d, "x") == 99


def test_value_is_none():
    d = {"a": None}
    # None is a valid value to return; further traversal on None returns None
    assert dig(d, "a") is None


def test_non_dict_intermediate_returns_none():
    d = {"a": 5}
    assert dig(d, "a.b") is None


def test_full_nomad_bandgap_path():
    d = {
        "properties": {
            "electronic": {
                "band_gap": {"value": 1.6e-19},
                "dos_electronic": {
                    "band_gap": {"value": 2.4e-19}
                }
            }
        }
    }
    assert dig(d, "properties.electronic.band_gap.value") == 1.6e-19
    assert dig(d, "properties.electronic.dos_electronic.band_gap.value") == 2.4e-19


def test_path_longer_than_structure():
    d = {"a": {"b": 1}}
    assert dig(d, "a.b.c.d") is None
