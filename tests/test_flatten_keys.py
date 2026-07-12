"""Tests for _flatten_keys in get_preparsed_metadata_nomad.py."""
import pytest

from get_preparsed_metadata_nomad import _flatten_keys


def test_flat_dict():
    obj = {"a": 1, "b": 2}
    keys = _flatten_keys(obj)
    assert "a" in keys
    assert "b" in keys


def test_nested_dict_produces_parent_and_child():
    obj = {"a": {"b": 1}}
    keys = _flatten_keys(obj)
    # parent key included
    assert "a" in keys
    # child key with full path included
    assert "a.b" in keys


def test_deeply_nested():
    obj = {"x": {"y": {"z": 99}}}
    keys = _flatten_keys(obj)
    assert "x" in keys
    assert "x.y" in keys
    assert "x.y.z" in keys


def test_list_recurses_into_first_element():
    obj = {"items": [{"value": 1.0}]}
    keys = _flatten_keys(obj)
    assert "items" in keys
    assert "items.value" in keys


def test_empty_list_not_recursed():
    obj = {"empty": []}
    keys = _flatten_keys(obj)
    assert "empty" in keys
    # no deeper key should appear
    assert not any(k.startswith("empty.") for k in keys)


def test_empty_dict():
    assert _flatten_keys({}) == []


def test_non_dict_non_list():
    # scalars at root produce no keys
    assert _flatten_keys(42) == []
    assert _flatten_keys("str") == []
    assert _flatten_keys(None) == []


def test_prefix_forwarded():
    obj = {"b": 1}
    keys = _flatten_keys(obj, prefix="a")
    assert "a.b" in keys


def test_realistic_nomad_structure():
    obj = {
        "properties": {
            "electronic": {
                "band_gap": [{"value": 1.6e-19, "type": "direct"}]
            }
        }
    }
    keys = _flatten_keys(obj)
    assert "properties" in keys
    assert "properties.electronic" in keys
    assert "properties.electronic.band_gap" in keys
    assert "properties.electronic.band_gap.value" in keys
    assert "properties.electronic.band_gap.type" in keys
