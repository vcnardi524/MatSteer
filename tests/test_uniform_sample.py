"""Tests for uniform_sample_by_value in plot_tsne_pca_bandgap.py."""
import numpy as np
import pytest

from plot_tsne_pca_bandgap import uniform_sample_by_value


RNG = np.random.default_rng(42)


def test_returns_valid_indices():
    values = np.linspace(0.0, 10.0, 200)
    idx = uniform_sample_by_value(values, n=50, bins=10)
    assert idx.dtype.kind == "i" or idx.dtype.kind == "u"
    assert len(idx) <= 50
    assert (idx >= 0).all() and (idx < len(values)).all()


def test_no_duplicate_indices():
    values = np.linspace(0.0, 5.0, 500)
    idx = uniform_sample_by_value(values, n=100, bins=10)
    assert len(idx) == len(set(idx.tolist()))


def test_respects_n_upper_bound():
    values = np.linspace(0.0, 1.0, 1000)
    for n in [10, 50, 100, 200]:
        idx = uniform_sample_by_value(values, n=n, bins=20)
        assert len(idx) <= n


def test_spreads_across_range():
    # Create values concentrated in [0,1]; uniform sampling should still
    # draw from the full range when possible.
    values = np.concatenate([
        np.linspace(0.0, 1.0, 900),
        np.linspace(9.0, 10.0, 100),
    ])
    idx = uniform_sample_by_value(values, n=100, bins=10)
    sampled = values[idx]
    # both ends of the range should be represented
    assert sampled.min() < 1.5
    assert sampled.max() > 8.5


def test_works_with_fewer_points_than_n():
    values = np.array([1.0, 2.0, 3.0])
    idx = uniform_sample_by_value(values, n=50, bins=3)
    # can't return more indices than available points
    assert len(idx) <= len(values)


def test_single_value():
    values = np.array([5.0] * 20)
    # all in the same bin; should not crash
    idx = uniform_sample_by_value(values, n=5, bins=5)
    assert len(idx) <= 5


def test_two_values():
    values = np.array([0.0, 1.0])
    idx = uniform_sample_by_value(values, n=2, bins=2)
    assert len(idx) <= 2
