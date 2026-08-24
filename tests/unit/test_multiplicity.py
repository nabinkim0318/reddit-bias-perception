"""Benjamini-Hochberg FDR helper."""

from __future__ import annotations

import pytest

from analysis.multiplicity import benjamini_hochberg, family_size


def test_benjamini_hochberg_known_values_preserve_order():
    p_values = [0.06, 0.001, 0.05]
    adjusted = benjamini_hochberg(p_values)
    assert adjusted[0] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.003)
    assert adjusted[2] == pytest.approx(0.06)
    assert family_size(p_values) == 3


def test_null_pvalues_excluded_from_family_and_kept_in_place():
    p_values = [0.01, None, float("nan"), 0.04]
    adjusted = benjamini_hochberg(p_values)
    assert adjusted[1] is None
    assert adjusted[2] is None
    assert adjusted[0] is not None
    assert adjusted[3] is not None
    assert family_size(p_values) == 2
    assert all(value is None or 0.0 <= value <= 1.0 for value in adjusted)


def test_bh_monotonicity_in_sorted_raw_p():
    p_values = [0.20, 0.01, 0.04, 0.03]
    adjusted = benjamini_hochberg(p_values)
    ranked = sorted(
        (p, q) for p, q in zip(p_values, adjusted) if p is not None and q is not None
    )
    q_in_rank_order = [q for _, q in ranked]
    assert q_in_rank_order == sorted(q_in_rank_order)
    assert all(0.0 <= q <= 1.0 for q in q_in_rank_order)


def test_invalid_pvalues_raise():
    with pytest.raises(ValueError):
        benjamini_hochberg([1.2])
