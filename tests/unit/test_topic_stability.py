"""Permutation-invariant topic-assignment stability."""

from __future__ import annotations

import pytest

from analysis.topic_stability import (
    compute_assignment_stability,
    summarize_run,
    summarize_topic_stability,
)


def test_identical_clusterings_under_permuted_labels_have_ari_one():
    left = [0, 0, 1, 1, -1, 2, 2]
    right = [7, 7, 4, 4, -1, 9, 9]
    agreement = compute_assignment_stability(left, right)
    assert agreement["ari_all"] == pytest.approx(1.0)
    assert agreement["ari_inliers_both"] == pytest.approx(1.0)
    assert agreement["inliers_both_denominator"] == 6


def test_clearly_different_clusterings_have_lower_ari():
    left = [0, 0, 0, 0, 1, 1, 1, 1]
    right = [0, 1, 0, 1, 0, 1, 0, 1]
    agreement = compute_assignment_stability(left, right)
    assert agreement["ari_all"] < 0.2


def test_outliers_are_not_silently_dropped_from_all_ari():
    left = [0, 0, -1, 1]
    right = [0, 0, -1, 1]
    agreement = compute_assignment_stability(left, right)
    assert agreement["n_documents"] == 4
    assert agreement["inliers_both_denominator"] == 3
    assert "ordinary label" in agreement["outlier_treatment"]["ari_all"]


def test_stability_report_topic_count_and_outlier_rate_variation():
    runs = [
        summarize_run(11, [0, 0, 1, 1, -1]),
        summarize_run(23, [2, 2, 3, 3, 3]),
        summarize_run(37, [0, 1, 2, -1, -1]),
    ]
    report = summarize_topic_stability(runs)
    assert report["n_runs"] == 3
    assert report["stability_declared"] is False
    assert report["stability_metric"] == "adjusted_rand_index"
    assert report["topic_count_distribution"]["values"] == [2.0, 2.0, 3.0] or set(
        report["topic_count_distribution"]["values"]
    ) == {2.0, 3.0}
    assert report["topic_count_distribution"]["min"] == 2.0
    assert report["topic_count_distribution"]["max"] == 3.0
    rates = report["outlier_rate_distribution"]["values"]
    assert min(rates) == pytest.approx(0.0)
    assert max(rates) == pytest.approx(0.4)
    assert report["pairwise_ari_all"]["min"] <= report["pairwise_ari_all"]["median"]
    assert report["pairwise_ari_all"]["median"] <= report["pairwise_ari_all"]["max"]
    pair_seeds = [
        (pair["seed_a"], pair["seed_b"]) for pair in report["pairwise_ari_all"]["pairs"]
    ]
    assert pair_seeds == sorted(pair_seeds)
    again = summarize_topic_stability(list(reversed(runs)))
    assert again["pairwise_ari_all"] == report["pairwise_ari_all"]
    assert again["seeds"] == [11, 23, 37]
