"""Cluster-aware exploratory emotion models. Synthetic rows only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.emotion_manova import run_exploratory_manova
from analysis.emotion_statistics import (
    EmotionStatisticsError,
    run_emotion_statistics,
    write_statistics_outputs,
)
from analysis.topic_mapping import TopicMappingError

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "synthetic" / "methodology"
)
EMOTION_SCORES = FIXTURE_DIR / "emotion_scores.csv"


def _synthetic_frame() -> pd.DataFrame:
    return pd.read_csv(EMOTION_SCORES)


def test_statistics_import_has_no_research_side_effects():
    import analysis.emotion_clustering as clustering
    import analysis.emotion_manova as manova
    import analysis.emotion_mean_plot as mean_plot
    import analysis.emotion_statistics as stats

    assert callable(stats.run_emotion_statistics)
    assert callable(manova.run_exploratory_manova)
    assert clustering.__doc__
    assert mean_plot.__doc__


def test_cluster_column_is_required_in_clustered_mode():
    frame = _synthetic_frame().drop(columns=["subreddit"])
    with pytest.raises(EmotionStatisticsError, match="cluster column"):
        run_emotion_statistics(frame)


def test_missing_cluster_labels_do_not_fall_back():
    frame = _synthetic_frame()
    frame.loc[0, "subreddit"] = None
    with pytest.raises(EmotionStatisticsError, match="missing cluster"):
        run_emotion_statistics(frame)


def test_all_emotions_are_reported_including_null_results():
    frame = _synthetic_frame()
    result = run_emotion_statistics(frame)
    tests = result["tests"]
    assert list(tests["emotion"]) == [
        "emotion_admiration",
        "emotion_anger",
        "emotion_joy",
    ]
    assert tests["p_raw"].notna().any()
    assert "p_fdr_bh" in tests.columns
    assert tests["p_fdr_bh"].notna().sum() == tests["p_raw"].notna().sum()
    assert int(result["manifest"]["n_clusters"]) == 12
    assert result["manifest"]["multiple_testing_method"] == "benjamini_hochberg"
    assert result["manifest"]["hypotheses_in_fdr_family"] == 3
    # Admiration was constructed with a large category gap; anger was not.
    admiration = tests.set_index("emotion").loc["emotion_admiration"]
    anger = tests.set_index("emotion").loc["emotion_anger"]
    assert admiration["p_raw"] < 0.05
    assert anger["n_posts"] > 0


def test_descriptives_and_contrasts_include_effects_and_ordered_cis():
    frame = _synthetic_frame()
    result = run_emotion_statistics(frame)
    descriptives = result["descriptives"]
    assert {"n_posts", "n_subreddits", "mean", "std", "category", "emotion"} <= set(
        descriptives.columns
    )
    assert descriptives["n_posts"].gt(0).all()
    contrasts = result["contrasts"]
    assert not contrasts.empty
    defined = contrasts.dropna(subset=["ci_95_low", "ci_95_high"])
    assert not defined.empty
    assert (defined["ci_95_low"] <= defined["ci_95_high"]).all()
    assert contrasts["estimate"].notna().all()


def test_few_clusters_are_flagged_not_silently_treated_as_solved():
    frame = _synthetic_frame().head(12).copy()
    result = run_emotion_statistics(frame)
    assert result["manifest"]["inference_status"] == "limited_few_clusters"
    assert result["manifest"]["n_clusters"] < 10
    assert any(
        "few clusters" in item.lower() for item in result["manifest"]["limitations"]
    )


def test_aggregate_outputs_exclude_source_text_and_record_ids():
    frame = _synthetic_frame()
    result = run_emotion_statistics(frame, input_filename="emotion_scores.csv")
    for table in (result["descriptives"], result["tests"], result["contrasts"]):
        joined = " ".join(table.astype(str).fillna("").values.ravel())
        assert "SYNTH-STAT-" not in joined
        assert "clean_text" not in table.columns
        assert "permalink" not in table.columns


def test_mismatched_mapping_is_rejected(tmp_path):
    frame = _synthetic_frame()
    mapping_path = tmp_path / "bad_mapping.json"
    mapping_path.write_text(
        """
        {
          "labels": {"0": "portrayal_stereotype"},
          "mapping_version": 1,
          "topic_assignment_checksum": "not-the-seed-11-checksum",
          "topic_run_id": "synthetic-topic-run-seed-11"
        }
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TopicMappingError):
        run_emotion_statistics(
            frame.drop(columns=["mapped_topic_category"]),
            mapping_path=mapping_path,
            topic_run_id="synthetic-topic-run-seed-11",
        )


def test_write_outputs_and_manifest(tmp_path):
    result = run_emotion_statistics(_synthetic_frame())
    paths = write_statistics_outputs(result, tmp_path)
    tests = pd.read_csv(paths["tests"])
    assert set(tests["emotion"]) == {
        "emotion_admiration",
        "emotion_anger",
        "emotion_joy",
    }
    manifest = (tmp_path / "analysis_manifest.json").read_text(encoding="utf-8")
    assert "benjamini_hochberg" in manifest
    assert "SYNTH-STAT-" not in manifest
    assert "cluster-robust" in manifest


def test_exploratory_manova_is_labeled_unadjusted():
    frame = _synthetic_frame().dropna(subset=["mapped_topic_category"])
    report = run_exploratory_manova(
        frame,
        emotion_columns=["emotion_admiration", "emotion_anger"],
        group_column="mapped_topic_category",
    )
    assert "UNADJUSTED EXPLORATORY MANOVA" in report
    assert "not cluster-aware" in report
