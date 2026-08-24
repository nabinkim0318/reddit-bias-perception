"""Offline synthetic methodology regression.

Topic assignments → structural stability → run-specific category mapping →
cluster-aware emotion models → BH correction → aggregate outputs.

No network, no BERTopic fit, no embedding download, no Reddit source text.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pandas as pd
import pytest

from analysis.emotion_statistics import run_emotion_statistics, write_statistics_outputs
from analysis.topic_mapping import (
    apply_topic_category_mapping,
    load_topic_category_mapping,
)
from analysis.topic_stability import (
    assignment_checksum,
    summarize_run,
    summarize_topic_stability,
    write_stability_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic" / "methodology"


def test_synthetic_methodology_e2e(tmp_path, monkeypatch):
    monkeypatch.setattr(
        socket,
        "socket",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("network disabled in methodology e2e")
        ),
    )

    assignments_payload = json.loads(
        (FIXTURE_DIR / "topic_assignments.json").read_text(encoding="utf-8")
    )
    runs = [
        summarize_run(int(item["seed"]), item["assignments"])
        for item in assignments_payload["runs"]
    ]
    stability = summarize_topic_stability(runs)
    write_stability_report(tmp_path / "topic_stability_report.json", stability)

    assert stability["n_runs"] == 5
    assert stability["stability_declared"] is False
    assert (
        stability["topic_count_distribution"]["min"]
        < stability["topic_count_distribution"]["max"]
    )
    assert (
        stability["outlier_rate_distribution"]["min"]
        < stability["outlier_rate_distribution"]["max"]
    )
    permuted = next(
        pair
        for pair in stability["pairwise_ari_all"]["pairs"]
        if pair["seed_a"] == 11 and pair["seed_b"] == 23
    )
    assert permuted["ari_all"] == pytest.approx(1.0)

    seed_11 = next(run for run in assignments_payload["runs"] if run["seed"] == 11)
    checksum = assignment_checksum(seed_11["assignments"])
    mapping = load_topic_category_mapping(FIXTURE_DIR / "topic_to_category.json")
    mapped = apply_topic_category_mapping(
        seed_11["assignments"],
        mapping,
        topic_run_id="synthetic-topic-run-seed-11",
        topic_assignment_checksum=checksum,
    )
    assert mapped[0] == "portrayal_stereotype"
    assert mapped[-1] is None

    frame = pd.read_csv(FIXTURE_DIR / "emotion_scores.csv")
    assert list(frame["topic"].astype(int)) == seed_11["assignments"]
    result = run_emotion_statistics(
        frame,
        mapping_path=FIXTURE_DIR / "topic_to_category.json",
        topic_run_id="synthetic-topic-run-seed-11",
        topic_assignment_checksum=checksum,
        input_filename="emotion_scores.csv",
    )
    paths = write_statistics_outputs(result, tmp_path / "stats")
    tests = pd.read_csv(paths["tests"])
    descriptives = pd.read_csv(paths["descriptives"])
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    stability_disk = json.loads(
        (tmp_path / "topic_stability_report.json").read_text(encoding="utf-8")
    )

    assert set(tests["emotion"]) == {
        "emotion_admiration",
        "emotion_anger",
        "emotion_joy",
    }
    assert tests["p_fdr_bh"].notna().all()
    assert manifest["multiple_testing_method"] == "benjamini_hochberg"
    assert manifest["n_clusters"] == 12
    assert manifest["mapping"]["mapping_provenance"] == "enforced"
    assert "cluster-robust" in manifest["covariance_specification"]
    assert "SYNTH-STAT-" not in tests.to_csv(index=False)
    assert "clean_text" not in descriptives.columns
    assert stability_disk["stability_metric"] == "adjusted_rand_index"
    assert "reddit.com" not in Path(paths["manifest"]).read_text(encoding="utf-8")
    assert result["tests"].equals(
        run_emotion_statistics(
            frame,
            mapping_path=FIXTURE_DIR / "topic_to_category.json",
            topic_run_id="synthetic-topic-run-seed-11",
            topic_assignment_checksum=checksum,
        )["tests"]
    )
