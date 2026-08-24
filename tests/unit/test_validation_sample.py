"""Deterministic blinded validation sampling. Offline only."""

from __future__ import annotations

import csv
from pathlib import Path

from validation.sample import (
    even_allocation,
    run_sampling,
    sample_validation_tasks,
    write_sampling_outputs,
)
from validation.schema import BLINDED_ANNOTATOR_COLUMNS, FORBIDDEN_ANNOTATOR_FIELDS

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "synthetic"
    / "validation"
    / "model_annotations.csv"
)


def _rows() -> list[dict[str, str]]:
    with FIXTURE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_even_allocation_spreads_across_strata() -> None:
    alloc = even_allocation({"a": 10, "b": 2, "c": 1}, 4)
    assert alloc["a"] >= 1
    assert alloc["b"] >= 1
    assert alloc["c"] >= 1
    assert sum(alloc.values()) == 4


def test_sampling_is_deterministic_for_fixed_seed() -> None:
    rows = _rows()
    first = sample_validation_tasks(rows, sample_size=6, seed=42)
    second = sample_validation_tasks(rows, sample_size=6, seed=42)
    third = sample_validation_tasks(rows, sample_size=6, seed=7)
    ids_first = [row["original_record_id"] for row in first.index_rows]
    ids_second = [row["original_record_id"] for row in second.index_rows]
    ids_third = [row["original_record_id"] for row in third.index_rows]
    assert ids_first == ids_second
    assert [row["task_id"] for row in first.index_rows] == [
        row["task_id"] for row in second.index_rows
    ]
    assert ids_first != ids_third or first.config["seed"] != third.config["seed"]


def test_requested_strata_are_represented_when_possible() -> None:
    records = []
    for i in range(8):
        records.append(
            {
                "id": f"SYNTH-VAL-{i:04d}",
                "subreddit": "SynthArt",
                "subreddit_group": "creative_AI_communities",
                "matched_bias_types": "gender",
                "clean_text": f"[SYNTHETIC] majority yes item {i}",
                "status": "success",
                "pred_label": "yes",
            }
        )
    records.append(
        {
            "id": "SYNTH-VAL-0090",
            "subreddit": "SynthTech",
            "subreddit_group": "technical",
            "matched_bias_types": "age",
            "clean_text": "[SYNTHETIC] minority no item",
            "status": "success",
            "pred_label": "no",
        }
    )
    result = sample_validation_tasks(records, sample_size=4, seed=1)
    labels = {row["model_pred_label"] for row in result.index_rows}
    groups = {row["subreddit_group"] for row in result.index_rows}
    assert labels == {"yes", "no"}
    assert "technical" in groups
    assert "creative_AI_communities" in groups


def test_duplicate_source_rows_do_not_create_duplicate_tasks() -> None:
    result = sample_validation_tasks(_rows(), sample_size=50, seed=42)
    original_ids = [row["original_record_id"] for row in result.index_rows]
    assert len(original_ids) == len(set(original_ids))
    assert result.diagnostics["duplicates_dropped"] >= 1
    assert "SYNTH-VAL-0001" in original_ids


def test_insufficient_stratum_size_is_reported() -> None:
    records = [
        {
            "id": "SYNTH-VAL-0101",
            "subreddit": "SynthArt",
            "subreddit_group": "creative_AI_communities",
            "matched_bias_types": "gender",
            "clean_text": "[SYNTHETIC] only row",
            "status": "success",
            "pred_label": "yes",
        }
    ]
    result = sample_validation_tasks(records, sample_size=10, seed=0)
    assert len(result.index_rows) == 1
    assert result.diagnostics["success_pool"]["actual"] == 1
    assert result.diagnostics["success_pool"]["requested"] == 10
    assert result.diagnostics["success_pool"]["stratum_shortfalls"]["__total__"] == 9


def test_annotator_export_is_blinded(tmp_path: Path) -> None:
    result = sample_validation_tasks(_rows(), sample_size=5, seed=42)
    paths = write_sampling_outputs(result, tmp_path)
    with paths["annotation_tasks"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        task_rows = list(reader)
    assert fieldnames == list(BLINDED_ANNOTATOR_COLUMNS)
    forbidden = {name.lower() for name in fieldnames} & {
        field.lower() for field in FORBIDDEN_ANNOTATOR_FIELDS
    }
    assert not forbidden
    for row in task_rows:
        for field in FORBIDDEN_ANNOTATOR_FIELDS:
            assert field not in row or not row[field]
        assert row["human_label"] == ""
        assert "yes" != row.get("model_pred_label", "")
        assert "raw_output" not in row
        assert "status" not in row


def test_private_index_keeps_linkage(tmp_path: Path) -> None:
    result = sample_validation_tasks(
        _rows(), sample_size=5, seed=42, failure_sample_size=2
    )
    paths = write_sampling_outputs(result, tmp_path)
    with paths["sampling_index"].open(newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    assert index_rows
    assert "original_record_id" in index_rows[0]
    assert "model_pred_label" in index_rows[0]
    assert "model_status" in index_rows[0]


def test_run_sampling_cli_paths(tmp_path: Path) -> None:
    result = run_sampling(
        FIXTURE,
        tmp_path,
        sample_size=8,
        seed=42,
        failure_sample_size=2,
    )
    assert (tmp_path / "sampling_index.csv").is_file()
    assert (tmp_path / "annotation_tasks.csv").is_file()
    assert result.config["seed"] == 42
    assert "config_hash" in result.config
    assert len(result.config["config_hash"]) == 64
