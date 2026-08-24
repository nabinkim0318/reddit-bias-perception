"""Offline tests for LLM annotation correctness. No model download or network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from processing.llm_annotation import (
    ERROR_GENERATION_EXCEPTION,
    ERROR_OUTPUT_CARDINALITY_MISMATCH,
    STATUS_MODEL_ERROR,
    STATUS_PARSE_ERROR,
    STATUS_SUCCESS,
    AnnotationModelError,
    classify_batch,
    parse_annotation_output,
    split_annotation_frames,
    summarize_annotation_counts,
    write_annotation_outputs,
)
from processing.schema import ANNOTATION_SCHEMA_VERSION, ClassificationResult


def _ids(*n: int) -> list[str]:
    return [f"SYNTH-ANN-{i}" for i in n]


# --- Parser: valid -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("yes", "yes"),
        ("no", "no"),
        (" YES ", "yes"),
        ("No", "no"),
        ('"yes"', "yes"),
        ("Label: yes", "yes"),
        ("Label: no", "no"),
        ("label: YES", "yes"),
        ('Label: "no"', "no"),
        ("Label: yes\nReasoning: short synthetic rationale.", "yes"),
    ],
)
def test_parser_accepts_contract_yes_no(raw: str, expected: str) -> None:
    parsed = parse_annotation_output(raw)
    assert parsed.status == STATUS_SUCCESS
    assert parsed.pred_label == expected
    assert parsed.error_type is None


# --- Parser: malformed -------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        None,
        "   ",
        "[EMPTY_OUTPUT]",
        "The synthetic item discusses representation.",
        "Probably yes, but I am not sure",
        "yes and no",
        "Label: yes and no",
        "Label: yes\nLabel: no",
        '{"label": "yes"}',
        "I cannot determine this",
        "Reasoning: looks like a positive case.",
        "Inference error: generation failed",
        "yes, with caveats",
    ],
)
def test_parser_rejects_malformed_without_label(raw) -> None:
    parsed = parse_annotation_output(raw)
    assert parsed.status == STATUS_PARSE_ERROR
    assert parsed.pred_label is None
    assert parsed.error_type is not None


def test_parser_empty_uses_empty_output_code() -> None:
    parsed = parse_annotation_output("")
    assert parsed.error_type == "empty_output"


def test_parser_ambiguous_both_labels() -> None:
    parsed = parse_annotation_output("Label: yes\nLabel: no")
    assert parsed.status == STATUS_PARSE_ERROR
    assert parsed.pred_label is None
    assert parsed.error_type == "ambiguous_output"


# --- Schema ------------------------------------------------------------------


def test_schema_success_requires_yes_or_no() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(
            id="SYNTH-ANN-0",
            subreddit="synthetic",
            clean_text="synthetic item",
            status="success",
            pred_label=None,
        )


def test_schema_failure_forbids_scientific_label() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(
            id="SYNTH-ANN-0",
            subreddit="synthetic",
            clean_text="synthetic item",
            status="parse_error",
            pred_label="no",
        )


def test_schema_version_constant() -> None:
    row = ClassificationResult(
        id="SYNTH-ANN-0",
        subreddit="synthetic",
        clean_text="synthetic item",
        status="success",
        pred_label="yes",
    )
    assert row.schema_version == ANNOTATION_SCHEMA_VERSION == 2


# --- Model / batch failures --------------------------------------------------


def test_generation_exception_is_model_error_with_null_label() -> None:
    def _raise(_texts: list[str]) -> list[str]:
        raise RuntimeError("synthetic generation failure")

    rows = classify_batch(
        ["synthetic a", "synthetic b"],
        _ids(1, 2),
        ["synthetic", "synthetic"],
        _raise,
    )
    assert len(rows) == 2
    for row, expected_id in zip(rows, _ids(1, 2)):
        assert row["id"] == expected_id
        assert row["status"] == STATUS_MODEL_ERROR
        assert row["pred_label"] is None
        assert row["error_type"] == ERROR_GENERATION_EXCEPTION


def test_typed_annotation_model_error_preserves_error_type() -> None:
    def _raise(_texts: list[str]) -> list[str]:
        raise AnnotationModelError("model_unavailable")

    rows = classify_batch(["synthetic a"], _ids(1), ["synthetic"], _raise)
    assert rows[0]["status"] == STATUS_MODEL_ERROR
    assert rows[0]["pred_label"] is None
    assert rows[0]["error_type"] == "model_unavailable"


def test_failed_batch_does_not_label_every_item_no() -> None:
    def _raise(_texts: list[str]) -> list[str]:
        raise RuntimeError("boom")

    rows = classify_batch(
        ["a", "b", "c"],
        _ids(1, 2, 3),
        ["s", "s", "s"],
        _raise,
    )
    assert [row["pred_label"] for row in rows] == [None, None, None]
    assert all(row["status"] == STATUS_MODEL_ERROR for row in rows)


def test_cardinality_mismatch_marks_all_inputs_model_error() -> None:
    def _short(_texts: list[str]) -> list[str]:
        return ["yes"]

    rows = classify_batch(
        ["a", "b"],
        _ids(1, 2),
        ["s", "s"],
        _short,
    )
    assert len(rows) == 2
    assert {row["id"] for row in rows} == set(_ids(1, 2))
    for row in rows:
        assert row["status"] == STATUS_MODEL_ERROR
        assert row["pred_label"] is None
        assert row["error_type"] == ERROR_OUTPUT_CARDINALITY_MISMATCH


def test_mixed_batch_stays_aligned() -> None:
    def _mixed(_texts: list[str]) -> list[str]:
        return ["yes", "not a label", "Label: no"]

    rows = classify_batch(
        ["synthetic yes", "synthetic malformed", "synthetic no"],
        _ids(1, 2, 3),
        ["s", "s", "s"],
        _mixed,
    )
    assert [row["id"] for row in rows] == _ids(1, 2, 3)
    assert rows[0]["status"] == STATUS_SUCCESS and rows[0]["pred_label"] == "yes"
    assert rows[1]["status"] == STATUS_PARSE_ERROR and rows[1]["pred_label"] is None
    assert rows[2]["status"] == STATUS_SUCCESS and rows[2]["pred_label"] == "no"


# --- Export / split ----------------------------------------------------------


def _frame_from_generate(outputs: list[str]) -> pd.DataFrame:
    n = len(outputs)
    rows = classify_batch(
        [f"synthetic {i}" for i in range(n)],
        _ids(*range(n)),
        ["synthetic"] * n,
        lambda _texts: outputs,
    )
    return pd.DataFrame(rows)


def test_split_keeps_failures_out_of_yes_no(tmp_path: Path) -> None:
    result_df = _frame_from_generate(["yes", "no", "garbled", "yes"])
    yes_df, no_df, unclassified_df = split_annotation_frames(result_df)

    assert list(yes_df["pred_label"]) == ["yes", "yes"]
    assert list(no_df["pred_label"]) == ["no"]
    assert len(unclassified_df) == 1
    assert (
        unclassified_df["pred_label"].isna().all()
        or unclassified_df["pred_label"].isnull().all()
    )
    assert set(unclassified_df["status"]) == {STATUS_PARSE_ERROR}

    paths = {
        "yes": tmp_path / "yes.csv",
        "no": tmp_path / "no.csv",
        "unclassified": tmp_path / "unclassified.csv",
        "combined": tmp_path / "combined.csv",
    }
    counts = write_annotation_outputs(result_df, paths)
    assert counts["success_yes"] == 2
    assert counts["success_no"] == 1
    assert counts["parse_error"] == 1
    assert counts["model_error"] == 0
    assert counts["unclassified"] == 1

    written_no = pd.read_csv(paths["no"])
    written_unclassified = pd.read_csv(paths["unclassified"])
    assert list(written_no["pred_label"]) == ["no"]
    assert len(written_no) == 1
    assert len(written_unclassified) == 1
    assert list(written_unclassified["status"]) == [STATUS_PARSE_ERROR]
    assert len(pd.read_csv(paths["combined"])) == 4


def test_counts_exclude_failures_from_success_denominator() -> None:
    result_df = _frame_from_generate(["yes", "no", "???"])
    counts = summarize_annotation_counts(result_df)
    assert counts["success_yes"] + counts["success_no"] == 2
    assert counts["parse_error"] == 1
    assert counts["total"] == 3


# --- Regression: failure never becomes "no" ----------------------------------


FAILURE_OUTPUTS = [
    "",
    "Probably yes, but I am not sure",
    "yes and no",
    "The synthetic text discusses bias",
    "I cannot determine this",
    '{"label": "no"}',
    "Inference error: CUDA",
    "Model not available",
]


def test_annotation_failure_never_becomes_pred_label_no() -> None:
    """Regression: unsuccessful classification must not become pred_label == 'no'."""
    for raw in FAILURE_OUTPUTS:
        parsed = parse_annotation_output(raw)
        assert parsed.pred_label is None
        assert parsed.pred_label != "no"
        assert parsed.status != STATUS_SUCCESS

    def _raise(_texts: list[str]) -> list[str]:
        raise RuntimeError("synthetic")

    rows = classify_batch(["synthetic"], _ids(1), ["s"], _raise)
    assert rows[0]["pred_label"] is None
    assert rows[0]["pred_label"] != "no"
    assert rows[0]["status"] == STATUS_MODEL_ERROR

    def _mismatch(_texts: list[str]) -> list[str]:
        return []

    rows = classify_batch(
        ["synthetic a", "synthetic b"], _ids(1, 2), ["s", "s"], _mismatch
    )
    for row in rows:
        assert row["pred_label"] is None
        assert row["status"] == STATUS_MODEL_ERROR
        assert row["error_type"] == ERROR_OUTPUT_CARDINALITY_MISMATCH


def test_successful_no_is_preserved() -> None:
    rows = classify_batch(
        ["synthetic"],
        _ids(1),
        ["s"],
        lambda _texts: ["no"],
    )
    assert rows[0]["status"] == STATUS_SUCCESS
    assert rows[0]["pred_label"] == "no"
    yes_df, no_df, unclassified_df = split_annotation_frames(pd.DataFrame(rows))
    assert len(yes_df) == 0
    assert len(no_df) == 1
    assert len(unclassified_df) == 0


def test_success_with_null_label_is_unclassified() -> None:
    """Malformed v2: status=success with a null label must not enter yes/no."""
    result_df = pd.DataFrame(
        [
            {
                "id": "SYNTH-ANN-0",
                "subreddit": "synthetic",
                "clean_text": "synthetic item",
                "status": STATUS_SUCCESS,
                "pred_label": None,
                "error_type": None,
                "llm_reasoning": "",
                "raw_output": "",
                "schema_version": ANNOTATION_SCHEMA_VERSION,
            }
        ]
    )

    yes_df, no_df, unclassified_df = split_annotation_frames(result_df)

    assert len(yes_df) == 0
    assert len(no_df) == 0
    assert len(unclassified_df) == 1
    assert unclassified_df.iloc[0]["status"] == STATUS_SUCCESS
    assert pd.isna(unclassified_df.iloc[0]["pred_label"])

    counts = summarize_annotation_counts(result_df)
    assert counts["success_yes"] == 0
    assert counts["success_no"] == 0
    assert counts["unclassified"] == 1
    assert counts["total"] == 1
