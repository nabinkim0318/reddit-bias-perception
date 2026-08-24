"""Human-validation schema rejects invalid labels instead of guessing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from validation.schema import (
    AdjudicationRecord,
    HumanAnnotation,
    ReferenceAnnotation,
    SamplingIndexRow,
)


def test_human_annotation_accepts_uncertainty_vocabulary() -> None:
    for label in ("yes", "no", "uncertain", "insufficient_context"):
        item = HumanAnnotation(task_id="VAL-0001", label=label)
        assert item.label == label


def test_human_annotation_rejects_invalid_label() -> None:
    with pytest.raises(ValidationError):
        HumanAnnotation(task_id="VAL-0001", label="maybe")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        HumanAnnotation(task_id="VAL-0001", label="bias")  # type: ignore[arg-type]


def test_human_annotation_rejects_empty_task_id() -> None:
    with pytest.raises(ValidationError):
        HumanAnnotation(task_id="  ", label="yes")


def test_sampling_index_success_requires_yes_no() -> None:
    with pytest.raises(ValidationError):
        SamplingIndexRow(
            task_id="VAL-0001",
            original_record_id="SYNTH-VAL-0001",
            model_status="success",
            model_pred_label=None,
            annotation_status="success",
        )


def test_sampling_index_failure_forbids_pred_label() -> None:
    with pytest.raises(ValidationError):
        SamplingIndexRow(
            task_id="VAL-0001",
            original_record_id="SYNTH-VAL-0008",
            model_status="parse_error",
            model_pred_label="no",
            annotation_status="parse_error",
        )


def test_reference_unresolved_has_no_binary_label() -> None:
    with pytest.raises(ValidationError):
        ReferenceAnnotation(
            task_id="VAL-0001",
            reference_label="yes",
            resolution_status="unresolved",
        )


def test_adjudication_resolved_requires_label() -> None:
    with pytest.raises(ValidationError):
        AdjudicationRecord(
            task_id="VAL-0001",
            annotator_a_label="yes",
            annotator_b_label="no",
            adjudicated_label=None,
            adjudication_status="resolved",
        )


def test_adjudication_unresolved_forbids_label() -> None:
    with pytest.raises(ValidationError):
        AdjudicationRecord(
            task_id="VAL-0001",
            annotator_a_label="yes",
            annotator_b_label="no",
            adjudicated_label="yes",
            adjudication_status="unresolved",
        )
