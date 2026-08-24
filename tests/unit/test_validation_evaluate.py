"""Agreement and model-vs-human evaluation. Offline synthetic cases only."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from validation.evaluate import (
    binary_confusion,
    classification_metrics,
    cohens_kappa,
    evaluate_validation,
    load_human_annotations,
)
from validation.schema import (
    AdjudicationRecord,
    HumanAnnotation,
    SamplingIndexRow,
    ValidationInputError,
)


def _ann(task_id: str, label: str) -> HumanAnnotation:
    return HumanAnnotation(task_id=task_id, label=label)  # type: ignore[arg-type]


def _index_success(task_id: str, pred: str, record_id: str) -> SamplingIndexRow:
    return SamplingIndexRow(
        task_id=task_id,
        original_record_id=record_id,
        model_status="success",
        model_pred_label=pred,  # type: ignore[arg-type]
        annotation_status="success",
        subreddit_group="technical",
        keyword_category="gender",
        strata_key=f"{pred}|technical|gender",
    )


def test_cohens_kappa_known_synthetic_case() -> None:
    labels_a = ["yes", "yes", "no", "no"]
    labels_b = ["yes", "no", "no", "no"]
    kappa = cohens_kappa(labels_a, labels_b)
    assert kappa == pytest.approx(0.5)


def test_undefined_kappa_is_null() -> None:
    labels = ["yes", "yes", "yes"]
    assert cohens_kappa(labels, labels) is None
    assert cohens_kappa([], []) is None


def test_exact_agreement_and_disagreement_counts() -> None:
    index = {
        "VAL-0001": _index_success("VAL-0001", "yes", "SYNTH-VAL-0001"),
        "VAL-0002": _index_success("VAL-0002", "no", "SYNTH-VAL-0002"),
    }
    annotations_a = {
        "VAL-0001": _ann("VAL-0001", "yes"),
        "VAL-0002": _ann("VAL-0002", "yes"),
    }
    annotations_b = {
        "VAL-0001": _ann("VAL-0001", "yes"),
        "VAL-0002": _ann("VAL-0002", "no"),
    }
    report = evaluate_validation(
        index=index,
        annotations_a=annotations_a,
        annotations_b=annotations_b,
        bootstrap_iterations=0,
    )
    agreement = report["inter_annotator_agreement"]
    assert agreement["total_doubly_annotated"] == 2
    assert agreement["disagreement_count"] == 1
    assert agreement["percent_agreement"] == pytest.approx(0.5)
    assert report["counts"]["agreed_yes"] == 1
    assert report["counts"]["human_disagreements"] == 1


def test_confusion_matrix_and_f1_on_known_case() -> None:
    confusion = binary_confusion(
        ["yes", "no", "yes", "no"],
        ["yes", "no", "no", "yes"],
    )
    assert confusion == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    metrics = classification_metrics(confusion)
    assert metrics["precision_yes"] == pytest.approx(0.5)
    assert metrics["recall_yes"] == pytest.approx(0.5)
    assert metrics["f1_yes"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_zero_denominator_metrics_are_unavailable() -> None:
    metrics = classification_metrics(
        {
            "true_positive": 0,
            "true_negative": 4,
            "false_positive": 0,
            "false_negative": 0,
        }
    )
    assert metrics["precision_yes"] is None
    assert metrics["recall_yes"] is None
    assert metrics["f1_yes"] is None
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] is None


def test_model_failures_excluded_from_binary_metrics() -> None:
    index = {
        "VAL-0001": SamplingIndexRow(
            task_id="VAL-0001",
            original_record_id="SYNTH-VAL-0008",
            model_status="parse_error",
            model_pred_label=None,
            annotation_status="parse_error",
            strata_key="failure|parse_error|unknown|unknown",
        ),
        "VAL-0002": SamplingIndexRow(
            task_id="VAL-0002",
            original_record_id="SYNTH-VAL-0009",
            model_status="model_error",
            model_pred_label=None,
            annotation_status="model_error",
            strata_key="failure|model_error|unknown|unknown",
        ),
    }
    labels = {
        "VAL-0001": _ann("VAL-0001", "no"),
        "VAL-0002": _ann("VAL-0002", "no"),
    }
    report = evaluate_validation(
        index=index,
        annotations_a=labels,
        annotations_b=labels,
        bootstrap_iterations=0,
    )
    matrix = report["confusion_matrix"]
    assert matrix["true_negative"] == 0
    assert matrix["false_negative"] == 0
    assert report["counts"]["binary_evaluable"] == 0
    assert report["counts"]["model_parse_error"] == 1
    assert report["counts"]["model_model_error"] == 1
    assert report["exclusions"]["model_execution_failure"] == 2


def test_parse_or_model_failure_cannot_become_negative() -> None:
    """Execution failure is not a scientific no."""
    index = {
        "VAL-0001": SamplingIndexRow(
            task_id="VAL-0001",
            original_record_id="SYNTH-VAL-0008",
            model_status="parse_error",
            model_pred_label=None,
            annotation_status="parse_error",
        ),
        "VAL-0002": SamplingIndexRow(
            task_id="VAL-0002",
            original_record_id="SYNTH-VAL-0009",
            model_status="model_error",
            model_pred_label=None,
            annotation_status="model_error",
        ),
        "VAL-0003": _index_success("VAL-0003", "yes", "SYNTH-VAL-0001"),
    }
    annotations_a = {
        "VAL-0001": _ann("VAL-0001", "no"),
        "VAL-0002": _ann("VAL-0002", "no"),
        "VAL-0003": _ann("VAL-0003", "yes"),
    }
    report = evaluate_validation(
        index=index,
        annotations_a=annotations_a,
        annotations_b=annotations_a,
        bootstrap_iterations=0,
    )
    assert report["confusion_matrix"]["true_negative"] == 0
    assert report["confusion_matrix"]["false_negative"] == 0
    assert report["confusion_matrix"]["true_positive"] == 1
    assert report["counts"]["binary_evaluable"] == 1


def test_uncertain_and_insufficient_are_counted_but_excluded() -> None:
    index = {
        "VAL-0001": _index_success("VAL-0001", "yes", "SYNTH-VAL-0006"),
        "VAL-0002": _index_success("VAL-0002", "no", "SYNTH-VAL-0007"),
        "VAL-0003": _index_success("VAL-0003", "yes", "SYNTH-VAL-0001"),
    }
    annotations_a = {
        "VAL-0001": _ann("VAL-0001", "uncertain"),
        "VAL-0002": _ann("VAL-0002", "insufficient_context"),
        "VAL-0003": _ann("VAL-0003", "yes"),
    }
    report = evaluate_validation(
        index=index,
        annotations_a=annotations_a,
        annotations_b=annotations_a,
        bootstrap_iterations=0,
    )
    assert report["counts"]["uncertain"] == 1
    assert report["counts"]["insufficient_context"] == 1
    assert report["counts"]["unresolved"] == 2
    assert report["counts"]["binary_evaluable"] == 1
    assert report["confusion_matrix"]["true_positive"] == 1


def test_adjudication_is_separate_and_resolves_disagreement() -> None:
    index = {"VAL-0001": _index_success("VAL-0001", "yes", "SYNTH-VAL-0005")}
    annotations_a = {"VAL-0001": _ann("VAL-0001", "yes")}
    annotations_b = {"VAL-0001": _ann("VAL-0001", "no")}
    adjudication = {
        "VAL-0001": AdjudicationRecord(
            task_id="VAL-0001",
            annotator_a_label="yes",
            annotator_b_label="no",
            adjudicated_label="yes",
            adjudication_status="resolved",
        )
    }
    without = evaluate_validation(
        index=index,
        annotations_a=annotations_a,
        annotations_b=annotations_b,
        bootstrap_iterations=0,
    )
    assert without["counts"]["binary_evaluable"] == 0
    assert without["counts"]["human_disagreements"] == 1
    with_adj = evaluate_validation(
        index=index,
        annotations_a=annotations_a,
        annotations_b=annotations_b,
        adjudication=adjudication,
        bootstrap_iterations=0,
    )
    assert with_adj["counts"]["binary_evaluable"] == 1
    assert with_adj["counts"]["adjudicated"] == 1
    assert with_adj["confusion_matrix"]["true_positive"] == 1
    assert annotations_a["VAL-0001"].label == "yes"
    assert annotations_b["VAL-0001"].label == "no"


def test_missing_annotation_is_detected() -> None:
    index = {
        "VAL-0001": _index_success("VAL-0001", "yes", "SYNTH-VAL-0001"),
        "VAL-0002": _index_success("VAL-0002", "no", "SYNTH-VAL-0002"),
    }
    annotations_a = {"VAL-0001": _ann("VAL-0001", "yes")}
    annotations_b = {
        "VAL-0001": _ann("VAL-0001", "yes"),
        "VAL-0002": _ann("VAL-0002", "no"),
    }
    with pytest.raises(ValidationInputError, match="missing in annotator A"):
        evaluate_validation(
            index=index,
            annotations_a=annotations_a,
            annotations_b=annotations_b,
            bootstrap_iterations=0,
        )


def test_missing_and_duplicate_annotations_fail(tmp_path: Path) -> None:
    path = tmp_path / "annotator_a.csv"
    path.write_text(
        "task_id,human_label\nVAL-0001,yes\nVAL-0001,no\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationInputError, match="duplicate"):
        load_human_annotations(path)

    empty = tmp_path / "annotator_b.csv"
    empty.write_text("task_id,human_label\nVAL-0002,\n", encoding="utf-8")
    with pytest.raises(ValidationInputError, match="missing label"):
        load_human_annotations(empty)


def test_invalid_human_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "annotator_a.csv"
    path.write_text("task_id,human_label\nVAL-0001,bias\n", encoding="utf-8")
    with pytest.raises(ValidationInputError):
        load_human_annotations(path)


def test_alignment_is_by_task_id_not_row_order(tmp_path: Path) -> None:
    index = {
        "VAL-0001": _index_success("VAL-0001", "yes", "SYNTH-VAL-0001"),
        "VAL-0002": _index_success("VAL-0002", "no", "SYNTH-VAL-0002"),
    }
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    with a_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "human_label"])
        writer.writeheader()
        writer.writerow({"task_id": "VAL-0002", "human_label": "no"})
        writer.writerow({"task_id": "VAL-0001", "human_label": "yes"})
    with b_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "human_label"])
        writer.writeheader()
        writer.writerow({"task_id": "VAL-0001", "human_label": "yes"})
        writer.writerow({"task_id": "VAL-0002", "human_label": "no"})
    report = evaluate_validation(
        index=index,
        annotations_a=load_human_annotations(a_path),
        annotations_b=load_human_annotations(b_path),
        bootstrap_iterations=0,
    )
    assert report["counts"]["agreed_yes"] == 1
    assert report["counts"]["agreed_no"] == 1
    assert report["confusion_matrix"]["true_positive"] == 1
    assert report["confusion_matrix"]["true_negative"] == 1


def test_aggregate_report_has_no_record_level_source_fields() -> None:
    index = {"VAL-0001": _index_success("VAL-0001", "yes", "SYNTH-VAL-0001")}
    labels = {"VAL-0001": _ann("VAL-0001", "yes")}
    report = evaluate_validation(
        index=index,
        annotations_a=labels,
        annotations_b=labels,
        bootstrap_iterations=0,
    )
    forbidden = {
        "clean_text",
        "text_to_annotate",
        "title",
        "selftext",
        "body",
        "comments",
        "permalink",
        "author",
        "username",
        "raw_output",
        "llm_reasoning",
        "notes",
        "original_record_id",
        "task_id",
    }
    assert forbidden.isdisjoint(report.keys())
    assert report["claims"]["human_validation_study_completed"] is False
    blob = str(report)
    assert "SYNTH-VAL-0001" not in blob
    assert "reddit.com/" not in blob.lower()
