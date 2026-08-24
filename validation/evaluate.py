"""Agreement, adjudication, and model-vs-human evaluation.

Binary scientific metrics use only successful model yes/no predictions and
resolved human yes/no references. Execution failures are never treated as
scientific ``no``. Uncertain / insufficient human labels are counted but
excluded from binary denominators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from pydantic import ValidationError

from processing.hashing import sha256_file, sha256_json
from processing.manifest import resolve_code_sha, utc_now_iso, write_json
from validation.schema import (
    CODEBOOK_VERSION,
    CONSTRUCT_ID,
    CONSTRUCT_VERSION,
    FORBIDDEN_ANNOTATOR_FIELDS,
    HUMAN_LABELS,
    PROTOCOL_VERSION,
    VALIDATION_SCHEMA_VERSION,
    AdjudicationRecord,
    HumanAnnotation,
    ReferenceAnnotation,
    SamplingIndexRow,
    ValidationInputError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

BINARY_LABELS = ("yes", "no")
POSITIVE = "yes"
REPORT_TEXT_FIELDS = {
    "clean_text",
    "text_to_annotate",
    "selftext",
    "title",
    "body",
    "comments",
    "notes",
    "raw_output",
    "llm_reasoning",
    "permalink",
    "author",
    "username",
    "original_record_id",
    "task_id",
}

REPORT_LIMITATIONS = [
    "This report does not establish that a human-validation study has been completed.",
    "Human agreement does not prove construct validity.",
    "Model-vs-human metrics do not prove that the construct captures objective AI bias.",
    "A Reddit post discussing the construct is not evidence that an AI system exhibits it.",
    "Keyword-filtered posts are not a random sample of Reddit.",
    "The validation sample is not necessarily representative of all Reddit discourse.",
    "Causal inference is not supported.",
    "Model execution failures are excluded from scientific yes/no denominators.",
    "Uncertain and insufficient-context human labels are excluded from binary metrics.",
]


def _empty_to_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise ValidationInputError(f"file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {str(k): ("" if v is None else str(v)) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def _require_unique_task_ids(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise ValidationInputError(f"{label}: missing task_id")
        if task_id in seen:
            duplicates.append(task_id)
        seen.add(task_id)
    if duplicates:
        raise ValidationInputError(
            f"{label}: duplicate task_id(s): {sorted(set(duplicates))}"
        )


def load_sampling_index(path: Path) -> dict[str, SamplingIndexRow]:
    rows = _read_csv_rows(path)
    _require_unique_task_ids(rows, "sampling index")
    parsed: dict[str, SamplingIndexRow] = {}
    errors: list[str] = []
    for row in rows:
        payload = {
            "task_id": row.get("task_id", ""),
            "original_record_id": row.get("original_record_id", ""),
            "model_status": _empty_to_none(row.get("model_status")) or "",
            "model_pred_label": _empty_to_none(row.get("model_pred_label")),
            "subreddit": row.get("subreddit", ""),
            "subreddit_group": row.get("subreddit_group") or "unknown",
            "keyword_category": row.get("keyword_category") or "unknown",
            "annotation_status": _empty_to_none(row.get("annotation_status"))
            or row.get("model_status", ""),
            "strata_key": row.get("strata_key", ""),
        }
        try:
            item = SamplingIndexRow.model_validate(payload)
        except ValidationError as exc:
            errors.append(f"{payload.get('task_id')}: {exc}")
            continue
        parsed[item.task_id] = item
    if errors:
        raise ValidationInputError("invalid sampling index rows: " + "; ".join(errors))
    if not parsed:
        raise ValidationInputError("sampling index is empty")
    return parsed


def load_human_annotations(path: Path) -> dict[str, HumanAnnotation]:
    rows = _read_csv_rows(path)
    _require_unique_task_ids(rows, "human annotations")
    parsed: dict[str, HumanAnnotation] = {}
    errors: list[str] = []
    for row in rows:
        leaked = set(k.lower() for k in row if row.get(k)) & {
            field.lower() for field in FORBIDDEN_ANNOTATOR_FIELDS
        }
        if leaked:
            raise ValidationInputError(
                f"human annotation file must not carry model fields: {sorted(leaked)}"
            )
        label = _empty_to_none(row.get("label") or row.get("human_label"))
        notes = _empty_to_none(row.get("notes"))
        task_id = str(row.get("task_id") or "").strip()
        if label is None:
            errors.append(f"{task_id}: missing label")
            continue
        try:
            item = HumanAnnotation.model_validate(
                {"task_id": task_id, "label": label, "notes": notes}
            )
        except ValidationError as exc:
            errors.append(f"{task_id}: unsupported or invalid label ({exc})")
            continue
        parsed[item.task_id] = item
    if errors:
        raise ValidationInputError("invalid human annotations: " + "; ".join(errors))
    return parsed


def load_adjudication(path: Optional[Path]) -> dict[str, AdjudicationRecord]:
    if path is None:
        return {}
    rows = _read_csv_rows(path)
    _require_unique_task_ids(rows, "adjudication")
    parsed: dict[str, AdjudicationRecord] = {}
    errors: list[str] = []
    for row in rows:
        payload = {
            "task_id": row.get("task_id", ""),
            "annotator_a_label": _empty_to_none(row.get("annotator_a_label")),
            "annotator_b_label": _empty_to_none(row.get("annotator_b_label")),
            "adjudicated_label": _empty_to_none(row.get("adjudicated_label")),
            "adjudication_status": _empty_to_none(row.get("adjudication_status")),
        }
        try:
            item = AdjudicationRecord.model_validate(payload)
        except ValidationError as exc:
            errors.append(f"{payload.get('task_id')}: {exc}")
            continue
        parsed[item.task_id] = item
    if errors:
        raise ValidationInputError("invalid adjudication rows: " + "; ".join(errors))
    return parsed


def align_double_annotations(
    index: Mapping[str, SamplingIndexRow],
    annotations_a: Mapping[str, HumanAnnotation],
    annotations_b: Mapping[str, HumanAnnotation],
) -> None:
    index_ids = set(index)
    missing_a = sorted(index_ids - set(annotations_a))
    missing_b = sorted(index_ids - set(annotations_b))
    extra_a = sorted(set(annotations_a) - index_ids)
    extra_b = sorted(set(annotations_b) - index_ids)
    problems: list[str] = []
    if missing_a:
        problems.append(f"missing in annotator A: {missing_a}")
    if missing_b:
        problems.append(f"missing in annotator B: {missing_b}")
    if extra_a:
        problems.append(f"unknown task_id in annotator A: {extra_a}")
    if extra_b:
        problems.append(f"unknown task_id in annotator B: {extra_b}")
    if problems:
        raise ValidationInputError("; ".join(problems))


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
    if len(labels_a) != len(labels_b):
        raise ValueError("kappa requires paired label sequences")
    n = len(labels_a)
    if n == 0:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    categories = sorted(set(labels_a) | set(labels_b))
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    expected = sum((count_a[cat] / n) * (count_b[cat] / n) for cat in categories)
    denom = 1.0 - expected
    if denom <= 1e-12:
        return None
    return (observed - expected) / denom


def binary_confusion(
    pred_labels: Sequence[str], ref_labels: Sequence[str]
) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for pred, ref in zip(pred_labels, ref_labels):
        if ref == POSITIVE and pred == POSITIVE:
            tp += 1
        elif ref != POSITIVE and pred != POSITIVE:
            tn += 1
        elif ref != POSITIVE and pred == POSITIVE:
            fp += 1
        else:
            fn += 1
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _safe_div(numer: float, denom: float) -> Optional[float]:
    if denom == 0:
        return None
    return numer / denom


def classification_metrics(confusion: Mapping[str, int]) -> dict[str, Optional[float]]:
    tp = int(confusion["true_positive"])
    tn = int(confusion["true_negative"])
    fp = int(confusion["false_positive"])
    fn = int(confusion["false_negative"])
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    if precision is None or recall is None:
        f1: Optional[float] = None
    elif precision + recall == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    if recall is None or specificity is None:
        balanced: Optional[float] = None
    else:
        balanced = 0.5 * (recall + specificity)
    return {
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "evaluable_sample_count": tp + tn + fp + fn,
    }


def _metric_from_pairs(pairs: Sequence[tuple[str, str]], name: str) -> Optional[float]:
    if not pairs:
        return None
    preds = [p for p, _ in pairs]
    refs = [r for _, r in pairs]
    metrics = classification_metrics(binary_confusion(preds, refs))
    return metrics[name]


def bootstrap_confidence_intervals(
    pairs: Sequence[tuple[str, str]],
    *,
    n_iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Optional[dict[str, Optional[float]]]]:
    names = ("precision_yes", "recall_yes", "f1_yes", "accuracy")
    if n_iterations <= 0 or len(pairs) == 0:
        return {name: None for name in names}
    rng = random.Random(int(seed))
    n = len(pairs)
    collected: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(int(n_iterations)):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        for name in names:
            value = _metric_from_pairs(sample, name)
            if value is not None and math.isfinite(value):
                collected[name].append(value)
    intervals: dict[str, Optional[dict[str, Optional[float]]]] = {}
    lo_q = alpha / 2.0
    hi_q = 1.0 - (alpha / 2.0)
    for name in names:
        stats = sorted(collected[name])
        if not stats:
            intervals[name] = None
            continue
        lo_idx = min(len(stats) - 1, max(0, int(math.floor(lo_q * len(stats)))))
        hi_idx = min(len(stats) - 1, max(0, int(math.ceil(hi_q * len(stats))) - 1))
        intervals[name] = {
            "low": stats[lo_idx],
            "high": stats[hi_idx],
            "n_defined_replicates": len(stats),
            "method": "percentile_bootstrap",
            "n_iterations": int(n_iterations),
            "seed": int(seed),
        }
    return intervals


def resolve_references(
    index: Mapping[str, SamplingIndexRow],
    annotations_a: Mapping[str, HumanAnnotation],
    annotations_b: Mapping[str, HumanAnnotation],
    adjudication: Mapping[str, AdjudicationRecord],
) -> dict[str, ReferenceAnnotation]:
    resolved: dict[str, ReferenceAnnotation] = {}
    for task_id, _row in index.items():
        label_a = annotations_a[task_id].label
        label_b = annotations_b[task_id].label
        adj = adjudication.get(task_id)
        if adj is not None:
            if adj.annotator_a_label != label_a or adj.annotator_b_label != label_b:
                raise ValidationInputError(
                    f"{task_id}: adjudication A/B labels do not match annotation files"
                )
            if (
                adj.adjudication_status == "resolved"
                and adj.adjudicated_label in BINARY_LABELS
            ):
                resolved[task_id] = ReferenceAnnotation(
                    task_id=task_id,
                    reference_label=adj.adjudicated_label,  # type: ignore[arg-type]
                    resolution_status="adjudicated",
                )
                continue
            resolved[task_id] = ReferenceAnnotation(
                task_id=task_id,
                reference_label=None,
                resolution_status="unresolved",
            )
            continue
        if label_a == label_b and label_a in BINARY_LABELS:
            resolved[task_id] = ReferenceAnnotation(
                task_id=task_id,
                reference_label=label_a,  # type: ignore[arg-type]
                resolution_status="agreed",
            )
        else:
            resolved[task_id] = ReferenceAnnotation(
                task_id=task_id,
                reference_label=None,
                resolution_status="unresolved",
            )
    return resolved


def _label_distribution(labels: Iterable[str]) -> dict[str, int]:
    counts = Counter(labels)
    return {label: int(counts.get(label, 0)) for label in HUMAN_LABELS}


def evaluate_validation(
    *,
    index: Mapping[str, SamplingIndexRow],
    annotations_a: Mapping[str, HumanAnnotation],
    annotations_b: Mapping[str, HumanAnnotation],
    adjudication: Optional[Mapping[str, AdjudicationRecord]] = None,
    sampling_config: Optional[Mapping[str, Any]] = None,
    checksums: Optional[Mapping[str, Any]] = None,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 42,
    code_sha: Optional[str] = None,
) -> dict[str, Any]:
    adjudication = dict(adjudication or {})
    align_double_annotations(index, annotations_a, annotations_b)
    references = resolve_references(index, annotations_a, annotations_b, adjudication)

    paired_a = [annotations_a[task_id].label for task_id in index]
    paired_b = [annotations_b[task_id].label for task_id in index]
    n_double = len(index)
    n_agree = sum(a == b for a, b in zip(paired_a, paired_b))
    disagreements = [
        task_id
        for task_id in index
        if annotations_a[task_id].label != annotations_b[task_id].label
    ]
    kappa = cohens_kappa(paired_a, paired_b)
    percent_agreement = (n_agree / n_double) if n_double else None

    agreed_yes = sum(
        1
        for task_id in index
        if annotations_a[task_id].label == "yes"
        and annotations_b[task_id].label == "yes"
    )
    agreed_no = sum(
        1
        for task_id in index
        if annotations_a[task_id].label == "no" and annotations_b[task_id].label == "no"
    )
    n_uncertain = sum(
        1
        for task_id in index
        if "uncertain"
        in {
            annotations_a[task_id].label,
            annotations_b[task_id].label,
        }
    )
    n_insufficient = sum(
        1
        for task_id in index
        if "insufficient_context"
        in {
            annotations_a[task_id].label,
            annotations_b[task_id].label,
        }
    )
    n_adjudicated = sum(
        1
        for record in adjudication.values()
        if record.adjudication_status == "resolved"
    )
    n_unresolved = sum(
        1 for item in references.values() if item.resolution_status == "unresolved"
    )

    model_success = sum(1 for row in index.values() if row.model_status == "success")
    model_parse_error = sum(
        1 for row in index.values() if row.model_status == "parse_error"
    )
    model_model_error = sum(
        1 for row in index.values() if row.model_status == "model_error"
    )

    pred_pairs: list[tuple[str, str]] = []
    excluded_failure = 0
    excluded_unresolved = 0
    for task_id, row in index.items():
        reference = references[task_id]
        model_ok = (
            row.model_status == "success" and row.model_pred_label in BINARY_LABELS
        )
        if not model_ok:
            excluded_failure += 1
            continue
        if reference.reference_label not in BINARY_LABELS:
            excluded_unresolved += 1
            continue
        pred_pairs.append((row.model_pred_label or "", reference.reference_label))

    confusion = binary_confusion([p for p, _ in pred_pairs], [r for _, r in pred_pairs])
    metrics = classification_metrics(confusion)
    intervals = bootstrap_confidence_intervals(
        pred_pairs,
        n_iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )

    subgroup_counts: dict[str, dict[str, int]] = {
        "subreddit_group": defaultdict(int),
        "keyword_category": defaultdict(int),
        "model_pred_label": defaultdict(int),
        "model_status": defaultdict(int),
    }
    for row in index.values():
        subgroup_counts["subreddit_group"][row.subreddit_group] += 1
        subgroup_counts["keyword_category"][row.keyword_category] += 1
        subgroup_counts["model_pred_label"][row.model_pred_label or "null"] += 1
        subgroup_counts["model_status"][row.model_status] += 1

    counts = {
        "sampled": n_double,
        "double_annotated": n_double,
        "agreed_yes": agreed_yes,
        "agreed_no": agreed_no,
        "human_disagreements": len(disagreements),
        "uncertain": n_uncertain,
        "insufficient_context": n_insufficient,
        "adjudicated": n_adjudicated,
        "unresolved": n_unresolved,
        "model_success": model_success,
        "model_parse_error": model_parse_error,
        "model_model_error": model_model_error,
        "binary_evaluable": len(pred_pairs),
    }

    config = dict(sampling_config or {})
    config_hash = config.get("config_hash") or (sha256_json(config) if config else None)

    report = {
        "claims": {
            "human_validation_study_completed": False,
            "labels_are_ground_truth": False,
            "construct_validity_established": False,
            "objective_ai_bias_measured": False,
        },
        "code_sha": code_sha,
        "codebook_version": CODEBOOK_VERSION,
        "confidence_intervals": intervals,
        "confusion_matrix": confusion,
        "construct_id": CONSTRUCT_ID,
        "construct_version": CONSTRUCT_VERSION,
        "counts": counts,
        "exclusions": {
            "model_execution_failure": excluded_failure,
            "unresolved_human_reference": excluded_unresolved,
        },
        "generated_at": utc_now_iso(),
        "inter_annotator_agreement": {
            "cohens_kappa": kappa,
            "disagreement_count": len(disagreements),
            "kappa_undefined_reason": (
                None
                if kappa is not None
                else (
                    "undefined because expected agreement is 1 "
                    "(degenerate marginal distributions)"
                    if n_double
                    else "no double-annotated items"
                )
            ),
            "label_distribution_a": _label_distribution(paired_a),
            "label_distribution_b": _label_distribution(paired_b),
            "percent_agreement": percent_agreement,
            "prevalence_limitation": (
                "Cohen's kappa depends on marginal label distributions; high "
                "percent agreement is not sufficient evidence of reliability."
            ),
            "total_doubly_annotated": n_double,
        },
        "limitations": list(REPORT_LIMITATIONS),
        "metrics": metrics,
        "model_execution_status": {
            "model_error": model_model_error,
            "parse_error": model_parse_error,
            "success": model_success,
        },
        "protocol_version": PROTOCOL_VERSION,
        "provenance": dict(checksums or {}),
        "sampling_config_hash": config_hash,
        "sampling_seed": config.get("seed"),
        "subgroup_counts": {
            key: dict(sorted(value.items())) for key, value in subgroup_counts.items()
        },
        "subgroup_metrics_omitted": True,
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
    }
    _assert_report_privacy(report)
    return report


def _walk_keys_and_strings(obj: Any) -> Iterable[tuple[Optional[str], str]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key), ""
            yield from _walk_keys_and_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_keys_and_strings(item)
    elif isinstance(obj, str):
        yield None, obj


def _assert_report_privacy(report: Mapping[str, Any]) -> None:
    for key, text in _walk_keys_and_strings(report):
        if key is not None and key in REPORT_TEXT_FIELDS:
            raise ValidationInputError(
                f"aggregate report must not include field {key!r}"
            )
        lowered = text.lower()
        if "reddit.com/" in lowered or "redd.it/" in lowered:
            raise ValidationInputError("aggregate report contains a Reddit URL")


def _load_json_if_exists(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def evaluate_from_paths(
    *,
    sampling_index: Path,
    annotations_a: Path,
    annotations_b: Path,
    output: Path,
    adjudication: Optional[Path] = None,
    sampling_config: Optional[Path] = None,
    source_artifact: Optional[Path] = None,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    index = load_sampling_index(sampling_index)
    labels_a = load_human_annotations(annotations_a)
    labels_b = load_human_annotations(annotations_b)
    adj = load_adjudication(adjudication)
    config = _load_json_if_exists(sampling_config)
    checksums = {
        "annotations_a_sha256": sha256_file(Path(annotations_a)),
        "annotations_b_sha256": sha256_file(Path(annotations_b)),
        "sampling_index_sha256": sha256_file(Path(sampling_index)),
    }
    if adjudication is not None:
        checksums["adjudication_sha256"] = sha256_file(Path(adjudication))
    if source_artifact is not None:
        checksums["source_artifact_sha256"] = sha256_file(Path(source_artifact))
        checksums["source_artifact_name"] = Path(source_artifact).name
    if sampling_config is not None and Path(sampling_config).is_file():
        checksums["sampling_config_sha256"] = sha256_file(Path(sampling_config))
    checksums["codebook_version"] = CODEBOOK_VERSION
    checksums["construct_id"] = CONSTRUCT_ID
    checksums["construct_version"] = CONSTRUCT_VERSION
    report = evaluate_validation(
        index=index,
        annotations_a=labels_a,
        annotations_b=labels_b,
        adjudication=adj,
        sampling_config=config,
        checksums=checksums,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        code_sha=resolve_code_sha(REPO_ROOT),
    )
    write_json(Path(output), report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute inter-annotator agreement and model-vs-human metrics. "
            "Writes an aggregate privacy-safe JSON report."
        )
    )
    parser.add_argument("--sampling-index", required=True)
    parser.add_argument("--annotations-a", required=True)
    parser.add_argument("--annotations-b", required=True)
    parser.add_argument("--adjudication", default=None)
    parser.add_argument("--sampling-config", default=None)
    parser.add_argument("--source-artifact", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        evaluate_from_paths(
            sampling_index=Path(args.sampling_index),
            annotations_a=Path(args.annotations_a),
            annotations_b=Path(args.annotations_b),
            output=Path(args.output),
            adjudication=Path(args.adjudication) if args.adjudication else None,
            sampling_config=(
                Path(args.sampling_config) if args.sampling_config else None
            ),
            source_artifact=(
                Path(args.source_artifact) if args.source_artifact else None
            ),
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
    except ValidationInputError as exc:
        print(f"validation input error: {exc}", file=sys.stderr)
        return 2
    print(
        "Wrote aggregate validation report. This does not by itself complete "
        "a human-validation study."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
