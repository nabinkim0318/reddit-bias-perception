"""Multi-seed structural stability for topic assignments.

Topic IDs are arbitrary labels. Pairwise agreement uses Adjusted Rand Index
(ARI), which is invariant to label permutation. ARI measures structural
consistency of partitions, not semantic validity of topics.

This module does not fit BERTopic. Fitting lives in ``analysis.bertopic_model``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from sklearn.metrics import adjusted_rand_score

from analysis.topic_probabilities import OUTLIER_TOPIC
from processing.hashing import sha256_json
from processing.manifest import resolve_code_sha, utc_now_iso, write_json

STABILITY_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]

TOPIC_STABILITY_LIMITATIONS = [
    "ARI measures structural assignment agreement, not semantic topic validity.",
    "Topic IDs are run-specific labels and are not stable scientific categories.",
    "nr_topics='auto' is an exploratory choice; inlier topic count may vary across seeds.",
    "A single BERTopic seed does not establish a unique latent topic structure.",
    "Passing synthetic ARI tests validates the metric implementation, not a real Reddit topic solution.",
    "Human topic naming is not ground truth.",
]


class TopicStabilityError(ValueError):
    """Raised when stability inputs are invalid."""


@dataclass(frozen=True)
class TopicRunSummary:
    seed: int
    assignments: tuple[int, ...]
    inlier_topic_count: int
    outlier_count: int
    outlier_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "inlier_topic_count": self.inlier_topic_count,
            "outlier_count": self.outlier_count,
            "outlier_rate": self.outlier_rate,
            "seed": self.seed,
        }


def assignment_checksum(assignments: Sequence[int]) -> str:
    return sha256_json([int(topic) for topic in assignments])


def summarize_run(seed: int, assignments: Sequence[int]) -> TopicRunSummary:
    labels = tuple(int(topic) for topic in assignments)
    n = len(labels)
    if n == 0:
        raise TopicStabilityError("assignments must not be empty")
    outlier_count = sum(1 for topic in labels if topic == OUTLIER_TOPIC)
    inliers = {topic for topic in labels if topic != OUTLIER_TOPIC}
    return TopicRunSummary(
        seed=int(seed),
        assignments=labels,
        inlier_topic_count=len(inliers),
        outlier_count=outlier_count,
        outlier_rate=outlier_count / n,
    )


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    floats = [float(value) for value in values]
    return {
        "max": max(floats) if floats else None,
        "median": _median(floats),
        "min": min(floats) if floats else None,
        "values": floats,
    }


def compute_assignment_stability(
    assignments_a: Sequence[int],
    assignments_b: Sequence[int],
    *,
    outlier_label: int = OUTLIER_TOPIC,
) -> dict[str, Any]:
    """Permutation-invariant agreement between two assignment vectors.

    ``ari_all`` treats ``outlier_label`` as an ordinary cluster label.
    ``ari_inliers_both`` restricts to documents that are inliers in both runs.
    The inlier denominator is the number of such documents.
    """
    left = [int(topic) for topic in assignments_a]
    right = [int(topic) for topic in assignments_b]
    if len(left) != len(right):
        raise TopicStabilityError(
            f"assignment lengths differ: {len(left)} vs {len(right)}"
        )
    if not left:
        raise TopicStabilityError("assignments must not be empty")

    ari_all = float(adjusted_rand_score(left, right))
    both_inlier_index = [
        i
        for i, (a, b) in enumerate(zip(left, right))
        if a != outlier_label and b != outlier_label
    ]
    n_inliers_both = len(both_inlier_index)
    if n_inliers_both == 0:
        ari_inliers = None
        inlier_reason = "no documents are inliers in both runs"
    else:
        ari_inliers = float(
            adjusted_rand_score(
                [left[i] for i in both_inlier_index],
                [right[i] for i in both_inlier_index],
            )
        )
        inlier_reason = None
    return {
        "ari_all": ari_all,
        "ari_inliers_both": ari_inliers,
        "inliers_both_denominator": n_inliers_both,
        "inliers_both_unavailable_reason": inlier_reason,
        "n_documents": len(left),
        "outlier_treatment": {
            "ari_all": (
                "ARI treats topic -1 as an ordinary label (the outlier cluster)."
            ),
            "ari_inliers_both": (
                "ARI is computed only on documents assigned a non-outlier "
                "topic in both runs. Denominator is the number of such documents."
            ),
        },
    }


def summarize_topic_stability(
    runs: Sequence[TopicRunSummary] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summaries = [_coerce_run(run) for run in runs]
    if len(summaries) < 1:
        raise TopicStabilityError("at least one run is required")
    n_docs = len(summaries[0].assignments)
    for summary in summaries:
        if len(summary.assignments) != n_docs:
            raise TopicStabilityError("all runs must cover the same ordered documents")

    ordered = sorted(
        summaries, key=lambda item: (item.seed, assignment_checksum(item.assignments))
    )
    pairs: list[dict[str, Any]] = []
    ari_all_values: list[float] = []
    ari_inlier_values: list[float] = []
    for left, right in combinations(ordered, 2):
        agreement = compute_assignment_stability(left.assignments, right.assignments)
        pair = {
            "ari_all": agreement["ari_all"],
            "ari_inliers_both": agreement["ari_inliers_both"],
            "inliers_both_denominator": agreement["inliers_both_denominator"],
            "seed_a": left.seed,
            "seed_b": right.seed,
        }
        pairs.append(pair)
        ari_all_values.append(float(agreement["ari_all"]))
        if agreement["ari_inliers_both"] is not None:
            ari_inlier_values.append(float(agreement["ari_inliers_both"]))

    report = {
        "limitations": list(TOPIC_STABILITY_LIMITATIONS),
        "n_documents": n_docs,
        "n_runs": len(ordered),
        "outlier_rate_distribution": _distribution(
            [summary.outlier_rate for summary in ordered]
        ),
        "outlier_treatment": {
            "ari_all": (
                "Pairwise ARI on all documents treats topic -1 as a cluster label. "
                "Outliers are not silently dropped."
            ),
            "ari_inliers_both": (
                "Pairwise ARI restricted to documents that are inliers in both "
                "compared runs. Denominator is that document count."
            ),
        },
        "pairwise_ari_all": {
            **_distribution(ari_all_values),
            "pairs": pairs,
        },
        "pairwise_ari_inliers_both": {
            **_distribution(ari_inlier_values),
            "n_defined_pairs": len(ari_inlier_values),
        },
        "runs": [summary.to_dict() for summary in ordered],
        "seeds": [summary.seed for summary in ordered],
        "stability_declared": False,
        "stability_metric": "adjusted_rand_index",
        "stability_schema_version": STABILITY_SCHEMA_VERSION,
        "topic_count_distribution": _distribution(
            [float(summary.inlier_topic_count) for summary in ordered]
        ),
    }
    return report


def annotate_stability_provenance(
    report: Mapping[str, Any],
    *,
    input_filename: Optional[str] = None,
    input_sha256: Optional[str] = None,
    config_identity: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Attach input/config identity without claiming a real-data stability score."""
    payload = dict(report)
    if input_filename is not None:
        payload["input_filename"] = Path(input_filename).name
    if input_sha256 is not None:
        payload["input_sha256"] = input_sha256
    if config_identity is not None:
        identity = dict(config_identity)
        payload["config_hash"] = sha256_json(identity)
        payload["config_identity"] = identity
    return payload


def _coerce_run(run: TopicRunSummary | Mapping[str, Any]) -> TopicRunSummary:
    if isinstance(run, TopicRunSummary):
        return run
    if "assignments" not in run:
        raise TopicStabilityError("run is missing assignments")
    seed = int(run["seed"])
    return summarize_run(seed, run["assignments"])


def write_stability_report(path: Path, report: Mapping[str, Any]) -> Path:
    write_json(path, report)
    return Path(path)


def _load_runs_from_json(path: Path) -> list[TopicRunSummary]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runs_raw: Iterable[Mapping[str, Any]]
    if isinstance(payload, dict) and "runs" in payload:
        runs_raw = payload["runs"]
    elif isinstance(payload, list):
        runs_raw = payload
    else:
        raise TopicStabilityError("stability input must be a run list or {runs: ...}")
    return [_coerce_run(run) for run in runs_raw]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multi-seed topic-assignment stability. "
            "Does not declare a model scientifically stable. "
            "Real BERTopic fitting is a separate local command."
        )
    )
    parser.add_argument(
        "--assignments-json",
        required=True,
        help="JSON list of {seed, assignments} for the same ordered documents.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        runs = _load_runs_from_json(Path(args.assignments_json))
        report = summarize_topic_stability(runs)
        report["code_sha"] = resolve_code_sha(REPO_ROOT)
        report["completed_at"] = utc_now_iso()
        write_stability_report(Path(args.output), report)
    except TopicStabilityError as exc:
        print(f"topic stability error: {exc}", file=sys.stderr)
        return 2
    print(
        "Wrote topic-stability report. Structural ARI is not semantic validation "
        "and does not by itself establish a real-data stability score."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
