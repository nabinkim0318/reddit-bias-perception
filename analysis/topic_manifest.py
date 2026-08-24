"""Aggregate provenance for a local BERTopic run.

Manifests must not contain Reddit text, stable Reddit IDs, usernames,
permalinks, topic-document rows, embeddings, or secrets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from processing.hashing import sha256_file, sha256_json
from processing.manifest import new_run_id, resolve_code_sha, utc_now_iso, write_json

TOPIC_RUN_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]

TOPIC_RUN_LIMITATIONS = [
    "Topic identities are run-specific and are not stable scientific categories.",
    "nr_topics='auto' is exploratory; one automatically reduced solution is not canonical.",
    "Structural stability requires a separate multi-seed evaluation.",
    "Topic stability is not semantic validity.",
    "This manifest is not a real-data stability score unless a governed multi-seed run was completed.",
]


def build_topic_run_manifest(
    *,
    run_id: str,
    started_at: str,
    completed_at: Optional[str],
    code_sha: Optional[str],
    input_filename: str,
    input_sha256: str,
    input_record_count: int,
    seed: int,
    bertopic_version: Optional[str],
    embedding_model: str,
    embedding_model_revision: Optional[str],
    umap_configuration: Mapping[str, Any],
    clustering_configuration: Mapping[str, Any],
    vectorizer_configuration_hash: str,
    nr_topics: Any,
    nr_topics_mode: str,
    discovered_inlier_topic_count: int,
    outlier_count: int,
    outlier_rate: float,
    topic_assignment_checksum: str,
    topic_summary_checksum: Optional[str],
    overall_status: str,
    calculate_probabilities: bool,
    language: str,
    limitations: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    return {
        "bertopic_package_version": bertopic_version,
        "calculate_probabilities": calculate_probabilities,
        "clustering_configuration": dict(clustering_configuration),
        "code_sha": code_sha,
        "completed_at": completed_at,
        "discovered_inlier_topic_count": discovered_inlier_topic_count,
        "embedding_model": embedding_model,
        "embedding_model_revision": embedding_model_revision,
        "input": {
            "filename": Path(input_filename).name,
            "record_count": int(input_record_count),
            "sha256": input_sha256,
        },
        "language": language,
        "limitations": list(limitations or TOPIC_RUN_LIMITATIONS),
        "nr_topics": nr_topics,
        "nr_topics_mode": nr_topics_mode,
        "outlier_count": int(outlier_count),
        "outlier_rate": float(outlier_rate),
        "overall_status": overall_status,
        "run_id": run_id,
        "seed": int(seed),
        "started_at": started_at,
        "topic_assignment_checksum": topic_assignment_checksum,
        "topic_run_schema_version": TOPIC_RUN_SCHEMA_VERSION,
        "topic_summary_checksum": topic_summary_checksum,
        "umap_configuration": dict(umap_configuration),
        "vectorizer_configuration_hash": vectorizer_configuration_hash,
    }


def write_topic_run_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    write_json(path, payload)
    return Path(path)


def new_topic_run_id() -> str:
    return new_run_id()


def topic_code_sha() -> Optional[str]:
    return resolve_code_sha(REPO_ROOT)


def now_utc() -> str:
    return utc_now_iso()


def file_checksum(path: Path) -> str:
    return sha256_file(path)


def snapshot_checksum(payload: Mapping[str, Any]) -> str:
    return sha256_json(dict(payload))
