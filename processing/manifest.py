"""Lightweight run/stage provenance for the canonical synthetic pipeline.

Manifests record what inputs, configuration, and code produced outputs.
They must not contain raw source text, secrets, or full exception dumps.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from processing.hashing import canonical_json, sha256_file, sorted_mapping

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "synthetic_demo_manifest.json"
AGGREGATE_FILENAME = "synthetic_demo_aggregate.json"

OVERALL_SUCCESS = "success"
OVERALL_FAILURE = "failure"
OVERALL_REUSED = "reused"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_run_id() -> str:
    return uuid.uuid4().hex


def resolve_code_sha(repo_root: Path) -> Optional[str]:
    env_sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT")
    if env_sha:
        return env_sha.strip() or None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(sorted_mapping(dict(payload)))
    # canonical_json is one line; pretty-print with sorted keys for readability
    parsed = json.loads(text)
    path.write_text(
        json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_entry(path: Path) -> dict[str, str]:
    resolved = Path(path)
    return {
        "name": resolved.name,
        "sha256": sha256_file(resolved) if resolved.is_file() else "",
    }


def build_manifest(
    *,
    run_id: str,
    started_at: str,
    completed_at: Optional[str],
    overall_status: str,
    code_sha: Optional[str],
    input_path_name: str,
    input_checksum: str,
    input_classification: str,
    input_record_count: int,
    output_record_count: int,
    config_snapshot: Mapping[str, Any],
    config_hash: str,
    stages: list[dict[str, Any]],
    counts: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    annotation_mode: str,
    annotator_id: Optional[str],
    annotation_status_counts: Mapping[str, int],
    error: Optional[Mapping[str, str]] = None,
    cache_reused: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "annotation_mode": annotation_mode,
        "annotation_status_counts": dict(annotation_status_counts),
        "annotator_id": annotator_id,
        "artifacts": dict(artifacts),
        "cache_reused": cache_reused,
        "code_sha": code_sha,
        "completed_at": completed_at,
        "config": dict(config_snapshot),
        "config_hash": config_hash,
        "counts": dict(counts),
        "error": dict(error) if error else None,
        "input": {
            "checksum_sha256": input_checksum,
            "classification": input_classification,
            "path_name": input_path_name,
            "record_count": input_record_count,
        },
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "output_record_count": output_record_count,
        "overall_status": overall_status,
        "run_id": run_id,
        "stages": stages,
        "started_at": started_at,
    }
    return payload


def write_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    write_json(path, payload)
    return Path(path)


def read_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("manifest is not a JSON object")
    return payload


def cached_run_is_reusable(
    manifest: Mapping[str, Any],
    *,
    input_checksum: str,
    config_hash: str,
    aggregate_path: Path,
) -> bool:
    """Reuse only when provenance matches the current input, config, and aggregate.

    ``code_sha`` is recorded for provenance but is not a cache key, so a
    documentation-only commit does not by itself invalidate a matching run.
    """
    if manifest.get("overall_status") not in {OVERALL_SUCCESS, OVERALL_REUSED}:
        return False
    if int(manifest.get("manifest_schema_version") or 0) != MANIFEST_SCHEMA_VERSION:
        return False
    if not Path(aggregate_path).is_file():
        return False
    recorded_input = manifest.get("input") or {}
    if recorded_input.get("checksum_sha256") != input_checksum:
        return False
    if manifest.get("config_hash") != config_hash:
        return False
    recorded_schema = (manifest.get("config") or {}).get("manifest_schema_version")
    if recorded_schema not in (None, MANIFEST_SCHEMA_VERSION):
        return False
    artifacts = manifest.get("artifacts") or {}
    aggregate_meta = artifacts.get("aggregate") or {}
    recorded_sha = (
        aggregate_meta.get("sha256") if isinstance(aggregate_meta, dict) else None
    )
    if not recorded_sha:
        return False
    if sha256_file(aggregate_path) != recorded_sha:
        return False
    return True
