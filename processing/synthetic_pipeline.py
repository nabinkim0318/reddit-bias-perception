"""Canonical offline synthetic pipeline.

Exercises preprocessing, keyword filtering, the annotation contract,
aggregation, and provenance using fully synthetic fixtures.

This path does not call Reddit, download models, or load a real LLM.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from config.config import PROJECT_ROOT, TEMPLATE_PATH
from processing.clean_text import (
    DATE_FILTER_END_UTC,
    DATE_FILTER_START_UTC,
    PREPROCESS_SCHEMA_VERSION,
    build_clean_text,
    build_clean_text_lc,
    in_canonical_date_window,
    is_valid_content,
)
from processing.hashing import sha256_file, sha256_json
from processing.keyword_filter import (
    filter_posts_dataframe,
    keyword_policy_payload,
    load_bias_keywords,
    load_subreddit_groups,
)
from processing.llm_annotation import summarize_annotation_counts
from processing.manifest import (
    AGGREGATE_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    OVERALL_FAILURE,
    OVERALL_REUSED,
    OVERALL_SUCCESS,
    artifact_entry,
    build_manifest,
    cached_run_is_reusable,
    new_run_id,
    read_json,
    read_manifest,
    resolve_code_sha,
    utc_now_iso,
    write_json,
    write_manifest,
)
from processing.schema import ANNOTATION_SCHEMA_VERSION
from processing.synthetic_annotator import (
    ANNOTATION_MODE,
    SYNTHETIC_ANNOTATOR_ID,
    SyntheticDemoAnnotator,
)

logger = logging.getLogger(__name__)

INPUT_CLASSIFICATION = "synthetic"
STAGE_SCHEMA_VERSION = 1
DEFAULT_SYNTHETIC_INPUT = (
    PROJECT_ROOT / "tests" / "fixtures" / "synthetic" / "posts.json"
)
DEFAULT_SYNTHETIC_GROUPS = (
    PROJECT_ROOT / "tests" / "fixtures" / "synthetic" / "subreddit_groups.csv"
)

REQUIRED_INPUT_FIELDS = ("id", "subreddit", "title", "selftext", "created_utc")


class SyntheticPipelineError(Exception):
    """Canonical pipeline failure with a compact machine-readable code."""

    def __init__(self, code: str, message: str, stage: str) -> None:
        self.code = code
        self.stage = stage
        super().__init__(message)


class PipelineInputRecord(BaseModel):
    """Public synthetic / crawler-shaped input. Extra fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    id: str
    subreddit: str
    title: str
    selftext: str
    created_utc: float


def _stage(
    name: str,
    status: str,
    *,
    input_count: int = 0,
    output_count: int = 0,
    excluded_count: int = 0,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "excluded_count": excluded_count,
        "input_count": input_count,
        "name": name,
        "output_count": output_count,
        "schema_version": STAGE_SCHEMA_VERSION,
        "status": status,
    }
    if extra:
        payload.update(extra)
    return payload


def load_input_records(input_path: Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    if not path.is_file():
        raise SyntheticPipelineError(
            "missing_input",
            f"Required input is missing: {path.name}",
            "load_input",
        )
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise SyntheticPipelineError(
                    "invalid_schema",
                    "Input JSON must be a list of record objects",
                    "validate_schema",
                )
            raw_records = payload
        elif suffix == ".jsonl":
            raw_records = []
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise SyntheticPipelineError(
                        "invalid_schema",
                        f"JSONL line {line_no} is not an object",
                        "validate_schema",
                    )
                raw_records.append(parsed)
        else:
            raise SyntheticPipelineError(
                "invalid_schema",
                f"Unsupported input format: {suffix or path.name}",
                "load_input",
            )
    except json.JSONDecodeError as exc:
        raise SyntheticPipelineError(
            "invalid_schema",
            "Input is not valid JSON",
            "validate_schema",
        ) from exc

    validated: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise SyntheticPipelineError(
                "invalid_schema",
                f"Input row {index} is not an object",
                "validate_schema",
            )
        missing = [field for field in REQUIRED_INPUT_FIELDS if field not in record]
        if missing:
            raise SyntheticPipelineError(
                "invalid_schema",
                f"Input row {index} missing fields: {missing}",
                "validate_schema",
            )
        try:
            item = PipelineInputRecord.model_validate(record)
        except ValidationError as exc:
            raise SyntheticPipelineError(
                "invalid_schema",
                f"Input row {index} failed schema validation",
                "validate_schema",
            ) from exc
        validated.append(item.model_dump())
    return validated


def preprocess_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    invalid = 0
    date_excluded = 0
    for record in records:
        title = str(record.get("title") or "")
        selftext = str(record.get("selftext") or "")
        if not (is_valid_content(title) or is_valid_content(selftext)):
            invalid += 1
            continue
        if not in_canonical_date_window(record.get("created_utc")):
            date_excluded += 1
            continue
        kept.append(
            {
                "id": str(record["id"]),
                "subreddit": str(record["subreddit"]),
                "clean_text": build_clean_text(title, selftext),
                "clean_text_lc": build_clean_text_lc(title, selftext),
                "matched_bias_types": [],
                "matched_keywords": [],
            }
        )
    counts = {
        "input_count": len(records),
        "output_count": len(kept),
        "excluded_count": invalid + date_excluded,
        "invalid_content_count": invalid,
        "date_excluded_count": date_excluded,
    }
    return kept, counts


def deduplicate_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    seen: dict[str, dict[str, Any]] = {}
    dropped = 0
    ordered: list[dict[str, Any]] = []
    for record in records:
        post_id = str(record["id"])
        if post_id in seen:
            dropped += 1
            continue
        copied = dict(record)
        seen[post_id] = copied
        ordered.append(copied)
    return ordered, dropped


def _bias_type_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for bias_type in record.get("matched_bias_types") or []:
            counter[str(bias_type)] += 1
    return dict(sorted(counter.items()))


def build_config_snapshot(
    *,
    groups_map: Mapping[str, str],
    groups_path_name: str,
) -> tuple[dict[str, Any], str]:
    bias_kw_dict = load_bias_keywords()
    policy = keyword_policy_payload(
        groups_map=dict(groups_map), bias_kw_dict=bias_kw_dict
    )
    prompt_hash = None
    template = Path(TEMPLATE_PATH)
    if template.is_file():
        prompt_hash = sha256_file(template)
    snapshot = {
        "annotation_mode": ANNOTATION_MODE,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotator_id": SYNTHETIC_ANNOTATOR_ID,
        "date_filter_end_utc": DATE_FILTER_END_UTC,
        "date_filter_start_utc": DATE_FILTER_START_UTC,
        "groups_path_name": groups_path_name,
        "keyword_policy_hash": sha256_json(policy),
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "model_id": None,
        "model_revision": None,
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "prompt_hash": prompt_hash,
        "stage_schema_version": STAGE_SCHEMA_VERSION,
    }
    return snapshot, sha256_json(snapshot)


def build_aggregate(
    *,
    annotation_counts: Mapping[str, int],
    bias_type_counts: Mapping[str, int],
    input_record_count: int,
    filtered_record_count: int,
) -> dict[str, Any]:
    successful = int(annotation_counts.get("success_yes", 0)) + int(
        annotation_counts.get("success_no", 0)
    )
    return {
        "artifact_type": "synthetic_demo_aggregate",
        "bias_type_counts": dict(bias_type_counts),
        "classification": INPUT_CLASSIFICATION,
        "disclaimer": (
            "Pipeline-validation artifact from fully synthetic fixtures. "
            "Not a research finding."
        ),
        "filtered_records": filtered_record_count,
        "input_records": input_record_count,
        "model_error": int(annotation_counts.get("model_error", 0)),
        "no": int(annotation_counts.get("success_no", 0)),
        "parse_error": int(annotation_counts.get("parse_error", 0)),
        "successful_annotations": successful,
        "total_records": int(annotation_counts.get("total", 0)),
        "unclassified": int(annotation_counts.get("unclassified", 0)),
        "yes": int(annotation_counts.get("success_yes", 0)),
    }


def _write_failure_manifest(
    *,
    output_dir: Path,
    run_id: str,
    started_at: str,
    code_sha: Optional[str],
    input_path_name: str,
    input_checksum: str,
    input_record_count: int,
    config_snapshot: Mapping[str, Any],
    config_hash: str,
    stages: list[dict[str, Any]],
    error: SyntheticPipelineError,
) -> None:
    payload = build_manifest(
        run_id=run_id,
        started_at=started_at,
        completed_at=utc_now_iso(),
        overall_status=OVERALL_FAILURE,
        code_sha=code_sha,
        input_path_name=input_path_name,
        input_checksum=input_checksum,
        input_classification=INPUT_CLASSIFICATION,
        input_record_count=input_record_count,
        output_record_count=0,
        config_snapshot=config_snapshot,
        config_hash=config_hash,
        stages=stages,
        counts={"input_records": input_record_count, "output_records": 0},
        artifacts={},
        annotation_mode=ANNOTATION_MODE,
        annotator_id=SYNTHETIC_ANNOTATOR_ID,
        annotation_status_counts={},
        error={"code": error.code, "message": str(error), "stage": error.stage},
        cache_reused=False,
    )
    write_manifest(output_dir / MANIFEST_FILENAME, payload)


def run_synthetic_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    groups_path: Optional[Path] = None,
    force: bool = False,
    annotator: Optional[SyntheticDemoAnnotator] = None,
) -> dict[str, Any]:
    """Run the canonical synthetic/offline workflow.

    Public outputs are the aggregate JSON and the provenance manifest only.
    Record-level source text is not written to the output directory.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    groups_file = Path(groups_path) if groups_path else DEFAULT_SYNTHETIC_GROUPS
    annotator = annotator or SyntheticDemoAnnotator()
    started_at = utc_now_iso()
    run_id = new_run_id()
    code_sha = resolve_code_sha(PROJECT_ROOT)
    stages: list[dict[str, Any]] = []

    if not input_path.is_file():
        raise SyntheticPipelineError(
            "missing_input",
            f"Required input is missing: {input_path.name}",
            "load_input",
        )
    if not groups_file.is_file():
        raise SyntheticPipelineError(
            "missing_input",
            f"Required keyword-groups file is missing: {groups_file.name}",
            "load_input",
        )

    input_checksum = sha256_file(input_path)
    groups_map = load_subreddit_groups(groups_file)
    config_snapshot, config_hash = build_config_snapshot(
        groups_map=groups_map, groups_path_name=groups_file.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    aggregate_path = output_dir / AGGREGATE_FILENAME

    if not force and manifest_path.is_file():
        try:
            existing = read_manifest(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            existing = None
        if existing and cached_run_is_reusable(
            existing,
            input_checksum=input_checksum,
            config_hash=config_hash,
            aggregate_path=aggregate_path,
        ):
            logger.info("Reusing synthetic demo outputs; provenance matches.")
            existing["cache_reused"] = True
            if existing.get("overall_status") == OVERALL_SUCCESS:
                existing["overall_status"] = OVERALL_REUSED
            write_manifest(manifest_path, existing)
            return {
                "aggregate": read_json(aggregate_path),
                "manifest": existing,
                "reused": True,
            }

    try:
        records = load_input_records(input_path)
        stages.append(
            _stage(
                "load_input",
                "success",
                input_count=len(records),
                output_count=len(records),
            )
        )
        stages.append(
            _stage(
                "validate_schema",
                "success",
                input_count=len(records),
                output_count=len(records),
            )
        )

        preprocessed, pre_counts = preprocess_records(records)
        stages.append(
            _stage(
                "preprocess",
                "success",
                input_count=pre_counts["input_count"],
                output_count=pre_counts["output_count"],
                excluded_count=pre_counts["excluded_count"],
                extra={
                    "date_excluded_count": pre_counts["date_excluded_count"],
                    "invalid_content_count": pre_counts["invalid_content_count"],
                },
            )
        )

        deduped, dup_count = deduplicate_records(preprocessed)
        stages.append(
            _stage(
                "deduplicate",
                "success",
                input_count=len(preprocessed),
                output_count=len(deduped),
                extra={"deduplicated_count": dup_count},
            )
        )

        if not deduped:
            filtered, kw_counts = [], {
                "input_count": 0,
                "output_count": 0,
                "excluded_count": 0,
                "deduplicated_count": 0,
            }
        else:
            filtered, kw_counts = filter_posts_dataframe(
                pd.DataFrame(deduped),
                groups_map=groups_map,
                strict=True,
                deduplicate=False,
            )
        stages.append(
            _stage(
                "keyword_filter",
                "success",
                input_count=kw_counts["input_count"],
                output_count=kw_counts["output_count"],
                excluded_count=kw_counts["excluded_count"],
            )
        )

        annotated_rows = annotator.annotate_records(filtered)
        annotation_counts = summarize_annotation_counts(
            pd.DataFrame(annotated_rows) if annotated_rows else pd.DataFrame()
        )
        stages.append(
            _stage(
                "annotate",
                "success",
                input_count=len(filtered),
                output_count=len(annotated_rows),
                extra={
                    "annotation_mode": ANNOTATION_MODE,
                    "annotator_id": annotator.identifier,
                    "status_counts": annotation_counts,
                },
            )
        )

        aggregate = build_aggregate(
            annotation_counts=annotation_counts,
            bias_type_counts=_bias_type_counts(filtered),
            input_record_count=len(records),
            filtered_record_count=len(filtered),
        )
        write_json(aggregate_path, aggregate)
        stages.append(
            _stage(
                "aggregate",
                "success",
                input_count=len(annotated_rows),
                output_count=1,
            )
        )
        stages.append(_stage("manifest", "success", output_count=1))

        counts = {
            "date_excluded": pre_counts["date_excluded_count"],
            "deduplicated": dup_count,
            "input_records": len(records),
            "invalid_content": pre_counts["invalid_content_count"],
            "keyword_excluded": kw_counts["excluded_count"],
            "keyword_kept": kw_counts["output_count"],
            "output_records": len(annotated_rows),
            "preprocessed": pre_counts["output_count"],
        }
        manifest = build_manifest(
            run_id=run_id,
            started_at=started_at,
            completed_at=utc_now_iso(),
            overall_status=OVERALL_SUCCESS,
            code_sha=code_sha,
            input_path_name=input_path.name,
            input_checksum=input_checksum,
            input_classification=INPUT_CLASSIFICATION,
            input_record_count=len(records),
            output_record_count=len(annotated_rows),
            config_snapshot=config_snapshot,
            config_hash=config_hash,
            stages=stages,
            counts=counts,
            artifacts={
                "aggregate": artifact_entry(aggregate_path),
            },
            annotation_mode=ANNOTATION_MODE,
            annotator_id=annotator.identifier,
            annotation_status_counts=annotation_counts,
            cache_reused=False,
        )
        write_manifest(manifest_path, manifest)
        return {"aggregate": aggregate, "manifest": manifest, "reused": False}
    except SyntheticPipelineError as exc:
        if exc.code != "missing_input":
            _write_failure_manifest(
                output_dir=output_dir,
                run_id=run_id,
                started_at=started_at,
                code_sha=code_sha,
                input_path_name=input_path.name,
                input_checksum=input_checksum,
                input_record_count=0,
                config_snapshot=config_snapshot,
                config_hash=config_hash,
                stages=stages
                + [_stage(exc.stage, "failure", extra={"error_code": exc.code})],
                error=exc,
            )
        raise


def run_from_cli(
    *,
    input_path: Path,
    output_dir: Path,
    groups_path: Optional[Path] = None,
    force: bool = False,
) -> int:
    try:
        result = run_synthetic_pipeline(
            input_path,
            output_dir,
            groups_path=groups_path,
            force=force,
        )
    except SyntheticPipelineError as exc:
        logger.error("[%s] %s (%s)", exc.stage, exc, exc.code)
        return 2 if exc.code == "missing_input" else 1
    status = result["manifest"].get("overall_status")
    logger.info(
        "Synthetic demo %s. aggregate=%s manifest=%s",
        status,
        AGGREGATE_FILENAME,
        MANIFEST_FILENAME,
    )
    return 0
