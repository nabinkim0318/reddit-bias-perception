"""Deterministic stratified sampling for human validation.

Draws a blinded annotator file and a private sampling index from a local
model-annotation artifact. Sample-size adequacy is a study-design decision;
this module does not claim that any N is statistically sufficient.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from processing.hashing import sha256_file, sha256_json
from processing.manifest import resolve_code_sha, utc_now_iso, write_json
from validation.schema import (
    BLINDED_ANNOTATOR_COLUMNS,
    CODEBOOK_VERSION,
    CONSTRUCT_ID,
    CONSTRUCT_VERSION,
    FORBIDDEN_ANNOTATOR_FIELDS,
    PROTOCOL_VERSION,
    SAMPLING_INDEX_COLUMNS,
    VALIDATION_SCHEMA_VERSION,
    SamplingIndexRow,
    ValidationInputError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SUCCESS_STATUSES = {"success"}
FAILURE_STATUSES = {"parse_error", "model_error"}
BINARY_LABELS = {"yes", "no"}

_STRATUM_SEP = "|"


def _cell(row: Mapping[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] is not None:
            value = row[name]
            if isinstance(value, float) and value != value:  # NaN
                continue
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "<na>"}:
                return text
    return default


def _flatten_listlike(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _primary_keyword_category(row: Mapping[str, Any]) -> str:
    explicit = _cell(row, "keyword_category")
    if explicit:
        return explicit
    values = _flatten_listlike(row.get("matched_bias_types", row.get("bias_types", [])))
    if not values:
        return "unknown"
    return sorted(values)[0]


def _normalize_pred_label(value: str) -> Optional[str]:
    text = value.strip().lower()
    if not text:
        return None
    if text not in BINARY_LABELS:
        raise ValidationInputError(f"unsupported model pred_label: {value!r}")
    return text


def _load_subreddit_groups(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        mapping: dict[str, str] = {}
        if isinstance(payload, dict):
            for group, names in payload.items():
                if isinstance(names, list):
                    for name in names:
                        mapping[str(name).lower()] = str(group)
                else:
                    mapping[str(group).lower()] = str(names)
        return mapping
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping = {}
        for row in reader:
            subreddit = _cell(row, "subreddit").lower()
            group = _cell(row, "group", "subreddit_group")
            if subreddit:
                mapping[subreddit] = group or "unknown"
        return mapping


def load_annotation_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise ValidationInputError(f"input artifact not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records", payload.get("rows", []))
        if not isinstance(payload, list):
            raise ValidationInputError("JSON annotation artifact must be a list")
        return [dict(row) for row in payload]
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _prepare_record(
    row: Mapping[str, Any],
    groups_map: Mapping[str, str],
) -> Optional[dict[str, Any]]:
    record_id = _cell(row, "id", "original_record_id")
    if not record_id:
        return None
    status = _cell(row, "status", "annotation_status", "model_status").lower()
    if status not in SUCCESS_STATUSES | FAILURE_STATUSES:
        return None
    pred_raw = _cell(row, "pred_label", "model_pred_label")
    pred_label = _normalize_pred_label(pred_raw) if pred_raw else None
    if status in FAILURE_STATUSES:
        pred_label = None
    elif pred_label not in BINARY_LABELS:
        return None
    subreddit = _cell(row, "subreddit", default="unknown")
    group = _cell(row, "subreddit_group", "group")
    if not group:
        group = groups_map.get(subreddit.lower(), "unknown")
    keyword_category = _primary_keyword_category(row)
    text = _cell(row, "clean_text", "text_to_annotate", "text")
    if status in SUCCESS_STATUSES:
        strata_key = _STRATUM_SEP.join(
            [pred_label or "unknown", group or "unknown", keyword_category]
        )
    else:
        strata_key = _STRATUM_SEP.join(
            ["failure", status, group or "unknown", keyword_category]
        )
    return {
        "original_record_id": record_id,
        "model_status": status,
        "model_pred_label": pred_label,
        "subreddit": subreddit,
        "subreddit_group": group or "unknown",
        "keyword_category": keyword_category,
        "annotation_status": status,
        "strata_key": strata_key,
        "text_to_annotate": text,
    }


def deduplicate_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    dropped = 0
    for row in records:
        record_id = str(row["original_record_id"])
        if record_id in seen:
            dropped += 1
            continue
        seen.add(record_id)
        unique.append(dict(row))
    return unique, dropped


def even_allocation(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Spread n draws across strata as evenly as possible, capped by size."""
    alloc = {key: 0 for key in sizes}
    keys = sorted(key for key, size in sizes.items() if size > 0)
    if not keys or n <= 0:
        return alloc
    remaining = dict(sizes)
    assigned = 0
    target = min(int(n), sum(sizes.values()))
    while assigned < target:
        progressed = False
        for key in keys:
            if assigned >= target:
                break
            if remaining[key] > 0:
                alloc[key] += 1
                remaining[key] -= 1
                assigned += 1
                progressed = True
        if not progressed:
            break
    return alloc


def _shuffle_take(
    items: Sequence[dict[str, Any]],
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pool = list(items)
    rng.shuffle(pool)
    return pool[: max(0, k)]


def _sample_pool(
    records: Sequence[dict[str, Any]],
    sample_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_stratum.setdefault(str(row["strata_key"]), []).append(row)
    sizes = {key: len(rows) for key, rows in by_stratum.items()}
    requested = max(0, int(sample_size))
    alloc = even_allocation(sizes, requested)
    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    for key in sorted(by_stratum):
        take = alloc.get(key, 0)
        available = sizes[key]
        if take > available:
            shortfalls[key] = take - available
            take = available
        elif requested > sum(alloc.values()) and take < available:
            pass
        selected.extend(_shuffle_take(by_stratum[key], take, rng))
        if take < min(requested, available) and alloc.get(key, 0) > available:
            shortfalls[key] = alloc[key] - available
    actual = len(selected)
    if requested > actual:
        shortfalls["__total__"] = requested - actual
    diagnostics = {
        "requested": requested,
        "actual": actual,
        "stratum_sizes": sizes,
        "stratum_allocated": alloc,
        "stratum_shortfalls": shortfalls,
        "sample_size_adequacy": (
            "Sample-size adequacy is a study-design decision; "
            "this sampler does not claim statistical sufficiency."
        ),
    }
    return selected, diagnostics


def _assign_task_ids(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["strata_key"]),
            str(row["original_record_id"]),
        ),
    )
    assigned: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        item = dict(row)
        item["task_id"] = f"VAL-{index:04d}"
        assigned.append(item)
    return assigned


def _index_row(row: Mapping[str, Any]) -> dict[str, Any]:
    parsed = SamplingIndexRow(
        task_id=str(row["task_id"]),
        original_record_id=str(row["original_record_id"]),
        model_status=row["model_status"],
        model_pred_label=row["model_pred_label"],
        subreddit=str(row.get("subreddit") or ""),
        subreddit_group=str(row.get("subreddit_group") or "unknown"),
        keyword_category=str(row.get("keyword_category") or "unknown"),
        annotation_status=row["annotation_status"],
        strata_key=str(row.get("strata_key") or ""),
    )
    return parsed.model_dump()


def _annotator_row(row: Mapping[str, Any]) -> dict[str, str]:
    payload = {
        "task_id": str(row["task_id"]),
        "text_to_annotate": str(row.get("text_to_annotate") or ""),
        "human_label": "",
        "notes": "",
    }
    leaked = set(payload) & FORBIDDEN_ANNOTATOR_FIELDS
    if leaked:
        raise ValidationInputError(f"annotator export leaked fields: {leaked}")
    return payload


@dataclass
class SamplingResult:
    index_rows: list[dict[str, Any]] = field(default_factory=list)
    annotator_rows: list[dict[str, str]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def sample_validation_tasks(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    failure_sample_size: int = 0,
    groups_map: Optional[Mapping[str, str]] = None,
) -> SamplingResult:
    prepared: list[dict[str, Any]] = []
    skipped = 0
    mapping = dict(groups_map or {})
    for row in records:
        item = _prepare_record(row, mapping)
        if item is None:
            skipped += 1
            continue
        prepared.append(item)
    unique, dropped = deduplicate_records(prepared)
    success_rows = [row for row in unique if row["model_status"] in SUCCESS_STATUSES]
    failure_rows = [row for row in unique if row["model_status"] in FAILURE_STATUSES]
    rng = random.Random(int(seed))
    sampled_success, success_diag = _sample_pool(success_rows, sample_size, rng)
    sampled_failure, failure_diag = _sample_pool(failure_rows, failure_sample_size, rng)
    assigned = _assign_task_ids(sampled_success + sampled_failure)
    index_rows = [_index_row(row) for row in assigned]
    annotator_rows = [_annotator_row(row) for row in assigned]
    config = {
        "codebook_version": CODEBOOK_VERSION,
        "construct_id": CONSTRUCT_ID,
        "construct_version": CONSTRUCT_VERSION,
        "failure_sample_size": int(failure_sample_size),
        "protocol_version": PROTOCOL_VERSION,
        "sample_size": int(sample_size),
        "seed": int(seed),
        "stratification_fields": [
            "model_pred_label",
            "subreddit_group",
            "keyword_category",
            "annotation_status",
        ],
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
    }
    diagnostics = {
        "duplicates_dropped": dropped,
        "failure_pool": failure_diag,
        "input_row_count": len(records),
        "prepared_row_count": len(prepared),
        "skipped_row_count": skipped,
        "success_pool": success_diag,
        "unique_record_count": len(unique),
    }
    return SamplingResult(
        index_rows=index_rows,
        annotator_rows=annotator_rows,
        config=config,
        diagnostics=diagnostics,
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_sampling_outputs(result: SamplingResult, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "sampling_index.csv"
    tasks_path = output_dir / "annotation_tasks.csv"
    config_path = output_dir / "sampling_config.json"
    diagnostics_path = output_dir / "sampling_diagnostics.json"
    _write_csv(index_path, result.index_rows, SAMPLING_INDEX_COLUMNS)
    _write_csv(tasks_path, result.annotator_rows, BLINDED_ANNOTATOR_COLUMNS)
    if set(result.annotator_rows[0] if result.annotator_rows else []) & (
        FORBIDDEN_ANNOTATOR_FIELDS
    ):
        raise ValidationInputError("annotator-facing file contains model fields")
    config_hash = sha256_json(result.config)
    payload = dict(result.config)
    payload["config_hash"] = config_hash
    write_json(config_path, payload)
    write_json(diagnostics_path, result.diagnostics)
    return {
        "annotation_tasks": tasks_path,
        "sampling_config": config_path,
        "sampling_diagnostics": diagnostics_path,
        "sampling_index": index_path,
    }


def run_sampling(
    input_path: Path,
    output_dir: Path,
    *,
    sample_size: int,
    seed: int,
    failure_sample_size: int = 0,
    subreddit_groups: Optional[Path] = None,
) -> SamplingResult:
    records = load_annotation_records(input_path)
    groups_map = _load_subreddit_groups(subreddit_groups)
    result = sample_validation_tasks(
        records,
        sample_size=sample_size,
        seed=seed,
        failure_sample_size=failure_sample_size,
        groups_map=groups_map,
    )
    paths = write_sampling_outputs(result, output_dir)
    result.config["config_hash"] = sha256_json(
        {key: value for key, value in result.config.items() if key != "config_hash"}
    )
    result.diagnostics["source_artifact_sha256"] = sha256_file(Path(input_path))
    result.diagnostics["source_artifact_name"] = Path(input_path).name
    result.diagnostics["code_sha"] = resolve_code_sha(REPO_ROOT)
    result.diagnostics["generated_at"] = utc_now_iso()
    write_json(paths["sampling_diagnostics"], result.diagnostics)
    config_payload = dict(result.config)
    write_json(paths["sampling_config"], config_payload)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a deterministic, blinded human-validation sample from a "
            "local model-annotation artifact. Output is private/local."
        )
    )
    parser.add_argument("--input", required=True, help="Local model-results CSV/JSON")
    parser.add_argument(
        "--output-dir", required=True, help="Local validation directory"
    )
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--failure-sample-size",
        type=int,
        default=0,
        help="Optional QA sample of parse_error/model_error rows (not in yes/no metrics).",
    )
    parser.add_argument(
        "--subreddit-groups",
        default=None,
        help="Optional JSON/CSV mapping subreddit → group for stratification.",
    )
    args = parser.parse_args(argv)
    if args.sample_size < 0 or args.failure_sample_size < 0:
        print("sample sizes must be >= 0", file=sys.stderr)
        return 2
    run_sampling(
        Path(args.input),
        Path(args.output_dir),
        sample_size=args.sample_size,
        seed=args.seed,
        failure_sample_size=args.failure_sample_size,
        subreddit_groups=(
            Path(args.subreddit_groups) if args.subreddit_groups else None
        ),
    )
    print(
        "Wrote private sampling outputs. Do not commit files that contain "
        "Reddit-derived text. Sample-size adequacy is a study-design decision."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
