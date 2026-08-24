"""LLM annotation contract: parse, status, and yes/no split helpers.

Scientific labels are ``yes`` / ``no`` only after a successful parse.
Execution failures are recorded as ``status`` with a null ``pred_label``.
Malformed model text is never guessed via substring heuristics.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
from pydantic import ValidationError

from processing.schema import (
    ANNOTATION_SCHEMA_VERSION,
    AnnotationStatus,
    ClassificationResult,
    PredLabel,
)

GenerateFn = Callable[[List[str]], List[str]]

STATUS_SUCCESS: AnnotationStatus = "success"
STATUS_PARSE_ERROR: AnnotationStatus = "parse_error"
STATUS_MODEL_ERROR: AnnotationStatus = "model_error"

ERROR_EMPTY_OUTPUT = "empty_output"
ERROR_MALFORMED_OUTPUT = "malformed_output"
ERROR_AMBIGUOUS_OUTPUT = "ambiguous_output"
ERROR_GENERATION_EXCEPTION = "generation_exception"
ERROR_MODEL_UNAVAILABLE = "model_unavailable"
ERROR_OUTPUT_CARDINALITY_MISMATCH = "output_cardinality_mismatch"
ERROR_VALIDATION_ERROR = "validation_error"
ERROR_MISSING_OUTPUT = "missing_output"

ANNOTATION_COLUMNS = [
    "id",
    "subreddit",
    "clean_text",
    "status",
    "pred_label",
    "error_type",
    "llm_reasoning",
    "raw_output",
    "schema_version",
]

_EMPTY_SENTINELS = {"", "[empty_output]"}
_LABEL_LINE_RE = re.compile(r"(?im)^\s*label\s*:\s*(.+?)\s*$")
_REASONING_LINE_RE = re.compile(r"(?im)^\s*reasoning\s*:\s*(.+?)\s*$")
_FENCE_OPEN_RE = re.compile(r"^```(?:\w+)?\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```$")


class AnnotationModelError(Exception):
    """Inference failed before a per-item parse could run."""

    def __init__(self, error_type: str = ERROR_GENERATION_EXCEPTION) -> None:
        self.error_type = error_type
        super().__init__(error_type)


@dataclass(frozen=True)
class AnnotationParse:
    status: AnnotationStatus
    pred_label: Optional[PredLabel]
    error_type: Optional[str] = None
    llm_reasoning: str = ""


def _normalize_token(value: str) -> str:
    token = value.strip().strip("\"'").strip()
    token = token.rstrip(".,;!")
    return token.lower()


def _strip_trivial_wrappers(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_OPEN_RE.sub("", stripped, count=1)
        stripped = _FENCE_CLOSE_RE.sub("", stripped, count=1)
        stripped = stripped.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        stripped = stripped[1:-1].strip()
    return stripped


def _extract_reasoning(text: str) -> str:
    match = _REASONING_LINE_RE.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def parse_annotation_output(decoded_output: Optional[str]) -> AnnotationParse:
    """Strictly parse a model response into yes/no or parse_error.

    Accepted forms (after trim, surrounding quotes, and case folding):
    - the entire response is ``yes`` or ``no``
    - a line ``Label: yes`` or ``Label: no`` (the prompt contract)

    Anything else, including hedging, both labels, JSON objects, and prose,
    is ``parse_error`` with a null scientific label.
    """
    if decoded_output is None:
        return AnnotationParse(
            status=STATUS_PARSE_ERROR,
            pred_label=None,
            error_type=ERROR_EMPTY_OUTPUT,
        )

    text = _strip_trivial_wrappers(str(decoded_output))
    if text.lower() in _EMPTY_SENTINELS:
        return AnnotationParse(
            status=STATUS_PARSE_ERROR,
            pred_label=None,
            error_type=ERROR_EMPTY_OUTPUT,
        )

    whole = _normalize_token(text)
    if whole in {"yes", "no"}:
        return AnnotationParse(
            status=STATUS_SUCCESS,
            pred_label=whole,  # type: ignore[arg-type]
            error_type=None,
            llm_reasoning="",
        )

    label_values: List[str] = []
    for raw_value in _LABEL_LINE_RE.findall(text):
        normalized = _normalize_token(raw_value)
        if normalized not in {"yes", "no"}:
            return AnnotationParse(
                status=STATUS_PARSE_ERROR,
                pred_label=None,
                error_type=ERROR_MALFORMED_OUTPUT,
            )
        label_values.append(normalized)

    if not label_values:
        return AnnotationParse(
            status=STATUS_PARSE_ERROR,
            pred_label=None,
            error_type=ERROR_MALFORMED_OUTPUT,
        )

    unique = set(label_values)
    if len(unique) > 1:
        return AnnotationParse(
            status=STATUS_PARSE_ERROR,
            pred_label=None,
            error_type=ERROR_AMBIGUOUS_OUTPUT,
        )

    label: PredLabel = "yes" if "yes" in unique else "no"
    return AnnotationParse(
        status=STATUS_SUCCESS,
        pred_label=label,
        error_type=None,
        llm_reasoning=_extract_reasoning(text),
    )


def build_annotation_record(
    *,
    post_id: str,
    subreddit: str,
    clean_text: str,
    status: AnnotationStatus,
    pred_label: Optional[PredLabel] = None,
    error_type: Optional[str] = None,
    llm_reasoning: str = "",
    raw_output: str = "",
) -> Dict[str, object]:
    if status != STATUS_SUCCESS:
        pred_label = None
    record = ClassificationResult(
        id=str(post_id),
        subreddit=str(subreddit),
        clean_text=clean_text if isinstance(clean_text, str) else str(clean_text),
        status=status,
        pred_label=pred_label,
        error_type=error_type,
        llm_reasoning=llm_reasoning or "",
        raw_output=raw_output or "",
        schema_version=ANNOTATION_SCHEMA_VERSION,
    )
    return record.model_dump()


def _failure_record(
    *,
    post_id: str,
    subreddit: str,
    clean_text: str,
    status: AnnotationStatus,
    error_type: str,
    raw_output: str = "",
) -> Dict[str, object]:
    return build_annotation_record(
        post_id=post_id,
        subreddit=subreddit,
        clean_text=clean_text,
        status=status,
        pred_label=None,
        error_type=error_type,
        llm_reasoning="",
        raw_output=raw_output,
    )


def postprocess_decoded_outputs(
    decoded_outputs: Sequence[str],
    batch_texts: Sequence[str],
    batch_ids: Sequence[str],
    batch_subreddits: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, decoded in enumerate(decoded_outputs):
        parsed = parse_annotation_output(decoded)
        raw = decoded if isinstance(decoded, str) else ""
        try:
            rows.append(
                build_annotation_record(
                    post_id=str(batch_ids[i]),
                    subreddit=str(batch_subreddits[i]),
                    clean_text=batch_texts[i],
                    status=parsed.status,
                    pred_label=parsed.pred_label,
                    error_type=parsed.error_type,
                    llm_reasoning=parsed.llm_reasoning,
                    raw_output=raw,
                )
            )
        except ValidationError:
            logging.error("Annotation schema validation failed for id=%s", batch_ids[i])
            rows.append(
                _failure_record(
                    post_id=str(batch_ids[i]),
                    subreddit=str(batch_subreddits[i]),
                    clean_text=batch_texts[i] if i < len(batch_texts) else "",
                    status=STATUS_PARSE_ERROR,
                    error_type=ERROR_VALIDATION_ERROR,
                    raw_output="",
                )
            )
    return rows


def classify_batch(
    batch_texts: List[str],
    batch_ids: List[str],
    batch_subreddits: List[str],
    generate_fn: GenerateFn,
) -> List[Dict[str, object]]:
    """Classify one batch using an injectable generation function.

    Generation is never allowed to collapse into ``pred_label="no"``.
    Input cardinality is preserved: every input row yields exactly one output.
    """
    n = len(batch_texts)
    if n == 0:
        return []

    try:
        decoded_outputs = generate_fn(batch_texts)
    except AnnotationModelError as exc:
        return [
            _failure_record(
                post_id=str(batch_ids[i]) if i < len(batch_ids) else f"unknown_{i}",
                subreddit=(
                    str(batch_subreddits[i]) if i < len(batch_subreddits) else "unknown"
                ),
                clean_text=batch_texts[i] if i < n else "",
                status=STATUS_MODEL_ERROR,
                error_type=exc.error_type,
            )
            for i in range(n)
        ]
    except Exception:
        logging.error("Model inference failed (%s)", ERROR_GENERATION_EXCEPTION)
        return [
            _failure_record(
                post_id=str(batch_ids[i]) if i < len(batch_ids) else f"unknown_{i}",
                subreddit=(
                    str(batch_subreddits[i]) if i < len(batch_subreddits) else "unknown"
                ),
                clean_text=batch_texts[i] if i < n else "",
                status=STATUS_MODEL_ERROR,
                error_type=ERROR_GENERATION_EXCEPTION,
            )
            for i in range(n)
        ]

    if not isinstance(decoded_outputs, list) or len(decoded_outputs) != n:
        logging.error("Generation output cardinality mismatch: expected %s", n)
        return [
            _failure_record(
                post_id=str(batch_ids[i]) if i < len(batch_ids) else f"unknown_{i}",
                subreddit=(
                    str(batch_subreddits[i]) if i < len(batch_subreddits) else "unknown"
                ),
                clean_text=batch_texts[i] if i < n else "",
                status=STATUS_MODEL_ERROR,
                error_type=ERROR_OUTPUT_CARDINALITY_MISMATCH,
            )
            for i in range(n)
        ]

    return postprocess_decoded_outputs(
        decoded_outputs, batch_texts, batch_ids, batch_subreddits
    )


def _scientific_split_masks(result_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Yes/no only when both status and pred_label match the scientific contract."""
    yes_mask = result_df["status"].eq(STATUS_SUCCESS) & result_df["pred_label"].eq(
        "yes"
    )
    no_mask = result_df["status"].eq(STATUS_SUCCESS) & result_df["pred_label"].eq("no")
    return yes_mask, no_mask


def split_annotation_frames(
    result_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split annotations into success-yes, success-no, and unclassified frames.

    Only ``status == success`` with ``pred_label`` ``yes``/``no`` enter those
    frames. ``success`` with a null label, missing ``status``, and all failures
    are unclassified — never coerced to ``no``.
    """
    if result_df.empty:
        empty = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        return empty.copy(), empty.copy(), empty.copy()

    if "status" not in result_df.columns:
        return (
            pd.DataFrame(columns=ANNOTATION_COLUMNS),
            pd.DataFrame(columns=ANNOTATION_COLUMNS),
            result_df.copy(),
        )

    yes_mask, no_mask = _scientific_split_masks(result_df)
    yes_df = result_df[yes_mask].copy()
    no_df = result_df[no_mask].copy()
    unclassified_df = result_df[~(yes_mask | no_mask)].copy()
    return yes_df, no_df, unclassified_df


def summarize_annotation_counts(result_df: pd.DataFrame) -> Dict[str, int]:
    if result_df.empty:
        return {
            "total": 0,
            "success_yes": 0,
            "success_no": 0,
            "parse_error": 0,
            "model_error": 0,
            "unclassified": 0,
        }

    if "status" not in result_df.columns:
        n = int(len(result_df))
        return {
            "total": n,
            "success_yes": 0,
            "success_no": 0,
            "parse_error": 0,
            "model_error": 0,
            "unclassified": n,
        }

    yes_mask, no_mask = _scientific_split_masks(result_df)
    success_yes = int(yes_mask.sum())
    success_no = int(no_mask.sum())
    parse_error = int(result_df["status"].eq(STATUS_PARSE_ERROR).sum())
    model_error = int(result_df["status"].eq(STATUS_MODEL_ERROR).sum())
    unclassified = int((~(yes_mask | no_mask)).sum())
    return {
        "total": int(len(result_df)),
        "success_yes": success_yes,
        "success_no": success_no,
        "parse_error": parse_error,
        "model_error": model_error,
        "unclassified": unclassified,
    }


def log_annotation_counts(counts: Dict[str, int]) -> None:
    successful = counts["success_yes"] + counts["success_no"]
    logging.info("Annotation counts:")
    logging.info("  success_yes: %s", counts["success_yes"])
    logging.info("  success_no: %s", counts["success_no"])
    logging.info("  parse_error: %s", counts["parse_error"])
    logging.info("  model_error: %s", counts["model_error"])
    logging.info("  unclassified: %s", counts["unclassified"])
    logging.info("  total: %s", counts["total"])
    if successful:
        logging.info(
            "  yes among successful: %s/%s (failures excluded from denominator)",
            counts["success_yes"],
            successful,
        )
    else:
        logging.info("  yes among successful: n/a (no successful annotations)")


def ensure_result_cardinality(
    results: List[Dict[str, object]],
    batch_ids: Sequence[str],
    batch_texts: Sequence[str],
    batch_subreddits: Sequence[str],
) -> List[Dict[str, object]]:
    """Keep one row per input. Never zip/truncate mismatched lists."""
    n = len(batch_texts)
    if len(results) == n:
        return results
    logging.error("Result count mismatch: got %s expected %s", len(results), n)
    aligned: List[Dict[str, object]] = []
    for i in range(n):
        post_id = str(batch_ids[i]) if i < len(batch_ids) else f"unknown_{i}"
        if i < len(results) and str(results[i].get("id")) == post_id:
            aligned.append(results[i])
        else:
            aligned.append(
                _failure_record(
                    post_id=post_id,
                    subreddit=(
                        str(batch_subreddits[i])
                        if i < len(batch_subreddits)
                        else "unknown"
                    ),
                    clean_text=batch_texts[i] if i < n else "",
                    status=STATUS_MODEL_ERROR,
                    error_type=ERROR_OUTPUT_CARDINALITY_MISMATCH,
                )
            )
    return aligned


def empty_annotation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def annotation_output_paths(out_dir: Path, subreddit: str) -> Dict[str, Path]:
    return {
        "yes": out_dir / f"{subreddit}_filtered_ai_bias.csv",
        "no": out_dir / f"{subreddit}_filtered_ai_non_bias.csv",
        "unclassified": out_dir / f"{subreddit}_filtered_ai_unclassified.csv",
        "combined": out_dir / f"{subreddit}_llm_classification_results.csv",
    }


def write_annotation_outputs(
    result_df: pd.DataFrame, paths: Dict[str, Path]
) -> Dict[str, int]:
    """Write combined, yes, no, and unclassified CSVs. Failures never enter yes/no."""
    yes_df, no_df, unclassified_df = split_annotation_frames(result_df)
    counts = summarize_annotation_counts(result_df)
    result_df.to_csv(paths["combined"], index=False)
    yes_df.to_csv(paths["yes"], index=False)
    no_df.to_csv(paths["no"], index=False)
    unclassified_df.to_csv(paths["unclassified"], index=False)
    log_annotation_counts(counts)
    logging.info("Annotation artifacts:")
    logging.info("  yes: %s (%s)", paths["yes"], counts["success_yes"])
    logging.info("  no: %s (%s)", paths["no"], counts["success_no"])
    logging.info(
        "  unclassified: %s (%s)", paths["unclassified"], counts["unclassified"]
    )
    logging.info("  combined: %s", paths["combined"])
    return counts
