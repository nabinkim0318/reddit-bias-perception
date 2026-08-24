"""Exploratory emotion inference with subreddit-clustered covariance.

Observational unit: Reddit post.
Posts from the same subreddit are not assumed independent.

The supported inferential path is, for each emotion score:

    emotion_score ~ C(mapped_topic_category)

with cluster-robust standard errors by subreddit when clustering is
supportable. Multiple emotion-level omnibus tests are Benjamini-Hochberg
FDR-corrected. Topic-derived categories are exploratory mappings, not
ground truth. These models do not support causal inference.

Importing this module does not read research files or run models.
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from analysis.multiplicity import MULTIPLICITY_METHOD, benjamini_hochberg, family_size
from analysis.topic_mapping import (
    TopicMappingError,
    apply_topic_category_mapping,
    load_topic_category_mapping,
)
from processing.hashing import sha256_file
from processing.manifest import resolve_code_sha, utc_now_iso, write_json

STATISTICS_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_COLUMN = "mapped_topic_category"
DEFAULT_CLUSTER_COLUMN = "subreddit"
DEFAULT_ALPHA = 0.05
# Project heuristic, not a universal statistical law. Cluster-robust SEs are
# often discussed as unreliable with very few clusters (literature commonly
# mentions tens of clusters). We flag n_clusters < 10.
FEW_CLUSTER_THRESHOLD = 10
MODEL_SPECIFICATION = "emotion_score ~ C(mapped_topic_category)"
COVARIANCE_SPECIFICATION = "cluster-robust OLS (cov_type=cluster, groups=subreddit)"
AGGREGATE_TEXT_FIELDS = {
    "clean_text",
    "clean_text_lc",
    "selftext",
    "title",
    "full_text",
    "body",
    "comments",
    "top_comments",
    "permalink",
    "author",
    "username",
    "text",
}

STATISTICS_LIMITATIONS = [
    "The observational unit is the Reddit post.",
    "Posts from the same subreddit are not assumed independent.",
    "Cluster-robust standard errors do not create a randomized design.",
    "Mapped topic categories are exploratory topic-derived groupings, not ground-truth bias categories.",
    "Statistical tests conditional on a mapped category do not validate the BERTopic solution.",
    "Statistical association is not causal inference.",
    "BH/FDR correction does not eliminate all researcher degrees of freedom.",
    "Effect estimates can be small even when a p-value is below a conventional threshold.",
    "Reddit discourse is not a representative population estimate.",
    "These analyses do not validate GoEmotions calibration.",
]


class EmotionStatisticsError(ValueError):
    """Raised when clustered emotion inference cannot be completed honestly."""


def _blank_to_na(value: Any) -> Any:
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return pd.NA
    return value


def _patsy_quote(name: str) -> str:
    escaped = str(name).replace("\\", "\\\\").replace("'", "\\'")
    return f"Q('{escaped}')"
    escaped = str(name).replace("\\", "\\\\").replace("'", "\\'")
    return f"Q('{escaped}')"


def infer_emotion_columns(
    frame: pd.DataFrame,
    explicit: Optional[Sequence[str]] = None,
) -> list[str]:
    if explicit:
        columns = [str(column) for column in explicit]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise EmotionStatisticsError(
                "missing emotion columns: " + ", ".join(missing)
            )
        return columns
    inferred = [
        column for column in frame.columns if str(column).startswith("emotion_")
    ]
    if inferred:
        return inferred
    raise EmotionStatisticsError(
        "no emotion columns found; pass --emotion-columns or include emotion_* fields"
    )


def expand_goemotions_probs(
    frame: pd.DataFrame, column: str = "goemotions_probs"
) -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    parsed = []
    for value in frame[column].tolist():
        if isinstance(value, str):
            parsed.append(ast.literal_eval(value))
        elif isinstance(value, (list, tuple)):
            parsed.append(list(value))
        else:
            parsed.append(None)
    n_emotions = next((len(item) for item in parsed if item is not None), 0)
    expanded = pd.DataFrame(
        [item if item is not None else [math.nan] * n_emotions for item in parsed],
        columns=[f"emotion_{index}" for index in range(n_emotions)],
        index=frame.index,
    )
    return pd.concat([frame, expanded], axis=1)


def summarize_emotions(
    frame: pd.DataFrame,
    *,
    emotion_columns: Sequence[str],
    group_column: str,
    cluster_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for emotion in emotion_columns:
        numeric = pd.to_numeric(frame[emotion], errors="coerce")
        work = frame.assign(_emotion_value=numeric)
        for category, subset in work.groupby(group_column, dropna=True, sort=True):
            values = subset["_emotion_value"]
            valid = subset.loc[values.notna()]
            n_valid = int(len(valid))
            rows.append(
                {
                    "category": str(category),
                    "emotion": emotion,
                    "mean": float(values.mean()) if n_valid else None,
                    "n_posts": n_valid,
                    "n_subreddits": (
                        int(valid[cluster_column].nunique(dropna=True))
                        if cluster_column in valid.columns
                        else None
                    ),
                    "std": float(values.std(ddof=1)) if n_valid > 1 else None,
                }
            )
    return pd.DataFrame(rows)


def _validate_cluster_column(
    frame: pd.DataFrame,
    cluster_column: str,
) -> tuple[int, str]:
    if cluster_column not in frame.columns:
        raise EmotionStatisticsError(
            f"cluster column {cluster_column!r} is required for clustered inference; "
            "refusing to fall back to independent-row ANOVA"
        )
    labels = frame[cluster_column]
    if labels.isna().any() or (labels.astype(str).str.strip() == "").any():
        raise EmotionStatisticsError(
            f"missing cluster labels in {cluster_column!r}; "
            "refusing to fall back to independent-row inference"
        )
    n_clusters = int(labels.nunique(dropna=True))
    if n_clusters < 2:
        raise EmotionStatisticsError(
            f"clustered inference requires at least two clusters in {cluster_column!r}"
        )
    if n_clusters < FEW_CLUSTER_THRESHOLD:
        status = "limited_few_clusters"
    else:
        status = "cluster_robust"
    return n_clusters, status


def _extract_wald_omnibus(fitted: Any, group_column: str) -> dict[str, Any]:
    unavailable = {
        "omnibus_df": None,
        "omnibus_statistic": None,
        "omnibus_unavailable_reason": "category term not found in Wald table",
        "p_raw": None,
    }
    contrast_names = [
        str(name)
        for name in getattr(fitted, "params", pd.Series(dtype=float)).index
        if group_column in str(name) and "[T." in str(name)
    ]
    if hasattr(fitted, "wald_test") and contrast_names:
        try:
            import numpy as np

            param_index = [str(name) for name in fitted.params.index]
            constraint = np.zeros((len(contrast_names), len(param_index)))
            for row_i, name in enumerate(contrast_names):
                constraint[row_i, param_index.index(name)] = 1.0
            wald = fitted.wald_test(constraint, scalar=True)
            statistic = float(np.asarray(wald.statistic).squeeze())
            p_raw = float(np.asarray(wald.pvalue).squeeze())
            df_value = getattr(wald, "df_constraint", None)
            if df_value is None:
                df_value = getattr(wald, "df_num", len(contrast_names))
            return {
                "omnibus_df": None if _is_missing_number(df_value) else float(df_value),
                "omnibus_statistic": statistic,
                "omnibus_unavailable_reason": None,
                "p_raw": p_raw,
            }
        except Exception:
            pass
    if not hasattr(fitted, "wald_test_terms"):
        unavailable["omnibus_unavailable_reason"] = "wald_test_terms is not available"
        return unavailable
    table = fitted.wald_test_terms().table
    if table is None:
        return unavailable
    frame = pd.DataFrame(table)
    if frame.empty:
        return unavailable
    match = None
    for label in (str(item) for item in frame.index):
        if group_column in label and "Intercept" not in label:
            match = label
            break
    if match is None:
        return unavailable
    row = frame.loc[match]
    statistic = row["statistic"] if "statistic" in row.index else row.iloc[0]
    p_raw = None
    for key in ("pvalue", "p-value", "P>|chi2|"):
        if key in row.index:
            p_raw = row[key]
            break
    if p_raw is None and len(row) > 1:
        p_raw = row.iloc[1]
    df_value = None
    for key in ("df_constraint", "df", "df_num"):
        if key in row.index:
            df_value = row[key]
            break
    return {
        "omnibus_df": None if _is_missing_number(df_value) else float(df_value),
        "omnibus_statistic": (
            None if _is_missing_number(statistic) else float(statistic)
        ),
        "omnibus_unavailable_reason": None,
        "p_raw": None if _is_missing_number(p_raw) else float(p_raw),
    }


def _is_missing_number(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value)) or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _contrast_rows(
    fitted: Any,
    *,
    emotion: str,
    group_column: str,
    reference_category: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = fitted.params
    bse = fitted.bse
    try:
        intervals = fitted.conf_int(alpha=DEFAULT_ALPHA)
    except Exception:
        intervals = None
    for name, estimate in params.items():
        name_str = str(name)
        if group_column not in name_str or "[T." not in name_str:
            continue
        category = name_str.split("[T.", 1)[-1].rstrip("]")
        std_error = bse.get(name) if hasattr(bse, "get") else bse[name]
        ci_low = None
        ci_high = None
        ci_reason = None
        if intervals is None:
            ci_reason = "cluster-robust confidence interval unavailable"
        else:
            try:
                ci_low = float(intervals.loc[name].iloc[0])
                ci_high = float(intervals.loc[name].iloc[1])
                if ci_low > ci_high:
                    ci_low, ci_high = ci_high, ci_low
            except Exception:
                ci_reason = "cluster-robust confidence interval unavailable"
                ci_low = None
                ci_high = None
        rows.append(
            {
                "category": category,
                "ci_95_high": ci_high,
                "ci_95_low": ci_low,
                "ci_unavailable_reason": ci_reason,
                "emotion": emotion,
                "estimate": float(estimate),
                "reference_category": reference_category,
                "std_error": (
                    None if _is_missing_number(std_error) else float(std_error)
                ),
            }
        )
    return rows


def fit_clustered_emotion_model(
    frame: pd.DataFrame,
    *,
    emotion: str,
    group_column: str,
    cluster_column: str,
    n_clusters: int,
    inference_status: str,
) -> dict[str, Any]:
    from statsmodels.formula.api import ols

    numeric = pd.to_numeric(frame[emotion], errors="coerce")
    work = frame.loc[numeric.notna()].copy()
    work[emotion] = numeric.loc[numeric.notna()]
    n_posts = int(len(work))
    n_groups = int(work[group_column].nunique(dropna=True))
    result = {
        "emotion": emotion,
        "excluded_missing_emotion": int(numeric.isna().sum()),
        "inference_status": inference_status,
        "n_clusters": n_clusters,
        "n_groups": n_groups,
        "n_posts": n_posts,
        "omnibus_df": None,
        "omnibus_statistic": None,
        "p_raw": None,
        "contrasts": [],
    }
    if n_posts < 3 or n_groups < 2:
        result["inference_status"] = "not_estimable"
        result["omnibus_unavailable_reason"] = (
            "need at least two mapped categories and three evaluable posts"
        )
        return result

    levels = sorted(work[group_column].astype(str).unique())
    reference = levels[0]
    formula = (
        f"{_patsy_quote(emotion)} ~ C({_patsy_quote(group_column)}, "
        f"Treatment(reference={reference!r}))"
    )
    try:
        fitted = ols(formula, data=work).fit(
            cov_type="cluster",
            cov_kwds={"groups": work[cluster_column]},
        )
    except Exception as exc:
        result["inference_status"] = "not_estimable"
        result["omnibus_unavailable_reason"] = f"cluster-robust OLS failed: {exc}"
        return result

    omnibus = _extract_wald_omnibus(fitted, group_column)
    result.update(omnibus)
    result["contrasts"] = _contrast_rows(
        fitted,
        emotion=emotion,
        group_column=group_column,
        reference_category=reference,
    )
    result["reference_category"] = reference
    return result


def prepare_analysis_frame(
    frame: pd.DataFrame,
    *,
    group_column: str,
    topic_column: str = "topic",
    mapping_path: Optional[Path] = None,
    topic_run_id: Optional[str] = None,
    topic_assignment_checksum: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = expand_goemotions_probs(frame.copy())
    mapping_meta: dict[str, Any] = {
        "group_column": group_column,
        "mapping_provenance": "group_column_precomputed",
        "topic_assignment_checksum": topic_assignment_checksum,
        "topic_run_id": topic_run_id,
    }
    if mapping_path is not None:
        mapping = load_topic_category_mapping(mapping_path)
        if topic_column not in work.columns:
            raise EmotionStatisticsError(
                f"topic column {topic_column!r} is required to apply a mapping"
            )
        topics = [int(value) for value in work[topic_column].tolist()]
        checksum = topic_assignment_checksum
        if checksum is None:
            from analysis.topic_stability import assignment_checksum

            checksum = assignment_checksum(topics)
        run_id = topic_run_id or mapping.topic_run_id
        categories = apply_topic_category_mapping(
            topics,
            mapping,
            topic_run_id=run_id,
            topic_assignment_checksum=checksum,
        )
        work[group_column] = categories
        mapping_meta = {
            "group_column": group_column,
            "mapping_path_name": Path(mapping_path).name,
            "mapping_provenance": "enforced",
            "mapping_version": mapping.mapping_version,
            "topic_assignment_checksum": checksum,
            "topic_run_id": run_id,
        }
    if group_column not in work.columns:
        raise EmotionStatisticsError(
            f"group column {group_column!r} is missing; supply a topic mapping "
            "bound to a specific topic run"
        )
    work[group_column] = work[group_column].map(_blank_to_na)
    return work, mapping_meta


def run_emotion_statistics(
    frame: pd.DataFrame,
    *,
    group_column: str = DEFAULT_GROUP_COLUMN,
    cluster_column: str = DEFAULT_CLUSTER_COLUMN,
    emotion_columns: Optional[Sequence[str]] = None,
    mapping_path: Optional[Path] = None,
    topic_run_id: Optional[str] = None,
    topic_assignment_checksum: Optional[str] = None,
    input_filename: str = "input.csv",
    input_sha256: Optional[str] = None,
) -> dict[str, Any]:
    prepared, mapping_meta = prepare_analysis_frame(
        frame,
        group_column=group_column,
        mapping_path=mapping_path,
        topic_run_id=topic_run_id,
        topic_assignment_checksum=topic_assignment_checksum,
    )
    emotions = infer_emotion_columns(prepared, emotion_columns)
    input_row_count = int(len(prepared))
    evaluable = prepared.dropna(subset=[group_column]).copy()
    excluded_ungrouped = input_row_count - int(len(evaluable))
    if evaluable.empty:
        raise EmotionStatisticsError(
            "no rows remain after dropping unmapped categories"
        )
    n_clusters, inference_status = _validate_cluster_column(evaluable, cluster_column)
    n_groups = int(evaluable[group_column].nunique(dropna=True))
    descriptives = summarize_emotions(
        evaluable,
        emotion_columns=emotions,
        group_column=group_column,
        cluster_column=cluster_column,
    )
    tests: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for emotion in emotions:
        fit = fit_clustered_emotion_model(
            evaluable,
            emotion=emotion,
            group_column=group_column,
            cluster_column=cluster_column,
            n_clusters=n_clusters,
            inference_status=inference_status,
        )
        contrasts.extend(fit.get("contrasts") or [])
        tests.append(
            {
                "emotion": emotion,
                "excluded_missing_emotion": fit["excluded_missing_emotion"],
                "inference_status": fit["inference_status"],
                "n_clusters": fit["n_clusters"],
                "n_groups": fit["n_groups"],
                "n_posts": fit["n_posts"],
                "omnibus_df": fit.get("omnibus_df"),
                "omnibus_statistic": fit.get("omnibus_statistic"),
                "p_raw": fit.get("p_raw"),
            }
        )
    p_raw_values = [row["p_raw"] for row in tests]
    q_values = benjamini_hochberg(p_raw_values)
    for row, q_value in zip(tests, q_values):
        row["p_fdr_bh"] = q_value
    tests_frame = pd.DataFrame(tests)
    contrasts_frame = pd.DataFrame(contrasts)
    limitations = list(STATISTICS_LIMITATIONS)
    if inference_status == "limited_few_clusters":
        limitations.append(
            f"Only {n_clusters} clusters were observed, below the project "
            f"heuristic of {FEW_CLUSTER_THRESHOLD}. Cluster-robust standard "
            "errors can be unstable with few clusters; this is flagged rather "
            "than treated as a solved design problem."
        )
    manifest = {
        "alpha": DEFAULT_ALPHA,
        "cluster_column": cluster_column,
        "code_sha": resolve_code_sha(REPO_ROOT),
        "completed_at": utc_now_iso(),
        "covariance_specification": COVARIANCE_SPECIFICATION,
        "emotion_columns": emotions,
        "evaluable_row_count": int(len(evaluable)),
        "excluded_row_counts": {
            "missing_mapped_category": excluded_ungrouped,
            "missing_emotion_by_outcome": {
                row["emotion"]: row["excluded_missing_emotion"] for row in tests
            },
        },
        "group_column": group_column,
        "hypotheses_in_fdr_family": family_size(p_raw_values),
        "inference_status": inference_status,
        "input": {
            "filename": Path(input_filename).name,
            "row_count": input_row_count,
            "sha256": input_sha256,
        },
        "limitations": limitations,
        "mapping": mapping_meta,
        "model_specification": MODEL_SPECIFICATION,
        "multiple_testing_method": MULTIPLICITY_METHOD,
        "n_clusters": n_clusters,
        "n_groups": n_groups,
        "observational_unit": "reddit_post",
        "overall_status": "success",
        "statistics_schema_version": STATISTICS_SCHEMA_VERSION,
    }
    return {
        "contrasts": contrasts_frame,
        "descriptives": descriptives,
        "manifest": manifest,
        "tests": tests_frame,
    }


def _assert_aggregate_privacy(frame: pd.DataFrame) -> None:
    overlap = AGGREGATE_TEXT_FIELDS.intersection(set(frame.columns))
    if overlap:
        raise EmotionStatisticsError(
            "aggregate output must not include source-text fields: "
            + ", ".join(sorted(overlap))
        )


def write_statistics_outputs(
    result: Mapping[str, Any], output_dir: Path
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptives = result["descriptives"]
    tests = result["tests"]
    contrasts = result["contrasts"]
    _assert_aggregate_privacy(descriptives)
    _assert_aggregate_privacy(tests)
    if not contrasts.empty:
        _assert_aggregate_privacy(contrasts)
    descriptives_path = output_dir / "emotion_descriptives.csv"
    tests_path = output_dir / "emotion_clustered_tests.csv"
    contrasts_path = output_dir / "emotion_contrasts.csv"
    manifest_path = output_dir / "analysis_manifest.json"
    descriptives.to_csv(descriptives_path, index=False)
    tests.to_csv(tests_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    manifest = dict(result["manifest"])
    manifest["output_checksums"] = {
        "emotion_clustered_tests.csv": sha256_file(tests_path),
        "emotion_contrasts.csv": sha256_file(contrasts_path),
        "emotion_descriptives.csv": sha256_file(descriptives_path),
    }
    write_json(manifest_path, manifest)
    manifest["output_checksums"]["analysis_manifest.json"] = sha256_file(manifest_path)
    write_json(manifest_path, manifest)
    return {
        "contrasts": str(contrasts_path),
        "descriptives": str(descriptives_path),
        "manifest": str(manifest_path),
        "tests": str(tests_path),
    }


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory cluster-aware emotion models for a local research table. "
            "Not part of make demo. Does not support causal inference."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--cluster-column", default=DEFAULT_CLUSTER_COLUMN)
    parser.add_argument("--emotion-columns", default=None)
    parser.add_argument(
        "--topic-mapping",
        default=None,
        help="JSON mapping bound to a specific topic run (required to map topic IDs).",
    )
    parser.add_argument("--topic-run-id", default=None)
    parser.add_argument("--topic-assignment-checksum", default=None)
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    emotion_columns = (
        [part.strip() for part in args.emotion_columns.split(",") if part.strip()]
        if args.emotion_columns
        else None
    )
    try:
        frame = _load_table(input_path)
        result = run_emotion_statistics(
            frame,
            group_column=args.group_column,
            cluster_column=args.cluster_column,
            emotion_columns=emotion_columns,
            mapping_path=Path(args.topic_mapping) if args.topic_mapping else None,
            topic_run_id=args.topic_run_id,
            topic_assignment_checksum=args.topic_assignment_checksum,
            input_filename=input_path.name,
            input_sha256=sha256_file(input_path),
        )
        write_statistics_outputs(result, Path(args.output_dir))
    except (EmotionStatisticsError, TopicMappingError) as exc:
        print(f"emotion statistics error: {exc}", file=sys.stderr)
        return 2
    print(
        "Wrote aggregate emotion statistics. Mapped topic categories are "
        "exploratory. Results are not causal findings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
