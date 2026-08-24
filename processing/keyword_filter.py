### processing/keyword_filter.py
"""
Keyword-based post-filter (strict regex recheck) for AI bias pipeline.

- Input : data/filtered/{subreddit}_filtered_cleaned.csv (from step 2)
- Action: Re-validate AI/bias keyword matches with stricter regex boundaries,
          apply subreddit-group-specific logic (consistent with step 2),
          and save JSON/CSV for downstream analysis.
- Output: data/filtered/{subreddit}_keyword_filtered.{json,csv}
"""


from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, ValidationError

from config.config import AI_KEYWORDS, BASE_DIR, CONFIG_DIR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


BASE_DIR = Path(BASE_DIR)
KEYWORDS_CSV = CONFIG_DIR / "bias_keywords.csv"
SUBREDDIT_GROUPS_CSV = CONFIG_DIR / "subreddit_groups.csv"


# ---------- Schema (Simple Pydantic) ----------
class FilteredAIBiasPost(BaseModel):
    id: str
    subreddit: str
    subreddit_group: str | None = None
    clean_text: str
    clean_text_lc: str
    matched_bias_types: List[str]
    matched_keywords: List[str]


# ---------- Utils ----------
def _flatten_listlike(x) -> List[str]:
    """stringified list -> list[str]"""
    if isinstance(x, list):
        return [str(v) for v in x]
    if x is None:
        return []
    try:
        if bool(pd.isna(x)):
            return []
    except (ValueError, TypeError):
        pass
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            arr = ast.literal_eval(x)
            if isinstance(arr, list):
                return [str(v) for v in arr]
        except Exception:
            pass
    # comma-separated fallback
    if isinstance(x, str) and "," in x:
        return [s.strip() for s in x.split(",") if s.strip()]
    return [str(x)] if str(x).strip() else []


def load_bias_keywords(path: Path | None = None) -> Dict[str, List[str]]:
    """bias_keywords.csv -> dict[bias_type] = [kw, ...]"""
    # Expected schema: columns: category,bias_type,keyword  (or category,keyword / bias_type inferred)
    csv_path = Path(path) if path is not None else KEYWORDS_CSV
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    # Priority: use bias_type column if present, otherwise use category
    if "keyword" not in df.columns:
        raise ValueError(f"'keyword' column is required in {csv_path}")
    bias_col = "bias_type" if "bias_type" in df.columns else "category"
    out: Dict[str, List[str]] = {}
    for bias_type, subdf in df.groupby(bias_col, dropna=False):
        kws = [str(k).lower().strip() for k in subdf["keyword"].astype(str)]
        out[str(bias_type).lower()] = sorted(list({k for k in kws if k}))
    return out


def load_subreddit_groups(path: Path | None = None) -> Dict[str, str]:
    """
    subreddit_groups.csv -> dict[subreddit_lower] = group_name
    Expected schema: subreddit,group
    """
    csv_path = Path(path) if path is not None else SUBREDDIT_GROUPS_CSV
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    if not {"subreddit", "group"} <= set(df.columns):
        raise ValueError(f"'subreddit' and 'group' columns are required in {csv_path}")
    return {str(r["subreddit"]).lower(): str(r["group"]) for _, r in df.iterrows()}


def keep_by_subreddit_group(
    group: str, subreddit: str, bias_matched: List[str], ai_matched: List[str]
) -> bool:
    """Same keep policy as the per-subreddit keyword-filter stage."""
    group_key = (group or "unknown").lower()
    sub = (subreddit or "").lower()
    if group_key == "technical":
        return bool(bias_matched)
    if group_key == "creative_ai_communities":
        return bool(bias_matched)
    if group_key == "critical_discussion":
        if sub == "twoxchromosomes":
            return bool(bias_matched and ai_matched)
        return bool(bias_matched)
    if group_key == "general_reddit":
        return bool(bias_matched and ai_matched)
    return False


def keyword_policy_payload(
    *,
    groups_map: Dict[str, str],
    bias_kw_dict: Dict[str, List[str]] | None = None,
) -> dict[str, Any]:
    bias = bias_kw_dict if bias_kw_dict is not None else load_bias_keywords()
    return {
        "ai_keywords": list(AI_KEYWORDS),
        "bias_keywords": {key: list(value) for key, value in sorted(bias.items())},
        "subreddit_groups": {key: groups_map[key] for key in sorted(groups_map)},
    }


def _strict_regex_patterns(keywords: List[str]) -> List[re.Pattern]:
    """
    'copilot','dalle' are exact word boundaries,
    others allow slight variations (es|s|ed|al|ical|y).
    """
    pats: List[re.Pattern] = []
    for kw in keywords:
        base = re.escape(str(kw).lower())
        if base in {"copilot", "dalle"}:
            pattern = rf"\b{base}\b"
        else:
            pattern = rf"\b{base}(?:es|s|ed|al|ical|y)?\b"
        pats.append(re.compile(pattern, flags=re.IGNORECASE))
    return pats


AI_PATTERNS = _strict_regex_patterns(AI_KEYWORDS)


def _build_bias_patterns(
    bias_kw_dict: Dict[str, List[str]],
) -> Dict[str, List[re.Pattern]]:
    return {btype: _strict_regex_patterns(kws) for btype, kws in bias_kw_dict.items()}


def strict_match_any_with_patterns(
    text: str, keywords: List[str], patterns: List[re.Pattern]
) -> List[str]:
    if not text:
        return []
    tl = str(text).lower()
    return [kw for kw, pat in zip(keywords, patterns) if pat.search(tl)]


# ---------- Core ----------
def filter_posts_dataframe(
    df: pd.DataFrame,
    *,
    groups_map: Dict[str, str] | None = None,
    bias_kw_dict: Dict[str, List[str]] | None = None,
    strict: bool = False,
    deduplicate: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the strict keyword-filter policy to an in-memory frame.

    ``strict=True`` raises on row validation errors instead of skipping them.
    Deduplication is opt-in so callers can keep a dedicated dedupe stage.
    """
    work = df.copy()
    required_cols = [
        "id",
        "subreddit",
        "clean_text",
        "clean_text_lc",
        "matched_bias_types",
        "matched_keywords",
    ]
    missing = [col for col in required_cols if col not in work.columns]
    if strict and missing:
        raise ValueError(f"keyword filter input missing columns: {missing}")
    for col in required_cols:
        if col not in work.columns:
            work[col] = None

    work["matched_bias_types"] = work["matched_bias_types"].apply(_flatten_listlike)
    work["matched_keywords"] = work["matched_keywords"].apply(_flatten_listlike)

    resolved_groups = groups_map if groups_map is not None else load_subreddit_groups()
    if "subreddit_group" not in work.columns:
        work["subreddit_group"] = None
    work["subreddit_group"] = work.apply(
        lambda r: (
            r.get("subreddit_group")
            or resolved_groups.get(str(r.get("subreddit") or "").lower(), "unknown")
        ),
        axis=1,
    )

    resolved_bias = bias_kw_dict if bias_kw_dict is not None else load_bias_keywords()
    bias_pat_dict = _build_bias_patterns(resolved_bias)

    out_records: List[dict[str, Any]] = []
    excluded = 0
    for _, r in work.iterrows():
        post_id = str(r.get("id") or "").strip()
        if not post_id or post_id.lower() == "nan":
            if strict:
                raise ValueError("keyword filter received a row with no id")
            excluded += 1
            continue

        text = str(r.get("clean_text") or "")
        text_lc = str(r.get("clean_text_lc") or "")
        sub = str(r.get("subreddit") or "")
        group = str(r.get("subreddit_group") or "unknown").lower()
        tl = text_lc.lower()

        re_ai_matched = strict_match_any_with_patterns(tl, AI_KEYWORDS, AI_PATTERNS)
        re_bias_matched: List[str] = []
        re_bias_types: List[str] = []
        for btype in resolved_bias.keys():
            kws = resolved_bias[btype]
            pats = bias_pat_dict[btype]
            hits = [kw for kw, pat in zip(kws, pats) if pat.search(tl)]
            if hits:
                re_bias_types.append(btype)
                re_bias_matched.extend(hits)

        if not keep_by_subreddit_group(group, sub, re_bias_matched, re_ai_matched):
            excluded += 1
            continue

        rec = {
            "id": str(r["id"]),
            "subreddit": sub,
            "subreddit_group": group,
            "clean_text": text,
            "clean_text_lc": text_lc,
            "matched_bias_types": sorted(
                set(re_bias_types or _flatten_listlike(r.get("matched_bias_types")))
            ),
            "matched_keywords": sorted(
                set(
                    re_bias_matched
                    + re_ai_matched
                    + _flatten_listlike(r.get("matched_keywords"))
                )
            ),
        }
        try:
            item = FilteredAIBiasPost(**rec)
            out_records.append(item.model_dump())
        except ValidationError as e:
            if strict:
                raise ValueError(
                    f"keyword filter validation failed for id={post_id}"
                ) from e
            logger.warning(f"Validation error for id={rec.get('id')}: {e}")
            excluded += 1

    deduped = 0
    if deduplicate:
        dedup: dict[str, dict[str, Any]] = {}
        for item in out_records:
            if item["id"] in dedup:
                deduped += 1
            dedup[item["id"]] = item
        out_records = list(dedup.values())

    counts = {
        "input_count": int(len(work)),
        "output_count": len(out_records),
        "excluded_count": excluded,
        "deduplicated_count": deduped,
    }
    return out_records, counts


def run_keyword_filter(subreddit: str) -> dict[str, Path]:
    """
    Read 2nd-step output CSV, re-validate with strict regex, apply subreddit-group-specific logic, and save JSON/CSV.
    """
    input_csv = Path(BASE_DIR) / "filtered" / f"{subreddit}_filtered_cleaned.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8", on_bad_lines="skip")
    out_records, _counts = filter_posts_dataframe(df, deduplicate=True)

    # Save
    out_dir = Path(BASE_DIR) / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{subreddit}_keyword_filtered.json"
    out_csv = out_dir / f"{subreddit}_keyword_filtered.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    pd.DataFrame(out_records).to_csv(out_csv, index=False)

    logger.info(f"✅ Keyword filter saved: {out_json}  ({len(out_records)} rows)")
    logger.info(f"✅ Keyword filter saved: {out_csv}   ({len(out_records)} rows)")

    return {"json": out_json, "csv": out_csv}


def main(subreddit: str):
    return run_keyword_filter(subreddit)
