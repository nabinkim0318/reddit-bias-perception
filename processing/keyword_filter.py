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
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
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


def load_bias_keywords() -> Dict[str, List[str]]:
    """bias_keywords.csv -> dict[bias_type] = [kw, ...]"""
    # Expected schema: columns: category,bias_type,keyword  (or category,keyword / bias_type inferred)
    df = pd.read_csv(KEYWORDS_CSV, encoding="utf-8", on_bad_lines="skip")
    # Priority: use bias_type column if present, otherwise use category
    if "keyword" not in df.columns:
        raise ValueError(f"'keyword' column is required in {KEYWORDS_CSV}")
    bias_col = "bias_type" if "bias_type" in df.columns else "category"
    out: Dict[str, List[str]] = {}
    for bias_type, subdf in df.groupby(bias_col, dropna=False):
        kws = [str(k).lower().strip() for k in subdf["keyword"].astype(str)]
        out[str(bias_type).lower()] = sorted(list({k for k in kws if k}))
    return out


def load_subreddit_groups() -> Dict[str, str]:
    """
    subreddit_groups.csv -> dict[subreddit_lower] = group_name
    Expected schema: subreddit,group
    """
    df = pd.read_csv(SUBREDDIT_GROUPS_CSV, encoding="utf-8", on_bad_lines="skip")
    if not {"subreddit", "group"} <= set(df.columns):
        raise ValueError(
            f"'subreddit' and 'group' columns are required in {SUBREDDIT_GROUPS_CSV}"
        )
    return {str(r["subreddit"]).lower(): str(r["group"]) for _, r in df.iterrows()}


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
    bias_kw_dict: Dict[str, List[str]]
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
def run_keyword_filter(subreddit: str) -> dict[str, Path]:
    """
    Read 2nd-step output CSV, re-validate with strict regex, apply subreddit-group-specific logic, and save JSON/CSV.
    """
    input_csv = Path(BASE_DIR) / "filtered" / f"{subreddit}_filtered_cleaned.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8", on_bad_lines="skip")

    # Ensure required columns exist
    required_cols = [
        "id",
        "subreddit",
        "clean_text",
        "clean_text_lc",
        "matched_bias_types",
        "matched_keywords",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Parse list columns
    df["matched_bias_types"] = df["matched_bias_types"].apply(_flatten_listlike)
    df["matched_keywords"] = df["matched_keywords"].apply(_flatten_listlike)

    # subreddit_group mapping (recalculate if missing)
    groups_map = load_subreddit_groups()
    if "subreddit_group" not in df.columns:
        df["subreddit_group"] = None
    df["subreddit_group"] = df.apply(
        lambda r: (
            r.get("subreddit_group")
            or groups_map.get(str(r.get("subreddit") or "").lower(), "unknown")
        ),
        axis=1,
    )

    # bias keyword dictionary (for strict regex re-validation)
    bias_kw_dict = load_bias_keywords()
    bias_pat_dict = _build_bias_patterns(bias_kw_dict)
    bias_kw_flat = sorted({kw for kws in bias_kw_dict.values() for kw in kws})

    out_records: List[dict[str, Any]] = []
    for _, r in df.iterrows():
        # id guard (not needed in python pipeline)
        post_id = str(r.get("id") or "").strip()
        if not post_id or post_id.lower() == "nan":
            continue

        # stringify required fields
        text = str(r.get("clean_text") or "")
        text_lc = str(r.get("clean_text_lc") or "")
        sub = str(r.get("subreddit") or "")
        group = str(r.get("subreddit_group") or "unknown").lower()
        tl = text_lc.lower()

        # Re-validation (strict matching)
        re_ai_matched = strict_match_any_with_patterns(tl, AI_KEYWORDS, AI_PATTERNS)

        # bias type estimation (based on dictionary: which type of word matched)
        re_bias_matched = []
        re_bias_types = []
        for btype in bias_kw_dict.keys():
            kws = bias_kw_dict[btype]
            pats = bias_pat_dict[btype]
            hits = [kw for kw, pat in zip(kws, pats) if pat.search(tl)]
            if hits:
                re_bias_types.append(btype)
                re_bias_matched.extend(hits)

        # Policy (same logic as 2nd-step, but strict version)
        if group == "technical":
            keep = bool(re_bias_matched)
        elif group == "creative_ai_communities":
            keep = bool(re_bias_matched)
        elif group == "critical_discussion":
            if sub.lower() == "twoxchromosomes":
                keep = bool(re_bias_matched and re_ai_matched)
            else:
                keep = bool(re_bias_matched)
        elif group == "general_reddit":
            keep = bool(re_bias_matched and re_ai_matched)
        else:
            keep = False

        if not keep:
            continue

        # Final record (Pydantic validation)
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
            logger.warning(f"Validation error for id={rec.get('id')}: {e}")

    # dedupe by id (optional)
    dedup = {}
    for it in out_records:
        dedup[it["id"]] = it
    out_records = list(dedup.values())

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
