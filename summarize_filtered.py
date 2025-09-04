# summarize_filtered.py
# -------------------------------------------
# data/filtered/*_keyword_filtered.csv 들을 읽어서
# 1) 서브레딧별 잔존 행수
# 2) 서브레딧×바이어스 카테고리 분포
# 3) 서브레딧별 상위 키워드 Top-K
# 4) (있으면) 월별 추이
# 5) (옵션) 서브레딧×카테고리별 예시 문장 샘플
# 을 CSV로 저장합니다.
# -------------------------------------------

import argparse
import ast
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ---------- Helpers ----------
def parse_listlike(val) -> List[str]:
    """JSON 리스트/문자열 'a, b, c' 모두를 안전하게 리스트로 변환."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []
    # JSON-like
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # fallback: comma-separated
    return [t.strip() for t in re.split(r",\s*", s) if t.strip()]


def safe_to_datetime(row) -> pd.Timestamp:
    """created_dt(문자열) 또는 created_utc(epoch) -> Timestamp."""
    created_dt = row.get("created_dt")
    created_utc = row.get("created_utc")
    if pd.notna(created_dt):
        try:
            ts = pd.to_datetime(created_dt, errors="coerce")
            if pd.notna(ts):
                return ts
        except Exception:
            pass
    if pd.notna(created_utc):
        try:
            ts = pd.to_datetime(float(created_utc), unit="s", errors="coerce")
            if pd.notna(ts):
                return ts
        except Exception:
            pass
    return pd.NaT


def load_all(pattern: str, in_dir: Path) -> pd.DataFrame:
    files = sorted(in_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {in_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"⚠️  Failed to read {f}: {e}")
            continue

        # 파일명에서 서브레딧 추정
        m = re.search(r"([^/\\]+)_keyword_filtered\.csv$", str(f))
        inferred = m.group(1) if m else f.stem.replace("_keyword_filtered", "")
        if "subreddit" not in df.columns:
            df["subreddit"] = inferred
        else:
            df["subreddit"] = df["subreddit"].fillna(inferred).astype(str)

        # 누락 컬럼 보정
        for col in ["matched_bias_types", "matched_keywords", "created_dt", "created_utc", "clean_text", "id", "title", "selftext"]:
            if col not in df.columns:
                df[col] = np.nan

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSVs could be loaded successfully.")
    return pd.concat(dfs, ignore_index=True)


# ---------- Core computations ----------
def compute_overview(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("subreddit", dropna=False)
          .size()
          .reset_index(name="rows")
          .sort_values("rows", ascending=False)
    )


def compute_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["subreddit", "matched_bias_types"]].copy()
    tmp["matched_bias_types"] = tmp["matched_bias_types"].apply(parse_listlike)
    exploded = tmp.explode("matched_bias_types").rename(columns={"matched_bias_types": "bias_type"})
    exploded = exploded[exploded["bias_type"].notna() & (exploded["bias_type"].astype(str).str.len() > 0)]
    return (
        exploded.groupby(["subreddit", "bias_type"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["subreddit", "cnt"], ascending=[True, False])
    )


def compute_top_keywords(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    tmp = df[["subreddit", "matched_keywords"]].copy()
    tmp["matched_keywords"] = tmp["matched_keywords"].apply(parse_listlike)
    exploded = tmp.explode("matched_keywords").rename(columns={"matched_keywords": "keyword"})
    exploded = exploded[exploded["keyword"].notna() & (exploded["keyword"].astype(str).str.len() > 0)]
    counts = exploded.groupby(["subreddit", "keyword"]).size().reset_index(name="cnt")
    counts["rank"] = counts.groupby("subreddit")["cnt"].rank(method="first", ascending=False)
    return (
        counts[counts["rank"] <= topk]
        .sort_values(["subreddit", "cnt"], ascending=[True, False])
        .drop(columns=["rank"])
    )


def compute_monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    # timestamp 열 구성
    if ("created_dt" not in df.columns) and ("created_utc" not in df.columns):
        return pd.DataFrame(columns=["subreddit", "month", "cnt"])

    ts = df[["created_dt", "created_utc"]].copy()
    ts = ts.apply(safe_to_datetime, axis=1)
    if ts.notna().any():
        month = ts.dt.to_period("M").dt.to_timestamp()
        tmp = pd.DataFrame({"subreddit": df["subreddit"], "month": month})
        tmp = tmp.dropna(subset=["month"])
        return (
            tmp.groupby(["subreddit", "month"])
               .size()
               .reset_index(name="cnt")
               .sort_values(["subreddit", "month"])
        )
    return pd.DataFrame(columns=["subreddit", "month", "cnt"])


def compute_examples(df: pd.DataFrame, per_group: int = 3) -> pd.DataFrame:
    """서브레딧×바이어스 카테고리별 예시 문장 N개."""
    tmp = df[["id", "subreddit", "clean_text", "matched_bias_types"]].copy()
    tmp["matched_bias_types"] = tmp["matched_bias_types"].apply(parse_listlike)
    exploded = tmp.explode("matched_bias_types").rename(columns={"matched_bias_types": "bias_type"})
    exploded = exploded[exploded["bias_type"].notna() & (exploded["bias_type"].astype(str).str.len() > 0)]
    # 그룹별 상위 N (단순 head 사용; 무작위면 .sample(per_group, replace=False) 사용)
    examples = (exploded
                .groupby(["subreddit", "bias_type"], group_keys=False)
                .head(per_group))
    return examples[["subreddit", "bias_type", "id", "clean_text"]]


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Summarize *_keyword_filtered.csv outputs.")
    ap.add_argument("--in-dir", default="data/filtered", help="Input dir (contains *_keyword_filtered.csv)")
    ap.add_argument("--pattern", default="*_keyword_filtered.csv", help="Glob pattern for input files")
    ap.add_argument("--out-dir", default="data/filtered/_summaries", help="Directory to write summaries")
    ap.add_argument("--topk", type=int, default=20, help="Top-K keywords per subreddit")
    ap.add_argument("--samples-per-group", type=int, default=3, help="Examples per (subreddit, category)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(args.pattern, in_dir)
    print(f"✅ Loaded {len(df):,} rows from {in_dir}")

    # 1) 서브레딧별 개수
    overview = compute_overview(df)
    overview.to_csv(out_dir / "overview_by_subreddit.csv", index=False)
    print("📄 overview_by_subreddit.csv saved.")

    # 2) 카테고리 분포
    cat = compute_category_breakdown(df)
    cat.to_csv(out_dir / "category_breakdown.csv", index=False)
    print("📄 category_breakdown.csv saved.")

    # 3) 상위 키워드
    topkw = compute_top_keywords(df, topk=args.topk)
    topkw.to_csv(out_dir / "top_keywords.csv", index=False)
    print("📄 top_keywords.csv saved.")

    # 4) 월별 추이
    monthly = compute_monthly_counts(df)
    if not monthly.empty:
        monthly.to_csv(out_dir / "monthly_counts.csv", index=False)
        print("📄 monthly_counts.csv saved.")
    else:
        print("ℹ️ No usable timestamps → monthly_counts skipped.")

    # 5) 예시 문장 (선택)
    if args.samples_per_group > 0:
        examples = compute_examples(df, per_group=args.samples_per_group)
        examples.to_csv(out_dir / "examples_per_category.csv", index=False)
        print("📄 examples_per_category.csv saved.")

    # 콘솔에서도 핵심 요약 한번 보여주기
    print("\n=== Overview (top 15) ===")
    print(overview.head(15).to_string(index=False))

    print("\n=== Category breakdown (first 30 rows) ===")
    print(cat.head(30).to_string(index=False))

    print("\n=== Top keywords (first 30 rows) ===")
    print(topkw.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
