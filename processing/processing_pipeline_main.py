import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

# ---- import your pipeline steps ----
from processing.duckdb_data_processing import main as duckdb_data_processing
from processing.keyword_filter import main as keyword_filter
from processing.llm_few_shot_pipeline import main as llm_filter
from processing.python_pipeline import main as python_pipeline

# ---------- Config ----------
PROJECT_ROOT = Path(__file__).resolve().parents[0].parents[0]
print(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
FILTERED_DIR = DATA_DIR / "filtered"

print(DATA_DIR)
print(FILTERED_DIR)


@dataclass
class Step:
    name: str
    fn: Callable[[str], object]
    expects: tuple[Path, ...] = ()


def run_step(step: Step, subreddit: str) -> object:
    logging.info(f"\n🚩 [{step.name}] started…")
    t0 = time.time()
    try:
        result = step.fn(subreddit)  # convention: each step takes subreddit
    except Exception:
        logging.exception(f"💥 [{step.name}] failed with exception")
        sys.exit(1)
    dt = time.time() - t0
    logging.info(f"✅ [{step.name}] completed in {dt:.2f}s")

    # existence checks (if any)
    missing = [p for p in step.expects if not p.exists()]
    if missing:
        for p in missing:
            logging.error(f"❌ Expected output not found: {p}")
        sys.exit(1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subreddit",
        type=str,
        required=True,
        help="Subreddit name (e.g., 'midjourney')",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only steps whose names contain these substrings",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Skip steps whose names contain these substrings",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def filter_steps(
    steps: Sequence[Step], only: Optional[list[str]], skip: Optional[list[str]]
) -> list[Step]:
    selected = list(steps)
    if only:
        keys = [k.lower() for k in only]
        selected = [s for s in selected if any(k in s.name.lower() for k in keys)]
    if skip:
        keys = [k.lower() for k in skip]
        selected = [s for s in selected if not any(k in s.name.lower() for k in keys)]
    return selected


def build_steps(subreddit: str) -> list[Step]:
    """
    Define the pipeline and expected artifacts produced by each step.
    """
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    duckdb_out = FILTERED_DIR / f"{subreddit}_duckdb_processed.csv"
    python_filtered = FILTERED_DIR / f"{subreddit}_filtered_cleaned.csv"
    keyword_filtered = FILTERED_DIR / f"{subreddit}_keyword_filtered.csv"
    llm_yes = FILTERED_DIR / f"{subreddit}_filtered_ai_bias.csv"
    llm_no = FILTERED_DIR / f"{subreddit}_filtered_ai_non_bias.csv"

    steps: list[Step] = [
        Step(
            "1. DuckDB Data Processing", duckdb_data_processing, expects=(duckdb_out,)
        ),
        Step("2. Python/DuckDB Filtering", python_pipeline, expects=(python_filtered,)),
        Step("3. Keyword Filtering", keyword_filter, expects=(keyword_filtered,)),
        Step("4. LLM Filtering", llm_filter, expects=(llm_yes, llm_no)),
    ]
    return steps


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    steps = filter_steps(build_steps(args.subreddit), args.only, args.skip)

    if not steps:
        logging.error("No steps to run after filtering. Check --only/--skip.")
        sys.exit(2)

    logging.info(f"🏁 Starting pipeline for r/{args.subreddit}")
    t0 = time.time()
    for step in steps:
        run_step(step, args.subreddit)
    logging.info(
        f"\n🎉 Pipeline completed successfully! (total {time.time() - t0:.2f}s)"
    )


if __name__ == "__main__":
    main()
