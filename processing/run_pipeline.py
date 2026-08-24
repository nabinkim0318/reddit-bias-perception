"""Pipeline runner.

Canonical public/offline command (synthetic fixtures only)::

    python -m processing.run_pipeline --synthetic \\
        --input tests/fixtures/synthetic/posts.json \\
        --output-dir artifacts/synthetic_demo

The ``--subreddit`` / ``--all`` interface is a local research runner. It
expects private extracted dumps under ``data/`` and may load a real LLM.
File-existence resumability for that path is unchanged and is not
provenance-aware.
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
FILTERED_DIR = DATA_DIR / "filtered"


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


def _legacy_step_fns() -> tuple[Callable[[str], object], ...]:
    """Import the private-data stages only when that runner is invoked."""
    from processing.duckdb_data_processing import main as duckdb_data_processing
    from processing.keyword_filter import main as keyword_filter
    from processing.llm_few_shot_pipeline import main as llm_filter
    from processing.python_pipeline import main as python_pipeline

    return duckdb_data_processing, python_pipeline, keyword_filter, llm_filter


def build_steps(subreddit: str) -> list[Step]:
    """
    Define the pipeline and expected artifacts produced by each step.
    """
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    duckdb_data_processing, python_pipeline, keyword_filter, llm_filter = (
        _legacy_step_fns()
    )

    duckdb_out = FILTERED_DIR / f"{subreddit}_duckdb_processed.csv"
    python_filtered = FILTERED_DIR / f"{subreddit}_filtered_cleaned.csv"
    keyword_filtered = FILTERED_DIR / f"{subreddit}_keyword_filtered.csv"
    llm_yes = FILTERED_DIR / f"{subreddit}_filtered_ai_bias.csv"
    llm_no = FILTERED_DIR / f"{subreddit}_filtered_ai_non_bias.csv"
    llm_unclassified = FILTERED_DIR / f"{subreddit}_filtered_ai_unclassified.csv"
    llm_combined = FILTERED_DIR / f"{subreddit}_llm_classification_results.csv"

    steps: list[Step] = [
        Step(
            "1. DuckDB Data Processing", duckdb_data_processing, expects=(duckdb_out,)
        ),
        Step("2. Python/DuckDB Filtering", python_pipeline, expects=(python_filtered,)),
        Step("3. Keyword Filtering", keyword_filter, expects=(keyword_filtered,)),
        Step(
            "4. LLM Filtering",
            llm_filter,
            expects=(llm_yes, llm_no, llm_unclassified, llm_combined),
        ),
    ]
    return steps


# ---------- Batch helpers ----------
def list_subreddits_from_extracted() -> list[str]:
    """
    Scan data/extracted/*.zst and infer subreddit names by stripping suffixes like
    *_submissions.zst or *_posts.zst
    """
    if not EXTRACTED_DIR.exists():
        return []
    subs: set[str] = set()
    for p in EXTRACTED_DIR.glob("*.zst"):
        name = p.stem
        for suf in ("_submissions", "_posts"):
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        subs.add(name)
    # Also allow explicitly dropped-in jsonl names if needed
    for p in EXTRACTED_DIR.glob("*.jsonl"):
        name = p.stem
        for suf in ("_submissions", "_posts"):
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        subs.add(name)
    return sorted(subs)


def already_done(subreddit: str, steps: Sequence[Step]) -> bool:
    """Check if all expected artifacts exist for the (filtered) steps."""
    for s in steps:
        for p in s.expects:
            if not p.exists():
                return False
    return True


def run_pipeline_for_subreddit(
    subreddit: str,
    only: Optional[list[str]] = None,
    skip: Optional[list[str]] = None,
    force: bool = False,
) -> tuple[str, bool, Optional[str]]:
    """Runs the pipeline for one subreddit. Returns (subreddit, success, error_msg)."""
    try:
        steps = filter_steps(build_steps(subreddit), only, skip)
        if not steps:
            return subreddit, False, "No steps selected (check --only/--skip)"
        if not force and already_done(subreddit, steps):
            logging.info(f"⏩ [{subreddit}] skipped (all outputs already exist).")
            return subreddit, True, None
        logging.info(f"🏁 [{subreddit}] Starting pipeline")
        t0 = time.time()
        for step in steps:
            run_step(step, subreddit)
        logging.info(
            f"🎉 [{subreddit}] Pipeline completed successfully in {time.time()-t0:.2f}s"
        )
        return subreddit, True, None
    except Exception as e:
        return subreddit, False, str(e)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical offline command: --synthetic --input PATH --output-dir PATH. "
            "Legacy --subreddit/--all runs require local private dumps."
        )
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run the canonical offline synthetic demo (no Reddit, no LLM)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input artifact for --synthetic (JSON list of records)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory for --synthetic aggregate and manifest",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help="Optional subreddit-groups CSV for --synthetic",
    )
    parser.add_argument(
        "--subreddit", type=str, help="Legacy: run a single subreddit (private data)"
    )
    parser.add_argument(
        "--subreddits", nargs="+", help="Legacy: run multiple subreddits"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Legacy: auto-discover all from {EXTRACTED_DIR}",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, help="Parallel processes (default: 1)"
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
        "--force", action="store_true", help="Ignore existing outputs and re-run"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args(argv)
    if args.synthetic:
        if args.input is None or args.output_dir is None:
            parser.error("--synthetic requires --input and --output-dir")
        return args
    if not args.subreddit and not args.subreddits and not args.all:
        parser.error(
            "specify --synthetic --input PATH --output-dir PATH "
            "(canonical offline demo) or a legacy --subreddit/--subreddits/--all run"
        )
    return args


def resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.subreddit:
        return [args.subreddit]
    if args.subreddits:
        return args.subreddits
    if args.all:
        subs = list_subreddits_from_extracted()
        if not subs:
            logging.error(f"No sources found in {EXTRACTED_DIR}")
            sys.exit(3)
        return subs
    return []


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.synthetic:
        from processing.synthetic_pipeline import run_from_cli

        raise SystemExit(
            run_from_cli(
                input_path=args.input,
                output_dir=args.output_dir,
                groups_path=args.groups,
                force=args.force,
            )
        )

    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    targets = resolve_targets(args)
    logging.info(f"🎯 Targets: {targets}")

    # Run serially or with N processes
    if args.jobs <= 1 or len(targets) == 1:
        failures = []
        for sub in targets:
            sub, ok, err = run_pipeline_for_subreddit(
                sub, args.only, args.skip, args.force
            )
            if not ok:
                logging.error(f"❌ [{sub}] failed: {err}")
                failures.append(sub)
        if failures:
            logging.error(f"\n⛔ Failed: {failures}")
            sys.exit(1)
        logging.info("\n✅ All done.")
        return

    # Parallel
    failures = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {
            ex.submit(
                run_pipeline_for_subreddit, sub, args.only, args.skip, args.force
            ): sub
            for sub in targets
        }
        for fut in as_completed(futures):
            sub = futures[fut]
            try:
                _, ok, err = fut.result()
                if not ok:
                    logging.error(f"❌ [{sub}] failed: {err}")
                    failures.append(sub)
                else:
                    logging.info(f"✅ [{sub}] finished.")
            except Exception as e:
                logging.exception(f"💥 [{sub}] crashed: {e}")
                failures.append(sub)
    logging.info(
        f"\n🧮 Done in {time.time()-start:.2f}s; failures: {failures if failures else 'none'}"
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
