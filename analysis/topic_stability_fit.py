"""Local multi-seed BERTopic fitting for stability evaluation.

This command fits real topic models and may download embedding assets. It is
not part of ``make demo`` and must not run in default CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from analysis.bertopic_model import run_topic_model
from analysis.topic_config import (
    load_topic_model_config,
    parse_stability_seeds,
    with_seed,
)
from analysis.topic_stability import (
    summarize_run,
    summarize_topic_stability,
    write_stability_report,
)
from processing.hashing import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit BERTopic on the same local corpus across configured seeds and "
            "write a structural stability report. Local/research-only."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "topic_model.json")
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--text-column", default="clean_text")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = load_topic_model_config(args.config)
    seeds = parse_stability_seeds(args.seeds or config.stability_seeds)
    input_path = Path(args.input)
    suffix = input_path.suffix.lower()
    frame = pd.read_json(input_path) if suffix == ".json" else pd.read_csv(input_path)
    if args.text_column not in frame.columns:
        print(f"missing text column {args.text_column!r}", file=sys.stderr)
        return 2
    docs = frame[args.text_column].fillna("").astype(str).tolist()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        logger.info("Fitting topic model with seed=%s", seed)
        result = run_topic_model(
            with_seed(config, seed),
            docs,
            input_filename=input_path.name,
            input_sha256=sha256_file(input_path),
        )
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "assignments.json").write_text(
            json.dumps({"seed": seed, "assignments": result.topics}, indent=2) + "\n",
            encoding="utf-8",
        )
        runs.append(summarize_run(seed, result.topics))
    report = summarize_topic_stability(runs)
    write_stability_report(output_dir / "topic_stability_report.json", report)
    print(
        "Wrote local multi-seed stability artifacts. ARI is structural only "
        "and is not semantic validation of a real Reddit topic solution."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
