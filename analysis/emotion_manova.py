"""Legacy unadjusted MANOVA diagnostic.

This is **not** the supported inferential path. MANOVA here does not account
for subreddit clustering and must not be treated as confirmatory.

Importing this module does not read research files or run models.

See ``analysis.emotion_statistics`` for the cluster-aware exploratory path.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from analysis.emotion_statistics import (
    EmotionStatisticsError,
    expand_goemotions_probs,
    infer_emotion_columns,
    prepare_analysis_frame,
)

MANOVA_LIMITATIONS = [
    "This MANOVA does not account for subreddit clustering.",
    "It is an unadjusted exploratory diagnostic, not confirmatory inference.",
    "Mapped topic categories are exploratory, not ground-truth bias categories.",
]


def run_exploratory_manova(
    frame: pd.DataFrame,
    *,
    emotion_columns: Sequence[str],
    group_column: str,
) -> str:
    """Return a string report. Does not write files or print at import time."""
    from statsmodels.multivariate.manova import MANOVA

    work = frame.dropna(subset=[group_column, *emotion_columns]).copy()
    if work.empty:
        raise EmotionStatisticsError("no rows available for exploratory MANOVA")
    formula = " + ".join(emotion_columns) + f" ~ {group_column}"
    manova = MANOVA.from_formula(formula, data=work)
    header = (
        "UNADJUSTED EXPLORATORY MANOVA (not cluster-aware; not confirmatory)\n"
        + "\n".join(f"- {item}" for item in MANOVA_LIMITATIONS)
        + "\n\n"
    )
    return header + str(manova.mv_test())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy unadjusted MANOVA. Not the supported inferential path. "
            "Does not account for subreddit clustering."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--group-column", default="mapped_topic_category")
    parser.add_argument("--emotion-columns", default=None)
    parser.add_argument("--topic-mapping", default=None)
    parser.add_argument("--topic-run-id", default=None)
    parser.add_argument("--topic-assignment-checksum", default=None)
    args = parser.parse_args(argv)
    warnings.warn(
        "analysis.emotion_manova is an unadjusted exploratory diagnostic and "
        "is not the supported cluster-aware inferential path.",
        UserWarning,
        stacklevel=1,
    )
    path = Path(args.input)
    frame = pd.read_csv(path) if path.suffix.lower() != ".json" else pd.read_json(path)
    try:
        prepared, _ = prepare_analysis_frame(
            expand_goemotions_probs(frame),
            group_column=args.group_column,
            mapping_path=Path(args.topic_mapping) if args.topic_mapping else None,
            topic_run_id=args.topic_run_id,
            topic_assignment_checksum=args.topic_assignment_checksum,
        )
        emotions = infer_emotion_columns(
            prepared,
            (
                [
                    part.strip()
                    for part in args.emotion_columns.split(",")
                    if part.strip()
                ]
                if args.emotion_columns
                else None
            ),
        )
        print(
            run_exploratory_manova(
                prepared, emotion_columns=emotions, group_column=args.group_column
            )
        )
    except EmotionStatisticsError as exc:
        print(f"exploratory MANOVA error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
