"""Mean emotion heatmap by mapped topic category.

Local visualization only. Mapped categories are exploratory topic-derived
groupings, not ground-truth bias labels. Importing this module has no
research-data side effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.emotion_statistics import (
    expand_goemotions_probs,
    infer_emotion_columns,
    prepare_analysis_frame,
)


def plot_emotion_means(
    frame: pd.DataFrame,
    *,
    group_column: str,
    emotion_columns: list[str],
    output_path: Path,
) -> None:
    work = frame.dropna(subset=[group_column, *emotion_columns])
    mean_df = work.groupby(group_column)[emotion_columns].mean().T
    plt.figure(figsize=(12, 6))
    sns.heatmap(mean_df, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Mean emotion scores by mapped topic category")
    plt.xlabel("Mapped topic category")
    plt.ylabel("Emotions")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local heatmap of mean emotion scores by mapped topic category. "
            "Not confirmatory."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/results/emotion_heatmap.png")
    parser.add_argument("--group-column", default="mapped_topic_category")
    parser.add_argument("--topic-mapping", default=None)
    parser.add_argument("--topic-run-id", default=None)
    parser.add_argument("--topic-assignment-checksum", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    prepared, _ = prepare_analysis_frame(
        expand_goemotions_probs(frame),
        group_column=args.group_column,
        mapping_path=Path(args.topic_mapping) if args.topic_mapping else None,
        topic_run_id=args.topic_run_id,
        topic_assignment_checksum=args.topic_assignment_checksum,
    )
    emotions = infer_emotion_columns(prepared)
    plot_emotion_means(
        prepared,
        group_column=args.group_column,
        emotion_columns=emotions,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
