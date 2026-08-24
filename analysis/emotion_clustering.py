"""Hierarchical clustering of mean emotion profiles by mapped topic category.

Local visualization only. Categories are exploratory topic-derived groupings,
not ground-truth bias labels. Importing this module has no research-data
side effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from analysis.emotion_statistics import (
    expand_goemotions_probs,
    infer_emotion_columns,
    prepare_analysis_frame,
)


def plot_category_dendrogram(
    frame: pd.DataFrame,
    *,
    group_column: str,
    emotion_columns: list[str],
    output_path: Path,
) -> None:
    work = frame.dropna(subset=[group_column, *emotion_columns])
    group_means = work.groupby(group_column)[emotion_columns].mean()
    linkage_matrix = linkage(group_means, method="ward")
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=group_means.index.tolist(), leaf_rotation=45)
    plt.title("Mapped topic-category clustering from mean emotion profiles")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local dendrogram of mapped topic categories. Not confirmatory "
            "and not a BERTopic semantic validation."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output", default="data/results/emotion_cluster_dendrogram.png"
    )
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
    plot_category_dendrogram(
        prepared,
        group_column=args.group_column,
        emotion_columns=emotions,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
