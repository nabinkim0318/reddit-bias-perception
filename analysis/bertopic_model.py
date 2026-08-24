"""Fit one configured BERTopic model and record run provenance.

This module fits a single topic model. Multi-seed stability measurement lives
in ``analysis.topic_stability``. Real fitting is local/research-only: it is not
part of ``make demo`` and must not run in default CI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from analysis.topic_config import (
    TopicModelConfig,
    build_cluster_model,
    build_umap_model,
    clustering_kwargs,
    config_snapshot,
    load_topic_model_config,
    umap_kwargs,
    vectorizer_identity,
)
from analysis.topic_manifest import (
    build_topic_run_manifest,
    new_topic_run_id,
    now_utc,
    topic_code_sha,
    write_topic_run_manifest,
)
from analysis.topic_probabilities import (
    ProbabilityAssignmentError,
    assigned_probabilities_from_topic_model,
)
from analysis.topic_stability import assignment_checksum, summarize_run
from processing.hashing import sha256_file, sha256_json
from processing.manifest import write_json
from utils.archive_vectorizer_config import vectorizer_model

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


class TopicModelRunError(RuntimeError):
    """Raised when a topic-model run cannot be completed safely."""


@dataclass
class TopicRunResult:
    run_id: str
    seed: int
    topics: list[int]
    topic_probabilities: list[Optional[float]]
    topic_info: pd.DataFrame
    document_topics: pd.DataFrame
    manifest: dict[str, Any]
    config: TopicModelConfig


def _bertopic_version() -> Optional[str]:
    try:
        import bertopic

        return str(getattr(bertopic, "__version__", None) or "") or None
    except Exception:
        return None


def _embedding_revision(topic_model: Any) -> Optional[str]:
    embedding = getattr(topic_model, "embedding_model", None)
    for attr in ("revision", "_revision"):
        value = getattr(embedding, attr, None)
        if value:
            return str(value)
    inner = getattr(embedding, "_first_module", None) or getattr(
        embedding, "auto_model", None
    )
    config = getattr(inner, "config", None)
    revision = getattr(config, "_commit_hash", None) or getattr(
        config, "revision", None
    )
    if revision:
        return str(revision)
    return None


def _representative_words(topic_model: Any, topic_id: int) -> list[str]:
    topic = topic_model.get_topic(topic_id)
    if not topic:
        return []
    return [word for word, _ in topic]


def build_topic_model(config: TopicModelConfig):
    """Construct BERTopic with explicit seeded UMAP and recorded clustering."""
    from bertopic import BERTopic

    umap_model = build_umap_model(config)
    hdbscan_model = build_cluster_model(config)
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        embedding_model=config.embedding_model,
        language=config.language,
        calculate_probabilities=config.calculate_probabilities,
        nr_topics=config.nr_topics,
        verbose=config.verbose,
    )


def run_topic_model(
    config: TopicModelConfig,
    documents: Sequence[str],
    *,
    metadata: Optional[pd.DataFrame] = None,
    input_filename: str = "documents",
    input_sha256: Optional[str] = None,
    run_id: Optional[str] = None,
) -> TopicRunResult:
    """Fit one configured topic model. Does not evaluate multi-seed stability."""
    docs = ["" if doc is None else str(doc) for doc in documents]
    n_documents = len(docs)
    if n_documents == 0:
        raise TopicModelRunError("documents must not be empty")

    started_at = now_utc()
    run_id = run_id or new_topic_run_id()
    topic_model = build_topic_model(config)
    topics_raw, probs = topic_model.fit_transform(docs)
    topics = [int(topic) for topic in topics_raw]
    try:
        topic_probabilities = assigned_probabilities_from_topic_model(
            topic_model,
            docs,
            topics,
            probability_matrix=probs,
        )
    except ProbabilityAssignmentError as exc:
        raise TopicModelRunError(str(exc)) from exc

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info.copy()
    topic_info["representative_words"] = topic_info["Topic"].apply(
        lambda topic_id: _representative_words(topic_model, int(topic_id))
    )
    prob_frame = pd.DataFrame({"topic": topics, "prob": topic_probabilities})
    avg_probs = prob_frame.groupby("topic")["prob"].mean().rename("avg_probability")
    topic_info = topic_info.merge(
        avg_probs, left_on="Topic", right_index=True, how="left"
    )

    document_topics = pd.DataFrame(
        {
            "document_index": list(range(n_documents)),
            "topic": topics,
            "topic_probability": topic_probabilities,
        }
    )
    if metadata is not None:
        if len(metadata) != n_documents:
            raise TopicModelRunError("metadata length must match documents")
        for column in metadata.columns:
            document_topics[column] = metadata[column].values

    run_summary = summarize_run(config.random_seed, topics)
    snapshot = config_snapshot(config, vectorizer=vectorizer_model)
    vectorizer_hash = snapshot.get("vectorizer_configuration_hash") or sha256_json(
        vectorizer_identity(vectorizer_model)
    )
    checksum = assignment_checksum(topics)
    summary_checksum = sha256_json(topic_info.astype(str).to_dict(orient="list"))
    manifest = build_topic_run_manifest(
        run_id=run_id,
        started_at=started_at,
        completed_at=now_utc(),
        code_sha=topic_code_sha(),
        input_filename=input_filename,
        input_sha256=input_sha256 or sha256_json(docs),
        input_record_count=n_documents,
        seed=config.random_seed,
        bertopic_version=_bertopic_version(),
        embedding_model=config.embedding_model,
        embedding_model_revision=_embedding_revision(topic_model),
        umap_configuration=umap_kwargs(config),
        clustering_configuration=clustering_kwargs(config),
        vectorizer_configuration_hash=str(vectorizer_hash),
        nr_topics=config.nr_topics,
        nr_topics_mode=config.nr_topics_mode,
        discovered_inlier_topic_count=run_summary.inlier_topic_count,
        outlier_count=run_summary.outlier_count,
        outlier_rate=run_summary.outlier_rate,
        topic_assignment_checksum=checksum,
        topic_summary_checksum=summary_checksum,
        overall_status="success",
        calculate_probabilities=config.calculate_probabilities,
        language=config.language,
    )
    return TopicRunResult(
        run_id=run_id,
        seed=config.random_seed,
        topics=topics,
        topic_probabilities=topic_probabilities,
        topic_info=topic_info,
        document_topics=document_topics,
        manifest=manifest,
        config=config,
    )


def run_bertopic_model(df: pd.DataFrame, config: Optional[TopicModelConfig] = None):
    """Compatibility wrapper around :func:`run_topic_model`."""
    if config is None:
        config = load_topic_model_config()
    if "clean_text" not in df.columns:
        raise TopicModelRunError("input frame must include clean_text")
    docs = df["clean_text"].fillna("").astype(str).tolist()
    result = run_topic_model(config, docs, metadata=df.reset_index(drop=True))
    return None, result.topic_info, result.document_topics


def _load_input_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one configured BERTopic model on a local research corpus. "
            "Not part of make demo. Topic IDs are run-specific."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "topic_model.json")
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nr-topics", default=None)
    parser.add_argument("--text-column", default="clean_text")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.nr_topics is not None:
        overrides["nr_topics"] = args.nr_topics
    config = load_topic_model_config(args.config, overrides=overrides)

    input_path = Path(args.input)
    df = _load_input_frame(input_path)
    if args.text_column not in df.columns:
        print(f"missing text column {args.text_column!r}", file=sys.stderr)
        return 2
    docs = df[args.text_column].fillna("").astype(str).tolist()
    try:
        result = run_topic_model(
            config,
            docs,
            metadata=df.reset_index(drop=True),
            input_filename=input_path.name,
            input_sha256=sha256_file(input_path),
        )
    except (TopicModelRunError, ProbabilityAssignmentError) as exc:
        print(f"topic model error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "bertopic_post_topics.csv"
    summary_path = output_dir / "bertopic_topic_info.csv"
    manifest_path = output_dir / "topic_run_manifest.json"
    result.document_topics.to_csv(assignment_path, index=False)
    result.topic_info.to_csv(summary_path, index=False)
    result.manifest["topic_assignment_artifact_checksum"] = sha256_file(assignment_path)
    result.manifest["topic_summary_artifact_checksum"] = sha256_file(summary_path)
    write_topic_run_manifest(manifest_path, result.manifest)
    write_json(output_dir / "topic_model_config_snapshot.json", config_snapshot(config))
    print(
        "Wrote local topic-model artifacts. Topic IDs are specific to this run "
        "and are not scientific categories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
