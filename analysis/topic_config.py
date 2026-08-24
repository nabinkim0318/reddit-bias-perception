"""Explicit BERTopic configuration contract.

Stochastic settings are recorded rather than left as hidden library defaults.
``nr_topics="auto"`` remains an exploratory choice: it is not a canonical
topic solution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from processing.hashing import sha256_json

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "topic_model.json"
)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_STABILITY_SEEDS: tuple[int, ...] = (11, 23, 37, 53, 71)
NR_TOPICS_AUTO = "auto"


@dataclass(frozen=True)
class UMAPSettings:
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"


@dataclass(frozen=True)
class ClusteringSettings:
    algorithm: str = "hdbscan"
    min_cluster_size: int = 10
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True
    accepts_random_state: bool = False
    notes: str = (
        "HDBSCAN is treated as deterministic given UMAP embeddings. "
        "The supported APIs do not take random_state."
    )


@dataclass(frozen=True)
class TopicModelConfig:
    """Run-level topic-model contract.

    ``nr_topics="auto"`` is exploratory. A single automatically reduced
    solution is not treated as canonical without a stability evaluation.
    """

    random_seed: int = 42
    nr_topics: Union[str, int] = NR_TOPICS_AUTO
    calculate_probabilities: bool = True
    language: str = "english"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    umap: UMAPSettings = field(default_factory=UMAPSettings)
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)
    vectorizer_source: str = "utils.archive_vectorizer_config.vectorizer_model"
    stability_seeds: tuple[int, ...] = DEFAULT_STABILITY_SEEDS
    verbose: bool = False

    @property
    def nr_topics_mode(self) -> str:
        if self.nr_topics == NR_TOPICS_AUTO or self.nr_topics == "auto":
            return "exploratory_auto"
        return "fixed"

    @property
    def uses_exploratory_auto_topics(self) -> bool:
        return self.nr_topics_mode == "exploratory_auto"


def parse_nr_topics(value: Any) -> Union[str, int]:
    if value is None:
        return NR_TOPICS_AUTO
    if isinstance(value, bool):
        raise ValueError("nr_topics must be 'auto' or a positive integer")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("fixed nr_topics must be a positive integer")
        return value
    text = str(value).strip()
    if text.lower() == NR_TOPICS_AUTO:
        return NR_TOPICS_AUTO
    if text.isdigit():
        parsed = int(text)
        if parsed < 1:
            raise ValueError("fixed nr_topics must be a positive integer")
        return parsed
    raise ValueError("nr_topics must be 'auto' or a positive integer")


def parse_stability_seeds(value: Any) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_STABILITY_SEEDS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        seeds = tuple(int(part) for part in parts)
    else:
        seeds = tuple(int(item) for item in value)
    if not seeds:
        raise ValueError("stability_seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("stability_seeds must be unique")
    return seeds


def load_topic_model_config(
    path: Optional[Union[str, Path]] = None,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> TopicModelConfig:
    payload: dict[str, Any] = {}
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("topic-model config must be a JSON object")
        payload.update(loaded)
    if overrides:
        payload.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

    umap_raw = payload.get("umap") or {}
    clustering_raw = payload.get("clustering") or {}
    vectorizer_raw = payload.get("vectorizer") or {}
    return TopicModelConfig(
        random_seed=int(payload.get("random_seed", 42)),
        nr_topics=parse_nr_topics(payload.get("nr_topics", NR_TOPICS_AUTO)),
        calculate_probabilities=bool(payload.get("calculate_probabilities", True)),
        language=str(payload.get("language", "english")),
        embedding_model=str(payload.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
        umap=UMAPSettings(
            n_neighbors=int(umap_raw.get("n_neighbors", 15)),
            n_components=int(umap_raw.get("n_components", 5)),
            min_dist=float(umap_raw.get("min_dist", 0.0)),
            metric=str(umap_raw.get("metric", "cosine")),
        ),
        clustering=ClusteringSettings(
            algorithm=str(clustering_raw.get("algorithm", "hdbscan")),
            min_cluster_size=int(clustering_raw.get("min_cluster_size", 10)),
            metric=str(clustering_raw.get("metric", "euclidean")),
            cluster_selection_method=str(
                clustering_raw.get("cluster_selection_method", "eom")
            ),
            prediction_data=bool(clustering_raw.get("prediction_data", True)),
            accepts_random_state=bool(
                clustering_raw.get("accepts_random_state", False)
            ),
            notes=str(
                clustering_raw.get(
                    "notes",
                    ClusteringSettings().notes,
                )
            ),
        ),
        vectorizer_source=str(
            vectorizer_raw.get("source", TopicModelConfig().vectorizer_source)
        ),
        stability_seeds=parse_stability_seeds(
            payload.get("stability_seeds", DEFAULT_STABILITY_SEEDS)
        ),
        verbose=bool(payload.get("verbose", False)),
    )


def umap_kwargs(config: TopicModelConfig) -> dict[str, Any]:
    """Keyword arguments for constructing a seeded UMAP model.

    UMAP is the stochastic dimensionality-reduction step. ``random_state`` is
    set from the topic-model seed. Do not assume other components accept a seed.
    """
    return {
        "n_neighbors": config.umap.n_neighbors,
        "n_components": config.umap.n_components,
        "min_dist": config.umap.min_dist,
        "metric": config.umap.metric,
        "random_state": config.random_seed,
    }


def clustering_kwargs(config: TopicModelConfig) -> dict[str, Any]:
    """Keyword arguments for HDBSCAN. No random_state is invented."""
    params: dict[str, Any] = {
        "min_cluster_size": config.clustering.min_cluster_size,
        "metric": config.clustering.metric,
        "cluster_selection_method": config.clustering.cluster_selection_method,
    }
    if config.clustering.prediction_data:
        params["prediction_data"] = True
    return params


def build_umap_model(config: TopicModelConfig) -> Any:
    from umap import UMAP

    return UMAP(**umap_kwargs(config))


def build_cluster_model(config: TopicModelConfig) -> Any:
    params = clustering_kwargs(config)
    try:
        from hdbscan import HDBSCAN

        return HDBSCAN(**params)
    except ImportError:
        from sklearn.cluster import HDBSCAN

        params.pop("prediction_data", None)
        return HDBSCAN(**params)


def vectorizer_identity(vectorizer: Any) -> dict[str, Any]:
    stop_words = getattr(vectorizer, "stop_words", None)
    if stop_words is None:
        stop_list: Optional[list[str]] = None
    else:
        stop_list = sorted(str(word) for word in stop_words)
    identity = {
        "class_name": type(vectorizer).__name__,
        "min_df": getattr(vectorizer, "min_df", None),
        "ngram_range": list(getattr(vectorizer, "ngram_range", (1, 1))),
        "source": "utils.archive_vectorizer_config.vectorizer_model",
        "stop_words_count": len(stop_list) if stop_list is not None else 0,
        "stop_words_sha256": sha256_json(stop_list) if stop_list is not None else None,
    }
    return identity


def config_snapshot(
    config: TopicModelConfig, *, vectorizer: Any = None
) -> dict[str, Any]:
    snapshot = {
        "calculate_probabilities": config.calculate_probabilities,
        "clustering": asdict(config.clustering),
        "embedding_model": config.embedding_model,
        "language": config.language,
        "nr_topics": config.nr_topics,
        "nr_topics_mode": config.nr_topics_mode,
        "random_seed": config.random_seed,
        "stability_seeds": list(config.stability_seeds),
        "umap": {**asdict(config.umap), "random_state": config.random_seed},
        "vectorizer_source": config.vectorizer_source,
    }
    if vectorizer is not None:
        identity = vectorizer_identity(vectorizer)
        snapshot["vectorizer"] = identity
        snapshot["vectorizer_configuration_hash"] = sha256_json(identity)
    return snapshot


def with_seed(config: TopicModelConfig, seed: int) -> TopicModelConfig:
    return TopicModelConfig(
        random_seed=int(seed),
        nr_topics=config.nr_topics,
        calculate_probabilities=config.calculate_probabilities,
        language=config.language,
        embedding_model=config.embedding_model,
        umap=config.umap,
        clustering=config.clustering,
        vectorizer_source=config.vectorizer_source,
        stability_seeds=config.stability_seeds,
        verbose=config.verbose,
    )


def format_seeds(seeds: Sequence[int]) -> str:
    return ",".join(str(seed) for seed in seeds)
