"""Topic-model configuration contract and seeded UMAP kwargs."""

from __future__ import annotations

from analysis.topic_config import (
    DEFAULT_STABILITY_SEEDS,
    TopicModelConfig,
    clustering_kwargs,
    config_snapshot,
    load_topic_model_config,
    umap_kwargs,
    with_seed,
)


def test_default_config_records_exploratory_auto_topics():
    config = load_topic_model_config()
    assert config.nr_topics == "auto"
    assert config.nr_topics_mode == "exploratory_auto"
    assert config.uses_exploratory_auto_topics is True
    assert config.stability_seeds == DEFAULT_STABILITY_SEEDS


def test_configured_seed_propagates_into_umap_kwargs():
    config = TopicModelConfig(random_seed=53)
    kwargs = umap_kwargs(config)
    assert kwargs["random_state"] == 53
    assert "random_state" not in clustering_kwargs(config)
    snapshot = config_snapshot(config)
    assert snapshot["umap"]["random_state"] == 53
    assert snapshot["clustering"]["accepts_random_state"] is False


def test_with_seed_does_not_change_nr_topics_mode():
    config = load_topic_model_config()
    seeded = with_seed(config, 71)
    assert seeded.random_seed == 71
    assert seeded.nr_topics == "auto"
    assert umap_kwargs(seeded)["random_state"] == 71


def test_fixed_nr_topics_is_available_for_controlled_runs():
    config = load_topic_model_config(overrides={"nr_topics": 12, "random_seed": 11})
    assert config.nr_topics == 12
    assert config.nr_topics_mode == "fixed"
    assert config.uses_exploratory_auto_topics is False
