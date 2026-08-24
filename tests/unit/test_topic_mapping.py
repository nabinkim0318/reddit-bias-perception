"""Topic-to-category mappings are bound to a specific topic run."""

from __future__ import annotations

import pytest

from analysis.topic_mapping import (
    TopicMappingError,
    apply_topic_category_mapping,
    mapping_from_dict,
)
from analysis.topic_stability import assignment_checksum


def _mapping(checksum: str, run_id: str = "synthetic-topic-run-seed-11"):
    return mapping_from_dict(
        {
            "labels": {"0": "portrayal_stereotype", "2": "policy_discussion"},
            "mapping_version": 1,
            "topic_assignment_checksum": checksum,
            "topic_run_id": run_id,
        }
    )


def test_mapping_applies_when_run_identity_matches():
    topics = [0, 2, -1, 0]
    checksum = assignment_checksum(topics)
    mapped = apply_topic_category_mapping(
        topics,
        _mapping(checksum),
        topic_run_id="synthetic-topic-run-seed-11",
        topic_assignment_checksum=checksum,
    )
    assert mapped == [
        "portrayal_stereotype",
        "policy_discussion",
        None,
        "portrayal_stereotype",
    ]


def test_mismatched_topic_run_is_rejected():
    topics = [0, 2]
    checksum = assignment_checksum(topics)
    with pytest.raises(TopicMappingError, match="different topic run"):
        apply_topic_category_mapping(
            topics,
            _mapping(checksum, run_id="other-run"),
            topic_run_id="synthetic-topic-run-seed-11",
            topic_assignment_checksum=checksum,
        )


def test_mismatched_assignment_checksum_is_rejected():
    topics = [0, 2]
    mapping = _mapping(assignment_checksum([0, 2, 2]))
    with pytest.raises(TopicMappingError, match="checksum"):
        apply_topic_category_mapping(
            topics,
            mapping,
            topic_run_id="synthetic-topic-run-seed-11",
            topic_assignment_checksum=assignment_checksum(topics),
        )


def test_mapping_without_provenance_fields_is_rejected():
    with pytest.raises(TopicMappingError, match="provenance"):
        mapping_from_dict({"labels": {"0": "portrayal_stereotype"}})
