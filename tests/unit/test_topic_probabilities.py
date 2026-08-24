"""Assigned-topic probabilities must not use topic IDs as array offsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.topic_probabilities import (
    ProbabilityAssignmentError,
    assigned_probabilities_from_document_info,
    assigned_probabilities_from_topic_model,
    assigned_topic_probabilities,
)


def _legacy_index_by_topic_id(probs, topics):
    return [p[t] if p is not None and t != -1 else None for p, t in zip(probs, topics)]


def test_noncontiguous_topic_ids_are_not_probability_offsets():
    topics = [-1, 2, 7]
    matrix = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
            [0.70, 0.80, 0.90],
        ]
    )
    assigned = [0.01, 0.55, 0.88]
    result = assigned_topic_probabilities(
        n_documents=3,
        topics=topics,
        assigned_probabilities=assigned,
        probability_matrix=matrix,
    )
    assert result[0] is None
    assert result[1] == pytest.approx(0.55)
    assert result[2] == pytest.approx(0.88)
    assert result[1] != pytest.approx(matrix[1, 2])
    with pytest.raises(IndexError):
        _legacy_index_by_topic_id(matrix, topics)


def test_probability_matrix_without_assigned_values_is_rejected():
    topics = [-1, 2, 7]
    matrix = np.zeros((3, 3))
    with pytest.raises(ProbabilityAssignmentError, match="topic IDs"):
        assigned_topic_probabilities(
            n_documents=3,
            topics=topics,
            probability_matrix=matrix,
        )


def test_cardinality_mismatch_fails_clearly():
    with pytest.raises(ProbabilityAssignmentError, match="length"):
        assigned_topic_probabilities(
            n_documents=3,
            topics=[-1, 2],
            assigned_probabilities=[0.1, 0.2, 0.3],
        )
    with pytest.raises(ProbabilityAssignmentError, match="probability length"):
        assigned_topic_probabilities(
            n_documents=2,
            topics=[0, 1],
            assigned_probabilities=[0.1],
        )
    with pytest.raises(ProbabilityAssignmentError, match="rows"):
        assigned_topic_probabilities(
            n_documents=2,
            topics=[0, 1],
            assigned_probabilities=[0.1, 0.2],
            probability_matrix=np.zeros((3, 2)),
        )


def test_outlier_and_missing_inlier_probabilities():
    result = assigned_topic_probabilities(
        n_documents=3,
        topics=[-1, 0, 2],
        assigned_probabilities=[0.9, None, float("nan")],
    )
    assert result == [None, None, None]


def test_document_info_probability_column_is_used():
    topics = [-1, 2, 7]
    document_info = pd.DataFrame(
        {
            "Topic": topics,
            "Probability": [0.0, 0.55, 0.88],
        }
    )
    result = assigned_probabilities_from_document_info(
        document_info, n_documents=3, topics=topics
    )
    assert result == [None, pytest.approx(0.55), pytest.approx(0.88)]


def test_document_info_topic_mismatch_and_missing_probability():
    topics = [0, 1]
    with pytest.raises(ProbabilityAssignmentError, match="does not match"):
        assigned_probabilities_from_document_info(
            pd.DataFrame({"Topic": [0, 2], "Probability": [0.1, 0.2]}),
            n_documents=2,
            topics=topics,
        )
    with pytest.raises(ProbabilityAssignmentError, match="no Probability"):
        assigned_probabilities_from_document_info(
            pd.DataFrame({"Topic": [0, 1]}),
            n_documents=2,
            topics=topics,
        )


class _FakeTopicModel:
    def __init__(self, topics, probabilities):
        self._topics = topics
        self._probabilities = probabilities

    def get_document_info(self, docs):
        assert len(docs) == len(self._topics)
        return pd.DataFrame(
            {
                "Document": list(docs),
                "Topic": self._topics,
                "Probability": self._probabilities,
            }
        )


def test_topic_model_document_info_api_is_preferred_over_matrix():
    topics = [-1, 2, 7]
    docs = ["alpha", "beta", "gamma"]
    model = _FakeTopicModel(topics, [0.0, 0.55, 0.88])
    matrix = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60],
            [0.70, 0.80, 0.90],
        ]
    )
    result = assigned_probabilities_from_topic_model(
        model, docs, topics, probability_matrix=matrix
    )
    assert result[0] is None
    assert result[1] == pytest.approx(0.55)
    assert result[2] == pytest.approx(0.88)
