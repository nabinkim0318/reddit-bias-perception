"""Assigned-topic probability extraction.

BERTopic topic identifiers are labels. They must never be used as offsets
into a probability matrix merely because they are integers.

Prefer the per-document assigned probability exposed by
``topic_model.get_document_info(docs)["Probability"]``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

OUTLIER_TOPIC = -1
DOCUMENT_INFO_TOPIC_COLUMN = "Topic"
DOCUMENT_INFO_PROBABILITY_COLUMN = "Probability"


class ProbabilityAssignmentError(ValueError):
    """Raised when assigned-topic probabilities cannot be recovered safely."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def _as_optional_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    return float(value)


def _length(value: Any) -> Optional[int]:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            return 0
        return int(shape[0])
    try:
        return len(value)
    except TypeError:
        return None


def assigned_topic_probabilities(
    *,
    n_documents: int,
    topics: Sequence[int],
    assigned_probabilities: Optional[Sequence[Any]] = None,
    probability_matrix: Any = None,
) -> list[Optional[float]]:
    """Return the probability/confidence of the assigned topic per document.

    Parameters
    ----------
    n_documents:
        Expected document count.
    topics:
        Assigned topic labels, including ``-1`` outliers.
    assigned_probabilities:
        Per-document assigned-topic probabilities (document-info
        ``Probability`` or a 1-D ``probabilities_`` vector). Indexed by
        document position, never by topic id.
    probability_matrix:
        Optional 2-D matrix returned by ``fit_transform``. Provided only so
        callers can validate cardinality. Columns are **not** indexed by
        topic id.

    Notes
    -----
    Outlier topic ``-1`` is recorded as ``None``. Missing inlier
    probabilities are also ``None``; they are not imputed.
    """
    if n_documents < 0:
        raise ProbabilityAssignmentError("n_documents must be non-negative")
    if len(topics) != n_documents:
        raise ProbabilityAssignmentError(
            f"topic assignment length {len(topics)} != document count {n_documents}"
        )

    matrix_rows = _length(probability_matrix)
    matrix_ndim = None
    shape = getattr(probability_matrix, "shape", None)
    if shape is not None:
        matrix_ndim = len(shape)
        if matrix_rows is not None and matrix_rows != n_documents:
            raise ProbabilityAssignmentError(
                f"probability matrix rows {matrix_rows} != document count {n_documents}"
            )
    elif probability_matrix is not None:
        try:
            first = probability_matrix[0]
        except Exception:
            first = None
        if first is not None and not isinstance(first, (int, float)):
            matrix_ndim = 2
        else:
            matrix_ndim = 1

    if assigned_probabilities is None:
        if probability_matrix is not None and matrix_ndim == 2:
            raise ProbabilityAssignmentError(
                "Cannot recover assigned-topic probabilities by indexing a "
                "probability matrix with numeric topic IDs. Topic IDs may be "
                "non-contiguous or remapped. Use get_document_info Probability "
                "values (or a 1-D assigned-probability vector)."
            )
        if probability_matrix is not None and matrix_ndim == 1:
            assigned_probabilities = probability_matrix
        else:
            return [None] * n_documents

    if len(assigned_probabilities) != n_documents:
        raise ProbabilityAssignmentError(
            f"assigned probability length {len(assigned_probabilities)} != "
            f"document count {n_documents}"
        )

    result: list[Optional[float]] = []
    for topic, prob in zip(topics, assigned_probabilities):
        topic_id = int(topic)
        if topic_id == OUTLIER_TOPIC:
            result.append(None)
            continue
        result.append(_as_optional_float(prob))
    return result


def assigned_probabilities_from_document_info(
    document_info: Mapping[str, Any] | Any,
    *,
    n_documents: int,
    topics: Sequence[int],
) -> list[Optional[float]]:
    """Extract assigned probabilities from a BERTopic document-info table."""
    info_len = len(document_info)
    if info_len != n_documents:
        raise ProbabilityAssignmentError(
            f"document-info length {info_len} != document count {n_documents}"
        )
    if DOCUMENT_INFO_TOPIC_COLUMN in getattr(document_info, "columns", []):
        info_topics = [
            int(value) for value in document_info[DOCUMENT_INFO_TOPIC_COLUMN]
        ]
        if info_topics != [int(topic) for topic in topics]:
            raise ProbabilityAssignmentError(
                "document-info Topic column does not match fit_transform assignments"
            )
    if DOCUMENT_INFO_PROBABILITY_COLUMN not in getattr(document_info, "columns", []):
        raise ProbabilityAssignmentError(
            "document-info table has no Probability column; refusing to index "
            "a probability matrix by topic id"
        )
    return assigned_topic_probabilities(
        n_documents=n_documents,
        topics=topics,
        assigned_probabilities=list(document_info[DOCUMENT_INFO_PROBABILITY_COLUMN]),
    )


def assigned_probabilities_from_topic_model(
    topic_model: Any,
    docs: Sequence[str],
    topics: Sequence[int],
    *,
    probability_matrix: Any = None,
) -> list[Optional[float]]:
    """Use BERTopic's document-information API for assigned probabilities."""
    n_documents = len(docs)
    if len(topics) != n_documents:
        raise ProbabilityAssignmentError(
            f"topic assignment length {len(topics)} != document count {n_documents}"
        )
    if not hasattr(topic_model, "get_document_info"):
        raise ProbabilityAssignmentError(
            "topic model does not expose get_document_info; refusing to index "
            "probability columns by topic id"
        )
    document_info = topic_model.get_document_info(docs)
    assigned = assigned_probabilities_from_document_info(
        document_info,
        n_documents=n_documents,
        topics=topics,
    )
    if probability_matrix is not None:
        assigned_topic_probabilities(
            n_documents=n_documents,
            topics=topics,
            assigned_probabilities=assigned,
            probability_matrix=probability_matrix,
        )
    return assigned
