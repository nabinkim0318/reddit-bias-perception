"""Run-specific topic-id → category mapping.

A topic number is local to one fitted topic solution. Mappings must not be
silently reused across unrelated stochastic BERTopic runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from analysis.topic_probabilities import OUTLIER_TOPIC
from analysis.topic_stability import assignment_checksum

MAPPING_SCHEMA_VERSION = 1


class TopicMappingError(ValueError):
    """Raised when a topic-to-category mapping is incompatible with a run."""


@dataclass(frozen=True)
class TopicCategoryMapping:
    mapping_version: int
    topic_run_id: str
    topic_assignment_checksum: str
    labels: dict[int, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": {str(key): value for key, value in sorted(self.labels.items())},
            "mapping_schema_version": MAPPING_SCHEMA_VERSION,
            "mapping_version": self.mapping_version,
            "topic_assignment_checksum": self.topic_assignment_checksum,
            "topic_run_id": self.topic_run_id,
        }


def load_topic_category_mapping(path: Union[str, Path]) -> TopicCategoryMapping:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TopicMappingError("topic mapping must be a JSON object")
    return mapping_from_dict(payload)


def mapping_from_dict(payload: Mapping[str, Any]) -> TopicCategoryMapping:
    required = ("topic_run_id", "topic_assignment_checksum", "labels")
    missing = [key for key in required if key not in payload]
    if missing:
        raise TopicMappingError(
            "topic mapping missing required provenance fields: " + ", ".join(missing)
        )
    labels_raw = payload["labels"]
    if not isinstance(labels_raw, Mapping):
        raise TopicMappingError("labels must be an object of topic_id -> category")
    labels: dict[int, str] = {}
    for key, value in labels_raw.items():
        topic_id = int(key)
        if topic_id == OUTLIER_TOPIC:
            raise TopicMappingError("outlier topic -1 must not be mapped as a category")
        label = str(value).strip()
        if not label:
            raise TopicMappingError(f"empty category label for topic {topic_id}")
        labels[topic_id] = label
    return TopicCategoryMapping(
        mapping_version=int(payload.get("mapping_version", 1)),
        topic_run_id=str(payload["topic_run_id"]),
        topic_assignment_checksum=str(payload["topic_assignment_checksum"]),
        labels=labels,
    )


def validate_mapping_for_run(
    mapping: TopicCategoryMapping,
    *,
    topic_run_id: str,
    topic_assignment_checksum: str,
) -> None:
    if mapping.topic_run_id != topic_run_id:
        raise TopicMappingError(
            "topic-to-category mapping was created for a different topic run "
            f"(mapping run_id={mapping.topic_run_id!r}, current={topic_run_id!r})"
        )
    if mapping.topic_assignment_checksum != topic_assignment_checksum:
        raise TopicMappingError(
            "topic-to-category mapping checksum does not match this topic solution"
        )


def apply_topic_category_mapping(
    topics: Sequence[int],
    mapping: TopicCategoryMapping,
    *,
    topic_run_id: str,
    topic_assignment_checksum: Optional[str] = None,
) -> list[Optional[str]]:
    checksum = topic_assignment_checksum or assignment_checksum(topics)
    validate_mapping_for_run(
        mapping,
        topic_run_id=topic_run_id,
        topic_assignment_checksum=checksum,
    )
    result: list[Optional[str]] = []
    for topic in topics:
        topic_id = int(topic)
        if topic_id == OUTLIER_TOPIC:
            result.append(None)
            continue
        result.append(mapping.labels.get(topic_id))
    return result


def write_topic_category_mapping(path: Path, mapping: TopicCategoryMapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(mapping.to_dict(), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    return path
