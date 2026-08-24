"""Schemas for human-validation artifacts.

Human labels include uncertainty and must not reuse the automated
``ClassificationResult`` contract (success + yes/no only).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from processing.schema import AnnotationStatus, PredLabel

VALIDATION_SCHEMA_VERSION = 1
CONSTRUCT_ID = "visual_identity_bias_in_ai_generated_images"
CONSTRUCT_VERSION = "1"
CODEBOOK_VERSION = "1"
PROTOCOL_VERSION = "1"

HUMAN_LABELS = ("yes", "no", "uncertain", "insufficient_context")
BINARY_LABELS = ("yes", "no")

HumanLabel = Literal["yes", "no", "uncertain", "insufficient_context"]
BinaryLabel = Literal["yes", "no"]
ResolutionStatus = Literal["agreed", "adjudicated", "unresolved"]
AdjudicationStatus = Literal["resolved", "unresolved"]

BLINDED_ANNOTATOR_COLUMNS = (
    "task_id",
    "text_to_annotate",
    "human_label",
    "notes",
)

FORBIDDEN_ANNOTATOR_FIELDS = frozenset(
    {
        "pred_label",
        "model_pred_label",
        "status",
        "model_status",
        "raw_output",
        "llm_reasoning",
        "original_record_id",
        "id",
        "error_type",
        "expected",
        "expected_label",
        "gold",
        "gold_label",
    }
)

SAMPLING_INDEX_COLUMNS = (
    "task_id",
    "original_record_id",
    "model_status",
    "model_pred_label",
    "subreddit",
    "subreddit_group",
    "keyword_category",
    "annotation_status",
    "strata_key",
)


class ValidationInputError(ValueError):
    """Malformed, incomplete, or inconsistent validation inputs."""


class HumanAnnotation(BaseModel):
    """One annotator's label for a blinded task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    label: HumanLabel
    notes: Optional[str] = None

    @field_validator("task_id")
    @classmethod
    def _strip_task_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("task_id must be non-empty")
        return stripped


class SamplingIndexRow(BaseModel):
    """Private linkage from blinded task_id back to the model artifact."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    original_record_id: str = Field(min_length=1)
    model_status: AnnotationStatus
    model_pred_label: Optional[PredLabel] = None
    subreddit: str = ""
    subreddit_group: str = "unknown"
    keyword_category: str = "unknown"
    annotation_status: AnnotationStatus
    strata_key: str = ""

    @model_validator(mode="after")
    def _status_label_contract(self) -> "SamplingIndexRow":
        if self.model_status == "success":
            if self.model_pred_label not in BINARY_LABELS:
                raise ValueError(
                    "success sampling rows require model_pred_label yes/no"
                )
        elif self.model_pred_label is not None:
            raise ValueError("failure sampling rows must have model_pred_label=None")
        if self.annotation_status != self.model_status:
            raise ValueError("annotation_status must match model_status")
        return self


class AdjudicationRecord(BaseModel):
    """Optional third-party resolution. Does not overwrite A/B labels."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    annotator_a_label: HumanLabel
    annotator_b_label: HumanLabel
    adjudicated_label: Optional[HumanLabel] = None
    adjudication_status: AdjudicationStatus

    @model_validator(mode="after")
    def _adjudication_contract(self) -> "AdjudicationRecord":
        if self.adjudication_status == "resolved":
            if self.adjudicated_label is None:
                raise ValueError("resolved adjudication requires adjudicated_label")
        return self


class ReferenceAnnotation(BaseModel):
    """Resolved human reference used for model-vs-human metrics."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    reference_label: Optional[BinaryLabel] = None
    resolution_status: ResolutionStatus

    @model_validator(mode="after")
    def _reference_contract(self) -> "ReferenceAnnotation":
        if self.resolution_status == "unresolved":
            if self.reference_label is not None:
                raise ValueError("unresolved items must not carry a binary reference")
        elif self.reference_label not in BINARY_LABELS:
            raise ValueError("agreed/adjudicated references require yes/no")
        return self
