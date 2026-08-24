# processing/schema.py

from typing import List, Literal, Optional

from pydantic import BaseModel, model_validator

ANNOTATION_SCHEMA_VERSION = 2

AnnotationStatus = Literal["success", "parse_error", "model_error"]
PredLabel = Literal["yes", "no"]


class ClassificationResult(BaseModel):
    """One automated annotation.

    ``pred_label`` is the scientific yes/no construct and is set only when
    ``status == "success"``. Unsuccessful runs leave ``pred_label`` null.
    Automated labels are model predictions, not human-validated ground truth.
    The operational construct is documented in docs/annotation_codebook.md.
    """

    id: str
    subreddit: str
    clean_text: str
    status: AnnotationStatus
    pred_label: Optional[PredLabel] = None
    error_type: Optional[str] = None
    llm_reasoning: str = ""
    raw_output: str = ""
    schema_version: int = ANNOTATION_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_label_status_contract(self) -> "ClassificationResult":
        if self.status == "success":
            if self.pred_label not in {"yes", "no"}:
                raise ValueError("success requires pred_label of 'yes' or 'no'")
        elif self.pred_label is not None:
            raise ValueError("unsuccessful annotations must have pred_label=None")
        return self


class FilteredAIBiasPost(BaseModel):
    id: str
    subreddit: str
    clean_text: str
    matched_keywords: List[str]
    bias_types: List[str]
