"""LLM annotation schema tests. Offline; no Hugging Face model is loaded."""

import pytest
from pydantic import ValidationError

from processing.schema import ClassificationResult


def test_success_yes_no_accepted():
    row = ClassificationResult(
        id="SYNTH-ANN-1",
        subreddit="synthetic",
        clean_text="synthetic item",
        status="success",
        pred_label="yes",
        llm_reasoning="",
        raw_output="yes",
    )
    assert row.pred_label == "yes"
    assert row.status == "success"


def test_parse_error_requires_null_pred_label():
    row = ClassificationResult(
        id="SYNTH-ANN-2",
        subreddit="synthetic",
        clean_text="synthetic item",
        status="parse_error",
        pred_label=None,
        error_type="malformed_output",
    )
    assert row.pred_label is None

    with pytest.raises(ValidationError):
        ClassificationResult(
            id="SYNTH-ANN-3",
            subreddit="synthetic",
            clean_text="synthetic item",
            status="parse_error",
            pred_label="no",
        )


def test_model_error_requires_null_pred_label():
    with pytest.raises(ValidationError):
        ClassificationResult(
            id="SYNTH-ANN-4",
            subreddit="synthetic",
            clean_text="synthetic item",
            status="model_error",
            pred_label="no",
        )
