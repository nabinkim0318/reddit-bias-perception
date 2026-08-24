"""Human-validation package.

Framework for sampling, blinding, agreement, and model-vs-human evaluation.
This package does not claim that a human-validation study has been completed.
"""

from validation.schema import (
    CODEBOOK_VERSION,
    CONSTRUCT_ID,
    CONSTRUCT_VERSION,
    PROTOCOL_VERSION,
    VALIDATION_SCHEMA_VERSION,
    HumanAnnotation,
    ReferenceAnnotation,
    ValidationInputError,
)

__all__ = [
    "CODEBOOK_VERSION",
    "CONSTRUCT_ID",
    "CONSTRUCT_VERSION",
    "PROTOCOL_VERSION",
    "VALIDATION_SCHEMA_VERSION",
    "HumanAnnotation",
    "ReferenceAnnotation",
    "ValidationInputError",
]
