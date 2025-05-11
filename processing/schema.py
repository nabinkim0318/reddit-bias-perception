# processing/schema.py

from typing import List, Literal

from pydantic import BaseModel, Field, ValidationError


class ClassificationResult(BaseModel):
    id: str
    subreddit: str
    clean_text: str
    pred_label: Literal["bias", "non-bias", "uncertain"]
    llm_reasoning: str


class FilteredAIBiasPost(BaseModel):
    id: str
    subreddit: str
    clean_text: str
    matched_keywords: List[str]
    bias_types: List[str]
