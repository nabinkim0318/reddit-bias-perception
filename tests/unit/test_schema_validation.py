"""## tests/unit/test_schema_validation.py

import ast
import json
import os
from typing import List, Literal, Optional

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from config.config import TOPIC_ASSIGNMENT_PATH, TOPIC_OUTPUT


def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_csv(path):
    return pd.read_csv(path)


# Reusable function to safely evaluate literal lists
def safe_eval_list(val):
    try:
        result = ast.literal_eval(val)
        return isinstance(result, list)
    except Exception:
        return False


@pytest.mark.skipif(
    not os.path.exists("data/raw/reddit_raw.json"), reason="Missing raw data"
)
def test_reddit_raw_schema():
    data = safe_load_json("data/raw/reddit_raw.json")
    required_keys = {
        "id",
        "subreddit",
        "title",
        "selftext",
        "comments",
        "score",
        "num_comments",
        "upvote_ratio",
        "created_utc",
    }
    for post in data:
        assert required_keys.issubset(
            post.keys()
        ), f"Missing keys in post: {set(post.keys())}"
        assert isinstance(post["id"], str)
        assert isinstance(post["comments"], list)
        assert 0.0 <= post["upvote_ratio"] <= 1.0


def test_reddit_bias_data_clean_schema():
    path = "data/processed/reddit_bias_data_clean.csv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pytest.skip("Missing or empty cleaned dataset")
    df = safe_load_csv(path)
    required_columns = {
        "id",
        "subreddit",
        "clean_text",
        "full_text",
        "comments",
        "top_comments",
        "score",
        "num_comments",
        "upvote_ratio",
        "flair",
        "created_utc",
        "title",
        "selftext",
    }
    assert required_columns.issubset(
        df.columns
    ), f"Missing columns: {required_columns - set(df.columns)}"
    assert df["clean_text"].apply(lambda x: isinstance(x, str)).all()
    assert df["comments"].apply(safe_eval_list).all()


class FilteredAIBiasPost(BaseModel):
    id: str
    subreddit: str
    clean_text: str
    matched_keywords: List[str]
    bias_types: List[str]


def test_filtered_ai_bias_output_schema():
    path = "data/processed/filtered_ai_bias.csv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pytest.skip("Filtered AI bias output not found or empty")

    df = safe_load_csv(path)
    for col in ["matched_keywords", "bias_types"]:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    for i, record in df.iterrows():
        try:
            FilteredAIBiasPost(**record.to_dict())
        except ValidationError as e:
            pytest.fail(f"Row {i} failed schema validation: {e}")


# ------------------ Pydantic Models for Topic Modeling ---------------------


class TopicInfo(BaseModel):
    topic: int
    count: int
    representative_words: List[str]


class PostTopic(BaseModel):
    id: str
    subreddit: str
    clean_text: str
    topic: int
    topic_probability: float
    bias_types: List[str]
    score: int
    num_comments: int
    upvote_ratio: float
    flair: Optional[str]
    created_utc: float


# --------------------- Helper Functions ---------------------


def safe_eval_list(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except Exception:
        return val  # Let Pydantic catch it


# --------------------- Tests ---------------------


def test_topic_info_schema():
    if not os.path.exists(TOPIC_OUTPUT):
        pytest.skip("Missing topic info output")

    df = pd.read_csv(TOPIC_OUTPUT)
    df["representative_words"] = df["representative_words"].apply(safe_eval_list)

    for i, row in df.iterrows():
        try:
            TopicInfo(**row.to_dict())
        except ValidationError as e:
            pytest.fail(f"Row {i} failed schema validation: {e}")


def test_post_topics_schema():
    if not os.path.exists(TOPIC_ASSIGNMENT_PATH):
        pytest.skip("Missing post-topic assignment output")

    df = pd.read_csv(TOPIC_ASSIGNMENT_PATH)
    df["bias_types"] = df["bias_types"].apply(safe_eval_list)

    for i, row in df.iterrows():
        try:
            PostTopic(**row.to_dict())
        except ValidationError as e:
            pytest.fail(f"Row {i} failed schema validation: {e}")


def test_sentiment_labeled_schema():
    path = "data/results/sentiment_labeled.csv"
    if not os.path.exists(path):
        pytest.skip("Missing sentiment output")
    df = safe_load_csv(path)
    for col in ["goemotions_probs", "vader"]:
        assert df[col].apply(safe_eval_list).all()
    if "top_comment_sentiment" in df.columns:
        assert df["top_comment_sentiment"].apply(safe_eval_list).all()


class ClassificationResult(BaseModel):
    id: str
    subreddit: str
    clean_text: str
    pred_label: Literal["bias", "non-bias", "uncertain"]
    llm_reasoning: str


def test_fewshot_classification_counts_and_schema(caplog):
    path = "data/processed/fewshot_classification_results.csv"
    if not os.path.exists(path):
        pytest.skip("Few-shot output not found")

    df = safe_load_csv(path)

    # Check required columns exist
    required_columns = {"id", "subreddit", "clean_text", "pred_label", "llm_reasoning"}
    assert required_columns.issubset(
        df.columns
    ), f"Missing columns: {required_columns - set(df.columns)}"

    # Check valid pred_label values
    valid_labels = {"bias", "non-bias", "uncertain"}
    assert set(df["pred_label"].dropna().unique()).issubset(valid_labels)

    # Optionally validate one or more rows
    sample = df.iloc[0].to_dict()
    ClassificationResult(**sample)  # Raises if schema violated

    # Check distribution of results
    label_counts = df["pred_label"].value_counts().to_dict()
    print("🔢 Classification label distribution:")
    for label in valid_labels:
        count = label_counts.get(label, 0)
        print(f" - {label}: {count}")
        assert count >= 0

    # Optional: assert at least some bias-related results exist
    assert label_counts.get("bias", 0) > 0, "No 'bias' classifications found"

    # Optional log capture
    if caplog.text:
        print("📋 Log output during test:\n", caplog.text)
"""
